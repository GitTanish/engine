"""Battle Evaluation Suite — statistically-backed ablation evidence.

Runs seeded headless episodes across scenario x variant cells and produces:
  - eval_results.csv        per-episode metrics (raw evidence)
  - eval_summary.json       aggregate stats + Defense Effectiveness Score
  - EVAL_REPORT.md          human-readable report with the score table

Composite Defense Effectiveness Score (0-100), weights documented:
  DES = 100 * ( 0.40 * win_rate
              + 0.20 * survival_rate      (1 - losses/defenders)
              + 0.15 * speed_score        (1 - mean_resolve_s / T_CAP)
              + 0.15 * integrity_score    (exp(-penetration_m / 50))
              + 0.10 * exchange_ratio     (kills / (kills + losses)) )

Variants (ablations, one factor off at a time):
  full            hungarian + PN + kalman prediction   (shipped default)
  no_kalman       constant-velocity aimpoint instead of KF predict
  pursuit         lead pursuit steering instead of PN
  greedy_assign   legacy swarming allocator instead of Hungarian
  legacy_all      greedy + pursuit + no prediction     (pre-Tier1 stack)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from gym_pybullet_drones.envs.BaseDefenseAviary import BaseDefenseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DefenseAgent import DefenseAgent
from gym_pybullet_drones.utils.tlog import JsonlLogger

SCENARIOS = {
    # hit_rate_per_sec: tuned DOWN from the shipped 12.3/s so battles last
    # long enough for guidance/assignment to matter (evidence: smoke run
    # showed all variants identical — battles resolved in <1s, inside
    # warm-up hover, purely by dice).
    "close_quarters": {"num_defenders": 5, "num_enemies": 3, "spawn_radius": 80.0,
                       "ratio": 0.5, "hit_rate": 2.5, "max_s": 8},
    "standard":       {"num_defenders": 8, "num_enemies": 5, "spawn_radius": 260.0,
                       "ratio": 0.4, "hit_rate": 0.6, "max_s": 12},
    "far_field":      {"num_defenders": 10, "num_enemies": 6, "spawn_radius": 320.0,
                       "ratio": 0.34, "hit_rate": 0.35, "max_s": 16},
}
VARIANTS = {
    "full":         dict(assignment_mode='hungarian', guidance='pn', use_prediction=True),
    "no_kalman":    dict(assignment_mode='hungarian', guidance='pn', use_prediction=False),
    "pursuit":      dict(assignment_mode='hungarian', guidance='pursuit', use_prediction=True),
    "greedy_assign":dict(assignment_mode='greedy', guidance='pn', use_prediction=True),
    "legacy_all":   dict(assignment_mode='greedy', guidance='pursuit', use_prediction=False),
}
T_CAP_S = 120.0


def run_episode(cfg, variant, seed):
    rng_seed = seed
    np.random.seed(rng_seed)
    env = BaseDefenseAviary(drone_model=DroneModel.HEAVY,
                            num_drones=cfg["num_defenders"],
                            physics=Physics.PYB, gui=False, record=False,
                            obstacles=False, neighbourhood_radius=100.0)
    # Scenario-tuned lethality (see SCENARIOS note): override the shipped rate
    env.HIT_RATE_PER_SEC = cfg.get("hit_rate", 12.31)
    max_steps = int(cfg.get("max_s", 20) / env.CTRL_TIMESTEP)
    agents = [DefenseAgent(i, cfg["num_defenders"], verbose=False, **variant)
              for i in range(cfg["num_defenders"])]
    obs, _ = env.reset(scenario_config={"num_enemies": cfg["num_enemies"],
                                        "spawn_radius": cfg["spawn_radius"],
                                        "ground_attacker_ratio": cfg["ratio"]})
    kills = losses = splash = rams = 0
    min_pen = float('inf')
    dup_assign_steps = 0        # defenders sharing one enemy in the same tick
    jerk_sum = jerk_n = 0       # control smoothness: ||Δv_set|| between ticks
    kal_se, kal_n = 0.0, 0      # Kalman velocity error accumulation
    prev_vset = {}
    outcome, steps = "TIMEOUT", max_steps
    for step in range(max_steps):
        states = [env._getDroneStateVector(k) for k in range(env.NUM_DRONES)]
        neighbors = [{'id': k, 'pos': states[k][0:3]} for k in range(env.NUM_DRONES)]
        enemies = [{'id': k, 'uid': env.enemy_uids[k], 'pos': env.enemy_positions[k],
                    'vel': env.enemy_velocities[k],
                    'mode': env.enemy_agents[k].mode}
                   for k in range(len(env.enemy_positions))]
        action = np.zeros((env.NUM_DRONES, 4))
        assigned_now = []
        for k, a in enumerate(agents):
            my = {'pos': states[k][0:3], 'vel': states[k][10:13],
                  'quat': states[k][3:7], 'ang_vel': states[k][13:16]}
            action[k] = a.compute_action(my, neighbors, enemies, env.BASE_POS,
                                         env.CTRL_TIMESTEP, step * env.CTRL_TIMESTEP)
            uid = getattr(a.strategy, 'last_uid', None)
            if uid is not None:
                assigned_now.append(uid)
            vs = getattr(a, 'last_vset', None)
            if vs is not None and k in prev_vset:
                jerk_sum += float(np.linalg.norm(vs - prev_vset[k])); jerk_n += 1
            if vs is not None:
                prev_vset[k] = vs
        if enemies and agents[0].tracker.tracks:
            for e in enemies:
                est = agents[0].tracker.velocity(e['uid'])
                if est is not None:
                    kal_se += float(np.sum((est - np.asarray(e['vel']))**2)); kal_n += 1
        dup_assign_steps += len(assigned_now) - len(set(assigned_now))
        if len(enemies):
            min_pen = min(min_pen, float(np.min(
                np.linalg.norm(env.enemy_positions[:, :2] - env.BASE_POS[:2], axis=1))))
        obs, _, _, _, info = env.step(action)
        evs = info.get("combat_events", [])
        kills += len(evs)
        splash += sum(e['type'] == 'splash' for e in evs)
        rams += sum(e['type'] == 'ram' for e in evs)
        losses = len(env.dead_friendly_indices)
        if info.get("asset_hit"):
            outcome, steps = "ASSET_HIT", step + 1
            break
        if losses >= cfg["num_defenders"]:
            outcome, steps = "WIPEOUT", step + 1
            break
        if len(env.enemy_positions) == 0:
            outcome, steps = "VICTORY", step + 1
            break
    else:
        outcome, steps = "TIMEOUT", max_steps
    env.close()
    return {
        "outcome": outcome, "steps": steps,
        "resolve_s": round(steps * env.CTRL_TIMESTEP, 3),
        "kills": kills, "losses": int(losses), "splash": splash, "rams": rams,
        "min_penetration_m": None if min_pen is float('inf') else round(min_pen, 2),
        "dup_assign_steps": dup_assign_steps,
        "jerk_mean": round(jerk_sum / jerk_n, 3) if jerk_n else None,
        "kalman_rmse": round(float(np.sqrt(kal_se / kal_n)), 3) if kal_n else None,
    }


def aggregate(df):
    g = df.groupby(['scenario', 'variant'])
    rows = []
    for (sc, va), d in g:
        n = len(d)
        win = (d.outcome == 'VICTORY').mean()
        rows.append({
            'scenario': sc, 'variant': va, 'episodes': n,
            'win_rate': round(win, 3),
            'mean_losses': round(d.losses.mean(), 2),
            'mean_kills': round(d.kills.mean(), 2),
            'mean_resolve_s': round(d[d.outcome != 'TIMEOUT'].resolve_s.mean(), 2) if (d.outcome != 'TIMEOUT').any() else None,
            'mean_penetration_m': round(d.min_penetration_m.dropna().mean(), 1) if d.min_penetration_m.notna().any() else None,
            'exchange_ratio': round(d.kills.sum() / max(1, d.kills.sum() + d.losses.sum()), 3),
            'dup_assign_per_ep': round(d.dup_assign_steps.mean(), 1),
            'jerk_mean': round(d.jerk_mean.dropna().mean(), 3) if d.jerk_mean.notna().any() else None,
            'kalman_rmse': round(d.kalman_rmse.dropna().mean(), 3) if d.kalman_rmse.notna().any() else None,
        })
    out = pd.DataFrame(rows)

    def des(r, nd):
        speed = 1.0 - (r['mean_resolve_s'] or T_CAP_S) / T_CAP_S
        integ = np.exp(-(r['mean_penetration_m'] if r['mean_penetration_m'] is not None else 999) / 50.0)
        surv = 1.0 - r['mean_losses'] / nd
        return round(100 * (0.40 * r['win_rate'] + 0.20 * max(surv, 0) +
                            0.15 * max(speed, 0) + 0.15 * integ +
                            0.10 * r['exchange_ratio']), 1)

    nd_map = {s: c['num_defenders'] for s, c in SCENARIOS.items()}
    out['DES'] = [des(r, nd_map[r['scenario']]) for _, r in out.iterrows()]
    return out.sort_values(['scenario', 'DES'], ascending=[True, False])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=12)
    ap.add_argument('--seed-base', type=int, default=1000)
    ap.add_argument('--outdir', default='eval_output')
    ap.add_argument('--cells', default='',
                    help="comma list scenario:variant to run (default: all)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    JsonlLogger.init(run_dir=os.path.join(args.outdir, 'logs'), console=False)

    # Resume support: skip episodes already recorded
    done = set()
    res_path = os.path.join(args.outdir, 'eval_results.csv')
    if os.path.exists(res_path):
        prev = pd.read_csv(res_path)
        done = {(r.scenario, r.variant, int(r.seed)) for r in prev.itertuples()}
        print(f"resuming: {len(done)} episodes already on file", flush=True)

    records = []
    t0 = time.time()
    for sc_name, sc_cfg in SCENARIOS.items():
        for va_name, va_cfg in VARIANTS.items():
            if args.cells and f"{sc_name}:{va_name}" not in args.cells.split(','):
                continue
            for ep in range(args.episodes):
                seed = args.seed_base + ep
                if (sc_name, va_name, seed) in done:
                    continue
                r = run_episode(sc_cfg, va_cfg, seed)
                r.update({'scenario': sc_name, 'variant': va_name, 'seed': seed})
                records.append(r)
                print(f"[{sc_name}/{va_name}] ep{ep}: {r['outcome']} "
                      f"k={r['kills']} l={r['losses']} t={r['resolve_s']}s", flush=True)
        # Incremental evidence dump (survives harness interruption)
        base = pd.read_csv(res_path) if os.path.exists(res_path) else pd.DataFrame()
        merged = pd.concat([base, pd.DataFrame(records)], ignore_index=True)
        merged.to_csv(res_path, index=False)
        records = []
    df = pd.read_csv(res_path)

    summary = aggregate(df)
    summary.to_csv(os.path.join(args.outdir, 'eval_summary.csv'), index=False)
    payload = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
               'episodes_per_cell': args.episodes, 'seed_base': args.seed_base,
               'wall_minutes': round((time.time() - t0) / 60, 1),
               'table': json.loads(summary.to_json(orient='records')),
               'des_formula': "100*(0.40*win + 0.20*survival + 0.15*speed + 0.15*integrity + 0.10*exchange)"}
    with open(os.path.join(args.outdir, 'eval_summary.json'), 'w') as f:
        json.dump(payload, f, indent=2)
    print(summary.to_string(index=False), flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
