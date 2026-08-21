import numpy as np
from scipy.optimize import linear_sum_assignment
from gym_pybullet_drones.utils.tlog import get_logger

# Cost added to Fighters (mode=1) so Bombers are always prioritized.
# Origin: tuned by examples/genetic_optimizer.py (GA) from a hardcoded 5000.0.
# NOTE: single-run GA output — treat as scenario-tuned, re-validate before reuse.
FIGHTER_PENALTY_DEFAULT = 790.2

class WolfPackStrategy:
    """
    Optimal decentralized-by-replication target assignment.

    Every agent observes the SAME world snapshot, so each agent independently
    solves the identical rectangular Linear Sum Assignment Problem (Hungarian,
    via scipy) and reads its own row from the shared solution. Result:
    - mathematically optimal 1:1 pairing (no duplicate engagements)
    - surplus defenders fall through to patrol
    - zero communication required (deterministic replication)
    - comms-denied modulo fallback preserved for the no-neighbors case
    """
    def __init__(self, intercept_limit=3, fighter_penalty=FIGHTER_PENALTY_DEFAULT,
                 w_dist=0.05, assignment_mode='hungarian'):
        self.intercept_limit = intercept_limit          # used by 'greedy' mode
        self.fighter_penalty = fighter_penalty
        self.w_dist = w_dist                            # spatial tie-break weight
        self.assignment_mode = assignment_mode          # 'hungarian' | 'greedy'
        self.log = get_logger("wolfpack")
        self.last_uid = None                            # instrumentation: last selection

    def _threat_cost(self, e, asset_pos):
        """Priority cost of an enemy (lower = more urgent). Bombers > Fighters."""
        e_pos = np.asarray(e['pos'], dtype=float)
        dist_asset = np.linalg.norm(e_pos - np.asarray(asset_pos, dtype=float))
        tti = dist_asset / 10.0                         # assume 10 m/s enemy speed
        if e.get('mode', 0) == 1:                       # Fighter penalty
            tti += self.fighter_penalty
        return tti

    def _select_greedy(self, my_id, my_pos, neighbors, enemies, asset_pos):
        """Legacy reference allocator (ablation baseline): priority-sorted
        threats with distance-rank admission control (allows swarming)."""
        limit = max(1, int(np.ceil((len(neighbors) + 1) / len(enemies))))
        occupied = set()
        ranked = sorted(enemies, key=lambda e: (
            self._threat_cost(e, asset_pos),
            np.linalg.norm(np.asarray(e['pos'], float) - np.asarray(my_pos, float))))
        for e in ranked:
            e_pos = np.asarray(e['pos'], dtype=float)
            dists = {int(n.get('id', -1)):
                     np.linalg.norm(np.asarray(n['pos'], float) - e_pos)
                     for n in neighbors}
            better = sum(1 for d in dists.values()
                         if d < np.linalg.norm(np.asarray(my_pos, float) - e_pos) - 1e-6)
            if better < limit:
                return e
            top = sorted(dists.items(), key=lambda kv: kv[1])[:limit]
            occupied.update(k for k, _ in top)
        return None

    def select_target(self, my_id, my_pos, neighbors, enemies, asset_pos):
        """
        Returns the enemy dict this defender should engage, or None → patrol.
        """
        if not enemies:
            return None

        # --- Comms-denied fallback: deterministic modulo allocation -------
        if not neighbors:
            threats = sorted(enemies, key=lambda e: self._threat_cost(e, asset_pos))
            bombers = [t for t in threats if t.get('mode', 0) == 0]
            pool = bombers if bombers else threats
            chosen = pool[int(my_id) % len(pool)]
            self.last_uid = chosen.get('uid')
            return chosen

        if self.assignment_mode == 'greedy':
            chosen = self._select_greedy(my_id, my_pos, neighbors, enemies, asset_pos)
            self.last_uid = None if chosen is None else chosen.get('uid')
            return chosen

        # --- Global optimal assignment, replicated locally -----------------
        # Deterministic defender ordering (sorted ids => every agent builds
        # the identical matrix and thus the identical solution).
        defender_ids = sorted([int(n.get('id', -1)) for n in neighbors] + [int(my_id)])
        my_row = defender_ids.index(int(my_id))

        # Column order: most urgent threats first so that when defenders are
        # scarce, the assignment covers the highest-priority enemies.
        order = sorted(range(len(enemies)),
                       key=lambda j: self._threat_cost(enemies[j], asset_pos))
        cost = np.zeros((len(defender_ids), len(order)), dtype=float)
        for a, did in enumerate(defender_ids):
            n_pos = next((np.asarray(n['pos'], dtype=float)
                          for n in neighbors if int(n.get('id', -1)) == did),
                         np.asarray(my_pos, dtype=float))
            for b, j in enumerate(order):
                e = enemies[j]
                dist_me = np.linalg.norm(n_pos - np.asarray(e['pos'], dtype=float))
                # Base threat priority + mild proximity preference as tie-break
                cost[a, b] = self._threat_cost(e, asset_pos) + self.w_dist * dist_me

        rows, cols = linear_sum_assignment(cost)

        assigned_col = None
        for a, b in zip(rows, cols):
            if a == my_row:
                assigned_col = b
                break

        if assigned_col is None:
            # More defenders than enemies: I am surplus → patrol
            self.last_uid = None
            return None

        chosen = enemies[order[assigned_col]]
        self.last_uid = chosen.get('uid')
        self.log.log("assignment", defender=int(my_id),
                     enemy_uid=self.last_uid,
                     cost=round(float(cost[my_row, assigned_col]), 4))
        return chosen

    def predict_battle_outcome(self, num_defenders, num_attackers):
        if num_attackers == 0: return "Victory (100%)"
        blue_str = num_defenders ** 2
        red_str = num_attackers ** 2
        if blue_str > red_str:
            return f"Victory (Pred. Remaining: {np.sqrt(blue_str - red_str):.1f})"
        else:
            return f"Defeat (Pred. Enemy Remaining: {np.sqrt(red_str - blue_str):.1f})"
