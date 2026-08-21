# Engine — Multi-Drone Base Defense System

Autonomous drone swarm defense simulation built on [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones):
a defender swarm protects a ground asset against mixed bomber/fighter raids, with a
Three.js mission-control dashboard wired to live telemetry over WebSocket.

## Quickstart (uv)

```powershell
uv sync                                   # install from uv.lock
uv run python gym_pybullet_drones/examples/test_defense_scenario.py   # GUI sim + WS telemetry :8765
uv run python serve_frontend.py           # dashboard on http://localhost:8000
```

In the dashboard, switch DATA SOURCE → LIVE WS to drive the 3D radar from the running
simulation. DEMO/REPLAY modes run an internal kinematic scenario.

## Stack

| Layer | Module | What it does |
|---|---|---|
| Env | `envs/BaseDefenseAviary.py` | Enemy lifecycle, splash + kinetic-ram weapons (frequency-independent Poisson rates), deaths, `info["combat_events"]`/`asset_hit`, stable enemy UIDs |
| Assignment | `strategies/WolfPackStrategy.py` | Hungarian optimal 1:1 target allocation, replicated locally per agent (decentralized-by-replication); TTI threat costing, bomber priority; comms-denied modulo fallback; `assignment_mode='greedy'` legacy baseline |
| Guidance | `control/DefenseAgent.py` | Proportional Navigation velocity setpoints (`guidance='pn'|'pursuit'`), standoff intercept geometry, boids patrol orbit |
| Estimation | `control/tracking.py` | uid-keyed constant-acceleration Kalman pool; predicts aimpoint τ ahead |
| Decisions | `control/bt.py` | Behavior tree: WARMUP > SAFETY > ENGAGE > PATROL, branch-path logged |
| Telemetry | `utils/tlog.py` | JSONL decision-level event log |
| Validation | `examples/eval_suite.py`, `monte_carlo_validator.py` | Seeded ablation evals with composite DES score; Monte Carlo win-rate runs |
| Dashboard | `ui/mission_control_console.html` + `serve_frontend.py` | Three.js radar, live WS client, sparklines, event stream |

## Evaluation (evidence-backed)

```powershell
uv run python gym_pybullet_drones/examples/eval_suite.py --episodes 6 --seed-base 42
```

Produces `eval_output/`: per-episode CSV (raw evidence), aggregates, and `EVAL_REPORT.md`.
Headline results (see full report): **Hungarian produces 0 duplicate engagements across all
episodes vs 588–3384/episode under legacy greedy**; Kalman velocity RMSE ≈ 1.0 m/s on
maneuvering targets; outcome parity across guidance variants in envelope-dominated battles
is documented honestly rather than hidden.

## Scenarios

- `test_defense_scenario.py` — main entry: 10 defenders, GUI, WebSocket telemetry.
- `MobileAssetSolution/` — moving-convoy variant with fog-of-war enemies.
- `monte_carlo_validator.py`, `batch_validator*.py` — headless statistical validation.
- `genetic_optimizer.py` — evolves the WolfPack fighter-priority penalty.

## Development

- Python ≥3.10; dependencies managed by **uv** (`pyproject.toml`, PEP 621).
- Tests: `uv run pytest tests/test_build.py`
- Branches: `master` (stable) / `dev` (active work).
