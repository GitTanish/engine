# Engine
### Multi-Drone Base Defense Simulation

Engine is an autonomous multi-drone defense simulation built on [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones).

It simulates a swarm of 10 defenders protecting a ground asset against mixed bomber and fighter raids. The system combines multi-agent coordination, target tracking, guidance, behavior-tree decision-making, and live mission telemetry in a single environment.

The goal is to make complex swarm behavior easier to build, observe, and evaluate.

## What It Does

- **Coordinates defenders.** Hungarian assignment allocates threats across agents, with time-to-impact costing, bomber prioritization, and a greedy baseline for comparison. A comms-denied fallback keeps allocation deterministic when agents cannot see one another.
- **Tracks moving targets.** A constant-acceleration Kalman filter maintains target state and predicts future aimpoints. Tracks are keyed to stable target identities, so kills cannot corrupt filter state.
- **Guides intercepts.** Proportional-navigation steering with a direct-pursuit baseline and standoff intercept geometry around the protected asset.
- **Makes decisions.** A behavior tree organizes warmup, safety, engagement, and patrol behavior.
- **Resolves combat.** Frequency-independent splash rates and kinetic ram-kills settle engagements, with defender losses and enemy lifecycles managed by the environment.
- **Exposes internal state.** Structured combat events, decision logs, and live WebSocket telemetry make it possible to inspect what the simulation is doing.
- **Evaluates changes.** Seeded ablation experiments and Monte Carlo validation help distinguish measurable improvements from assumptions.

## Results So Far

Engine includes a reproducible evaluation pipeline for testing coordination and tracking behavior.

In the current reported evaluation:

- Hungarian assignment produced **zero duplicate engagements** across the evaluated episodes.
- The legacy greedy baseline produced **588–8,179 duplicate engagements per episode**.
- Kalman velocity RMSE was approximately **1.0 m/s** on maneuvering targets.
- Different guidance variants sometimes produced similar outcomes in battles dominated by the engagement envelope and lethality model.

These are scenario-specific simulation results, not guarantees of real-world performance. The evaluation artifacts provide the basis for reproducing and investigating them.

## Architecture

| Component | Implementation |
|---|---|
| Simulation environment | `envs/BaseDefenseAviary.py` |
| Multi-agent assignment | `strategies/WolfPackStrategy.py` |
| Guidance | `control/DefenseAgent.py` |
| State estimation | `control/tracking.py` |
| Decision-making | `control/bt.py` |
| Decision telemetry | `utils/tlog.py` |
| Evaluation suite | `examples/eval_suite.py`, `monte_carlo_validator.py` |
| Mission-control dashboard | `ui/mission_control_console.html`, `serve_frontend.py` |

The assignment layer uses decentralized replication: agents independently compute the same allocation from shared inputs. The simulation records combat outcomes and decision-level events so that coordination behavior can be examined rather than inferred from visual output alone.

## Quickstart

Requirements: Python 3.10 or newer and [`uv`](https://docs.astral.sh/uv/).

Install dependencies:

```bash
uv sync
```

Start the simulation:

```bash
uv run python gym_pybullet_drones/examples/test_defense_scenario.py
```

In a second terminal, start the dashboard:

```bash
uv run python serve_frontend.py
```

Open [http://localhost:8000](http://localhost:8000).

Select **DATA SOURCE → LIVE WS** to connect the dashboard to the running simulation. The simulator streams telemetry over WebSocket on port `8765`. DEMO and REPLAY modes use an internal kinematic scenario.

## Run an Evaluation

```bash
uv run python gym_pybullet_drones/examples/eval_suite.py \
  --episodes 6 --seed-base 42
```

The evaluation generates per-episode CSV files, aggregate metrics, and an `EVAL_REPORT.md` under `eval_output/`.

To run the build test:

```bash
uv run pytest tests/test_build.py
```

## Additional Experiments

- `MobileAssetSolution/` explores defense of a moving convoy under fog of war.
- `monte_carlo_validator.py` and `batch_validator*.py` support repeated headless evaluation.
- `genetic_optimizer.py` explores optimization of the WolfPack fighter-priority penalty.

## Project Status

Engine is an experimental simulation focused on coordination, observability, and repeatable evaluation. Its behavior and results depend on the simulated dynamics, sensing assumptions, engagement model, and tested scenarios.

The project is an ongoing effort to understand how multi-agent systems behave when individual decisions interact, and how to make those interactions measurable.

Contributions, experiments, and reproducible evaluations are welcome.
