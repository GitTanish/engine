# Engine
### Multi-Drone Base Defense Simulation

Engine is a multi-agent drone defense simulation built on [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones).

It simulates 10 defenders protecting a ground asset against mixed bomber and fighter raids. The system brings coordination, target tracking, interception guidance, behavior-tree decisions, and live telemetry into one environment.

The idea is straightforward: **build complex multi-agent behavior, then make it possible to understand whether it actually works.**

## Why Engine

Coordinated systems are difficult to evaluate by watching them alone. A swarm can look organized while making redundant decisions, tracking targets poorly, or failing to improve mission outcomes.

Engine is built to make those behaviors observable and testable. Its components can be examined independently, compared against baselines, and evaluated across repeatable scenarios.

## What It Does

- **Coordinates a swarm.** Hungarian assignment allocates threats across defenders using time-to-impact costs and bomber prioritization. A greedy baseline supports comparison, while a deterministic fallback handles denied communications.
- **Predicts target motion.** A constant-acceleration Kalman filter estimates target state and predicts future aimpoints. Stable target identities prevent target destruction from corrupting tracking state.
- **Guides interceptions.** Proportional Navigation drives interception, with direct pursuit available as a baseline and standoff geometry around the protected asset.
- **Makes decisions explicitly.** A behavior tree organizes warmup, safety, engagement, and patrol behavior, with decision paths available for inspection.
- **Simulates combat.** The environment manages enemy lifecycles, defender losses, splash effects, and kinetic ramming.
- **Makes behavior observable.** Structured combat events, decision logs, and live WebSocket telemetry expose what is happening inside the simulation.
- **Measures changes.** Seeded ablations and Monte Carlo validation provide a way to test improvements rather than assume them.

## Results

Engine includes an evaluation pipeline for studying coordination and target tracking under controlled simulation conditions.

In the current reported evaluation:

| Metric | Result |
|---|---|
| Duplicate engagements with Hungarian assignment | **0 across evaluated episodes** |
| Duplicate engagements with legacy greedy assignment | **588–8,179 per episode** |
| Kalman velocity RMSE on maneuvering targets | **Approximately 1.0 m/s** |

Different guidance variants also produced similar outcomes in some battles dominated by the engagement envelope and lethality model. That result matters: changing an algorithm does not necessarily improve the system it belongs to.

These measurements are specific to the evaluated scenarios and simulation assumptions. They are not evidence of equivalent real-world performance. The evaluation artifacts are intended to make the results reproducible and open to further investigation.

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

The assignment layer uses decentralized replication: agents independently compute the same allocation from shared inputs. Combat outcomes and decision-level events are recorded so coordination can be analyzed from the underlying data, not inferred from visual output alone.

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

In a second terminal, launch the dashboard:

```bash
uv run python serve_frontend.py
```

Open [http://localhost:8000](http://localhost:8000).

Select **DATA SOURCE → LIVE WS** to connect to the running simulation. Telemetry is streamed over WebSocket on port `8765`. DEMO and REPLAY modes use an internal kinematic scenario.

## Run an Evaluation

```bash
uv run python gym_pybullet_drones/examples/eval_suite.py \
  --episodes 6 --seed-base 42
```

The evaluation generates per-episode CSV files, aggregate metrics, and an `EVAL_REPORT.md` under `eval_output/`.

Run the build test:

```bash
uv run pytest tests/test_build.py
```

## Further Experiments

- `MobileAssetSolution/` explores defense of a moving convoy under fog of war.
- `monte_carlo_validator.py` and `batch_validator*.py` support repeated headless evaluation.
- `genetic_optimizer.py` explores optimization of the WolfPack fighter-priority penalty.

## What Comes Next

Engine is an ongoing experiment in multi-agent coordination, observability, and repeatable evaluation.

The broader question is how individual decisions interact to produce system-level behavior, and how we can measure those interactions well enough to improve them.

There is more to explore in coordination strategies, guidance, tracking, and evaluation. Contributions, experiments, and reproducible results are welcome.
