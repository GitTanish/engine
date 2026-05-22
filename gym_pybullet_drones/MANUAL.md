# Drone Defense Simulation: User Manual

## 1. Overview
This project simulates a swarm of **15kg Heavy Defense Drones** protecting a high-value asset from waves of attacking enemies. It enforces strict physics, "Red-on-Blue" mortality, and specific tactical behaviors.

## 2. Key Core Files (The Engine)
These files contain the critical logic that drives the simulation. **Do not modify these unless you understand the physics/logic implications.**

### `envs/BaseDefenseAviary.py` (The World)
*   **Role**: The "God Class" of the simulation.
*   **Key Features**:
    *   **Mortality Logic**: Handles `_process_red_engagements`. If a Friendly is hit, it is disabled and teleported underground.
    *   **Weapon System**: Defines Hit Probability (5%) and Range (30-250m).
    *   **Spawning**: Randomizes enemy altitudes (90-120m) and types (Bombers vs Fighters).

### `control/DefenseAgent.py` (The Brain)
*   **Role**: The decision-making unit for each Blue Drone.
*   **Key Features**:
    *   **OODA Loop**: Observe -> Orient -> Decide -> Act.
    *   **Dynamic Scaling**: Automatically calculates the Patrol Ring radius based on swarm size (`Radius = Circumference / 2pi`) to prevent collisions.
    *   **Safety Override**: The "Lizard Brain" that prevents crashing into neighbors.

### `strategies/WolfPackStrategy.py` (The Logic)
*   **Role**: Target prioritization logic.
*   **Key Behavior**: **"Bomber Fixation"**. It prioritizes distant Bombers (Mode 0) over nearby Fighters (Mode 1). This is the source of the "unfair" casualty rates in Scenario B.

### `utils/heavy_controller.py` (The Muscles)
*   **Role**: Custom PID controller for the 15kg drone.
*   **Key Feature**: "Surgical Fix" that scales acceleration by mass to generate correct Force output, ensuring the heavy drone can hover and maneuver.

---

## 3. Running Scenarios
We have created specific launcher scripts for the required MDR scenarios.

### Scenario A: Parity
*   **Command**: `python examples/run_scenario_A.py`
*   **Config**: 10 Defenders vs 10 Enemies (3 Bombers, 7 Fighters).
*   **Expectation**: **Victory**. A balanced fight where the Blue Swarm usually wins with low-to-moderate casualties.

### Scenario B: Asymmetric (The Stress Test)
*   **Command**: `python examples/run_scenario_B.py`
*   **Config**: 10 Defenders vs 13 Enemies (6 Bombers, 7 Fighters).
*   **Expectation**: **Pyrrhic Victory**. The Blue Swarm will win (Asset Safe), but expect **30-90% Casualties**. The drones will chase Bombers while getting shot in the back by Fighters.

---

## 4. Validation (The Proof)
To scientifically prove the system works, we use the Monte Carlo Validator.

### `examples/batch_validator.py`
*   **Command**: `python examples/batch_validator.py`
*   **Role**: Runs the simulation **Headless** (no GUI) for 100 iterations per scenario.
*   **Importance**:
    *   It eliminates "lucky runs".
    *   It generates a CSV report (`validation_results.csv`) proving the Win Rate and Casualty Rate over time.
    *   **Use this before submitting any code changes** to ensure you haven't broken the logic.

### `validation_results.csv`
*   The output file containing the raw data of every simulation run (Win/Loss, Friendly Deaths, Steps Taken).
