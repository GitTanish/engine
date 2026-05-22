# Project Documentation: gym-pybullet-drones

## 1. Project Overview

This project, `gym-pybullet-drones`, is a sophisticated and flexible framework for simulating multi-drone systems using the PyBullet physics engine. It is designed with a strong focus on reinforcement learning (RL) research but is also well-suited for developing and testing classical control algorithms and high-level, multi-agent behaviors.

The framework provides a set of `gymnasium`-compatible environments, various drone models, and a modular architecture that allows users to easily define new tasks, controllers, and scenarios. A key design philosophy is the abstraction of low-level flight dynamics, enabling RL agents or scripted policies to focus on high-level goals like navigation, trajectory tracking, or complex mission execution.

## 2. Core Concepts

### 2.1. Three-Tier Environment Architecture

The simulation environments are built on a three-tiered inheritance structure, which cleanly separates concerns:

1.  **`BaseAviary` (The Physics Layer):** This is the foundational abstract class for all environments. It manages the direct interaction with the PyBullet physics server, loads drone and environment assets (URDF files), steps the physics simulation, and handles collision detection. It provides the core, low-level simulation loop but is not tied to any specific task or learning algorithm.

2.  **`BaseRLAviary` (The RL Interface Layer):** Inheriting from `BaseAviary`, this class serves as the bridge to reinforcement learning. Its primary role is to create the action and observation spaces required by the `gymnasium` API. It translates the RL agent's high-level actions (e.g., target position, velocity) into low-level motor commands by using an underlying PID controller. This significantly simplifies the learning problem for the agent.

3.  **Concrete Environments (The Task Layer):** These classes (e.g., `HoverAviary`, `VelocityAviary`) inherit from `BaseRLAviary`. They define a specific task or goal by implementing the final pieces of the `gymnasium` interface:
    *   `_computeReward()`: The reward function for the task.
    *   `_computeTerminated()`: The logic for when an episode ends in a terminal state (e.g., a crash).
    *   `_computeTruncated()`: The logic for when an episode is cut short for other reasons (e.g., time limit).

### 2.2. Control Abstraction

A powerful feature of this framework is its handling of drone control. Instead of requiring RL agents to learn direct motor control (i.e., outputting RPMs for each rotor), the system uses an abstraction layer. The `ActionType` enum allows a user to configure what the agent's actions represent.

For high-level action types like `ActionType.POS` (target 3D position), the `BaseRLAviary` employs the `DSLPIDControl` class. This robust PID controller takes the agent's high-level target and the drone's current state, and computes the. precise rotor RPMs needed to achieve that target. This allows the agent to learn *what to do* rather than *how to fly*.

## 3. Directory Structure

-   `assets/`: Contains all physical models and simulation assets. The most important are the `.urdf` (Unified Robot Description Format) files, which define the drones' physical properties like mass, inertia, motor constants, and collision shapes. These are not just visual models; they are critical to the physics simulation.

-   `control/`: Implements the various controllers and high-level agent behaviors.
    -   **PID Controllers (`DSLPIDControl.py`):** The core flight controller for translating high-level commands into motor RPMs.
    -   **Scripted Agents (`DefenseAgent.py`, `EnemyAgent.py`):** These are not general-purpose controllers but rather high-level, scripted behaviors for specific scenarios. For instance, `DefenseAgent` uses a state machine and boids-like flocking algorithms (`WolfPackStrategy.py`) to execute complex defensive maneuvers.

-   `envs/`: Home to the simulation environments. This is where the three-tier architecture (`BaseAviary`, `BaseRLAviary`, and concrete task environments) is defined.

-   `examples/`: Contains standalone scripts that demonstrate how to instantiate and run the various environments and controllers. These are the best starting point for new users. Scripts like `pid.py` show how to test a single drone with a PID controller, while `test_defense_scenario.py` likely showcases a complex multi-agent simulation.

-   `utils/`: Provides helper classes and utility functions used throughout the project.
    -   **`enums.py`:** A critical file for configuration. It defines enums like `DroneModel`, `Physics`, `ActionType`, and `ObservationType`, which allow for easy, readable configuration of the simulation environments.
    -   **`Logger.py`:** A utility for logging simulation data for later analysis and plotting.

## 4. Key Components & Classes

### 4.1. Environments (`gym_pybullet_drones/envs/`)

-   **`BaseAviary`:** The core physics simulator. Don't use directly.
-   **`BaseRLAviary`:** The base for all RL-ready environments. Provides `gymnasium`-compatible action/observation spaces and integrates the PID controller.
-   **`HoverAviary`:** A simple RL task where the goal is to make the drone hover at a target location.
-   **`VelocityAviary`:** An RL task where the agent must learn to command the drone to a specific velocity.
-   **`CtrlAviary`:** A non-RL environment for testing and debugging controllers directly.
-   **`BaseDefenseAviary`:** A specialized environment for multi-agent defense scenarios, likely involving `DefenseAgent` and `EnemyAgent`.

### 4.2. Controllers & Agents (`gym_pybullet_drones/control/`)

-   **`BaseControl`:** The abstract interface that all controller classes must implement, ensuring they can be used interchangeably.
-   **`DSLPIDControl`:** The workhorse PID controller that provides stable, low-level flight control.
-   **`DefenseAgent` / `EnemyAgent`:** High-level, scripted agents that implement specific behaviors for complex scenarios rather than learning them. They use a standard controller (like PID) for execution but make decisions based on a programmed strategy.

### 4.3. Configuration (`gym_pybullet_drones/utils/enums.py`)

-   **`DroneModel`:** Selects the drone model to be loaded (e.g., `CF2X`, `RACER`).
-   **`ActionType`:** Defines what the agent's output actions represent (e.g., `RPM`, `PID`, `POS`, `VEL`). This is one of the most important settings for an RL environment.
-   **`ObservationType`:** Defines the content of the observation vector provided to the agent (e.g., `KIN` for kinematics, `RGB` for camera images).

## 5. Getting Started

The easiest way to get started is to run one of the scripts in the `examples/` directory.

For example, to run a simple PID-controlled flight, you would likely execute:

```bash
python gym_pybullet_drones/examples/pid.py
```

To train an RL agent, you would start with a script like `learn.py` (if it exists) or adapt an existing example by wrapping the environment with an RL library's training loop (e.g., using Stable Baselines3). The key is to instantiate the desired environment (e.g., `HoverAviary`) and then use it like any other `gymnasium` environment.

## 6. Heuristic Brain: Scripted Agent Analysis

For scenarios requiring deterministic, rule-based behaviors, the project includes a "heuristic brain" composed of several scripted agents. These agents are not trained via reinforcement learning but instead follow a sophisticated set of rules to achieve their objectives. They are primarily used in the defense scenario to simulate an attacker and a coordinated group of defenders.

### 6.1. `EnemyAgent.py` - The Attacker

This agent is designed with a single, simple objective: attack the asset. Its behavior is direct and predictable.

**Logic Breakdown:**

1.  **Role:** Ground Attacker (GA). Its sole target is the green asset block.
2.  **Initialization:** It initializes a `HeavyDSLPIDControl` instance, which it uses for all flight control.
3.  **Core Logic (`compute_action`)**:
    *   It determines its current position (`my_pos`) and the vector towards the `asset_pos`.
    *   It calculates a `target_pos` for the current timestep. If the asset is far away, the target is a point along the direct path at `MAX_SPEED`. If the asset is close, the target becomes the asset itself.
    *   It sets a corresponding `target_vel` (either pointing towards the asset at `MAX_SPEED` or zero if the asset is the final destination).
    *   It completely ignores the presence of any defending drones.
    *   The calculated `target_pos` and `target_vel` are passed to the PID controller, which computes the required motor RPMs to execute the move.

### 6.2. `WolfPackStrategy.py` - The Decision Core

This class is not an agent itself but a utility that contains the core decision-making logic for the `DefenseAgent`. It implements a decentralized target assignment strategy, allowing a group of defenders to coordinate without a central commander.

**Logic Breakdown (`select_target`):**

1.  **Filter Threats:** The strategy first identifies which enemies are relevant by filtering for any enemy within a predefined `THREAT_RANGE`. If no enemies are in range, it returns `None`.
2.  **Rank by Urgency:** It then sorts the filtered threats by "urgency," which is defined as their distance to the asset being defended. The enemy closest to the asset is considered the `most_urgent`.
3.  **Calculate Swarm Ratio:** It determines how many defenders should engage each enemy by calculating `n_friends // n_enemies`. This ensures that as more enemies appear, the defenders spread out, while a numbers advantage allows them to swarm a single target.
4.  **Decentralized Bidding:** This is the key to coordination. The agent running the strategy checks how far it and all its allies (`neighbors`) are from the `most_urgent` threat.
5.  **Win the Bid:** It sorts allies by their distance to the target and selects the top `swarm_ratio` defenders. If the current agent's `my_id` is in this list of closest defenders, it "wins the bid" and is assigned the target.
6.  **Return Assignment:** If the agent is assigned the target, the function returns the `target_enemy` dictionary. Otherwise, it returns `None`, signaling to the agent that it should continue its default behavior (patrolling).

### 6.3. `DefenseAgent.py` - The Defender

This is the most complex agent, acting as the "brain" for the defensive drones. It combines the `WolfPackStrategy` with a state machine and flocking behaviors to create a robust and coordinated defense. Its logic follows an **OODA Loop (Observe -> Orient -> Decide -> Act)**.

**Logic Breakdown (`compute_action`):**

1.  **Observe:** The agent receives its own state, the state of its `neighbors` (other defenders), `enemies`, and the `asset_pos`.

2.  **Safety Override (Highest Priority):** Before any other logic, it runs a safety check. If any neighboring defender is dangerously close (within `min_safe` distance), it immediately calculates and applies an escape velocity, overriding all other behaviors to prevent a collision.

3.  **Decide (`WolfPackStrategy`):** The agent calls `self.strategy.select_target()` to determine if there is an enemy it should be attacking. This is the primary decision-making step. The result determines its state.

4.  **Act (State Machine):**
    *   **State: `INTERCEPT` (A target was assigned)**
        *   The agent calculates an intercept point by predicting the enemy's future position based on its velocity.
        *   It sets a `target_vel` towards this predicted point, capped at `MAX_SPEED`.
        *   **Collision Avoidance:** Even during an attack, it maintains situational awareness. It calculates a gentle "separation push" away from any nearby friendly drones and adds it to its `target_vel`, ensuring it doesn't collide with allies while pursuing an enemy.
        *   The final `target_pos` and `target_vel` are sent to the PID controller.

    *   **State: `PATROL` (No target was assigned)**
        *   **Deterministic Ring Slot:** To ensure the defenders form a stable patrol formation without clumping, each agent calculates a unique "slot" on a ring around the asset. This position is determined deterministically using the agent's ID, which breaks symmetry and creates an organized patrol.
        *   **Boids Flocking:** The agent then applies a modified Boids algorithm to fine-tune its movement:
            *   **Separation (High Priority):** A strong repulsive force pushes it away from any neighbors that are too close.
            *   **Cohesion & Alignment (Lower Priority):** Weaker forces pull it towards the center of mass of its neighbors and encourage it to align its velocity with them.
        *   **Vector Combination:** The final `desired_vel` is a weighted sum of vectors: the strong push from **separation**, the pull towards its assigned **ring slot**, a gentle **tangential** force to keep the patrol circulating, an **altitude correction** force, and the weaker pushes from **alignment** and **cohesion**.
        *   The final velocity is capped and sent to the PID controller.

5.  **PID Control:** In all cases, the final computed `target_pos` and `target_vel` are passed to the `HeavyDSLPIDControl` instance, which calculates the required motor RPMs to execute the agent's decision.

## 7. Full Project File Structure

```
C:.
│   .gitignore
│   CITATION.cff
│   LICENSE
│   PROJECT_DOCUMENTATION.md
│   pyproject.toml
│   README.md
│   test_defense_fix.md
│
├───.github
│   │   dependabot.yml
│   │
│   └───workflows
│           test.yml
│
├───gym_pybullet_drones
│   │   README.md
│   │   __init__.py
│   │
│   ├───assets
│   │       architrave.urdf
│   │       beta-presets-bak.txt
│   │       beta-traj.csv
│   │       box.urdf
│   │       cf2.dae
│   │       cf2p.urdf
│   │       cf2x.urdf
│   │       clone_bfs.sh
│   │       eeprom.bin
│   │       ffmpeg_png2mp4.sh
│   │       heavy_drone.urdf
│   │       helix.gif
│   │       helix.png
│   │       marl.gif
│   │       racer.urdf
│   │       rl.gif
│   │       sphere2.urdf
│   │
│   ├───control
│   │   │   BaseControl.py
│   │   │   CTBRControl.py
│   │   │   DefenseAgent.py
│   │   │   DSLPIDControl.py
│   │   │   EnemyAgent.py
│   │   │   MRAC.py
│   │   │   WolfPackStrategy.py
│   │   │   __init__.py
│   │   │
│   │   └───strategies
│   │       │   WolfPackStrategy.py
│   │       │   __init__.py
│   │       │
│   │       └───__pycache__
│   │               WolfPackStrategy.cpython-310.pyc
│   │               WolfPackStrategy.cpython-312.pyc
│   │               __init__.cpython-310.pyc
│   │               __init__.cpython-312.pyc
│   │
│   ├───envs
│   │   │   BaseAviary.py
│   │   │   BaseDefenseAviary.py
│   │   │   BaseRLAviary.py
│   │   │   BetaAviary.py
│   │   │   CFAviary.py
│   │   │   CtrlAviary.py
│   │   │   HoverAviary.py
│   │   │   MultiHoverAviary.py
│   │   │   VelocityAviary.py
│   │   │   __init__.py
│   │   │
│   ├───examples
│   │   │   beta.py
│   │   │   cf.py
│   │   │   debug.py
│   │   │   debug_path.py
│   │   │   debug_test_env.py
│   │   │   downwash.py
│   │   │   learn.py
│   │   │   mrac.py
│   │   │   output.log
│   │   │   output.txt
│   │   │   pid.py
│   │   │   pid_velocity.py
│   │   │   play.py
│   │   │   test_defense_scenario.py
│   │   │   test_heavy_drone_render.py
│   │   │   test_heavy_hover.py
│   │   │   __init__.py
│   │   │
│   │   └───results
│   │       │   save-flight-11.29.2025_18.28.18.npy
│   │       │   save-flight-11.29.2025_18.35.17.npy
│   │       │
│   │       ├───save-flight-pid-11.29.2025_18.28.18
│   │       ├───save-flight-pid-11.29.2025_18.35.17
│   │       ├───save-flight-vel-11.29.2025_18.30.00
│   │       └───save-flight-vel-11.29.2025_18.31.16
│   │
│   └───utils
│       │   drone_constants.py
│       │   enums.py
│       │   heavy_controller.py
│       │   Logger.py
│       │   utils.py
│       │   __init__.py
│
└───tests
        test_build.py
        test_examples.py
```