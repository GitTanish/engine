# Engine - Autonomous Drone Swarm Defense

A decentralized multi-agent system for autonomous drone swarm defense simulation using `gym-pybullet-drones` and PyBullet physics.

## 🎯 Overview

**Engine** implements a **Wolf Pack Strategy** where 10 Friendly Drones (Blue) defend a ground asset (Green Box) from 4 Enemy Drones (Red) using decentralized decision-making and emergent swarm behavior.

### Key Features

-   ⚙️ **Heavy Drone Physics**: 8kg drones with realistic inertia (custom `HeavyDSLPIDControl`)
-   🐺 **Wolf Pack Strategy**: Decentralized target assignment based on threat urgency
-   🤖 **Modular Strategy Pattern**: Clean separation between decision logic and execution
-   🎨 **Visual Debugging**: Real-time visualization of agent states (Patrol = Blue, Intercept = Red)
-   📊 **Automatic Scoring**: Tracks Wins/Losses with auto-reset for continuous evaluation

---

## 📁 Architecture

### Strategy Pattern Design

```
gym_pybullet_drones/
├── control/
│   ├── strategies/               # 🆕 Decision Logic Modules
│   │   ├── __init__.py
│   │   └── WolfPackStrategy.py   # Target assignment logic
│   ├── DefenseAgent.py           # Defensive drone "pilot"
│   └── EnemyAgent.py             # Attacker drone
├── envs/
│   └── BaseDefenseAviary.py      # Physics environment
├── utils/
│   └── heavy_controller.py       # HeavyDSLPIDControl (8kg PID)
└── examples/
    └── test_defense_scenario.py  # Simulation runner
```

### Component Responsibilities

| Component | Responsibility | Key Methods |
|-----------|---------------|-------------|
| **WolfPackStrategy** | Target assignment logic | `select_target(my_id, my_pos, neighbors, enemies, asset_pos)` |
| **DefenseAgent** | Execute strategy + flight control | `compute_action(my_state, neighbors, enemies, asset_pos, dt, time_now)` |
| **EnemyAgent** | Attack the asset | `compute_action(my_state, asset_pos, control_timestep)` |
| **BaseDefenseAviary** | Physics simulation | `step(action)`, `reset()` |

---

## 🐺 Wolf Pack Strategy

### Algorithm

The `WolfPackStrategy` implements a 4-step decentralized decision process:

```python
def select_target(my_id, my_pos, neighbors, enemies, asset_pos):
    # 1. FILTER: Identify enemies within threat range (30m)
    threats = [e for e in enemies if dist(e, my_pos) < 30]
    
    # 2. SORT: Rank by urgency (closest to asset = most urgent)
    threats.sort(key=lambda e: dist(e, asset_pos))
    
    # 3. RATIO: Calculate swarm ratio (defenders per enemy)
    swarm_ratio = max(1, num_defenders // num_enemies)
    
    # 4. BID: Am I one of the closest swarm_ratio defenders?
    if my_id in get_closest(swarm_ratio, neighbors, target):
        return target  # INTERCEPT
    else:
        return None    # PATROL
```

### Behavior States

#### 🔵 PATROL (Boids Flocking)
-   **Separation**: Avoid crowding neighbors (< 2m)
-   **Cohesion**: Steer towards group center of mass (< 10m)
-   **Alignment**: Match velocity with neighbors
-   **Base Attraction**: Hover at 1.5m altitude above the Green Box

#### 🔴 INTERCEPT (Lead Pursuit)
-   **Target**: Enemy drone assigned by `WolfPackStrategy`
-   **Navigation**: Lead pursuit (aim where enemy will be)
-   **Mode**: Kamikaze (no separation forces, direct ram)
-   **Outcome**: Collision (< 2.0m) → Enemy deleted

---

## 🚀 Usage

### Quick Start

```bash
# Navigate to examples folder
cd gym_pybullet_drones/examples

# Run the defense scenario
python test_defense_scenario.py
```

### Expected Output

```
[INFO] Starting Episode 1
[SCORE] Wins: 0 | Losses: 0
[DEBUG] Enemy 0 Dist: 94.34m
[DEBUG] Enemy 0 Dist: 92.11m
...
[COMBAT] Defender 2 neutralized Enemy 0!
[COMBAT] Defender 5 neutralized Enemy 1!
...
[VICTORY] All enemies neutralized!
[SCORE] Wins: 1 | Losses: 0
```

### Controls

-   **Press `q`**: Exit simulation
-   **Auto-Reset**: Enabled (continuous episodes)

---

## ⚙️ Configuration

### Drone Parameters (`BaseDefenseAviary.py`)

```python
HEAVY_DRONE_MASS = 8.0  # kg
MAX_SPEED = 10.0        # m/s
HOVER_RPM = 23950       # RPM for 8kg hover
```

### Strategy Parameters (`WolfPackStrategy.py`)

```python
THREAT_RANGE = 30.0     # Distance at which enemies become threats
MAX_SPEED = 10.0        # Maximum drone speed
swarm_ratio = max(1, n_friends // n_enemies)  # Defenders per enemy
```

### Scenario Setup (`test_defense_scenario.py`)

```python
NUM_DEFENDERS = 10      # Blue drones
NUM_ENEMIES = 4         # Red drones
COLLISION_RADIUS = 2.0  # Neutralization distance
ASSET_HIT_RADIUS = 1.0  # Failure condition
```

---

## 🎨 Visual Debugging

### Debug Lines

-   **Blue Line** (Patrol): Drone is in flocking mode, circling the asset
-   **Red Line** (Intercept): Drone is pursuing an enemy target
-   **Green Line**: Marks the asset position (vertical indicator)

### Drone Colors

-   **Blue**: Friendly defenders
-   **Red**: Enemy attackers
-   **Green**: Ground asset (Green Box)

---

## 📊 Win/Loss Conditions

### ✅ VICTORY
-   **Condition**: All Red drones neutralized (collision within 2.0m of Blue drone)
-   **Action**: Increment wins, auto-reset

### ❌ FAILURE
-   **Condition**: Any Red drone reaches the Green Box (< 1.0m)
-   **Action**: Increment losses, auto-reset

---

## 🔧 Extending the System

### Adding a New Strategy

1. Create `gym_pybullet_drones/control/strategies/YourStrategy.py`:
```python
class YourStrategy:
    def select_target(self, my_id, my_pos, neighbors, enemies, asset_pos):
        # Your decision logic here
        return target_enemy or None
```

2. Update `DefenseAgent.py`:
```python
from gym_pybullet_drones.control.strategies.YourStrategy import YourStrategy

self.strategy = YourStrategy()
```

### Strategy Ideas

-   **Ant Swarm**: Pheromone-based coordination
-   **Market-Based**: Auction-style task allocation
-   **Predictive**: Model enemy behavior and preempt
-   **Formation**: Maintain geometric defense patterns

---

## 🧪 Testing

### Unit Tests (Manual Verification)

```bash
# Test 1: Patrol behavior
python test_defense_scenario.py
# Expected: Blue drones circle at 1.5m altitude

# Test 2: Intercept behavior
# Expected: When Red enters range, closest Blues break off

# Test 3: Neutralization
# Expected: Combat messages and Red drones vanish

# Test 4: Win/Loss tracking
# Expected: Scoreboard updates after each episode
```

---

## 📝 Technical Details

### OODA Loop (Observe-Orient-Decide-Act)

```python
def compute_action(my_state, neighbors, enemies, asset_pos, dt, time_now):
    # OBSERVE: Parse current state
    my_pos = my_state['pos']
    my_vel = my_state['vel']
    
    # DECIDE: Use strategy
    target = self.strategy.select_target(my_id, my_pos, neighbors, enemies, asset_pos)
    
    # ACT: Execute behavior
    if target:
        intercept_logic()  # Lead pursuit
    else:
        patrol_logic()     # Boids flocking
    
    # Output: PID control
    rpm = self.ctrl.computeControlFromState(...)
    return rpm
```

### Physics Constraints

-   **Mass**: 8kg (high inertia, no instant stops)
-   **Controller**: HeavyDSLPIDControl (tuned PID for heavy drones)
-   **Velocity Control**: Moving waypoint to prevent overshoot

---

## 🛠️ Troubleshooting

### Issue: Drones fall to the ground
-   **Cause**: PID not tuned for 8kg mass
-   **Fix**: Ensure `HeavyDSLPIDControl` is used, not `DSLPIDControl`

### Issue: Import errors
-   **Cause**: Package structure mismatch
-   **Fix**: Ensure `sys.path` includes project root (handled in `test_defense_scenario.py`)

### Issue: Enemies not deleted
-   **Cause**: Collision radius too small
-   **Fix**: Increase `COLLISION_RADIUS` in `test_defense_scenario.py`

---

## 📚 References

-   [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)
-   [PyBullet Documentation](https://pybullet.org/)
-   [Boids Algorithm (Craig Reynolds)](https://www.red3d.com/cwr/boids/)
-   [Wolf Pack Hunting Strategy](https://en.wikipedia.org/wiki/Pack_hunter)

---

## 👥 Contributor

Tanish Saroj

## 👥 Contribution

This repository contains my implementation work for the Smart India Hackathon (SIH) drone swarm defense project, including:

- Rule-based swarm coordination
- Wolf Pack targeting strategy
- Heuristic scheduling/intercept logic
- Heavy drone simulation tuning
- PyBullet integration and control systems

---

## 📄 License

This project uses the same license as `gym-pybullet-drones`.

---

**Built with ❤️ using PyBullet and decentralized swarm intelligence**
