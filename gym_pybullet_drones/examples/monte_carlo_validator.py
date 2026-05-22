"""
Monte Carlo Validator Script
Runs N simulations in HEADLESS mode (no GUI) to calculate success rates.
"""
import time
import numpy as np
import pandas as pd
import sys
import os

# --- PATH SETUP ---
current_file_path = os.path.abspath(__file__)
examples_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(examples_dir)
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from gym_pybullet_drones.envs.BaseDefenseAviary import BaseDefenseAviary
    from gym_pybullet_drones.utils.enums import DroneModel, Physics
    from gym_pybullet_drones.control.DefenseAgent import DefenseAgent
except ImportError:
    # Fallback for flattened structure
    inner_package_dir = os.path.join(project_root, 'gym_pybullet_drones')
    sys.path.append(inner_package_dir) 
    from envs.BaseDefenseAviary import BaseDefenseAviary
    from utils.enums import DroneModel, Physics
    from control.DefenseAgent import DefenseAgent

# --- CONFIGURATION ---
NUM_SIMULATIONS = 100  # Run 100 battles
NUM_DEFENDERS = 10     # Fixed Swarm Size

def run_monte_carlo():
    results = []
    print(f"[INFO] Starting {NUM_SIMULATIONS} Monte Carlo runs in HEADLESS mode...")
    
    # Initialize Environment (Headless)
    env = BaseDefenseAviary(
        drone_model=DroneModel.HEAVY,
        num_drones=NUM_DEFENDERS, 
        physics=Physics.PYB,
        gui=False,       # CRITICAL: False for speed
        record=False,   
        obstacles=False,
        neighbourhood_radius=100.0
    )
    
    # Dynamic Max Steps (60 seconds of flight)
    MAX_STEPS = int(60 * env.CTRL_FREQ)
    print(f"[INFO] Simulation Frequency: {env.CTRL_FREQ}Hz | Max Steps: {MAX_STEPS}")

    start_time_global = time.time()

    for run_id in range(NUM_SIMULATIONS):
        # 1. Randomize Scenario (Stochastic Setup)
        num_enemies = np.random.randint(4, 8) # Randomize threat count
        scenario_config = {
            "num_enemies": num_enemies,
            "spawn_radius": np.random.uniform(150.0, 250.0), # Varied distances
            "ground_attacker_ratio": np.random.uniform(0.5, 0.8) # Varied composition
        }
        
        # 2. Reset Environment
        obs, info = env.reset(scenario_config=scenario_config)

        # 3. Init Agents (Defenders)
        friendlies = [DefenseAgent(i, NUM_DEFENDERS, verbose=False) for i in range(NUM_DEFENDERS)]
        
        # Metrics for this run
        violation_count = 0
        step = 0
        terminated = False
        outcome = "TIMEOUT" # Default
        
        # --- EPISODE LOOP ---
        while not terminated and step < MAX_STEPS:
            
            # A. State Parsing (Read directly from ENV to ensure sync)
            friendly_states = []
            for idx in range(NUM_DEFENDERS):
                state = env._getDroneStateVector(idx)
                friendly_states.append({
                    'id': idx,
                    'pos': state[0:3],
                    'vel': state[10:13],
                    'quat': state[3:7],
                    'ang_vel': state[13:16]
                })

            # B. Parse Enemies (Read from ENV)
            # This ensures we never process a "Ghost Drone" because the env list shrinks when they die
            current_enemies = []
            if len(env.enemy_positions) > 0:
                for e_idx, e_pos in enumerate(env.enemy_positions):
                    current_enemies.append({
                        'id': e_idx,
                        'pos': e_pos,
                        'vel': np.zeros(3) # Velocity approx
                    })

            # C. Compute Actions (Defenders)
            action = np.zeros((NUM_DEFENDERS, 4))
            for idx, agent in enumerate(friendlies):
                # Send full enemy list to strategy
                rpm = agent.compute_action(
                    my_state=friendly_states[idx],
                    neighbors=friendly_states,
                    enemies=current_enemies,
                    asset_pos=env.BASE_POS,
                    dt=env.CTRL_TIMESTEP,
                    time_now=step * env.CTRL_TIMESTEP
                )
                action[idx, :] = rpm

            # D. Step Physics
            obs, reward, terminated, truncated, info = env.step(action)
            
            # E. Combat Logic (Kinetic Ramming)
            # We iterate backwards to allow deletion without index errors
            enemies_to_remove = []
            if len(env.enemy_positions) > 0:
                for e_idx, e_pos in enumerate(env.enemy_positions):
                    # Check Victory/Loss Logic
                    dist_to_base = np.linalg.norm(e_pos - env.BASE_POS)
                    
                    # 1. LOSS CONDITION: Enemy hits base
                    if dist_to_base < 1.0:
                        outcome = "LOSS"
                        terminated = True
                        break
                    
                    # 2. NEUTRALIZATION: Defender Rams Enemy
                    for f_state in friendly_states:
                        # Ramming distance (2.0m for 8kg drones)
                        if np.linalg.norm(f_state['pos'] - e_pos) < 2.0:
                            enemies_to_remove.append(e_idx)
                            break # Enemy dead, move to next enemy

            # Apply Removals (Physically remove from PyBullet)
            if enemies_to_remove:
                # Remove duplicates and sort descending to keep indices valid
                for e_idx in sorted(list(set(enemies_to_remove)), reverse=True):
                    env.neutralize_enemy(e_idx) # CALLS METHOD IN BaseDefenseAviary

            # F. Check Victory (All enemies gone)
            if len(env.enemy_positions) == 0:
                outcome = "WIN"
                terminated = True

            # G. Check Strategy Violation (Unattended)
            if info.get("unattended_violation", False):
                violation_count += 1
            
            step += 1

        # --- RUN COMPLETE ---
        # Calculate friendly losses (drones that crashed/flipped)
        # Simple heuristic: If Z < 0.2, drone is crashed
        friendly_losses = sum(1 for f in friendly_states if f['pos'][2] < 0.2)

        run_data = {
            "Run_ID": run_id,
            "Enemies": num_enemies,
            "Outcome": outcome,
            "Success": 1 if outcome == "WIN" else 0,
            "Steps": step,
            "Violations": violation_count,
            "Friendly_Losses": friendly_losses
        }
        results.append(run_data)
        
        # Progress Bar (Modified to print every run for small batch)
        print(f"Run {run_id + 1}/{NUM_SIMULATIONS} | Last: {outcome} | Violations: {violation_count}")

    # --- SAVE & REPORT ---
    env.close()
    
    df = pd.DataFrame(results)
    df.to_csv("monte_carlo_results.csv", index=False)
    
    success_rate = (df['Success'].mean() * 100)
    print("\n" + "="*40)
    print("FINAL MONTE CARLO REPORT")
    print("="*40)
    print(f"Total Simulations:   {NUM_SIMULATIONS}")
    print(f"Success Rate:        {success_rate:.2f}%")
    print(f"Avg Time per Battle: {df['Steps'].mean() / 240:.2f} sec") # Assuming 240Hz
    print(f"Avg Violations:      {df['Violations'].mean():.2f}")
    print(f"Total Time Elapsed:  {time.time() - start_time_global:.2f} sec")
    print("="*40)

if __name__ == "__main__":
    run_monte_carlo()
