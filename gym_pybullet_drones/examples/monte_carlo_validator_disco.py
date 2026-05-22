"""
Monte Carlo Validator Script (WITH COMBAT LOGIC)
Runs N simulations in HEADLESS mode (no GUI) to calculate success rates.
"""
import time
import numpy as np
import pandas as pd
import sys
import os

# --- PATH SETUP (Same as your visualizer) ---
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
    # We do NOT import EnemyAgent here because the Environment handles them now
except ImportError as e:
    # Fallback for flattened structure
    inner_package_dir = os.path.join(project_root, 'gym_pybullet_drones')
    sys.path.append(inner_package_dir) 
    from envs.BaseDefenseAviary import BaseDefenseAviary
    from utils.enums import DroneModel, Physics
    from control.DefenseAgent import DefenseAgent

# --- CONFIGURATION ---
NUM_SIMULATIONS = 100
NUM_DEFENDERS = 10

def run_monte_carlo():
    results = []
    print(f"[INFO] Starting {NUM_SIMULATIONS} Monte Carlo runs in HEADLESS mode...")
    
    # Initialize Environment
    # We use Dynamic Steps based on frequency (e.g. 240Hz * 60s = 14400 steps)
    env = BaseDefenseAviary(
        drone_model=DroneModel.HEAVY,
        num_drones=NUM_DEFENDERS, 
        physics=Physics.PYB,
        gui=False,       
        record=False,   
        obstacles=False
    )
    
    # DYNAMIC MAX STEPS CALCULATION
    DESIRED_FLIGHT_TIME_SEC = 60
    MAX_STEPS = int(DESIRED_FLIGHT_TIME_SEC * env.CTRL_FREQ)
    print(f"[INFO] Simulation Frequency: {env.CTRL_FREQ}Hz")
    print(f"[INFO] Dynamic MAX_STEPS set to: {MAX_STEPS}")

    start_time = time.time()

    for run_id in range(NUM_SIMULATIONS):
        # 1. Randomize Scenario
        num_enemies = np.random.randint(3, 8) 
        scenario_config = {
            "num_enemies": num_enemies,
            "spawn_radius": np.random.uniform(200.0, 300.0),
            "ground_attacker_ratio": np.random.uniform(0.6, 1.0)
        }
        
        # 2. Reset
        try:
            obs, info = env.reset(scenario_config=scenario_config)
        except TypeError:
            obs, info = env.reset()

        # 3. Init Agents
        friendlies = [DefenseAgent(i, NUM_DEFENDERS, verbose=False) for i in range(NUM_DEFENDERS)]
        
        # --- TRACKING ACTIVE UNITS ---
        active_enemies = [True] * num_enemies
        active_friendlies = [True] * NUM_DEFENDERS

        # Metrics
        violation_count = 0
        enemies_hit_base = 0
        step = 0
        terminated = False
        
        # --- EPISODE LOOP ---
        while not terminated and step < MAX_STEPS:
            # Prepare State for Agents
            friendly_states = []
            enemy_states = [] 
            
            # Parse Friendlies
            for idx in range(NUM_DEFENDERS):
                state_vec = env._getDroneStateVector(idx)
                data = {
                    'id': idx,
                    'pos': state_vec[0:3],
                    'vel': state_vec[10:13],
                    'quat': state_vec[3:7],
                    'ang_vel': state_vec[13:16]
                }
                friendly_states.append(data)

            # Parse Enemies (Only Active Ones)
            if hasattr(env, 'enemy_positions'):
                for e_idx, e_pos in enumerate(env.enemy_positions):
                    # Only parse if active (optimization)
                    if active_enemies[e_idx]:
                        e_data = {
                            'id': NUM_DEFENDERS + e_idx,
                            'pos': e_pos,
                            'vel': np.zeros(3), 
                            'quat': np.array([0,0,0,1]), 
                            'ang_vel': np.zeros(3)
                        }
                        enemy_states.append(e_data)
                    else:
                        # Append dummy or skip, but list index alignment matters for logic below
                        enemy_states.append(None) 

            # Compute Actions (Friendly Only)
            action = np.zeros((NUM_DEFENDERS, 4))
            
            # Filter enemies list for the agent (remove Nones)
            visible_enemies = [e for e in enemy_states if e is not None]

            for idx, agent in enumerate(friendlies):
                if active_friendlies[idx]: # Only alive drones act
                    rpm = agent.compute_action(
                        my_state=friendly_states[idx],
                        neighbors=friendly_states,
                        enemies=visible_enemies,
                        asset_pos=np.array([0,0,0]),
                        dt=env.CTRL_TIMESTEP,
                        time_now=step * env.CTRL_TIMESTEP
                    )
                    action[idx, :] = rpm
                else:
                    action[idx, :] = np.zeros(4)

            # Step Environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # --- DYNAMIC COMBAT LOGIC ---
            FIRING_RANGE = 250.0
            
            # Blue shoots Red
            for f_idx, friendly in enumerate(friendly_states):
                if not active_friendlies[f_idx]: continue
                
                for e_idx, enemy in enumerate(enemy_states):
                    if enemy is not None and active_enemies[e_idx]:
                        dist = np.linalg.norm(friendly['pos'] - enemy['pos'])
                        if dist < FIRING_RANGE:
                            # Accuracy Falloff
                            hit_chance = (1.0 - (dist / FIRING_RANGE)) * 0.10
                            
                            if np.random.random() < hit_chance:
                                active_enemies[e_idx] = False
                                # Teleport dead enemy (Visual hack, not strictly needed for logic but good for consistency)
                                # p.resetBasePositionAndOrientation(...) # Skipped in headless

            # Red shoots Blue
            for e_idx, enemy in enumerate(enemy_states):
                if enemy is not None and active_enemies[e_idx]:
                    for f_idx, friendly in enumerate(friendly_states):
                        if active_friendlies[f_idx]:
                            dist = np.linalg.norm(enemy['pos'] - friendly['pos'])
                            if dist < FIRING_RANGE:
                                # Accuracy Falloff
                                hit_chance = (1.0 - (dist / FIRING_RANGE)) * 0.10
                                
                                if np.random.random() < hit_chance:
                                    active_friendlies[f_idx] = False

            # --- CHECK FAILURE CONDITIONS ---
            
            # 1. Victory Check
            if not any(active_enemies):
                terminated = True # Victory!
                break

            # 2. Unattended Rule Violation
            if info.get("unattended_violation", False):
                violation_count += 1
            
            # 3. Base Collision Check
            for e_idx, enemy in enumerate(enemy_states):
                if enemy is not None and active_enemies[e_idx]:
                    if np.linalg.norm(enemy['pos']) < 1.0: # Base Hit
                        enemies_hit_base += 1
                        terminated = True
            
            step += 1

        # --- END OF EPISODE ---
        is_success = (enemies_hit_base == 0) and (not any(active_enemies))
        
        # Log Data
        run_data = {
            "Run_ID": run_id,
            "Enemies": num_enemies,
            "Success": is_success,
            "Unattended_Violations": violation_count,
            "Steps": step,
            "Friendly_Losses": NUM_DEFENDERS - sum(active_friendlies)
        }
        results.append(run_data)
        
        # Print progress
        if (run_id + 1) % 10 == 0:
            print(f"Run {run_id + 1}/{NUM_SIMULATIONS} | Success: {is_success} | Violations: {violation_count}")

        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv("monte_carlo_results_combat.csv", index=False)

    env.close()
    
    # --- REPORTING ---
    print("\n" + "="*30)
    print("MONTE CARLO RESULTS (COMBAT)")
    print("="*30)
    print(f"Total Simulations: {NUM_SIMULATIONS}")
    print(f"Success Rate:      {(df['Success'].mean() * 100):.2f}%")
    print(f"Avg Violations:    {df['Unattended_Violations'].mean():.2f}")
    print(f"Avg Friendly Loss: {df['Friendly_Losses'].mean():.2f}")
    print(f"Time Elapsed:      {time.time() - start_time:.2f}s")
    print("="*30)
    print("Detailed results saved to 'monte_carlo_results_combat.csv'")

if __name__ == "__main__":
    run_monte_carlo()
