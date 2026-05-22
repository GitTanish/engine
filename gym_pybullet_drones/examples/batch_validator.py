import numpy as np
import time
import pandas as pd
import sys
import os

# --- PATH SETUP ---
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
# Add the PARENT of the project root to sys.path to allow 'import gym_pybullet_drones'
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path: sys.path.insert(0, parent_dir)

try:
    from gym_pybullet_drones.envs.BaseDefenseAviary import BaseDefenseAviary
    from gym_pybullet_drones.control.DefenseAgent import DefenseAgent
    from gym_pybullet_drones.utils.enums import DroneModel, Physics
except ImportError as e:
    print(f"[ERROR] Import failed: {e}. Check your PYTHONPATH.")
    sys.exit(1)

def run_batch(scenario_type, num_runs=100):
    """
    Executes Monte Carlo Validation.
    Scenario A: Parity (10 vs 10).
    Scenario B: Asymmetric (10 vs 13).
    """
    
    # --- CONFIGURATION FROM MDR ---
    NUM_DEFENDERS = 10
    
    if scenario_type == 'A':
        # "Same number of drones... 30% ground attack"
        NUM_ENEMIES = 10
        GROUND_RATIO = 0.3  # 3 Bombers, 7 Fighters
    elif scenario_type == 'B':
        # "30% extra drone with ground attack capability"
        # 10 Base + 3 Extra = 13 Total.
        # Bombers = 3 (Base) + 3 (Extra) = 6. Fighters = 7.
        NUM_ENEMIES = 13
        GROUND_RATIO = 0.462 # 6 / 13
    
    results = []
    
    print(f"\n=== STARTING BATCH: SCENARIO {scenario_type} ===")
    print(f"Defenders: {NUM_DEFENDERS} | Enemies: {NUM_ENEMIES} | Ground Ratio: {GROUND_RATIO:.2f}")
    print(f"Iterations: {num_runs} | Mode: Headless (Fast)")

    start_batch_time = time.time()

    for run_id in range(num_runs):
        # 1. Setup Env (Headless)
        env = BaseDefenseAviary(drone_model=DroneModel.HEAVY, num_drones=NUM_DEFENDERS, 
                                initial_xyzs=None, initial_rpys=None,
                                physics=Physics.PYB, gui=False, record=False) # No GUI
        
        # 2. Setup Agents
        agents = [DefenseAgent(i, NUM_DEFENDERS, verbose=False) for i in range(NUM_DEFENDERS)]
        
        # 3. Scenario Config
        config = {
            "num_enemies": NUM_ENEMIES,
            "spawn_radius": 400.0,
            "ground_attacker_ratio": GROUND_RATIO
        }
        
        obs, info = env.reset(scenario_config=config)
        
        step = 0
        max_steps = 240 * 50 # 50 seconds max (Time limit)
        
        violation_count = 0
        outcome = "DRAW" # Default if time runs out
        
        # Simulation Loop
        while step < max_steps:
            # A. Parse Friendly State
            friendly_states = []
            for idx in range(NUM_DEFENDERS):
                s = env._getDroneStateVector(idx)
                friendly_states.append({'id': idx, 'pos': s[0:3], 'quat': s[3:7], 'vel': s[10:13], 'ang_vel': s[13:16]})

            # B. Parse Enemy State (With Mode!)
            current_enemies = []
            for e_idx, e_pos in enumerate(env.enemy_positions):
                current_enemies.append({
                    'id': e_idx, 'pos': e_pos, 'vel': np.zeros(3),
                    'mode': env.enemy_agents[e_idx].mode 
                })

            # C. Compute Actions
            actions = np.zeros((NUM_DEFENDERS, 4))
            for idx, agent in enumerate(agents):
                if idx not in env.dead_friendly_indices:
                    actions[idx, :] = agent.compute_action(
                        friendly_states[idx], friendly_states, current_enemies, 
                        np.array([0,0,0]), env.CTRL_TIMESTEP, step/240.0
                    )
            
            # D. Step Env
            obs, _, _, _, info = env.step(actions)
            
            # E. Metrics Logic
            if info.get("unattended_violation", False):
                violation_count += 1
            
            # Win Condition: All enemies dead
            if len(env.enemy_positions) == 0:
                outcome = "WIN"
                break
            
            # Loss Condition: Asset Destroyed
            if len(env.enemy_positions) > 0:
                dists = np.linalg.norm(env.enemy_positions, axis=1)
                if np.any(dists < 1.0):
                    outcome = "LOSS_ASSET"
                    break
            
            # Loss Condition: All Defenders Dead
            if len(env.dead_friendly_indices) >= NUM_DEFENDERS:
                outcome = "LOSS_ANNIHILATION"
                break
                
            step += 1
            
        # End of Run Processing
        friendly_losses = info.get("friendly_losses", 0)
        env.close()
        
        # Log Progress
        if (run_id + 1) % 10 == 0:
            print(f"Run {run_id+1}/{num_runs} Complete. Last Result: {outcome}")

        results.append({
            "Scenario": scenario_type,
            "Run_ID": run_id,
            "Outcome": outcome,
            "Friendly_Losses": friendly_losses,
            "Violations": violation_count,
            "Steps": step
        })
        
    batch_duration = time.time() - start_batch_time
    print(f"Batch Complete in {batch_duration:.1f}s")
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Initialize Monte Carlo Validator...")
    
    # 1. Run Scenario A (Parity)
    df_a = run_batch('A', 100)
    
    # 2. Run Scenario B (Asymmetric)
    df_b = run_batch('B', 100)
    
    # 3. Combine & Save
    full_report = pd.concat([df_a, df_b])
    full_report.to_csv("validation_results.csv", index=False)
    
    # 4. Generate Report
    print("\n" + "="*40)
    print("FINAL VALIDATION REPORT")
    print("="*40)
    
    for scen in ['A', 'B']:
        df = full_report[full_report['Scenario'] == scen]
        total = len(df)
        wins = len(df[df['Outcome'] == 'WIN'])
        asset_losses = len(df[df['Outcome'] == 'LOSS_ASSET'])
        annihilation_losses = len(df[df['Outcome'] == 'LOSS_ANNIHILATION'])
        draws = len(df[df['Outcome'] == 'DRAW'])
        
        avg_losses = df['Friendly_Losses'].mean()
        avg_violations = df['Violations'].mean()
        
        print(f"\nSCENARIO {scen} ({total} runs):")
        print(f"  Win Rate:       {(wins/total)*100:.1f}%")
        print(f"  Asset Losses:   {asset_losses}")
        print(f"  Annihilations:  {annihilation_losses}")
        print(f"  Draws:          {draws}")
        print(f"  Avg Friendly Dead: {avg_losses:.1f} / 10")
        print(f"  Avg Unattended Frames: {avg_violations:.1f}")
        print("-" * 20)
