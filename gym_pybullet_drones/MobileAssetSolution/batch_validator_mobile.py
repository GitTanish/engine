import os
import time
import numpy as np
import pandas as pd
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path: sys.path.insert(0, root_dir)

from gym_pybullet_drones.control.DefenseAgent import DefenseAgent
from MobileDefenseAviary import MobileDefenseAviary


SCENARIOS = {
    "MOBILE_A": {"num_enemies": 10, "spawn_radius": 500.0, "ground_attacker_ratio": 0.3},
    "MOBILE_B": {"num_enemies": 13, "spawn_radius": 500.0, "ground_attacker_ratio": 0.46}
}
NUM_EPISODES_PER_SCENARIO = 5
OUTPUT_FILE = "validation_mobile_results.csv"

def run_batch():
    results = []
    
    # Headless Mobile Env
    env = MobileDefenseAviary(num_drones=10, gui=False)
    
    for scenario_name, config in SCENARIOS.items():
        print(f"\n--- Starting {scenario_name} ({NUM_EPISODES_PER_SCENARIO} Episodes) ---")
        print(f"Config: {config}", flush=True)

        for i in range(NUM_EPISODES_PER_SCENARIO):
            agents = [DefenseAgent(k, 10, verbose=False) for k in range(10)]
            obs, _ = env.reset(scenario_config=config)
            step = 0
            outcome = "TIMEOUT"
            
            while step < 2000:
                step += 1
                
                # Mock Perception
                all_states = [env._getDroneStateVector(k) for k in range(10)]
                neighbors = [{'id': k, 'pos': all_states[k][0:3]} for k in range(10) if k not in env.dead_friendly_indices]
                enemies = [{'id': k, 'pos': env.enemy_positions[k], 'mode': env.enemy_agents[k].mode} for k in range(len(env.enemy_agents)) if k < len(env.enemy_positions)]
                
                actions = np.zeros((10, 4))
                for k, agent in enumerate(agents):
                    if k not in env.dead_friendly_indices:
                        my_state = {'pos': all_states[k][0:3], 'vel': all_states[k][10:13], 'quat': all_states[k][3:7], 'ang_vel': all_states[k][13:16]}
                        # CRITICAL: Pass dynamic env.BASE_POS
                        actions[k, :] = agent.compute_action(my_state, neighbors, enemies, env.BASE_POS, env.CTRL_TIMESTEP, step*env.CTRL_TIMESTEP)
                
                obs, _, _, _, info = env.step(actions)
                
                if info.get("unattended_violation", False):
                    outcome = "DEFEAT_UNATTENDED"
                    break
                if len(env.enemy_positions) > 0:
                    dists = np.linalg.norm(env.enemy_positions - env.BASE_POS, axis=1) # Check dist to moving base
                    if np.any(dists < 2.0):
                        outcome = "DEFEAT_ASSET_HIT"
                        break
                if len(env.dead_friendly_indices) == 10:
                    outcome = "DEFEAT_WIPEOUT"
                    break
                if len(env.enemy_positions) == 0:
                    outcome = "VICTORY"
                    break
            
            print(f"[{scenario_name}] Ep {i+1} | Result: {outcome} | Casualties: {len(env.dead_friendly_indices)}", flush=True)
            results.append({
                "Scenario": scenario_name,
                "Episode": i+1, 
                "Outcome": outcome, 
                "Casualties": len(env.dead_friendly_indices), 
                "Win": 1 if outcome == "VICTORY" else 0
            })
        
    env.close()
    
    df = pd.DataFrame(results)
    print("\n--- Summary ---")
    print(df.groupby("Scenario")["Win"].mean())
    df.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    run_batch()
