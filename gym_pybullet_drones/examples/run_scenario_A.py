"""
SCENARIO A: PARITY
Requirement: "Attacker and defender have same number of drones... 30% are ground attack."
"""
import time
import numpy as np
import pybullet as p
import sys
import os

# --- PATH SETUP ---
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
# Add the PARENT of the project root to sys.path to allow 'import gym_pybullet_drones'
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path: sys.path.insert(0, parent_dir)

from gym_pybullet_drones.envs.BaseDefenseAviary import BaseDefenseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DefenseAgent import DefenseAgent
from gym_pybullet_drones.utils.utils import sync



def run():
    # 1. CONFIGURATION
    NUM_DEFENDERS = 10
    NUM_ENEMIES = 10  # Parity
    SPAWN_RADIUS = 400.0
    GROUND_RATIO = 0.3 # 30% Bombers (3 Bombers, 7 Fighters)

    # 2. INIT ENVIRONMENT
    print(f"[SCENARIO A] Initializing: {NUM_DEFENDERS} Blue vs {NUM_ENEMIES} Red (Parity)")
    
    # Init Locations (Ring)
    # Dynamic Spawn Radius to prevent collision on start (same logic as Agent)
    spawn_ring_radius = max(5.0, (NUM_DEFENDERS * 4.0) / (2 * np.pi))
    init_xyzs = np.array([[spawn_ring_radius*np.cos((2*np.pi/NUM_DEFENDERS)*i), spawn_ring_radius*np.sin((2*np.pi/NUM_DEFENDERS)*i), 1.0] for i in range(NUM_DEFENDERS)])
    
    env = BaseDefenseAviary(drone_model=DroneModel.HEAVY, num_drones=NUM_DEFENDERS, 
                            initial_xyzs=init_xyzs, initial_rpys=np.zeros((NUM_DEFENDERS, 3)),
                            physics=Physics.PYB, gui=True, user_debug_gui=True)

    # 3. AGENTS
    agents = [DefenseAgent(i, NUM_DEFENDERS) for i in range(NUM_DEFENDERS)]

    # 4. RUN LOOP
    for episode in range(3):
        print(f"\n[INFO] Starting Episode {episode+1}")
        
        # STRICT CONFIG INJECTION
        scenario_config = {
            "num_enemies": NUM_ENEMIES,
            "spawn_radius": SPAWN_RADIUS,
            "ground_attacker_ratio": GROUND_RATIO
        }
        obs, info = env.reset(scenario_config=scenario_config)
        

        

        
        # Visuals
        for i in range(NUM_DEFENDERS):
            p.changeVisualShape(env.DRONE_IDS[i], -1, rgbaColor=[0, 0, 1, 1], physicsClientId=env.CLIENT)

        start_time = time.time()
        step = 0
        while True:
            # Parse State
            friendly_states = []
            for idx in range(NUM_DEFENDERS):
                s = env._getDroneStateVector(idx)
                friendly_states.append({'id': idx, 'pos': s[0:3], 'quat': s[3:7], 'vel': s[10:13], 'ang_vel': s[13:16]})

            current_enemies = []
            for e_idx, e_pos in enumerate(env.enemy_positions):
                current_enemies.append({
                    'id': e_idx, 'pos': e_pos, 'vel': np.zeros(3),
                    'mode': env.enemy_agents[e_idx].mode # Essential for Priority Logic
                })

            # Compute Actions
            actions = np.zeros((NUM_DEFENDERS, 4))
            

            for idx, agent in enumerate(agents):
                # Only alive drones act
                if idx not in env.dead_friendly_indices:
                    actions[idx, :] = agent.compute_action(
                        friendly_states[idx], friendly_states, current_enemies, 
                        np.array([0,0,0]), env.CTRL_TIMESTEP, step/240.0
                    )
            
            # Step
            obs, _, _, _, info = env.step(actions)
            sync(step, start_time, env.CTRL_TIMESTEP)
            
            # End Conditions
            if len(env.enemy_positions) == 0:
                print("[VICTORY] Swarm Neutralized.")
                break
            if info.get("unattended_violation", False):
                # print("[WARN] Unattended Violation Detected") 
                pass
            
            # Loss Check (Asset Hit)
            dists = np.linalg.norm(env.enemy_positions, axis=1)
            if np.any(dists < 1.0):
                print("[FAILURE] Asset Destroyed.")
                break
            
            # Loss Check (All Defenders Dead)
            if len(env.dead_friendly_indices) >= NUM_DEFENDERS:
                 print("[FAILURE] All Defenders Destroyed.")
                 break
                
            step += 1
            
        time.sleep(2)
    env.close()

if __name__ == "__main__":
    run()
