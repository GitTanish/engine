"""
TEST SCRIPT: DYNAMIC SCALING
Verifies that 50 drones form a stable ring without jamming.
"""
import time
import numpy as np
import pybullet as p
import sys
import os

# --- PATH SETUP ---
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path: sys.path.insert(0, parent_dir)

from gym_pybullet_drones.envs.BaseDefenseAviary import BaseDefenseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DefenseAgent import DefenseAgent
from gym_pybullet_drones.utils.utils import sync

def run():
    # 1. CONFIGURATION
    NUM_DEFENDERS = 50 # Stress Test
    NUM_ENEMIES = 0    # No enemies, just formation check
    
    print(f"[TEST] Initializing Scaling Test with {NUM_DEFENDERS} Drones")
    
    # Init Locations (Ring) - Start them in a smaller ring to watch them expand
    init_xyzs = np.array([[10*np.cos((2*np.pi/NUM_DEFENDERS)*i), 10*np.sin((2*np.pi/NUM_DEFENDERS)*i), 1.0] for i in range(NUM_DEFENDERS)])
    
    env = BaseDefenseAviary(drone_model=DroneModel.HEAVY, num_drones=NUM_DEFENDERS, 
                            initial_xyzs=init_xyzs, initial_rpys=np.zeros((NUM_DEFENDERS, 3)),
                            physics=Physics.PYB, gui=True, user_debug_gui=True)

    # 3. AGENTS
    agents = [DefenseAgent(i, NUM_DEFENDERS) for i in range(NUM_DEFENDERS)]

    # 4. RUN LOOP
    print(f"\n[INFO] Starting Formation Test...")
    
    # STRICT CONFIG INJECTION
    scenario_config = {
        "num_enemies": 0,
        "spawn_radius": 400.0,
        "ground_attacker_ratio": 0.0
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

        # Compute Actions
        actions = np.zeros((NUM_DEFENDERS, 4))
        for idx, agent in enumerate(agents):
            actions[idx, :] = agent.compute_action(
                friendly_states[idx], friendly_states, [], 
                np.array([0,0,0]), env.CTRL_TIMESTEP, step/240.0
            )
        
        # Step
        obs, _, _, _, info = env.step(actions)
        sync(step, start_time, env.CTRL_TIMESTEP)
        
        # Print Radius Check
        if step % 240 == 0:
            # Calculate average radius of swarm
            positions = np.array([s['pos'] for s in friendly_states])
            radii = np.linalg.norm(positions[:, :2], axis=1)
            avg_radius = np.mean(radii)
            print(f"[STEP {step}] Swarm Radius: {avg_radius:.2f}m")

        step += 1
        
    env.close()

if __name__ == "__main__":
    run()
