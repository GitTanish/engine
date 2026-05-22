"""
SCENARIO B: ASYMMETRIC
Requirement: "Attacker have 30% extra drone... with ground attack capability."
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



def draw_enemy_tags(env, enemies):
    """
    Draws text tags above enemies for better visibility.
    """
    for enemy in enemies:
        pos = enemy['pos']
        vel = enemy['vel']
        
        # Tag above drone
        p.addUserDebugText(
            text="ENEMY",
            textPosition=[pos[0], pos[1], pos[2] + 2.0],
            textColorRGB=[1, 0, 0],
            textSize=1.5,
            lifeTime=0.1,
            physicsClientId=env.CLIENT
        )
        
        # Velocity Vector (Yellow)
        if np.linalg.norm(vel) > 0.1:
            p.addUserDebugLine(
                lineFromXYZ=pos,
                lineToXYZ=pos + (vel * 2.0),
                lineColorRGB=[1, 1, 0],
                lineWidth=2,
                lifeTime=0.1,
                physicsClientId=env.CLIENT
            )

def run():
    # 1. CONFIGURATION
    NUM_DEFENDERS = 10
    NUM_ENEMIES = 13  # 30% Extra (10 + 3)
    SPAWN_RADIUS = 400.0
    
    # Math: 10 Base (3 Bombers) + 3 Extra (3 Bombers) = 6 Bombers Total.
    # Total Enemies = 13.
    # Ratio = 6 / 13 = 0.4615
    GROUND_RATIO = 0.462 

    # 2. INIT ENVIRONMENT
    print(f"[SCENARIO B] Initializing: {NUM_DEFENDERS} Blue vs {NUM_ENEMIES} Red (Asymmetric)")
    
    init_xyzs = np.array([[5*np.cos((2*np.pi/NUM_DEFENDERS)*i), 5*np.sin((2*np.pi/NUM_DEFENDERS)*i), 1.0] for i in range(NUM_DEFENDERS)])
    
    env = BaseDefenseAviary(drone_model=DroneModel.HEAVY, num_drones=NUM_DEFENDERS, 
                            initial_xyzs=init_xyzs, initial_rpys=np.zeros((NUM_DEFENDERS, 3)),
                            physics=Physics.PYB, gui=True, user_debug_gui=True)

    # 3. AGENTS
    agents = [DefenseAgent(i, NUM_DEFENDERS) for i in range(NUM_DEFENDERS)]

    # 4. RUN LOOP
    for episode in range(3):
        print(f"\n[INFO] Starting Episode {episode+1}")
        
        scenario_config = {
            "num_enemies": NUM_ENEMIES,
            "spawn_radius": SPAWN_RADIUS,
            "ground_attacker_ratio": GROUND_RATIO
        }
        obs, info = env.reset(scenario_config=scenario_config)
        
        # --- CAMERA FIX: GOD VIEW ---
        # Distance: 150m (Close enough to see drones, far enough to see formation)
        # Yaw: 0 (North)
        # Pitch: -45 (Bird's Eye angled down)
        # Target: [0, 0, 90] (The center of the battle at Operational Altitude)
        p.resetDebugVisualizerCamera(
            cameraDistance=200.0, 
            cameraYaw=0, 
            cameraPitch=-45, 
            cameraTargetPosition=[0, 0, 90], 
            physicsClientId=env.CLIENT
        )

        

        
        for i in range(NUM_DEFENDERS):
            p.changeVisualShape(env.DRONE_IDS[i], -1, rgbaColor=[0, 0, 1, 1], physicsClientId=env.CLIENT)

        start_time = time.time()
        step = 0
        last_enemy_positions = {}
        
        while True:
            friendly_states = []
            for idx in range(NUM_DEFENDERS):
                s = env._getDroneStateVector(idx)
                friendly_states.append({'id': idx, 'pos': s[0:3], 'quat': s[3:7], 'vel': s[10:13], 'ang_vel': s[13:16]})

            current_enemies = []
            for e_idx, e_pos in enumerate(env.enemy_positions):
                # Calculate Velocity
                vel = np.zeros(3)
                if e_idx in last_enemy_positions:
                    dt = env.CTRL_TIMESTEP
                    vel = (e_pos - last_enemy_positions[e_idx]) / dt
                
                last_enemy_positions[e_idx] = e_pos
                
                current_enemies.append({
                    'id': e_idx, 'pos': e_pos, 'vel': vel,
                    'mode': env.enemy_agents[e_idx].mode 
                })

            actions = np.zeros((NUM_DEFENDERS, 4))
            

            actions = np.zeros((NUM_DEFENDERS, 4))
            
            # Draw Tags
            draw_enemy_tags(env, current_enemies)

            for idx, agent in enumerate(agents):
                if idx not in env.dead_friendly_indices:
                    actions[idx, :] = agent.compute_action(
                        friendly_states[idx], friendly_states, current_enemies, 
                        np.array([0,0,0]), env.CTRL_TIMESTEP, step/240.0
                    )
            
            obs, _, _, _, info = env.step(actions)
            sync(step, start_time, env.CTRL_TIMESTEP)
            
            if len(env.enemy_positions) == 0:
                print("[VICTORY] Swarm Neutralized.")
                break
            
            dists = np.linalg.norm(env.enemy_positions, axis=1)
            if np.any(dists < 1.0):
                print("[FAILURE] Asset Destroyed.")
                break
                
            step += 1
            
        time.sleep(2)
    env.close()

if __name__ == "__main__":
    run()
