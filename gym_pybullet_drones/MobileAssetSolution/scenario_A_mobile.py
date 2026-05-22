import time
import numpy as np
import pybullet as p
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path: sys.path.insert(0, root_dir)

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DefenseAgent import DefenseAgent
from MobileDefenseAviary import MobileDefenseAviary

def run():
    print("[MOBILE SCENARIO A] 10 vs 10 Parity | Moving Asset")
    
    NUM_DEFENDERS = 10
    NUM_ENEMIES = 10
    
    env = MobileDefenseAviary(
        drone_model=DroneModel.HEAVY, 
        num_drones=NUM_DEFENDERS, 
        physics=Physics.PYB, 
        gui=True,
        user_debug_gui=True
    )

    agents = [DefenseAgent(i, NUM_DEFENDERS) for i in range(NUM_DEFENDERS)]
    
    # 30% Bombers
    config = {"num_enemies": NUM_ENEMIES, "spawn_radius": 500.0, "ground_attacker_ratio": 0.3}
    obs, info = env.reset(scenario_config=config)
    
    step = 0
    while True:
        # Get States
        all_states = [env._getDroneStateVector(k) for k in range(NUM_DEFENDERS)]
        friendly_states = [{'id': k, 'pos': all_states[k][0:3], 'vel': all_states[k][10:13], 'quat': all_states[k][3:7], 'ang_vel': all_states[k][13:16]} for k in range(NUM_DEFENDERS)]
        
        # Get Mobile Enemies
        current_enemies = []
        for e_idx, e_pos in enumerate(env.enemy_positions):
            mode = env.enemy_agents[e_idx].mode
            current_enemies.append({'id': e_idx, 'pos': e_pos, 'vel': np.zeros(3), 'mode': mode})

        # Compute Actions (Passing DYNAMIC Base Pos)
        actions = np.zeros((NUM_DEFENDERS, 4))
        for idx, agent in enumerate(agents):
            if idx not in env.dead_friendly_indices:
                actions[idx, :] = agent.compute_action(
                    friendly_states[idx], friendly_states, current_enemies, 
                    env.BASE_POS, env.CTRL_TIMESTEP, step*env.CTRL_TIMESTEP
                )

        obs, _, _, _, info = env.step(actions)
        p.stepSimulation()
        
        # Camera Follow
        if step % 50 == 0:
            p.resetDebugVisualizerCamera(cameraDistance=30, cameraYaw=-40, cameraPitch=-30, cameraTargetPosition=env.BASE_POS)

        time.sleep(env.CTRL_TIMESTEP)
        step += 1
        
        if len(env.enemy_positions) == 0:
            print("[VICTORY] Swarm Neutralized.")
            break

if __name__ == "__main__":
    run()
