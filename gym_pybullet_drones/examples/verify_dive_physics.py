import pybullet as p
import numpy as np
import time
import sys
import os

# Path hack to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gym_pybullet_drones.envs.BaseDefenseAviary import BaseDefenseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DefenseAgent import DefenseAgent

def verify_dive():
    print("TEST: Initializing Diving Verification...")
    
    # 1. Setup Environment (Minimal)
    # 1 Defender, 1 Enemy specific config
    env = BaseDefenseAviary(drone_model=DroneModel.HEAVY, num_drones=1, gui=False, physics=Physics.PYB)
    
    # Manually overwrite the Enemy setup to force a Low-Altitude Bomber
    # We want a static bomber at Z=20m (Well below the old 60m floor)
    target_z_test = 20.0
    
    # Reset env first
    env.reset()
    
    # Hack: Inject our custom enemy state
    # We will manually mock the enemy_positions in the loop to be stationary at low altitude
    
    agent = DefenseAgent(0, 1)
    
    print(f"TEST: Spawning Defender high (100m) and Enemy Bomber low ({target_z_test}m)...")
    
    min_z_reached = 100.0
    
    # Run for 5 seconds
    for i in range(240 * 5):
        # Fake Enemy State: 1 Bomber at [10, 10, 20]
        # Close enough to trigger intercept, low enough to test diving
        enemy_pos = np.array([10, 10, target_z_test]) 
        
        # Override environment enemy state
        env.enemy_positions = np.array([enemy_pos])
        env.enemy_agents = [] # Don't need actual agents, just the list length/state for the defense agent
        
        # We need to construct a 'fake' enemy object dictionary for the DefenseAgent
        # DefenseAgent expects dict with 'pos', 'vel', 'mode'
        mock_enemies = [{
            'id': 0,
            'pos': enemy_pos,
            'vel': np.zeros(3),
            'mode': 0 # MODE 0 -> BOMBER (Should trigger dive)
        }]
        
        # Get Defender State
        state_vec = env._getDroneStateVector(0)
        my_state = {
            'pos': state_vec[0:3],
            'quat': state_vec[3:7],
            'vel': state_vec[10:13],
            'ang_vel': state_vec[13:16]
        }
        
        current_z = my_state['pos'][2]
        if current_z < min_z_reached:
            min_z_reached = current_z
            
        # print(f"Step {i}: Defender Z = {current_z:.2f} m")
        
        # Compute Action
        # neighbors is empty (1 defender)
        action = agent.compute_action(
            my_state, 
            neighbors=[], 
            enemies=mock_enemies, 
            asset_pos=np.array([0,0,0]), 
            dt=env.CTRL_TIMESTEP, 
            time_now=i*env.CTRL_TIMESTEP
        )
        
        # Step Env
        obs, reward, terminated, truncated, info = env.step(np.array([action]))
        
        # Fast Fail/Pass
        if current_z < 50.0:
            print(f"SUCCESS: Defender broke the 60m floor! Current Z: {current_z:.2f}m")
            env.close()
            return True

    env.close()
    
    if min_z_reached < 55.0:
        print(f"SUCCESS: Minimum Z reached: {min_z_reached:.2f}m")
        return True
    else:
        print(f"FAILURE: Defender never went below 60m. Min Z: {min_z_reached:.2f}m")
        return False

if __name__ == "__main__":
    verify_dive()
