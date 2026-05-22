"""
Debug script to isolate environment and visualization issues.
Run this to verify:
1. Green Box (Asset) creation.
2. Drone spawning (Heavy model).
3. Basic Physics (Hover).
"""
import time
import numpy as np
import pybullet as p
import sys
import os

# --- PATH SETUP ---
current_file_path = os.path.abspath(__file__)
examples_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(examples_dir)
parent_root = os.path.dirname(project_root)

if parent_root not in sys.path:
    sys.path.insert(0, parent_root)
if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORTS ---
try:
    from gym_pybullet_drones.envs.BaseDefenseAviary import BaseDefenseAviary
    from gym_pybullet_drones.utils.enums import DroneModel, Physics
    from gym_pybullet_drones.utils.heavy_controller import HeavyDSLPIDControl
    print("[INFO] Imports successful.")
except ImportError as e:
    print(f"[FATAL] Import failed: {e}")
    sys.exit(1)

def debug_run():
    # --- CONFIG ---
    ASSET_POS = np.array([0, 0, 0])
    INIT_XYZ = np.array([[0, 0, 1.0]]) # Start at 1m height
    
    print("[INFO] Initializing Debug Environment...")
    env = BaseDefenseAviary(
        drone_model=DroneModel.HEAVY,
        num_drones=1,
        initial_xyzs=INIT_XYZ,
        initial_rpys=np.zeros((1, 3)),
        physics=Physics.PYB,
        gui=True,
        record=False,
        obstacles=False,
        user_debug_gui=True
    )

    # --- ASSET CREATION DEBUG ---
    print("[INFO] Attempting to create Green Box Asset...")
    try:
        # Create Collision Shape
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], physicsClientId=env.CLIENT)
        if col_id < 0:
            print("[ERROR] Failed to create collision shape.")
        
        # Create Visual Shape
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[0, 1, 0, 1], physicsClientId=env.CLIENT)
        if vis_id < 0:
            print("[ERROR] Failed to create visual shape.")
            
        # Create Body
        body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=ASSET_POS, physicsClientId=env.CLIENT)
        if body_id < 0:
            print("[ERROR] Failed to create multi body.")
        else:
            print(f"[SUCCESS] Green Box created with Body ID: {body_id}")
            
    except Exception as e:
        print(f"[EXCEPTION] Asset creation crashed: {e}")

    # --- CONTROLLER SETUP ---
    ctrl = HeavyDSLPIDControl(drone_model=DroneModel.HEAVY)
    
    # --- SIMULATION LOOP ---
    print("[INFO] Starting 5s Hover Test...")
    obs, info = env.reset()
    
    for i in range(5 * env.CTRL_FREQ): # 5 seconds
        
        # Check GUI connection
        if not p.isConnected(physicsClientId=env.CLIENT):
            print("[WARN] GUI closed.")
            break
            
        # Simple Hover Action
        state = env._getDroneStateVector(0)
        # Construct state vector for PID [pos, quat, rpy, vel, ang_vel, last_action]
        # Note: HeavyDSLPIDControl expects full state
        if len(state) >= 16: # Ensure we have enough state data
             # BaseDefenseAviary state vector might be different than standard, check implementation
             # Standard: pos(3), quat(4), rpy(3), vel(3), ang_vel(3), last_clipped_action(4) -> 20
             pass

        # For debug, let's reconstruct it manually to be safe
        pos = state[0:3]
        quat = state[3:7]
        rpy = p.getEulerFromQuaternion(quat)
        vel = state[10:13]
        ang_vel = state[13:16]
        
        full_state = np.hstack([pos, quat, rpy, vel, ang_vel, np.zeros(3)])
        
        # Hover at 1m
        target_pos = np.array([0, 0, 1.0])
        target_vel = np.zeros(3)
        
        rpm, _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=full_state,
            target_pos=target_pos,
            target_vel=target_vel
        )
        
        action = np.array([rpm])
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.render()
        p.stepSimulation()
        time.sleep(1/60)
    
    print("[INFO] Test Complete.")
    env.close()

if __name__ == "__main__":
    debug_run()
