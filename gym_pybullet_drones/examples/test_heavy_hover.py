"""Script to test the HeavyDSLPIDControl with the heavy drone."""
import time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import sys
import os

# --- PATH FIX ---
# Force import from local source instead of installed package
current_file_path = os.path.abspath(__file__)
examples_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(examples_dir)
parent_dir = os.path.dirname(project_root)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# ----------------

from gym_pybullet_drones.envs.BaseDefenseAviary import BaseDefenseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.heavy_controller import HeavyDSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

def run():
    """
    Test hover control of the heavy drone.
    """
    # Initialize the environment
    env = BaseDefenseAviary(drone_model=DroneModel.HEAVY,
                            num_drones=1,
                            initial_xyzs=np.array([[0, 0, 0.1]]), # Start on ground
                            initial_rpys=np.array([[0, 0, 0]]),
                            physics=Physics.PYB,
                            gui=True,
                            record=False,
                            obstacles=False,
                            user_debug_gui=True
                            )

    # Initialize the logger
    logger = Logger(logging_freq_hz=int(env.CTRL_FREQ),
                    num_drones=1,
                    duration_sec=10
                    )

    # Initialize the custom heavy controller
    ctrl = HeavyDSLPIDControl(drone_model=DroneModel.HEAVY)

    # Reset the environment
    obs, info = env.reset()

    # Target Position
    TARGET_POS = np.array([0, 0, 1.0])
    TARGET_RPY = np.array([0, 0, 0])

    # Run the simulation
    START = time.time()
    action = np.zeros((1, 4))
    
    print("\n[INFO] Simulation started.")
    print(f"[INFO] Target Position: {TARGET_POS}")
    print("[INFO] Press Ctrl+C to exit early.\n")

    for i in range(0, int(10 * env.CTRL_FREQ)):
        
        # Step the simulation
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Compute control
        # Use raw environment state, not the RL observation
        action, _, _ = ctrl.computeControl(control_timestep=env.CTRL_TIMESTEP,
                                           cur_pos=env.pos[0],
                                           cur_quat=env.quat[0],
                                           cur_vel=env.vel[0],
                                           cur_ang_vel=env.ang_v[0],
                                           target_pos=TARGET_POS,
                                           target_rpy=TARGET_RPY
                                           )
        
        # Reshape action for environment
        action = action.reshape(1, 4)

        # Construct full state vector for Logger
        # pos (3), quat (4), rpy (3), vel (3), ang_v (3), last_clipped_action (4)
        full_state = np.hstack([
            env.pos[0],
            env.quat[0],
            env.rpy[0],
            env.vel[0],
            env.ang_v[0],
            env.last_clipped_action[0]
        ])

        # Log data
        logger.log(drone=0,
                   timestamp=i/env.CTRL_FREQ,
                   state=full_state,
                   control=np.hstack([TARGET_POS, TARGET_RPY, np.zeros(6)]) # Dummy control log
                   )

        # Sync the simulation
        sync(i, START, env.CTRL_TIMESTEP)

    # Close the environment
    env.close()
    
    # Plot results
    logger.plot()

if __name__ == "__main__":
    run()
