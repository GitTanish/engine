"""Script to test the rendering of the heavy drone model."""
import time
import argparse
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

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

def run():
    """
    Test rendering of the heavy drone.
    """
    # Initialize the simulation
    env = CtrlAviary(drone_model=DroneModel.HEAVY,
                     num_drones=1,
                     neighbourhood_radius=10,
                     initial_xyzs=np.array([[0, 0, 1.0]]),
                     initial_rpys=np.array([[0, 0, 0]]),
                     physics=Physics.PYB,
                     gui=True,
                     record=False,
                     obstacles=True, # Add some obstacles to see scale
                     user_debug_gui=True
                     )

    # Initialize the logger
    logger = Logger(logging_freq_hz=int(env.CTRL_FREQ),
                    num_drones=2,
                    duration_sec=60
                    )

    # Initialize the controller
    ctrl = DSLPIDControl(drone_model=DroneModel.HEAVY)

    # Run the simulation
    START = time.time()
    action = np.zeros((1, 4))
    
    print("\n[INFO] Simulation started. Look at the PyBullet GUI window.")
    print("[INFO] The drone should be visible as a large scaled Crazyflie mesh.")
    print("[INFO] Press Ctrl+C to exit early.\n")

    i = 0
    while True:

        
        # Step the simulation
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Compute control for hover (just to keep it somewhat stable if we were controlling it, 
        # but here we just want to see it. DSLPID might need tuning for heavy drone, 
        # so it might crash, but we'll see the mesh first)
        
        # For now, let's just apply hover RPMs based on the new constants if we wanted to fly,
        # but let's just let it fall or sit there to inspect the mesh.
        # Or better, let's try to hover it at 1m.
        
        action, _, _ = ctrl.computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                    state=obs[0],
                                                    target_pos=np.array([0, 0, 1.0]),
                                                    target_rpy=np.array([0, 0, 0])
                                                    )
        action = action.reshape(1, 4)

        # Sync the simulation
        sync(i, START, env.CTRL_TIMESTEP)

        keys = p.getKeyboardEvents()
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            print("[INFO] 'q' pressed. Exiting simulation.")
            break

        i += 1
    # Close the environment
    env.close()

if __name__ == "__main__":
    run()
