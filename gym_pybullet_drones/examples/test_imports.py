import sys
import os
print("Starting imports...")
import numpy as np
print("Numpy imported")
import pandas as pd
print("Pandas imported")
import pybullet as p
print("Pybullet imported")

# Path setup
current_file_path = os.path.abspath(__file__)
examples_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(examples_dir)
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from gym_pybullet_drones.envs.BaseDefenseAviary import BaseDefenseAviary
    print("BaseDefenseAviary imported")
except ImportError:
    inner_package_dir = os.path.join(project_root, 'gym_pybullet_drones')
    sys.path.append(inner_package_dir) 
    from envs.BaseDefenseAviary import BaseDefenseAviary
    print("BaseDefenseAviary imported (fallback)")

print("All imports successful")
