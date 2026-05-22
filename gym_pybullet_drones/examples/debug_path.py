import sys
import os

current_file_path = os.path.abspath(__file__)
examples_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(examples_dir)

print(f"Project Root: {project_root}")

parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print(f"Parent Dir: {parent_dir}")
print(f"Sys Path: {sys.path}")

try:
    import gym_pybullet_drones
    print(f"Successfully imported gym_pybullet_drones: {gym_pybullet_drones}")
    from gym_pybullet_drones.utils import drone_constants
    print(f"Successfully imported drone_constants: {drone_constants}")
    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
    print(f"Successfully imported CtrlAviary: {CtrlAviary}")
    from gym_pybullet_drones.control.DefenseAgent import DefenseAgent
    print(f"Successfully imported DefenseAgent: {DefenseAgent}")
except ImportError as e:
    print(f"Import failed: {e}")
