import numpy as np
import pybullet as p
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path: sys.path.insert(0, root_dir)

from gym_pybullet_drones.envs.BaseDefenseAviary import BaseDefenseAviary
from gym_pybullet_drones.utils.enums import DroneModel
from MobileEnemyAgent import MobileEnemyAgent

class MobileDefenseAviary(BaseDefenseAviary):
    """
    Subclass that adds Mobile Asset physics and Swaps Enemy Logic.
    """
    def __init__(self, num_drones=3, **kwargs):
        super().__init__(num_drones=num_drones, **kwargs)
        
        # Define Convoy Physics
        # Start at X=-200 so it travels across the map
        self.BASE_POS = np.array([-200.0, 0.0, 0.0]) 
        self.BASE_VEL = np.array([3.5, 0.0, 0.0]) # Moving East at 3.5 m/s
        self.base_body_id = -1

    def reset(self, seed=None, options=None, scenario_config=None):
        # 1. Parent Reset (Clears physics)
        obs, info = super().reset(seed=seed, options=options, scenario_config=scenario_config)
        
        # 2. Override Enemies with MOBILE Agents
        self.enemy_agents = [] # Clear standard agents
        config = scenario_config if scenario_config else {"num_enemies": 3, "spawn_radius": 400.0, "ground_attacker_ratio": 0.3}
        
        # Re-populate with "Smart" Enemies
        for i in range(len(self.enemy_ids)):
            # Randomly assign mode based on config ratio
            mode = 0 if np.random.rand() < config["ground_attacker_ratio"] else 1
            self.enemy_agents.append(MobileEnemyAgent(mode=mode))

        # 3. Create Mobile Asset Body
        # Reset Position to start of convoy route
        self.BASE_POS = np.array([-200.0, 0.0, 0.0]) 
        
        # Visual: Green Truck
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[1.5, 0.8, 0.8], rgbaColor=[0, 1, 0, 1])
        col_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[1.5, 0.8, 0.8])
        
        self.base_body_id = p.createMultiBody(
            baseMass=1000,
            baseVisualShapeIndex=visual_shape_id,
            baseCollisionShapeIndex=col_shape_id,
            basePosition=self.BASE_POS
        )
        
        return obs, info

    def step(self, action):
        # 1. Move the Asset
        self.BASE_POS += self.BASE_VEL * self.CTRL_TIMESTEP
        
        # Update Physics Body
        if self.base_body_id != -1:
            p.resetBasePositionAndOrientation(
                self.base_body_id, 
                self.BASE_POS, 
                [0,0,0,1], 
                physicsClientId=self.CLIENT
            )

        # 2. Call Parent Step (Handles Drones, Enemies, Shooting)
        return super().step(action)
