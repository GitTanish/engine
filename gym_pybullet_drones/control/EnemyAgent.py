import numpy as np

class EnemyAgent:
    """
    Modes:
    0: Ground Attacker (Target: Base)
    1: Air Interceptor (Target: Nearest Friendly Drone)
    """
    def __init__(self, mode=0, speed=10.0):
        self.mode = mode
        self.max_speed = speed
        
    def compute_action(self, my_pos, asset_pos, friendly_drones_pos):
        target_pos = asset_pos # Default to Base

        # --- LOGIC: Choose Target based on Mode ---
        if self.mode == 1 and len(friendly_drones_pos) > 0:
            # Find nearest friendly drone to attack
            dists = np.linalg.norm(friendly_drones_pos - my_pos, axis=1)
            nearest_idx = np.argmin(dists)
            target_pos = friendly_drones_pos[nearest_idx]

        # --- KINEMATICS: Calculate Velocity Vector ---
        direction = target_pos - my_pos
        dist = np.linalg.norm(direction)
        
        if dist < 0.1:
            return np.zeros(3)
            
        dir_norm = direction / dist
        vel = dir_norm * self.max_speed
        
        return vel