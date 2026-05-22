import numpy as np
import sys
import os

# Add root to path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path: sys.path.insert(0, root_dir)

from gym_pybullet_drones.control.EnemyAgent import EnemyAgent

class MobileEnemyAgent(EnemyAgent):
    """
    Advanced Enemy Agent with 'Fog of War' Logic.
    Implements: "enemy drones can spot an asset only if they happen to come in the close proximity"
    """
    def __init__(self, mode=0, speed=10.0, detection_range=300.0):
        super().__init__(mode=mode, speed=speed)
        self.detection_range = detection_range
        self.state = "SEARCH" 
        
        # Persistent Patrol Vector
        # Fly generally towards center (theater of war) but with randomness
        self.patrol_dir = np.random.uniform(-1, 1, 3)
        self.patrol_dir[2] = -0.1 # Slight downward bias
        self.patrol_dir = self.patrol_dir / np.linalg.norm(self.patrol_dir)

    def compute_action(self, my_pos, asset_pos, friendly_drones_pos):
        target_pos = None
        
        # --- MODE 0: GROUND ATTACKER (The Bomber) ---
        if self.mode == 0:
            dist_to_asset = np.linalg.norm(asset_pos - my_pos)
            
            # [LOGIC] Visual Contact Transition
            if dist_to_asset < self.detection_range:
                self.state = "ATTACK"
            
            # [LOGIC] State Behavior
            if self.state == "ATTACK":
                target_pos = asset_pos # Dive on asset
            else:
                # SEARCH: Fly generic patrol pattern.
                # Assume they know the "Theater" is roughly (0,0) but not the convoy's X/Y.
                center_bias = (np.array([0,0,100]) - my_pos) * 0.05
                target_pos = my_pos + (self.patrol_dir * 100) + center_bias

        # --- MODE 1: AIR INTERCEPTOR (The Fighter) ---
        elif self.mode == 1:
            if len(friendly_drones_pos) > 0:
                # Find nearest friendly drone
                dists = np.linalg.norm(friendly_drones_pos - my_pos, axis=1)
                nearest_idx = np.argmin(dists)
                target_drone = friendly_drones_pos[nearest_idx]
                
                # Check Visual Range
                if dists[nearest_idx] < self.detection_range:
                    self.state = "ATTACK"
                    target_pos = target_drone
                else:
                    self.state = "SEARCH"
                    target_pos = my_pos + (self.patrol_dir * 100)

        # --- KINEMATICS ---
        if target_pos is None: 
             target_pos = np.array([0,0,100])

        direction = target_pos - my_pos
        dist = np.linalg.norm(direction)
        
        if dist < 0.1: return np.zeros(3)
            
        dir_norm = direction / dist
        vel = dir_norm * self.max_speed
        
        # Add slight jitter to Search Mode to simulate scanning
        if self.state == "SEARCH":
            vel += np.random.normal(0, 0.5, 3)
        
        return vel
