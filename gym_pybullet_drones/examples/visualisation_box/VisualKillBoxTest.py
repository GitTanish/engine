import os
import time
import sys
import numpy as np
import pybullet as p

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# visualisation_box -> examples -> gym_pybullet_drones (root) -> (parent)
# We need to ensure package imports work.
root_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(root_dir)
parent_dir = os.path.dirname(project_root)

if parent_dir not in sys.path: sys.path.insert(0, parent_dir) # Add repo root
if current_dir not in sys.path: sys.path.insert(0, current_dir) # Add local folder

from gym_pybullet_drones.envs.BaseDefenseAviary import BaseDefenseAviary
from gym_pybullet_drones.control.DefenseAgent import DefenseAgent
from gym_pybullet_drones.utils.enums import DroneModel, Physics

# IMPORT LOCAL CUSTOM CONTROLLER
try:
    from heavy_controller import HeavyDSLPIDControl
except ImportError:
    from visualisation_box.heavy_controller import HeavyDSLPIDControl

# ==============================================================================
# 1. THE VISUAL ENVIRONMENT (Knife Fight Rules)
# ==============================================================================
class VisualDefenseAviary(BaseDefenseAviary):
    def __init__(self, num_drones=3, **kwargs):
        super().__init__(num_drones=num_drones, **kwargs)
        # TIGHT ENGAGEMENT RULES
        self.THREAT_RANGE = 40.0       
        self.FIRING_RANGE_MIN = 5.0    
        self.FIRING_RANGE_MAX = 30.0   
        self.BASE_POS = np.array([0, 0, 0])

# ==============================================================================
# 2. THE VISUAL PILOT (With Ramp Logic)
# ==============================================================================
class VisualDefenseAgent(DefenseAgent):
    def __init__(self, id, num_defenders, verbose=True):
        super().__init__(id, num_defenders, verbose)
        # We aim for 10m, but we will ramp to it.
        self.PATROL_ALTITUDE = 10.0  
        self.W_BASE = 1.5    
        
        # OVERRIDE CONTROLLER with Local Stabilized One
        self.ctrl = HeavyDSLPIDControl()

    def compute_action(self, my_state, neighbors, enemies, asset_pos, dt, time_now):
        my_pos = np.array(my_state['pos'], dtype=float)
        
        # --- GRADUAL TAKEOFF RAMP (The Fix) ---
        # Instead of demanding 10m instantly, we demand:
        # 0m at t=0s, 5m at t=2.5s, 10m at t=5.0s
        if time_now < 5.0:
            target_z = (time_now / 5.0) * 10.0
            # Override target to be directly above asset, rising slowly
            target_pos = np.array(asset_pos) + np.array([0, 0, target_z])
            target_vel = np.zeros(3)
            return self._send_pid(my_state, target_pos, target_vel, dt)

        # --- STANDARD LOGIC AFTER TAKEOFF ---
        # Dynamic Allocation
        if len(enemies) > 0:
            num_allies = len(neighbors) + 1
            dynamic_limit = int(np.ceil(num_allies / len(enemies)))
            self.strategy.intercept_limit = max(1, dynamic_limit)
        
        target_enemy = self.strategy.select_target(self.id, my_pos, neighbors, enemies, asset_pos)
        
        # Safety
        safety_vel, in_danger = self._safety_override(my_pos, neighbors)
        if in_danger:
            return self._send_pid(my_state, my_pos + safety_vel, safety_vel, dt)

        if target_enemy is not None:
            self.state = "INTERCEPT"
            e_pos = np.array(target_enemy['pos'], dtype=float)
            e_vel = np.array(target_enemy.get('vel', [0,0,0]), dtype=float)
            e_mode = target_enemy.get('mode', 0) 

            # Altitude Clamping (10m Doctrine)
            if e_mode == 0: 
                target_z = np.clip(e_pos[2], 2.0, 12.0) # Dive allowed
            else:
                target_z = np.clip(e_pos[2], 8.0, 12.0) # Stay high

            # Vector Math
            my_rank = 0
            for n in neighbors:
                n_pos = np.array(n['pos'], dtype=float)
                if np.linalg.norm(n_pos - e_pos) < np.linalg.norm(my_pos - e_pos):
                    my_rank += 1
            
            angle_step = np.deg2rad(20)
            slot_idx = (my_rank + 1) // 2
            sign = 1 if my_rank % 2 != 0 else -1
            if my_rank == 0: sign = 0 
            yaw_offset = slot_idx * angle_step * sign

            e_pos_virtual = np.array([e_pos[0], e_pos[1], target_z])
            base_pos_virtual = np.array([asset_pos[0], asset_pos[1], 0])
            
            attack_vector = base_pos_virtual - e_pos_virtual
            c, s = np.cos(yaw_offset), np.sin(yaw_offset)
            rot_x = attack_vector[0]*c - attack_vector[1]*s
            rot_y = attack_vector[0]*s + attack_vector[1]*c
            offset_vector = np.array([rot_x, rot_y, 0]) 
            dist_to_base = np.linalg.norm(offset_vector)
            
            if dist_to_base > 1e-6:
                attack_dir = offset_vector / dist_to_base
            else:
                attack_dir = np.array([1, 0, 0])

            # Standoff: 15m default, 5m close quarters
            standoff_dist = 15.0 
            if dist_to_base < 30.0:
                standoff_dist = 5.0 

            intercept_point = e_pos_virtual + (attack_dir * standoff_dist)
            dist_to_intercept = np.linalg.norm(intercept_point - my_pos)
            t_lead = min(dist_to_intercept / self.MAX_SPEED, 1.0)
            target_pos = intercept_point + (e_vel * t_lead)
            target_pos[2] = target_z 
            
            drive_dir = target_pos - my_pos
            drive_dist = np.linalg.norm(drive_dir)
            target_vel = (drive_dir / drive_dist) * self.MAX_SPEED if drive_dist > 1e-6 else np.zeros(3)

        else:
            # Patrol
            self.state = "PATROL"
            circumference = self.num_defenders * 4.0 
            dynamic_radius = max(3.5, circumference / (2 * np.pi))
            target_pos = self._compute_ring_slot(self.id, self.num_defenders, asset_pos, ring_radius=dynamic_radius, altitude=self.PATROL_ALTITUDE)
            target_vel = (target_pos - my_pos)
            if np.linalg.norm(target_vel) > self.MAX_SPEED:
                target_vel = (target_vel / np.linalg.norm(target_vel)) * self.MAX_SPEED

        return self._send_pid(my_state, target_pos, target_vel, dt)

# ==============================================================================
# 3. THE RUNNER
# ==============================================================================
def run():
    NUM_DEFENDERS = 10
    NUM_ENEMIES = 10
    SPAWN_RADIUS = 60.0 # User specified 60m Spawn
    GROUND_RATIO = 0.3 

    print(f"[TEST] Initializing Knife Fight with Ramp Takeoff")
    print(f"[CONFIG] Ramp: 0-10m over 5s | Ctrl: Balanced (P=20)")
    
    spawn_ring_radius = max(5.0, (NUM_DEFENDERS * 3.0) / (2 * np.pi))
    init_xyzs = np.array([[spawn_ring_radius*np.cos((2*np.pi/NUM_DEFENDERS)*i), spawn_ring_radius*np.sin((2*np.pi/NUM_DEFENDERS)*i), 1.0] for i in range(NUM_DEFENDERS)])
    
    # Use VisualDefenseAviary (Local Class)
    env = VisualDefenseAviary(
        drone_model=DroneModel.HEAVY, 
        num_drones=NUM_DEFENDERS, 
        initial_xyzs=init_xyzs, 
        initial_rpys=np.zeros((NUM_DEFENDERS, 3)),
        physics=Physics.PYB, 
        gui=True, 
        user_debug_gui=True
    )

    # Use VisualDefenseAgent (Local Class)
    agents = [VisualDefenseAgent(i, NUM_DEFENDERS) for i in range(NUM_DEFENDERS)]

    scenario_config = {
        "num_enemies": NUM_ENEMIES,
        "spawn_radius": SPAWN_RADIUS,
        "ground_attacker_ratio": GROUND_RATIO
    }
    obs, info = env.reset(scenario_config=scenario_config)
    
    for i in range(NUM_DEFENDERS):
        p.changeVisualShape(env.DRONE_IDS[i], -1, rgbaColor=[0, 0, 1, 1], physicsClientId=env.CLIENT)

    step = 0
    while True:
        all_states = [env._getDroneStateVector(k) for k in range(NUM_DEFENDERS)]
        friendly_states = [{'id': k, 'pos': s[0:3], 'vel': s[10:13], 'quat': s[3:7], 'ang_vel': s[13:16]} for k, s in enumerate(all_states)]

        current_enemies = []
        for e_idx, e_pos in enumerate(env.enemy_positions):
            current_enemies.append({
                'id': e_idx, 'pos': e_pos, 'vel': np.zeros(3),
                'mode': env.enemy_agents[e_idx].mode if e_idx < len(env.enemy_agents) else 0
            })

        actions = np.zeros((NUM_DEFENDERS, 4))
        for idx, agent in enumerate(agents):
            if idx not in env.dead_friendly_indices:
                actions[idx, :] = agent.compute_action(
                    friendly_states[idx], friendly_states, current_enemies, 
                    env.BASE_POS, env.CTRL_TIMESTEP, step*env.CTRL_TIMESTEP
                )

        obs, _, _, _, info = env.step(actions)
        
        # DEBUG: Full swarm status
        if step % 50 == 0:
            alts = [env._getDroneStateVector(k)[2] for k in range(NUM_DEFENDERS)]
            avg_alt = np.mean(alts)
            print(f"[STEP {step}] Avg Alt: {avg_alt:.2f}m | Ag0: {alts[0]:.2f}m")

        if len(env.enemy_positions) == 0:
            print("[VICTORY] Close Quarters Threat Neutralized.")
            break
        
        step += 1
        time.sleep(env.CTRL_TIMESTEP)

    env.close()

if __name__ == "__main__":
    run()
