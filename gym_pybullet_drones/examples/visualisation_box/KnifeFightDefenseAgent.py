import numpy as np
import pybullet as p
from gym_pybullet_drones.strategies.WolfPackStrategy import WolfPackStrategy

try:
    from heavy_controller import HeavyDSLPIDControl
except ImportError:
    from visualisation_box.heavy_controller import HeavyDSLPIDControl

class DefenseAgent:
    """
    Defensive Drone Agent using Wolf Pack Strategy.
    Implements OODA Loop: Observe -> Decide (via Strategy) -> Act (via PID)
    """
    def __init__(self, id, num_defenders, verbose=True):
        self.id = id
        self.num_defenders = num_defenders
        self.verbose = verbose
        self.state = "PATROL"
        self.is_goalkeeper = False
        
        self.ctrl = HeavyDSLPIDControl()
        self.strategy = WolfPackStrategy(intercept_limit=3)
        
        self.SEPARATION_DIST = 4.0
        self.COHESION_DIST = 15.0
        self.W_SEP = 2.0
        self.W_ALI = 0.8
        self.W_COH = 0.6
        self.W_BASE = 1.5
        self.W_ALT = 2.0
        
        self.MAX_SPEED = 10.0
        self.PATROL_ALTITUDE = 10.0 

    def _compute_ring_slot(self, agent_id, n_def, asset_pos, ring_radius=3.5, altitude=2.0, jitter_deg=6.0):
        idx = int(agent_id) % max(1, int(n_def))
        base_angle = (2.0 * np.pi * idx) / float(max(1, int(n_def)))
        jitter = (((agent_id * 1103) % 1000) / 1000.0 - 0.5) * np.deg2rad(jitter_deg)
        angle = base_angle + jitter
        x = asset_pos[0] + ring_radius * np.cos(angle)
        y = asset_pos[1] + ring_radius * np.sin(angle)
        z = asset_pos[2] + altitude
        return np.array([x, y, z], dtype=float)

    def _send_pid(self, my_state, target_pos, target_vel, dt):
        current_rpy = p.getEulerFromQuaternion(my_state['quat'])
        full_state = np.hstack([
            my_state['pos'],
            my_state['quat'],
            current_rpy,
            my_state['vel'],
            my_state['ang_vel'],
            np.zeros(3)
        ])
        rpm, _, _ = self.ctrl.computeControlFromState(
            control_timestep=dt,
            state=full_state,
            target_pos=target_pos,
            target_vel=target_vel
        )
        return rpm

    def _safety_override(self, agent_pos, neighbors, min_safe=1.5, escape_speed=2.5):
        pos = agent_pos
        avoid = np.zeros(3, dtype=float)
        for n in neighbors:
            if n['id'] == self.id:
                continue
            r = pos - np.array(n['pos'], dtype=float)
            d = np.linalg.norm(r) + 1e-8
            if d < min_safe:
                avoid += (r / d) * ((min_safe - d) / min_safe)
        if np.linalg.norm(avoid) > 1e-6:
            v = avoid / (np.linalg.norm(avoid)+1e-8) * escape_speed
            return v, True
        return None, False

    def compute_action(self, my_state, neighbors, enemies, asset_pos, dt, time_now):
        my_pos = np.array(my_state['pos'], dtype=float)
        
        # --- 1. STABILITY: GRADUAL TAKEOFF ---
        if time_now < 3.0: 
            warmup_alt = min(10.0, (time_now / 3.0) * 10.0)
            target_pos = np.array(asset_pos) + np.array([0, 0, warmup_alt]) 
            target_vel = np.zeros(3)
            return self._send_pid(my_state, target_pos, target_vel, dt)

        # --- 2. DYNAMIC LANCHESTER ALLOCATION ---
        num_enemies = len(enemies)
        if num_enemies > 0:
            num_allies = len(neighbors) + 1
            dynamic_limit = int(np.ceil(num_allies / num_enemies))
            self.strategy.intercept_limit = max(1, dynamic_limit)
        
        # --- 3. TARGETING ---
        target_enemy = self.strategy.select_target(self.id, my_pos, neighbors, enemies, asset_pos)
        
        # --- 4. SAFETY ---
        safety_vel, in_danger = self._safety_override(my_pos, neighbors)
        if in_danger:
            return self._send_pid(my_state, my_pos + safety_vel, safety_vel, dt)

        # --- 5. MISSION ---
        if target_enemy is not None:
            self.state = "INTERCEPT"
            e_pos = np.array(target_enemy['pos'], dtype=float)
            e_vel = np.array(target_enemy.get('vel', [0,0,0]), dtype=float)
            e_mode = target_enemy.get('mode', 0) 

            # [A] THE DIVING CATCH LOGIC
            if e_mode == 0: 
                target_z = np.clip(e_pos[2], 2.0, 12.0)
            else:
                target_z = np.clip(e_pos[2], 8.0, 12.0)

            # [B] SPATIAL DECONFLICTION
            my_rank = 0
            for n in neighbors:
                if np.linalg.norm(np.array(n['pos']) - e_pos) < np.linalg.norm(my_pos - e_pos):
                    my_rank += 1
            
            angle_step = np.deg2rad(15)
            slot_idx = (my_rank + 1) // 2
            sign = 1 if my_rank % 2 != 0 else -1
            if my_rank == 0: sign = 0 
            yaw_offset = slot_idx * angle_step * sign

            # [C] VECTOR MATH
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

            # [D] Standoff Distance
            standoff_dist = 15.0 
            if dist_to_base < 30.0:
                standoff_dist = 5.0

            intercept_point = e_pos_virtual + (attack_dir * standoff_dist)
            
            dist_to_intercept = np.linalg.norm(intercept_point - my_pos)
            t_lead = min(dist_to_intercept / self.MAX_SPEED, 2.0)
            target_pos = intercept_point + (e_vel * t_lead)
            target_pos[2] = target_z 
            
            drive_dir = target_pos - my_pos
            drive_dist = np.linalg.norm(drive_dir)
            if drive_dist > 1e-6:
                target_vel = (drive_dir / drive_dist) * self.MAX_SPEED
            else:
                target_vel = np.zeros(3)

        else:
            # --- 6. PATROL ---
            self.state = "PATROL"
            circumference = self.num_defenders * 4.0 
            dynamic_radius = max(3.5, circumference / (2 * np.pi))
            target_pos = self._compute_ring_slot(self.id, self.num_defenders, asset_pos, ring_radius=dynamic_radius, altitude=self.PATROL_ALTITUDE)
            target_vel = (target_pos - my_pos)
            if np.linalg.norm(target_vel) > self.MAX_SPEED:
                target_vel = (target_vel / np.linalg.norm(target_vel)) * self.MAX_SPEED

        # --- DEBUG: Print Logic Internals for Agent 0 ---
        if self.id == 0 and int(time_now*240) % 240 == 0: # Once per second approx
             print(f"  > [AG0 LOGIC] State: {self.state} | Tgt Pos: {target_pos} | Tgt Vel: {target_vel} | My Pos: {my_pos}")

        return self._send_pid(my_state, target_pos, target_vel, dt)
