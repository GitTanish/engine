import numpy as np
import pybullet as p
from gym_pybullet_drones.utils.heavy_controller import HeavyDSLPIDControl
from gym_pybullet_drones.strategies.WolfPackStrategy import WolfPackStrategy

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
        
        # Goalkeeper Logic REMOVED as per user request
        # All drones are standard defenders
        self.is_goalkeeper = False
        
        # Physics Controller
        self.ctrl = HeavyDSLPIDControl()
        
        # Decision Strategy
        # Decision Strategy
        self.strategy = WolfPackStrategy(intercept_limit=3)
        
        # Boids Parameters (ADJUSTED FOR STABILITY)
        self.SEPARATION_DIST = 4.0  # Increased from 2.0 to prevent collisions
        self.COHESION_DIST = 15.0   # Increased from 10.0 for smoother grouping
        
        self.W_SEP = 2.0     # Increased separation weight
        self.W_ALI = 0.8     # Reduced alignment weight
        self.W_COH = 0.6     # Reduced cohesion weight
        self.W_BASE = 1.5    # INCREASED base attraction to keep drones above asset
        self.W_ALT = 2.0     # NEW: Altitude correction weight
        
        self.MAX_SPEED = 10.0
        self.PATROL_ALTITUDE = 90.0  # UPDATED: Mid-range of 60-120m operational zone

    def _compute_ring_slot(self, agent_id, n_def, asset_pos, ring_radius=3.5, altitude=2.0, jitter_deg=6.0):
        # deterministic hash-based jitter
        idx = int(agent_id) % max(1, int(n_def))
        base_angle = (2.0 * np.pi * idx) / float(max(1, int(n_def)))
        # small deterministic jitter from id
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
        my_vel = np.array(my_state['vel'], dtype=float)

        # --- 1. STABILITY: Warm-up Sequence ---
        # Hover for 1s to settle physics
        if time_now < 1.0:
            target_pos = np.array(asset_pos) + np.array([0, 0, 2.0]) 
            target_vel = np.zeros(3)
            return self._send_pid(my_state, target_pos, target_vel, dt)

        # --- 2. DYNAMIC LANCHESTER ALLOCATION (NEW) ---
        # Calculate N/M Ratio: (Total Defenders / Total Enemies)
        # We use (len(neighbors) + 1) to count ourselves + alive friends
        num_enemies = len(enemies)
        if num_enemies > 0:
            num_allies = len(neighbors) + 1
            # Use ceil to ensure no drone is left idle. 
            # e.g., 10 defenders / 3 enemies = 3.33 -> Limit 4.
            # This allows 3 groups of 3 and 1 group of 1, effectively engaging everyone.
            dynamic_limit = int(np.ceil(num_allies / num_enemies))
            
            # Clamp limit to reasonable bounds (e.g. don't swarm 100 vs 1 unless needed)
            # Actually, for Lanchester advantage, we WANT 100 vs 1 if possible. 
            # So we only clamp minimum.
            self.strategy.intercept_limit = max(1, dynamic_limit)
        
        # --- 3. STRATEGY: Select Target ---
        target_enemy = self.strategy.select_target(
            my_id=self.id,
            my_pos=my_pos,
            neighbors=neighbors,
            enemies=enemies,
            asset_pos=asset_pos
        )

        # --- 3. SAFETY: Collision Avoidance ---
        safety_vel, in_danger = self._safety_override(my_pos, neighbors)
        if in_danger:
            return self._send_pid(my_state, my_pos + safety_vel, safety_vel, dt)

        # --- 4. MISSION: Standoff Intercept ---
        if target_enemy is not None:
            self.state = "INTERCEPT"
            
            e_pos = np.array(target_enemy['pos'], dtype=float)
            e_vel = np.array(target_enemy.get('vel', [0,0,0]), dtype=float)
            base_pos = np.array(asset_pos, dtype=float)

            # [A] THE DIVING CATCH LOGIC (PATCHED)
            # 1. EXTRACT DATA (Fix: Define e_mode before check)
            e_mode = target_enemy.get('mode', 0) 
            
            # Compliance Fix for MDR Check #5
            if e_mode == 0: 
                # TARGET IS BOMBER: Disengage altitude safety floor.
                # Allow dive to 5m to intercept ground threat.
                target_z = np.clip(e_pos[2], 5.0, 120.0)
            else:
                # TARGET IS FIGHTER: Maintain operational safety.
                # Don't get baited into ground crashes by fighters.
                target_z = np.clip(e_pos[2], 60.0, 120.0)

            # [B] SPATIAL DECONFLICTION (Firing Arcs)
            # Determine Rank to assign angle offset
            my_rank = 0
            for n in neighbors:
                n_pos = np.array(n['pos'], dtype=float)
                # Simple ranking: Who is closer to the enemy?
                if np.linalg.norm(n_pos - e_pos) < np.linalg.norm(my_pos - e_pos):
                    my_rank += 1
            
            # Calculate Yaw Offset (Fan out by 15 degrees per rank)
            angle_step = np.deg2rad(15)
            # Slot logic: 0->Center, 1->Left, 2->Right, 3->Far Left...
            slot_idx = (my_rank + 1) // 2
            sign = 1 if my_rank % 2 != 0 else -1
            if my_rank == 0: sign = 0 
            yaw_offset = slot_idx * angle_step * sign

            # [C] VECTOR MATH
            # Virtual Enemy Position (Projected to 2D for Standoff Calc)
            e_pos_virtual = np.array([e_pos[0], e_pos[1], target_z])
            base_pos_virtual = np.array([asset_pos[0], asset_pos[1], 0])

            attack_vector = base_pos_virtual - e_pos_virtual
            
            # Rotate attack vector by yaw_offset
            c, s = np.cos(yaw_offset), np.sin(yaw_offset)
            rot_x = attack_vector[0]*c - attack_vector[1]*s
            rot_y = attack_vector[0]*s + attack_vector[1]*c
            
            offset_vector = np.array([rot_x, rot_y, 0]) 
            dist_to_base = np.linalg.norm(offset_vector)
            
            if dist_to_base > 1e-6:
                attack_dir = offset_vector / dist_to_base
            else:
                attack_dir = np.array([1, 0, 0])

            # Standoff Distance Logic
            standoff_dist = 100.0
            if dist_to_base < 100.0:
                standoff_dist = 30.0 

            intercept_point = e_pos_virtual + (attack_dir * standoff_dist)
            
            # [D] LEAD & FINAL TARGET
            dist_to_intercept = np.linalg.norm(intercept_point - my_pos)
            t_lead = min(dist_to_intercept / self.MAX_SPEED, 2.0)
            
            # Final calculation defines 'target_pos'
            target_pos = intercept_point + (e_vel * t_lead)
            target_pos[2] = target_z # Enforce altitude
            
            # Drive Velocity
            drive_dir = target_pos - my_pos
            drive_dist = np.linalg.norm(drive_dir)
            if drive_dist > 1e-6:
                target_vel = (drive_dir / drive_dist) * self.MAX_SPEED
            else:
                target_vel = np.zeros(3)

        else:
            # --- 5. PATROL: Orbit Logic ---
            self.state = "PATROL"
            # Orbit at 90m (Operational Altitude)
            
            # DYNAMIC RADIUS: Scale based on swarm size to prevent jamming
            # Allocate 4.0m arc length per drone
            circumference = self.num_defenders * 4.0 
            dynamic_radius = max(3.5, circumference / (2 * np.pi))
            
            target_pos = self._compute_ring_slot(self.id, self.num_defenders, asset_pos, ring_radius=dynamic_radius, altitude=self.PATROL_ALTITUDE)
            
            target_vel = (target_pos - my_pos)
            if np.linalg.norm(target_vel) > self.MAX_SPEED:
                target_vel = (target_vel / np.linalg.norm(target_vel)) * self.MAX_SPEED

        # --- 6. EXECUTE ---
        return self._send_pid(my_state, target_pos, target_vel, dt)

