import numpy as np
import pybullet as p
from gym_pybullet_drones.utils.heavy_controller import HeavyDSLPIDControl
from gym_pybullet_drones.strategies.WolfPackStrategy import WolfPackStrategy, FIGHTER_PENALTY_DEFAULT
from gym_pybullet_drones.control.tracking import TrackerPool


def proportional_nav(my_pos, my_vel, tgt_pos, tgt_vel, max_speed,
                     n_const=4.0, dt=1.0 / 240.0):
    """True Proportional Navigation → commanded velocity setpoint.

    Classic missile guidance: a_cmd = N * Vc * LOS_rate, applied laterally.
    We integrate the lateral accel command for one control step and rotate
    it into a velocity setpoint that the downstream PID can track.

    Returns a velocity vector with |v| = min(max_speed, current speed scale).
    """
    my_pos = np.asarray(my_pos, dtype=float)
    my_vel = np.asarray(my_vel, dtype=float)
    tgt_pos = np.asarray(tgt_pos, dtype=float)
    tgt_vel = np.asarray(tgt_vel, dtype=float)

    r_vec = tgt_pos - my_pos
    r = np.linalg.norm(r_vec)
    if r < 1e-6:
        return np.zeros(3)
    r_hat = r_vec / r

    v_rel = tgt_vel - my_vel
    closing = -np.dot(r_hat, v_rel)                       # Vc (>0 when closing)

    # LOS rate vector: omega = (r x v_rel) / |r|^2
    los_rate = np.cross(r_vec, v_rel) / max(r * r, 1e-6)
    # Lateral accel command perpendicular to LOS
    a_cmd = n_const * max(closing, 0.0) * np.cross(los_rate, r_hat)

    speed = float(np.clip(np.linalg.norm(my_vel), 0.3 * max_speed, max_speed))
    # Integrate accel into velocity setpoint, renormalize to cruise speed
    v_set = my_vel + a_cmd * dt
    norm = np.linalg.norm(v_set)
    if norm < 1e-6:
        v_set = r_hat                                     # launch from hover
        norm = 1.0
    return (v_set / norm) * min(max_speed, max(speed, 0.5 * max_speed))


class DefenseAgent:
    """
    Defensive Drone Agent using Wolf Pack Strategy.
    Implements OODA Loop: Observe -> Decide (via Strategy) -> Act (via PID)
    """
    def __init__(self, id, num_defenders, verbose=True, fighter_penalty=FIGHTER_PENALTY_DEFAULT):
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
        self.strategy = WolfPackStrategy(intercept_limit=3, fighter_penalty=fighter_penalty)
        
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

        # Proportional Navigation gain (3-5 typical; 4 balances smoothness vs aggression)
        self.PN_GAIN = 4.0

        # Per-agent replicated Kalman tracker (uid-keyed, CA model)
        self.tracker = TrackerPool(dt=1.0 / 240.0)

        # Behavior tree: WARMUP > SAFETY > ENGAGE > PATROL
        from gym_pybullet_drones.control.bt import build_defense_behavior_tree
        self.bt = build_defense_behavior_tree(self)
        self._last_bt_path = None

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

    # ================= BEHAVIOR TREE LEAVES =================
    # compute_action() ticks the BT (self.bt); these are its leaves.

    def _act_warmup(self, bb):
        self.state = "STANDBY"
        target_pos = np.array(bb['asset_pos']) + np.array([0, 0, 2.0])
        return self._send_pid(bb['my_state'], target_pos, np.zeros(3), bb['dt'])

    def _cond_safety(self, bb):
        safety_vel, in_danger = self._safety_override(
            np.array(bb['my_state']['pos'], dtype=float), bb['neighbors'])
        bb['safety_vel'] = safety_vel
        return in_danger

    def _act_escape(self, bb):
        self.state = "ESCAPE"
        pos = np.array(bb['my_state']['pos'], dtype=float)
        v = bb['safety_vel']
        return self._send_pid(bb['my_state'], pos + v, v, bb['dt'])

    def _act_intercept(self, bb):
        self.state = "INTERCEPT"
        my_pos = np.array(bb['my_state']['pos'], dtype=float)
        my_vel = np.array(bb['my_state']['vel'], dtype=float)
        dt = bb['dt']
        target_enemy = bb['target']
        neighbors = bb['neighbors']

        e_pos = np.array(target_enemy['pos'], dtype=float)
        e_vel = np.array(target_enemy.get('vel', [0,0,0]), dtype=float)

        # [A] ALTITUDE POLICY: dive low for bombers, stay safe vs fighters
        if target_enemy.get('mode', 0) == 0:
            target_z = np.clip(e_pos[2], 5.0, 120.0)
        else:
            target_z = np.clip(e_pos[2], 60.0, 120.0)

        # [B] SPATIAL DECONFLICTION: fan out firing arcs by proximity rank
        my_rank = sum(
            1 for n in neighbors
            if int(n.get('id', -1)) != int(self.id)
            and np.linalg.norm(np.array(n['pos'], dtype=float) - e_pos)
                < np.linalg.norm(my_pos - e_pos))
        angle_step = np.deg2rad(15)
        slot_idx = (my_rank + 1) // 2
        sign = 1 if my_rank % 2 != 0 else -1
        if my_rank == 0: sign = 0
        yaw_offset = slot_idx * angle_step * sign

        # [C] STANDOFF GEOMETRY (attack corridor toward the asset)
        e_pos_virtual = np.array([e_pos[0], e_pos[1], target_z])
        base_pos_virtual = np.array([bb['asset_pos'][0], bb['asset_pos'][1], 0])
        attack_vector = base_pos_virtual - e_pos_virtual
        c, s = np.cos(yaw_offset), np.sin(yaw_offset)
        offset_vector = np.array([attack_vector[0]*c - attack_vector[1]*s,
                                  attack_vector[0]*s + attack_vector[1]*c, 0])
        dist_to_base = np.linalg.norm(offset_vector)
        attack_dir = offset_vector / dist_to_base if dist_to_base > 1e-6 else np.array([1, 0, 0])
        standoff_dist = 30.0 if dist_to_base < 100.0 else 100.0
        intercept_point = e_pos_virtual + (attack_dir * standoff_dist)

        # [D] GUIDANCE: Kalman-predicted aimpoint + Proportional Navigation
        dist_now = np.linalg.norm(e_pos - my_pos)
        t_go = float(np.clip(dist_now / self.MAX_SPEED, 0.1, 2.0))
        e_pred = self.tracker.predict(target_enemy.get('uid'), t_go)
        if e_pred is None:
            e_pred = e_pos + e_vel * t_go   # fallback: constant velocity

        target_pos = intercept_point + (e_pred - e_pos) * min(
            np.linalg.norm(intercept_point - my_pos) / self.MAX_SPEED, 2.0)
        target_pos[2] = np.clip(target_pos[2], 5.0, 120.0)
        target_vel = proportional_nav(my_pos, my_vel, target_pos, e_vel,
                                      self.MAX_SPEED,
                                      n_const=self.PN_GAIN, dt=dt)
        return self._send_pid(bb['my_state'], target_pos, target_vel, dt)

    def _act_patrol(self, bb):
        self.state = "PATROL"
        my_pos = np.array(bb['my_state']['pos'], dtype=float)
        # DYNAMIC RADIUS: 4.0m arc length per drone prevents ring jamming
        circumference = self.num_defenders * 4.0
        dynamic_radius = max(3.5, circumference / (2 * np.pi))
        target_pos = self._compute_ring_slot(self.id, self.num_defenders,
                                             bb['asset_pos'],
                                             ring_radius=dynamic_radius,
                                             altitude=self.PATROL_ALTITUDE)
        target_vel = target_pos - my_pos
        if np.linalg.norm(target_vel) > self.MAX_SPEED:
            target_vel = (target_vel / np.linalg.norm(target_vel)) * self.MAX_SPEED
        return self._send_pid(bb['my_state'], target_pos, target_vel, bb['dt'])

    def compute_action(self, my_state, neighbors, enemies, asset_pos, dt, time_now):
        """Tick the behavior tree; returns motor RPMs."""
        bb = {'my_state': my_state, 'neighbors': neighbors or [],
              'enemies': enemies or [], 'asset_pos': asset_pos,
              'dt': dt, 'time_now': time_now, 'path': []}

        # Observe: strategy + tracker refresh (shared by engage/patrol branches)
        my_pos = np.array(my_state['pos'], dtype=float)
        neighbors_excl_self = [n for n in (neighbors or [])
                               if int(n.get('id', -1)) != int(self.id)]
        bb['target'] = self.strategy.select_target(
            my_id=self.id, my_pos=my_pos, neighbors=neighbors_excl_self,
            enemies=enemies or [], asset_pos=asset_pos)
        if enemies:
            self.tracker.update(enemies)

        self.bt(bb)
        return bb.get('rpm')

