import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.EnemyAgent import EnemyAgent

class BaseDefenseAviary(CtrlAviary):
    def __init__(self, num_drones=3, **kwargs):
        super().__init__(num_drones=num_drones, **kwargs)
        
        # Defense Parameters
        self.BASE_POS = np.array([0, 0, 0])
        self.THREAT_RANGE = 100.0 
        self.FIRING_RANGE_MIN = 30.0
        self.FIRING_RANGE_MAX = 250.0

        # Weapons are tuned as frequency-independent Poisson rates (kills/sec).
        # Legacy constant was HIT_PROBABILITY_PER_STEP = 0.05 @ 240Hz CTRL_FREQ;
        # we derive the equivalent rate so behavior is preserved if CTRL_FREQ changes.
        self.HIT_RATE_PER_SEC = -np.log(1.0 - 0.05) / self.CTRL_TIMESTEP
        self.RAM_KILL_RADIUS = 2.0

        # Swarm State
        self.enemy_ids = []
        self.enemy_agents = []
        self.enemy_positions = []
        self.enemy_velocities = np.zeros((0, 3))
        # Stable spawn-time identities (array indices shift when enemies die,
        # so trackers must key on these UIDs, never on list position)
        self.enemy_uids = []
        self._next_enemy_uid = 0

        # Track dead friendlies to disable them
        self.dead_friendly_indices = set()

        # Managed visual bodies (removed/recreated on reset to avoid leaks)
        self.asset_body_id = -1

        # Kill events recorded during the last step (consumed by scripts/UI)
        self.combat_events = []

    def _observationSpace(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.NUM_DRONES, 42), dtype=np.float32)

    def _computeObs(self):
        obs = np.zeros((self.NUM_DRONES, 42))
        for i in range(self.NUM_DRONES):
            # If dead, return zero observation (blind)
            if i in self.dead_friendly_indices:
                continue
                
            state = self._getDroneStateVector(i)
            obs[i, 0:12] = state[0:12]
            
            if len(self.enemy_positions) > 0:
                rel_pos = np.array(self.enemy_positions) - state[0:3]
                dists = np.linalg.norm(rel_pos, axis=1)
                sorted_indices = np.argsort(dists)
                closest_enemies = rel_pos[sorted_indices][:10]
                flat_enemies = closest_enemies.flatten()
                obs[i, 12:12+len(flat_enemies)] = flat_enemies
        return obs

    def reset(self, seed=None, options=None, scenario_config=None):
        # Clear enemies
        for eid in self.enemy_ids:
            try: p.removeBody(eid, physicsClientId=self.CLIENT)
            except: pass
        self.enemy_ids = []
        self.enemy_agents = []
        self.enemy_positions = []
        self.enemy_velocities = np.zeros((0, 3))
        self.enemy_uids = []
        self.dead_friendly_indices = set()

        super().reset(seed=seed, options=options)
        
        # Default config if none provided (Safe default)
        config = scenario_config if scenario_config else {
            "num_enemies": 3,
            "spawn_radius": 400.0,
            "ground_attacker_ratio": 0.3 
        }
        
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        drone_path = os.path.join(current_dir, '..', 'assets', 'heavy_drone.urdf')

        for i in range(config["num_enemies"]):
            # Scenario Logic: Determine Mode
            is_ground = np.random.rand() < config["ground_attacker_ratio"]
            mode = 0 if is_ground else 1
            color = [1, 0, 0, 1] if mode == 0 else [1, 0.5, 0, 1] # Red=Ground, Orange=Air
            
            theta = np.random.uniform(0, 2*np.pi)
            r = config["spawn_radius"]
            z = np.random.uniform(90, 120) # Spawn High
            pos = [r*np.cos(theta), r*np.sin(theta), z]
            
            eid = p.loadURDF(drone_path, pos, physicsClientId=self.CLIENT)
            p.changeVisualShape(eid, -1, rgbaColor=color, physicsClientId=self.CLIENT)
            
            self.enemy_ids.append(eid)
            self.enemy_positions.append(np.array(pos))
            self.enemy_uids.append(self._next_enemy_uid)
            self._next_enemy_uid += 1
            self.enemy_agents.append(EnemyAgent(mode=mode))
            
        self.enemy_positions = np.array(self.enemy_positions) if self.enemy_positions else np.zeros((0, 3))
        self.enemy_velocities = np.zeros((len(self.enemy_positions), 3))
        
        # Base Visual. NOTE: super().reset() calls p.resetSimulation(), which
        # destroys every body — old body ids are invalid, so recreate from scratch.
        self.asset_body_id = -1
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[0, 1, 0, 1])
        self.asset_body_id = p.createMultiBody(baseVisualShapeIndex=visual_shape_id, basePosition=self.BASE_POS)

        return self._computeObs(), {}

    def step(self, action):
        # Kill events for this step (exposed via info["combat_events"])
        self.combat_events = []

        # 1. Enforce Death (The Graveyard)
        # If a drone is dead, we override its physics to stay underground
        if len(self.dead_friendly_indices) > 0:
            for dead_idx in self.dead_friendly_indices:
                # Override action to stop motors
                action[dead_idx, :] = 0 
                # Teleport to graveyard to prevent collision physics
                p.resetBasePositionAndOrientation(
                    self.DRONE_IDS[dead_idx], 
                    [0, 0, -100 - dead_idx], # Underground stack
                    [0,0,0,1], 
                    physicsClientId=self.CLIENT
                )

        # 2. Move Enemies
        current_drone_pos = np.array([self._getDroneStateVector(i)[0:3] for i in range(self.NUM_DRONES)])
        new_positions = []
        new_velocities = []
        for idx, agent in enumerate(self.enemy_agents):
            vel = agent.compute_action(
                my_pos=self.enemy_positions[idx],
                asset_pos=self.BASE_POS,
                friendly_drones_pos=current_drone_pos
            )
            # DEBUG: Trace Enemy Movement
            # if idx == 0: print(f"[DEBUG] Enemy 0 Vel: {vel} | Pos: {self.enemy_positions[idx]}")
            
            new_pos = self.enemy_positions[idx] + (vel * self.CTRL_TIMESTEP)
            new_positions.append(new_pos)
            new_velocities.append(np.asarray(vel, dtype=float))
            p.resetBasePositionAndOrientation(self.enemy_ids[idx], new_pos, [0,0,0,1], physicsClientId=self.CLIENT)
        self.enemy_positions = np.array(new_positions) if new_positions else np.zeros((0, 3))
        self.enemy_velocities = np.array(new_velocities) if new_velocities else np.zeros((0, 3))

        # 3. Physics Step
        obs, reward, term, trunc, info = super().step(action)
        
        # 4. WEAPONS: Blue shooting Red
        self._process_blue_engagements(current_drone_pos)
        
        # 5. WEAPONS: Red shooting Blue
        self._process_red_engagements(current_drone_pos)

        # 5b. WEAPONS: Kinetic ram-kill (last-resort intercept)
        self._process_ram_kills(current_drone_pos)

        # 6. Check Unattended
        violation, details = self._check_unattended_violation(current_drone_pos)
        if violation:
            info["unattended_violation"] = True
            
        # Add Friendly Loss count to info for metrics
        info["friendly_losses"] = len(self.dead_friendly_indices)

        # Combat bookkeeping for scripts/UI
        info["combat_events"] = self.combat_events
        info["asset_hit"] = False
        if len(self.enemy_positions) > 0:
            dists_to_base = np.linalg.norm(self.enemy_positions - self.BASE_POS, axis=1)
            if np.any(dists_to_base < 1.0):
                info["asset_hit"] = True

        return obs, reward, term, trunc, info

    def _hit_probability(self):
        """Frequency-independent per-step kill probability (Poisson)."""
        return 1.0 - np.exp(-self.HIT_RATE_PER_SEC * self.CTRL_TIMESTEP)

    def _process_blue_engagements(self, friendly_positions):
        """ Friendlies shooting Enemies """
        if len(self.enemy_positions) == 0: return
        alive_indices = [i for i in range(self.NUM_DRONES) if i not in self.dead_friendly_indices]
        if not alive_indices: return

        alive_pos = friendly_positions[alive_indices]
        kill_prob = self._hit_probability()
        indices_to_remove = []
        for e_idx, e_pos in enumerate(self.enemy_positions):
            dists = np.linalg.norm(alive_pos - e_pos, axis=1)
            in_range = (dists >= self.FIRING_RANGE_MIN) & (dists <= self.FIRING_RANGE_MAX)
            if np.any(in_range) and np.random.rand() < kill_prob:
                shooter = alive_indices[int(np.argmin(np.where(in_range, dists, np.inf)))]
                print(f"[SPLASH] Enemy {e_idx} neutralized by Defender {shooter}!")
                self.combat_events.append({"type": "splash", "shooter": int(shooter), "victim": int(e_idx),
                                           "victim_uid": int(self.enemy_uids[e_idx])})
                indices_to_remove.append(e_idx)

        for idx in sorted(list(set(indices_to_remove)), reverse=True):
            self.neutralize_enemy(idx)

    def _process_ram_kills(self, friendly_positions):
        """ Kinetic intercept: defender destroys enemy on contact. """
        if len(self.enemy_positions) == 0: return
        alive_indices = [i for i in range(self.NUM_DRONES) if i not in self.dead_friendly_indices]
        if not alive_indices: return

        alive_pos = friendly_positions[alive_indices]
        kills = []
        for e_idx, e_pos in enumerate(self.enemy_positions):
            dists = np.linalg.norm(alive_pos - e_pos, axis=1)
            j = int(np.argmin(dists))
            if dists[j] < self.RAM_KILL_RADIUS:
                print(f"[RAM] Defender {alive_indices[j]} neutralized Enemy {e_idx}!")
                self.combat_events.append({"type": "ram", "shooter": int(alive_indices[j]), "victim": int(e_idx),
                                           "victim_uid": int(self.enemy_uids[e_idx])})
                kills.append(e_idx)

        for idx in sorted(list(set(kills)), reverse=True):
            self.neutralize_enemy(idx)

    def _process_red_engagements(self, friendly_positions):
        """ Enemies shooting Friendlies """
        if len(self.enemy_positions) == 0: return
        
        for e_idx, agent in enumerate(self.enemy_agents):
            # Only Mode 1 (Air Interceptors) shoot drones
            if agent.mode == 1:
                e_pos = self.enemy_positions[e_idx]
                
                for f_idx in range(self.NUM_DRONES):
                    if f_idx in self.dead_friendly_indices: continue
                    
                    dist = np.linalg.norm(friendly_positions[f_idx] - e_pos)
                    
                    # Same firing range for fairness
                    if self.FIRING_RANGE_MIN <= dist <= self.FIRING_RANGE_MAX:
                        if np.random.rand() < self._hit_probability():
                            print(f"[ALERT] Friendly {f_idx} DOWN! Hit by Enemy {e_idx}")
                            self.dead_friendly_indices.add(f_idx)

    def _check_unattended_violation(self, drone_positions):
        for i, pos in enumerate(self.enemy_positions):
            agent = self.enemy_agents[i]
            dist_to_base = np.linalg.norm(pos - self.BASE_POS)
            
            if agent.mode == 0 and dist_to_base < self.THREAT_RANGE:
                # Only use alive friendlies for coverage check
                alive_pos = [drone_positions[k] for k in range(self.NUM_DRONES) if k not in self.dead_friendly_indices]
                if not alive_pos: return True, "All defenders dead"
                
                dists_to_friendly = np.linalg.norm(alive_pos - pos, axis=1)
                in_firing_range = np.any((dists_to_friendly >= self.FIRING_RANGE_MIN) & 
                                         (dists_to_friendly <= self.FIRING_RANGE_MAX))
                is_intercepting = np.any(dists_to_friendly < 50.0)
                
                if not (in_firing_range or is_intercepting):
                    return True, f"Enemy {i} UNATTENDED"
        return False, ""

    def neutralize_enemy(self, index):
        if 0 <= index < len(self.enemy_ids):
            try: p.removeBody(self.enemy_ids[index], physicsClientId=self.CLIENT)
            except: pass
            del self.enemy_ids[index]
            del self.enemy_agents[index]
            del self.enemy_uids[index]
            if len(self.enemy_positions) > 0:
                self.enemy_positions = np.delete(self.enemy_positions, index, axis=0)
            if len(self.enemy_velocities) > 0:
                self.enemy_velocities = np.delete(self.enemy_velocities, index, axis=0)