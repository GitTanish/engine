import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.EnemyAgent import EnemyAgent

class BaseDefenseAviary(CtrlAviary):
    def __init__(self, num_drones=3, **kwargs):
        super().__init__(num_drones=num_drones, **kwargs)
        
        # Defense Parameters (KNIFE FIGHT DOCTRINE)
        self.BASE_POS = np.array([0, 0, 0])
        self.THREAT_RANGE = 40.0       # Reduced from 100.0
        self.FIRING_RANGE_MIN = 5.0    # Reduced from 30.0
        self.FIRING_RANGE_MAX = 30.0   # Reduced from 250.0 (Visual Kill Box)
        self.HIT_PROBABILITY_PER_STEP = 0.05 
        
        # Swarm State
        self.enemy_ids = []
        self.enemy_agents = []
        self.enemy_positions = []
        
        # NEW: Track dead friendlies to disable them
        self.dead_friendly_indices = set()

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
        self.dead_friendly_indices = set()

        super().reset(seed=seed, options=options)
        
        # Default config if none provided (Safe default)
        config = scenario_config if scenario_config else {
            "num_enemies": 3,
            "spawn_radius": 100.0,
            "ground_attacker_ratio": 0.3 
        }
        
        import os
        # Loading heavy_drone.urdf from assets folder (relative to this file location in examples/visualisation_box)
        # current_file = .../examples/visualisation_box/BaseDefenseAviary.py
        # root is .../gym_pybullet_drones
        # assets is .../gym_pybullet_drones/assets
        # So we need to go up TWO levels: .. (examples) / .. (gym_pybullet_drones) 
        # But wait, gym_pybullet_drones package is usually installed or in python path. 
        # The original code used relative path from `envs` folder. 
        # `envs` is at `gym_pybullet_drones/envs`.
        # `visualisation_box` is at `gym_pybullet_drones/examples/visualisation_box`.
        # So `..` is `examples`, `../..` is `gym_pybullet_drones`, `../../assets` is correct.
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        drone_path = os.path.join(current_dir, '..', '..', 'assets', 'heavy_drone.urdf')

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
            self.enemy_agents.append(EnemyAgent(mode=mode))
            
        self.enemy_positions = np.array(self.enemy_positions)
        
        # Base Visual
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[0, 1, 0, 1])
        p.createMultiBody(baseVisualShapeIndex=visual_shape_id, basePosition=self.BASE_POS)

        return self._computeObs(), {}

    def step(self, action):
        # 1. Enforce Death (The Graveyard)
        if len(self.dead_friendly_indices) > 0:
            for dead_idx in self.dead_friendly_indices:
                action[dead_idx, :] = 0 
                p.resetBasePositionAndOrientation(
                    self.DRONE_IDS[dead_idx], 
                    [0, 0, -100 - dead_idx], 
                    [0,0,0,1], 
                    physicsClientId=self.CLIENT
                )

        # 2. Move Enemies
        current_drone_pos = np.array([self._getDroneStateVector(i)[0:3] for i in range(self.NUM_DRONES)])
        new_positions = []
        for idx, agent in enumerate(self.enemy_agents):
            vel = agent.compute_action(
                my_pos=self.enemy_positions[idx],
                asset_pos=self.BASE_POS,
                friendly_drones_pos=current_drone_pos
            )
            new_pos = self.enemy_positions[idx] + (vel * self.CTRL_TIMESTEP)
            new_positions.append(new_pos)
            p.resetBasePositionAndOrientation(self.enemy_ids[idx], new_pos, [0,0,0,1], physicsClientId=self.CLIENT)
        self.enemy_positions = np.array(new_positions)

        # 3. Physics Step
        obs, reward, term, trunc, info = super().step(action)
        
        # 4. WEAPONS: Blue shooting Red
        self._process_blue_engagements(current_drone_pos)
        
        # 5. WEAPONS: Red shooting Blue
        self._process_red_engagements(current_drone_pos)

        # 6. Check Unattended
        violation, details = self._check_unattended_violation(current_drone_pos)
        if violation:
            info["unattended_violation"] = True
            
        info["friendly_losses"] = len(self.dead_friendly_indices)
            
        return obs, reward, term, trunc, info

    def _process_blue_engagements(self, friendly_positions):
        if len(self.enemy_positions) == 0: return
        indices_to_remove = []
        for e_idx, e_pos in enumerate(self.enemy_positions):
            alive_indices = [i for i in range(self.NUM_DRONES) if i not in self.dead_friendly_indices]
            if not alive_indices: break
            
            alive_pos = friendly_positions[alive_indices]
            dists = np.linalg.norm(alive_pos - e_pos, axis=1)
            
            in_range = (dists >= self.FIRING_RANGE_MIN) & (dists <= self.FIRING_RANGE_MAX)
            if np.any(in_range):
                if np.random.rand() < self.HIT_PROBABILITY_PER_STEP:
                    indices_to_remove.append(e_idx)
                    print(f"[SPLASH] Enemy {e_idx} neutralized!")

        for idx in sorted(list(set(indices_to_remove)), reverse=True):
            self.neutralize_enemy(idx)

    def _process_red_engagements(self, friendly_positions):
        if len(self.enemy_positions) == 0: return
        for e_idx, agent in enumerate(self.enemy_agents):
            if agent.mode == 1:
                e_pos = self.enemy_positions[e_idx]
                for f_idx in range(self.NUM_DRONES):
                    if f_idx in self.dead_friendly_indices: continue
                    dist = np.linalg.norm(friendly_positions[f_idx] - e_pos)
                    if self.FIRING_RANGE_MIN <= dist <= self.FIRING_RANGE_MAX:
                        if np.random.rand() < self.HIT_PROBABILITY_PER_STEP:
                            print(f"[ALERT] Friendly {f_idx} DOWN! Hit by Enemy {e_idx}")
                            self.dead_friendly_indices.add(f_idx)

    def _check_unattended_violation(self, drone_positions):
        for i, pos in enumerate(self.enemy_positions):
            agent = self.enemy_agents[i]
            dist_to_base = np.linalg.norm(pos - self.BASE_POS)
            
            if agent.mode == 0 and dist_to_base < self.THREAT_RANGE:
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
            if len(self.enemy_positions) > 0:
                self.enemy_positions = np.delete(self.enemy_positions, index, axis=0)
