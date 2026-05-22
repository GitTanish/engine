
import numpy as np
import copy
import random
import time
import sys
import os

# --- PATH SETUP ---
# Ensure we can import the gym_pybullet_drones package
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
# Add the PARENT of the project root to sys.path to allow 'import gym_pybullet_drones'
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path: sys.path.insert(0, parent_dir)

from gym_pybullet_drones.envs.BaseDefenseAviary import BaseDefenseAviary
from gym_pybullet_drones.control.DefenseAgent import DefenseAgent
from gym_pybullet_drones.strategies.WolfPackStrategy import WolfPackStrategy
from gym_pybullet_drones.utils.enums import DroneModel

# --- 1. THE EVOLVABLE BRAIN ---
class EvolvableStrategy(WolfPackStrategy):
    """
    A genetic subclass of WolfPackStrategy.
    Overrides 'select_target' to use a genetically tuned 'fighter_penalty'.
    """
    def __init__(self, fighter_penalty):
        # We initialize with a default limit; DefenseAgent updates it dynamically.
        super().__init__(intercept_limit=3) 
        self.fighter_penalty = fighter_penalty

    def select_target(self, my_id, my_pos, neighbors, enemies, asset_pos):
        """
        Modified target selection that uses self.fighter_penalty instead of hardcoded 5000.0.
        """
        if not enemies: return None
        
        threats = []
        for e in enemies:
            e_pos = np.array(e['pos'], dtype=float)
            e_mode = e.get('mode', 0)
            
            # Distance & TTI Calculation
            dist_asset = np.linalg.norm(e_pos - np.array(asset_pos))
            tti = dist_asset / 10.0 
            
            # --- THE GENE: Tunable Penalty ---
            cost = tti 
            if e_mode == 1: # Air Interceptor (Fighter)
                cost += self.fighter_penalty # Evolved value
            
            dist_to_me = np.linalg.norm(e_pos - my_pos)
            
            threats.append({
                'cost': cost,
                'dist_me': dist_to_me,
                'data': e,
                'id': e['id']
            })

        # Sort by Cost (Priority), then Efficiency
        threats.sort(key=lambda x: (x['cost'], x['dist_me']))
        
        # --- DECENTRALIZED ALLOCATION ---
        best_target = None
        
        # State A: Comms Up
        if len(neighbors) > 0:
            for threat in threats:
                e_pos = np.array(threat['data']['pos'])
                # Count allies closer to this threat than me
                better_allies = sum(1 for n in neighbors if np.linalg.norm(np.array(n['pos']) - e_pos) < threat['dist_me'])
                
                if better_allies < self.intercept_limit:
                    best_target = threat['data']
                    break
        
        # State B: Comms Down (Fallback)
        else:
            bombers = [t for t in threats if t['cost'] < 2000]
            if bombers: best_target = bombers[my_id % len(bombers)]['data']
            else: best_target = threats[my_id % len(threats)]['data']

        return best_target if best_target else threats[0]['data']

# --- 2. THE SIMULATION RUNNER ---
def run_generation(population, generation_id):
    results = []
    print(f"\n--- GENERATION {generation_id} START ---")
    
    for i, genome in enumerate(population):
        penalty = genome
        
        # Setup Headless Environment (Scenario B: Asymmetric Stress Test)
        # 10 Defenders vs 13 Attackers (6 Bombers, 7 Fighters)
        # gui=False for speed
        env = BaseDefenseAviary(
            drone_model=DroneModel.HEAVY,
            num_drones=10, 
            gui=False
        )
        
        # Init Agents with Evolved Brains
        agents = []
        for k in range(10):
            agent = DefenseAgent(k, 10, verbose=False)
            # HOT SWAP THE STRATEGY
            agent.strategy = EvolvableStrategy(fighter_penalty=penalty)
            agents.append(agent)
            
        # Run Episode
        obs, _ = env.reset(scenario_config={"num_enemies": 13, "spawn_radius": 400.0, "ground_attacker_ratio": 0.46})
        total_reward = 0
        steps = 0
        
        # Max steps 14400 (60s) to allow engagement
        for _ in range(14400):
            steps += 1
            actions = np.zeros((10, 4))
            
            # Construct Perception Data (Mocking Sensors)
            all_drone_states = [env._getDroneStateVector(k) for k in range(10)]
            
            # Build Neighbors List
            neighbors = []
            for k in range(10):
                if k not in env.dead_friendly_indices: # Only alive drones talk
                    neighbors.append({'id': k, 'pos': all_drone_states[k][0:3]})
            
            # Build Enemies List
            enemies = []
            for k in range(len(env.enemy_agents)):
                # BaseDefenseAviary handles enemy removal, so we just check existing lists
                if k < len(env.enemy_positions):
                    enemies.append({
                        'id': env.enemy_ids[k], 
                        'pos': env.enemy_positions[k], 
                        'mode': env.enemy_agents[k].mode
                    })
            
            # Compute Actions
            for k, agent in enumerate(agents):
                if k in env.dead_friendly_indices: continue
                
                state_vec = all_drone_states[k]
                my_state = {'pos': state_vec[0:3], 'vel': state_vec[10:13], 'quat': state_vec[3:7], 'ang_vel': state_vec[13:16]}
                
                # DefenseAgent handles Dynamic Intercept Limit internally
                rpm = agent.compute_action(my_state, neighbors, enemies, env.BASE_POS, env.CTRL_TIMESTEP, steps*env.CTRL_TIMESTEP)
                actions[k, :] = rpm
                
            obs, _, _, _, info = env.step(actions)
            
            # --- EARLY TERMINATION CHECK (SPEED OPTIMIZATION) ---
            if len(env.dead_friendly_indices) == 10:
                # All defenders dead. Stop wasting CPU.
                # Penalty will be applied naturally by casualty count below.
                break

            # --- FITNESS FUNCTION ---
            
            # 1. FAIL CONDITION: Asset Unattended
            if info.get("unattended_violation", False):
                total_reward -= 5000 # "Death Sentence" for violating Mission Critical Rule
                break
            
            # 2. WIN CONDITION: All Enemies Neutralized
            if len(enemies) == 0:
                total_reward += 2000 # Victory Bonus
                # Survival Bonus: High value on keeping drones alive
                total_reward += (10 - len(env.dead_friendly_indices)) * 200 
                break
                
        # Partial Credit if Time Runs Out but Asset Safe
        if len(enemies) > 0 and not info.get("unattended_violation", False):
             total_reward += 500 
             
        # Casualties Penalty (Minimize Loss)
        casualties = len(env.dead_friendly_indices)
        total_reward -= (casualties * 100)
        
        results.append((genome, total_reward))
        env.close()
        
        print(f"Ind {i}: Penalty={penalty:.1f} | Casualties: {casualties} | Score: {total_reward}")
        
    return results

# --- 3. THE OPTIMIZER ---
def optimize():
    # Initial Population: Test extremes and random values
    pop_size = 10
    # Include the current baseline (5000.0) and the dangerous low-end (0.0)
    population = [5000.0, 2500.0, 1000.0, 500.0, 100.0, 0.0] 
    while len(population) < pop_size:
        population.append(random.uniform(0.0, 5000.0))
        
    generations = 5 # Run 5 generations
    
    for gen in range(generations):
        fitness_scores = run_generation(population, gen)
        
        # Sort by Score (Descending)
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        top_performers = fitness_scores[:3] # Keep top 3 Elites
        
        print(f"--> BEST OF GEN {gen}: Penalty {top_performers[0][0]:.1f} (Score: {top_performers[0][1]})")
        
        # Breeding Strategy
        new_pop = [x[0] for x in top_performers] # Elitism
        
        while len(new_pop) < pop_size:
            # Parent Selection
            parent_a = random.choice(top_performers)[0]
            parent_b = random.choice(top_performers)[0]
            
            # Crossover (Average)
            child = (parent_a + parent_b) / 2.0
            
            # Mutation (Random Drift)
            if random.random() < 0.3:
                child += random.uniform(-500, 500)
                child = max(0.0, child) # Clamp to 0
                
            new_pop.append(child)
            
        population = new_pop
        
    print("\n=== OPTIMIZATION COMPLETE ===")
    print(f"Recommended Fighter Penalty: {population[0]:.1f}")
    print("Action: Update 'cost += 5000.0' in WolfPackStrategy.py with this value.")

if __name__ == "__main__":
    optimize()
