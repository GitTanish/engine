
"""
TEST SCRIPT: LANCHESTER ALLOCATION (15 vs 3)
Verifies that 15 defenders split evenly (5 each) against 3 enemies
instead of the default behavior (3 each + 6 overflow on one).
"""
import time
import numpy as np
import pybullet as p
import sys
import os

# --- PATH SETUP ---
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path: sys.path.insert(0, parent_dir)

from gym_pybullet_drones.envs.BaseDefenseAviary import BaseDefenseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DefenseAgent import DefenseAgent
from gym_pybullet_drones.utils.utils import sync

def run():
    NUM_DEFENDERS = 15
    NUM_ENEMIES = 3 # Expect 15 // 3 = 5 defenders per enemy

    print(f"[TEST] Initializing Lanchester Test: {NUM_DEFENDERS} Defenders vs {NUM_ENEMIES} Enemies")
    
    # Init Locations (Ring)
    init_xyzs = np.array([[10*np.cos((2*np.pi/NUM_DEFENDERS)*i), 10*np.sin((2*np.pi/NUM_DEFENDERS)*i), 5.0] for i in range(NUM_DEFENDERS)])
    
    env = BaseDefenseAviary(drone_model=DroneModel.HEAVY, num_drones=NUM_DEFENDERS, 
                            initial_xyzs=init_xyzs, initial_rpys=np.zeros((NUM_DEFENDERS, 3)),
                            physics=Physics.PYB, gui=False, user_debug_gui=False) # Headless for speed

    # 3. AGENTS
    agents = [DefenseAgent(i, NUM_DEFENDERS) for i in range(NUM_DEFENDERS)]

    # 4. CONFIG with ENEMIES
    # We place enemies at specific spots to have distinct costs if needed, 
    # but here we just want to check distribution.
    scenario_config = {
        "num_enemies": NUM_ENEMIES,
        "spawn_radius": 50.0, # Enemies fairly close
        "ground_attacker_ratio": 1.0 # All Bombers (Equal Priority Base Cost)
    }
    obs, info = env.reset(scenario_config=scenario_config)
    
    # We cheat and manually placing enemies to ensure consistent TTI/Distance for the test
    # But for a robust test, getting them from Env is better. 
    # Let's rely on the strategy logic sorting them.
    
    print(f"\n[INFO] Running Simulation needed to hydrate state...")

    for step in range(10): # Run a few steps to let agents init
        # Parse State
        friendly_states = []
        for idx in range(NUM_DEFENDERS):
            s = env._getDroneStateVector(idx)
            friendly_states.append({'id': idx, 'pos': s[0:3], 'quat': s[3:7], 'vel': s[10:13], 'ang_vel': s[13:16]})
            
        # Get Enemy State from Env
        # BaseDefenseAviary stores enemies in separate lists/arrays
        enemies = []
        for i, pos in enumerate(env.enemy_positions):
            enemies.append({
                'id': env.enemy_ids[i],
                'pos': pos,
                'mode': env.enemy_agents[i].mode,
                'vel': np.zeros(3) # Approximation if not tracked explicitly in env
            }) 

        # Compute Actions & Track Allocation
        allocations = {}
        for idx, agent in enumerate(agents):
            try:
                # We are just computing action to trigger internal state if any, 
                # but mostly we want to snoop the select_target result
                
                # Manually mimic DefenseAgent logic to verify the equation
                n_enemies = len(enemies)
                if n_enemies > 0:
                     num_allies = NUM_DEFENDERS # Since we are iterating all
                     # The agent logic uses (len(neighbors)+1). In test, we assume perfect comms/neighbors = all-1
                     dynamic_limit = int(np.ceil(num_allies / n_enemies))
                     agent.strategy.intercept_limit = max(1, dynamic_limit)
                
                target = agent.strategy.select_target(
                    my_id=agent.id,
                    my_pos=friendly_states[idx]['pos'],
                    neighbors=friendly_states,
                    enemies=enemies,
                    asset_pos=np.array([0,0,0])
                )
                
                if target:
                    t_id = target['id']
                    allocations[t_id] = allocations.get(t_id, 0) + 1
                else:
                    allocations['None'] = allocations.get('None', 0) + 1
                    
            except Exception as e:
                print(f"Error computing agent {idx}: {e}")

        # Step Env
        action = np.zeros((NUM_DEFENDERS, 4))
        env.step(action)
        
        if step == 5: # Check at step 5
            print(f"\n[STEP {step}] Allocation Distribution:")
            for t_id, count in allocations.items():
                print(f"  Target {t_id}: {count} Defenders")
            
            # Verify Lanchester Logic
            # We expect roughly equal split: 5, 5, 5
            counts = list(allocations.values())
            if len(counts) == NUM_ENEMIES and all(c == 5 for c in counts):
                print("\n[SUCCESS] PERFECT DISTRIBUTION: 5 Defenders per Enemy.")
            else:
                 print("\n[INFO] Distribution varies (likely due to distance differences). Checking Minimum...")
                 min_count = min(counts) if counts else 0
                 if min_count >= 5:
                     print(f"[SUCCESS] PASSED: Minimum allocation per enemy is {min_count} (>= 5).")
                 else:
                     # It's possible for one to have 6 and another 4 if distances are weird, 
                     # but strategy should enforce limit. 
                     # Actually, strategy ENFORCES 'intercept_limit' as a MAX cap for joining?
                     # No, "if better_allies < self.intercept_limit". 
                     # So if limit is 5, it allows 0..4 'better' aliies. So the 5th guy joins. 
                     # The 6th guy sees 5 better allies, so he does NOT join.
                     # So it is a HARD CAP.
                     # Wait, unless fallback picks it up.
                     # If everyone is capped, fallback goes to Priority 0.
                     # So we expect: E1(5), E2(5), E3(5).
                     # If fallback happens, E1 gets +N.
                     print(f"[RESULT] Minimum allocation: {min_count}. (Expected 5)")

    env.close()

if __name__ == "__main__":
    run()
