"""
Visualizer: Kinetic Interception (The "Chase")
Forces defenders to physically chase and intercept limited enemies.
"""
import time
import numpy as np
import pybullet as p
import sys
import os

# --- PATH SETUP ---
current_file_path = os.path.abspath(__file__)
examples_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(examples_dir)
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path: sys.path.insert(0, parent_dir)

try:
    from gym_pybullet_drones.envs.BaseDefenseAviary import BaseDefenseAviary
    from gym_pybullet_drones.utils.enums import DroneModel, Physics
    from gym_pybullet_drones.control.DefenseAgent import DefenseAgent
except ImportError:
    # Fallback import
    inner_package_dir = os.path.join(project_root, 'gym_pybullet_drones')
    sys.path.append(inner_package_dir) 
    from envs.BaseDefenseAviary import BaseDefenseAviary
    from utils.enums import DroneModel, Physics
    from control.DefenseAgent import DefenseAgent

# --- CONFIGURATION ---
NUM_DEFENDERS = 10
NUM_ENEMIES = 3   # Reduced to create "Swarm" dynamic (10 vs 3)
ASSET_POS = np.array([0, 0, 0])

def run():
    print(f"[INFO] Initializing Kinetic Intercept: {NUM_DEFENDERS} Defenders vs {NUM_ENEMIES} Attackers")

    # --- SPAWN LOGIC: RING FORMATION ---
    init_xyzs = []
    RING_RADIUS = 5.0 
    
    # [FIX] Set Patrol Altitude to match Spec Sheet (60-120m)
    PATROL_ALTITUDE = 60.0 
    
    for i in range(NUM_DEFENDERS):
        angle = (2 * np.pi / NUM_DEFENDERS) * i
        pos = [
            RING_RADIUS * np.cos(angle), 
            RING_RADIUS * np.sin(angle), 
            PATROL_ALTITUDE  # <--- Changed from 1.0
        ]
        init_xyzs.append(pos)
    
    init_xyzs = np.array(init_xyzs)

    # Initialize Env
    # Note: num_drones=NUM_DEFENDERS because enemies are managed internally by BaseDefenseAviary
    env = BaseDefenseAviary(
        drone_model=DroneModel.HEAVY,
        num_drones=NUM_DEFENDERS,
        initial_xyzs=init_xyzs,
        initial_rpys=np.zeros((NUM_DEFENDERS, 3)),
        physics=Physics.PYB,
        gui=True,
        record=False,
        obstacles=False
    )

    # --- SPAWN ENEMIES VIA RESET CONFIG ---
    # This is where we tell the environment how many enemies to create
    scenario_config = {
        "num_enemies": NUM_ENEMIES,
        "spawn_radius": 200.0,
        "ground_attacker_ratio": 1.0 # All enemies rush the base
    }
    
    # Init Defenders
    friendlies = [DefenseAgent(i, NUM_DEFENDERS) for i in range(NUM_DEFENDERS)]
    
    # Reset with config
    env.reset(scenario_config=scenario_config)
    
    # --- SIMULATION LOOP ---
    START = time.time()
    step = 0
    while True: 
        # 1. Prepare State
        friendly_states = []
        enemy_states = []
        
        # Get Friendly States (Blue Team)
        for idx in range(NUM_DEFENDERS):
            state = env._getDroneStateVector(idx)
            friendly_states.append({
                'id': idx, 'pos': state[0:3], 'vel': state[10:13], 'quat': state[3:7], 'ang_vel': state[13:16]
            })

        # Get Enemy States (Red Team - From Env Internals)
        # Source of Truth: env.enemy_positions (Active enemies only)
        if hasattr(env, 'enemy_positions'):
            for e_idx, e_pos in enumerate(env.enemy_positions):
                enemy_states.append({
                    'id': NUM_DEFENDERS + e_idx, 'pos': e_pos, 'vel': np.zeros(3) 
                })

        # 2. Compute Actions (Defenders Chase)
        action = np.zeros((NUM_DEFENDERS, 4))
        # visible_enemies is just enemy_states now
        visible_enemies = enemy_states

        for idx, agent in enumerate(friendlies):
            rpm = agent.compute_action(
                my_state=friendly_states[idx],
                neighbors=friendly_states,
                enemies=visible_enemies,
                asset_pos=ASSET_POS,
                dt=env.CTRL_TIMESTEP,
                time_now=step * env.CTRL_TIMESTEP
            )
            action[idx, :] = rpm

        # 3. Step Physics
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- HYBRID LOGIC: LASER LOCK + KINETIC KILL ---
        LOCK_RANGE = 250.0
        KILL_RANGE = 10.0
        KILL_PROB = 0.2
        
        current_enemy_positions = env.enemy_positions # Numpy array
        
        # Debug: Print Enemy 0 Distance to confirm movement
        if len(current_enemy_positions) > 0 and step % 60 == 0:
            d0 = np.linalg.norm(current_enemy_positions[0] - ASSET_POS)
            print(f"[DEBUG] Step {step}: Enemy 0 Dist to Base: {d0:.2f}m")

        enemies_to_remove = []

        for f_idx, friendly in enumerate(friendly_states):
            # Check against all remaining enemies
            for e_idx, e_pos in enumerate(current_enemy_positions):
                if e_idx in enemies_to_remove: continue
                
                dist = np.linalg.norm(friendly['pos'] - e_pos)
                
                # A. LOCK (Visual)
                if dist < LOCK_RANGE:
                    p.addUserDebugLine(friendly['pos'], e_pos, [0, 1, 1], 1, 0.1, physicsClientId=env.CLIENT)

                    # B. KILL (Kinetic)
                    if dist < KILL_RANGE:
                        p.addUserDebugLine(friendly['pos'], e_pos, [1, 1, 0], 3, 0.1, physicsClientId=env.CLIENT)
                        
                        if np.random.random() < KILL_PROB:
                            print(f"[SPLASH] Defender {f_idx} neutralized Enemy {e_idx} at {dist:.1f}m!")
                            enemies_to_remove.append(e_idx)
                            break # Defender can only kill one per step

        # Process Removals (Reverse order to maintain indices)
        if enemies_to_remove:
            for e_idx in sorted(list(set(enemies_to_remove)), reverse=True):
                env.neutralize_enemy(e_idx)

        # Victory Check
        if len(env.enemy_positions) == 0:
            print("\n[VICTORY] Swarm successfully neutralized all intruders.")
            time.sleep(2)
            break
            
        # Base Hit Check
        for e_pos in env.enemy_positions:
            if np.linalg.norm(e_pos) < 2.0:
                print(f"\n[FAILURE] An Enemy impacted the Asset!")
                time.sleep(2)
                break

        # Sync visual speed
        from gym_pybullet_drones.utils.utils import sync
        sync(step, START, env.CTRL_TIMESTEP)
        step += 1
        
        # Check for 'q' key press to exit
        keys = p.getKeyboardEvents()
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            print("[INFO] 'q' pressed. Exiting simulation.")
            break

    env.close()

if __name__ == "__main__":
    run()
