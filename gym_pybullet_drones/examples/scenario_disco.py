"""Script to visualize the Base Defense scenario with a moving threat."""
import time
import numpy as np
import pybullet as p
import sys
import os

# --- 1. PATH FIX (ABSOLUTE ROBUSTNESS) ---
# Goal: Find the 'gym_pybullet_drones' folder that contains 'envs', 'control', etc.
current_file_path = os.path.abspath(__file__)
examples_dir = os.path.dirname(current_file_path) # .../examples
project_root = os.path.dirname(examples_dir)      # .../gym_pybullet_drones (Outer)

# We want to add the PARENT of 'project_root' to sys.path so we can do "from gym_pybullet_drones.envs..."
# This is required because the code uses absolute imports like 'gym_pybullet_drones.utils...'
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Debug: Print where we are looking
print(f"[DEBUG] Project Root added to path: {project_root}")

# --- 2. AUTO-INIT FIX ---
# Check if the inner 'gym_pybullet_drones' folder exists
inner_package_dir = os.path.join(project_root, 'gym_pybullet_drones')

if not os.path.exists(inner_package_dir):
    # If the folder 'gym_pybullet_drones' doesn't exist inside the root, 
    # it means your structure is flattened (envs is directly in root).
    # This handles the case where you might have moved files around.
    # We will try to import directly from the root if that's the case.
    print(f"[WARN] Inner package '{inner_package_dir}' not found.")
else:
    # Ensure __init__.py exists in the inner package
    init_path = os.path.join(inner_package_dir, '__init__.py')
    if not os.path.exists(init_path):
        print(f"[INFO] Creating missing __init__.py at: {init_path}")
        try:
            with open(init_path, 'w') as f:
                f.write("# Auto-generated package init\n")
        except Exception as e:
            print(f"[ERROR] Could not create __init__.py: {e}")

# --- 3. IMPORTS ---
try:
    # Try standard package import
    from gym_pybullet_drones.envs.BaseDefenseAviary import BaseDefenseAviary
    from gym_pybullet_drones.utils.enums import DroneModel, Physics
    from gym_pybullet_drones.control.DefenseAgent import DefenseAgent
    from gym_pybullet_drones.utils.utils import sync
    print("[INFO] Imported modules successfully via 'gym_pybullet_drones' package.")
except ImportError as e:
    print(f"[INFO] Package import failed: {e}")
    print("[INFO] Trying direct import (for flattened structure)...")
    try:
        # Fallback: Maybe 'envs' is directly reachable?
        sys.path.append(inner_package_dir) 
        from envs.BaseDefenseAviary import BaseDefenseAviary
        from utils.enums import DroneModel, Physics
        from control.DefenseAgent import DefenseAgent
        from utils.utils import sync
        print("[INFO] Imported modules successfully via direct import.")
    except ImportError as e:
        print(f"\n[FATAL ERROR] Could not import modules.")
        print(f"PYTHONPATH: {sys.path}")
        print(f"Error details: {e}")
        sys.exit(1)

# --- CONFIG ---
NUM_DEFENDERS = 10
NUM_ENEMIES = 10 # This is now just for the scenario config
ASSET_POS = np.array([0, 0, 0])



def run():
    """
    Run the simulation loop to visualize the Base Defense scenario.
    """
    # --- SPAWN LOGIC ---
    init_xyzs = []
    
    # 1. BLUE TEAM (Defenders) - Ring Formation
    for i in range(NUM_DEFENDERS):
        angle = (2*np.pi / NUM_DEFENDERS) * i
        # Start in air (1.0m)
        init_xyzs.append([5 * np.cos(angle), 5 * np.sin(angle), 1.0])
    
    init_xyzs = np.array(init_xyzs)
    
    # --- ENVIRONMENT SETUP ---
    print(f"[INFO] Initializing SimAstra Demo")

    # Initialize the environment with ONLY Defenders
    # Enemies are now managed internally by BaseDefenseAviary
    env = BaseDefenseAviary(drone_model=DroneModel.HEAVY,
                            num_drones=NUM_DEFENDERS,
                            initial_xyzs=init_xyzs,
                            initial_rpys=np.zeros((NUM_DEFENDERS, 3)),
                            physics=Physics.PYB,
                            gui=False,
                            record=False,
                            obstacles=False,
                            user_debug_gui=True,
                            neighbourhood_radius=100.0
                            )

    # --- AGENT SETUP ---
    friendlies = [DefenseAgent(i, NUM_DEFENDERS) for i in range(NUM_DEFENDERS)]

    # --- SCOREBOARD ---
    wins = 0
    losses = 0
    episode_count = 0

    while True: # Episode Loop
        episode_count += 1
        print(f"\n[INFO] Starting Episode {episode_count}")
        print(f"[SCORE] Wins: {wins} | Losses: {losses}")
        
        # Reset the environment (spawns the enemies via config)
        scenario_config = {
            "num_enemies": NUM_ENEMIES,
            "spawn_radius": 400.0, # CHANGE: 400m is the tactical sweet spot (40s flight time)
            "ground_attacker_ratio": 0.6 # Mix of types
        }
        obs, info = env.reset(scenario_config=scenario_config)
        

        

        
        # --- COLORING ---
        for i in range(NUM_DEFENDERS):
            p.changeVisualShape(env.DRONE_IDS[i], -1, rgbaColor=[0, 0, 1, 1], physicsClientId=env.CLIENT) # Blue
        
        # --- ASSET VISUALS ---
        try:
            # Visual Dome Line for debugging
            p.addUserDebugLine([0,0,0], [0,0,5], [0,1,0], 2, physicsClientId=env.CLIENT)
        except:
            pass

        START = time.time()
        action = np.zeros((NUM_DEFENDERS, 4))
        
        # Run Episode
        i = 0
        episode_over = False
        
        while not episode_over:
            
            # 1. State Parsing (Friendly)
            friendly_states = []
            for idx in range(NUM_DEFENDERS):
                state = env._getDroneStateVector(idx)
                # State structure: pos(3), quat(4), rpy(3), vel(3), ang_vel(3)
                data = {
                    'id': idx,
                    'pos': state[0:3],
                    'quat': state[3:7],
                    'vel': state[10:13],
                    'ang_vel': state[13:16]
                }
                friendly_states.append(data)

            # 2. Parse Enemies from Environment
            # BaseDefenseAviary manages enemies internally now.
            # We construct a list of dicts for the DefenseAgent.
            current_enemies = []
            if len(env.enemy_positions) > 0:
                for e_idx, e_pos in enumerate(env.enemy_positions):
                    # FETCH AGENT MODE DIRECTLY
                    agent_mode = env.enemy_agents[e_idx].mode
                    
                    current_enemies.append({
                        'id': e_idx,
                        'pos': e_pos,
                        'vel': np.zeros(3), # Velocity not easily accessible, assuming 0 for now
                        'mode': agent_mode # NEW: Required for Priority Logic
                    })

            # 3. Compute Actions (Defenders Only)

            for idx, agent in enumerate(friendlies):
                rpm = agent.compute_action(
                    my_state=friendly_states[idx],
                    neighbors=friendly_states,
                    enemies=current_enemies,
                    asset_pos=ASSET_POS,
                    dt=env.CTRL_TIMESTEP,
                    time_now=i/env.CTRL_FREQ
                )
                action[idx, :] = rpm
                
                # Debug Line: Patrol (Blue) vs Intercept (Red)
                my_pos = friendly_states[idx]['pos']
                if agent.state == "INTERCEPT":
                     p.addUserDebugLine(my_pos, my_pos + [0,0,2], [1,0,0], 1, lifeTime=0.1, physicsClientId=env.CLIENT)
                elif agent.state == "PATROL":
                     p.addUserDebugLine(my_pos, my_pos + [0,0,1], [0,0,1], 1, lifeTime=0.1, physicsClientId=env.CLIENT)

            # --- RED TEAM ---
            # The Environment (BaseDefenseAviary) now handles Enemy movement internally.
            # We just pass zeros for their motor commands so the physics engine doesn't interfere.
            for idx in range(NUM_ENEMIES):
                # NOTE: In this script, 'action' is size (NUM_DEFENDERS, 4).
                # We do NOT need to set actions for enemies because the environment
                # step() only expects actions for the controlled drones (NUM_DEFENDERS).
                # Wait, BaseDefenseAviary inherits from CtrlAviary.
                # If num_drones=NUM_DEFENDERS, then step() expects (NUM_DEFENDERS, 4).
                # The enemies are separate bodies managed by PyBullet directly in step().
                # So we don't need to add them to 'action' at all.
                pass

            # 4. Step the simulation
            # env.step() now handles enemy movement internally
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Sync the simulation
            sync(i, START, env.CTRL_TIMESTEP)
            
            # --- WIN/LOSS LOGIC ---
            # Check Loss: Enemy hits base
            # We check env.enemy_positions
            if len(env.enemy_positions) > 0:
                dists_to_base = np.linalg.norm(env.enemy_positions - ASSET_POS, axis=1)
                if np.any(dists_to_base < 1.0):
                    print(f"\n[FAILURE] An Enemy hit the Asset!")
                    losses += 1
                    episode_over = True

            if episode_over: break

            # Check Win: All enemies neutralized
            # Neutralization condition: Defender gets very close (Ramming)
            # We need to manually check this and modify env state to "kill" enemies
            
            enemies_to_remove = []
            for e_idx, e_pos in enumerate(env.enemy_positions):
                for f_idx, friendly in enumerate(friendly_states):
                    dist = np.linalg.norm(friendly['pos'] - e_pos)
                    if dist < 2.0: # Ram radius
                        print(f"[COMBAT] Defender {f_idx} neutralized Enemy {e_idx}!")
                        enemies_to_remove.append(e_idx)
                        break # One defender can only kill one enemy per step (simplification)
            
            # Remove killed enemies (in reverse order to preserve indices)
            if enemies_to_remove:
                enemies_to_remove = sorted(list(set(enemies_to_remove)), reverse=True)
                for e_idx in enemies_to_remove:
                    # Remove from PyBullet
                    try:
                        p.removeBody(env.enemy_ids[e_idx], physicsClientId=env.CLIENT)
                    except:
                        pass
                    # Remove from Env Lists
                    del env.enemy_ids[e_idx]
                    del env.enemy_agents[e_idx]
                    # Numpy delete
                    env.enemy_positions = np.delete(env.enemy_positions, e_idx, axis=0)
            
            if len(env.enemy_positions) == 0:
                print("\n[VICTORY] All enemies neutralized!")
                wins += 1
                episode_over = True
                break

            # Optional: Print distance every second
            if i % env.CTRL_FREQ == 0 and len(env.enemy_positions) > 0:
                # Debug: Nearest Enemy Dist
                dists = np.linalg.norm(env.enemy_positions - ASSET_POS, axis=1)
                
                # Battle Prediction
                prediction = friendlies[0].strategy.predict_battle_outcome(len(friendlies), len(env.enemy_positions))
                
                print(f"[DEBUG] Nearest Enemy Dist: {np.min(dists):.2f}m | Prediction: {prediction}")
                p.addUserDebugText(f"Battle Prediction: {prediction}", [0, 0, 10], [1, 1, 1], 1.5, 1.0, physicsClientId=env.CLIENT)
                
            # Check for 'q' key press to exit
            keys = p.getKeyboardEvents()
            if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                print("[INFO] 'q' pressed. Exiting simulation.")
                sys.exit(0)
                
            i += 1
        
        # Pause briefly before reset
        time.sleep(2.0)

    # Close the environment
    env.close()

if __name__ == "__main__":
    run()
