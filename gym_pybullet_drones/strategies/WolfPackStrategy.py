import numpy as np

class WolfPackStrategy:
    """
    Advanced Decentralized Strategy.
    Fixes:
    1. TTI-based Scoring (Prioritizes imminent threats).
    2. Deterministic Tie-Breaking (Solves Comms-Denied Suicide).
    3. Proper Priority (Bombers > Fighters).
    """
    def __init__(self, intercept_limit=3):
        self.intercept_limit = intercept_limit

    def select_target(self, my_id, my_pos, neighbors, enemies, asset_pos):
        """
        Selects target based on Time-To-Impact (TTI) and Implicit Coordination.
        """
        if not enemies:
            return None

        if neighbors is None:
            neighbors = []

        # --- 1. THREAT ASSESSMENT (TTI Based) ---
        threats = []
        for e in enemies:
            e_pos = np.array(e['pos'], dtype=float)
            e_mode = e.get('mode', 0)
            
            # Distance to Asset
            dist_asset = np.linalg.norm(e_pos - np.array(asset_pos))
            
            # Time To Impact (TTI)
            # Assuming max enemy speed 10m/s. TTI = Dist / Speed
            tti = dist_asset / 10.0 
            
            # COST CALCULATION (Lower = Higher Priority)
            # We want to minimize Cost.
            
            # Base Cost: Bomber = TTI. Fighter = TTI + Penalty.
            # This ensures we always prioritize a Bomber 10s away over a Fighter 1s away.
            cost = tti 
            
            if e_mode == 1: # Air Interceptor (Fighter)
                cost += 790.2 # Optimized via Genetic Algorithm (was 5000.0)
            
            # Secondary Sort: Distance to Me (Efficiency)
            dist_to_me = np.linalg.norm(e_pos - my_pos)
            
            threats.append({
                'cost': cost,
                'dist_me': dist_to_me,
                'data': e,
                'id': e['id'] # Enemy ID is needed for hash
            })

        # Sort by Cost (Priority), then Efficiency
        threats.sort(key=lambda x: (x['cost'], x['dist_me']))
        
        # --- 2. DECENTRALIZED ALLOCATION (The "Comms-Denied" Fix) ---
        best_target = None
        
        # STATE A: We have neighbors (Comms UP) -> Use Distance Ranking
        if len(neighbors) > 0:
            # Track neighbors who have likely taken a higher-priority target
            occupied_neighbor_ids = set()

            for threat in threats:
                e_pos = np.array(threat['data']['pos'])
                
                # Identify "Better Allies" for THIS threat
                better_allies_count = 0
                potential_takers = []

                for n in neighbors:
                    n_id = n.get('id', -1)
                    if n_id in occupied_neighbor_ids:
                        continue # This neighbor is busy with a higher priority target
                    
                    n_pos = np.array(n['pos'])
                    dist_n = np.linalg.norm(n_pos - e_pos)
                    dist_me = threat['dist_me']
                    
                    # Check if 'n' is better than 'me'
                    is_better = False
                    if dist_n < dist_me - 1e-6:
                        is_better = True
                    elif abs(dist_n - dist_me) < 1e-6:
                         # Tie Breaker
                         if n_id != -1 and n_id < my_id:
                             is_better = True
                    
                    if is_better:
                        better_allies_count += 1
                    
                    # Store (dist, id) to determine who effectively takes this target
                    potential_takers.append((dist_n, n_id))

                # If I am in the top 'limit' available drones, I take it
                if better_allies_count < self.intercept_limit:
                    best_target = threat['data']
                    break
                else:
                    # I didn't take it.
                    # Mark the TOP 'limit' neighbors as occupied for future checks.
                    # We assume the best positioned neighbors took this target.
                    potential_takers.sort() # Sort by distance
                    for i in range(min(len(potential_takers), self.intercept_limit)):
                        occupied_neighbor_ids.add(potential_takers[i][1])
        
        # STATE B: Neighbors empty (Comms DOWN) -> Use Deterministic Hashing
        else:
            # "I am alone. I must pick a target that no one else is picking."
            # Algorithm: Modulo Allocation.
            # We assume my peers are also running this algorithm.
            # If there are 3 bombers, and I am ID 4: 4 % 3 = 1 -> I take Bomber #1.
            
            # Filter only Bombers (High Priority) to ensure we cover them first
            bombers = [t for t in threats if t['cost'] < 2000]
            
            if bombers:
                target_idx = my_id % len(bombers)
                best_target = bombers[target_idx]['data']
            else:
                # No bombers? Pick closest fighter based on ID hash to spread fire
                target_idx = my_id % len(threats)
                best_target = threats[target_idx]['data']

        # Fallback
        if best_target is None and threats:
            best_target = threats[0]['data']
            
        return best_target

    def predict_battle_outcome(self, num_defenders, num_attackers):
        if num_attackers == 0: return "Victory (100%)"
        blue_str = num_defenders ** 2
        red_str = num_attackers ** 2
        if blue_str > red_str:
            return f"Victory (Pred. Remaining: {np.sqrt(blue_str - red_str):.1f})"
        else:
            return f"Defeat (Pred. Enemy Remaining: {np.sqrt(red_str - blue_str):.1f})"
