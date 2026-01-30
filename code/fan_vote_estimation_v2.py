"""
Fan Vote Estimation Model for Dancing with the Stars (Version 2)
=================================================================
Enhanced model with proper handling of:
- Withdrew cases
- No-elimination weeks  
- Multi-elimination weeks
- Finals ranking constraints
- Correct Rank and Bottom2 constraints

Output:
- fan_vote_results.csv: Estimated fan vote shares with uncertainty measures
- consistency_metrics.csv: Consistency and certainty metrics per season
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.special import softmax
from itertools import permutations
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(filepath):
    """Load data and compute weekly judge totals."""
    df = pd.read_csv(filepath)
    
    # Extract week columns
    week_cols = {}
    for week in range(1, 12):
        judge_cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
        week_cols[week] = [c for c in judge_cols if c in df.columns]
    
    # Compute weekly judge totals (J_{i,t})
    for week in range(1, 12):
        cols = week_cols[week]
        if cols:
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df[f'J_week{week}'] = df[cols].sum(axis=1, skipna=True)
    
    return df

def get_season_data(df, season):
    """Extract season-specific data structures with enhanced parsing."""
    season_df = df[df['season'] == season].copy()
    
    def parse_result(result):
        """Parse results field to get elimination info."""
        result = str(result)
        import re
        
        info = {
            'is_withdrew': 'Withdrew' in result,
            'is_finalist': False,
            'elim_week': None,
            'placement': None
        }
        
        # Check for elimination week
        match = re.search(r'Eliminated Week (\d+)', result)
        if match:
            info['elim_week'] = int(match.group(1))
        
        # Check for placement (finalist)
        if 'Place' in result or result in ['1st Place', '2nd Place', '3rd Place', '4th Place', '5th Place']:
            info['is_finalist'] = True
            place_match = re.search(r'(\d+)', result)
            if place_match:
                info['placement'] = int(place_match.group(1))
        
        return info
    
    # Parse all results
    for idx, row in season_df.iterrows():
        parsed = parse_result(row['results'])
        for k, v in parsed.items():
            season_df.loc[idx, k] = v
    
    # Get last active week for each contestant
    def get_last_active_week(row):
        for week in range(11, 0, -1):
            col = f'J_week{week}'
            if col in row and pd.notna(row[col]) and row[col] > 0:
                return week
        return 0
    
    season_df['last_active_week'] = season_df.apply(get_last_active_week, axis=1)
    
    # For Withdrew without explicit elimination week, infer from last_active_week
    mask = season_df['is_withdrew'] & season_df['elim_week'].isna()
    season_df.loc[mask, 'elim_week'] = season_df.loc[mask, 'last_active_week']
    
    # Max week for this season
    max_week = int(season_df['last_active_week'].max())
    
    # Build weekly data
    weekly_data = {}
    for week in range(1, max_week + 1):
        col = f'J_week{week}'
        if col not in season_df.columns:
            continue
            
        active = season_df[(season_df[col] > 0) & pd.notna(season_df[col])].copy()
        if len(active) == 0:
            continue
        
        J_scores = active[col].values.astype(float)
        names = active['celebrity_name'].values.tolist()
        
        # Classify eliminated contestants
        eliminated = []  # Vote-determined eliminations
        withdrew = []    # Withdrew (not vote-determined)
        
        for idx, row in active.iterrows():
            if row['elim_week'] == week:
                if row['is_withdrew']:
                    withdrew.append(row['celebrity_name'])
                else:
                    eliminated.append(row['celebrity_name'])
        
        survivors = [n for n in names if n not in eliminated and n not in withdrew]
        
        # Determine week type
        if week == max_week:
            week_type = 'finals'
        elif len(eliminated) == 0:
            week_type = 'none' if len(withdrew) == 0 else 'withdrew_only'
        elif len(eliminated) == 1:
            week_type = 'normal'
        else:
            week_type = 'multi'
        
        weekly_data[week] = {
            'names': names,
            'J_scores': J_scores,
            'eliminated': eliminated,
            'withdrew': withdrew,
            'survivors': survivors,
            'n_active': len(names),
            'week_type': week_type
        }
    
    # Get final rankings (placement 1, 2, 3, ...)
    finalists = season_df[season_df['is_finalist'] == True].copy()
    finalists = finalists.sort_values('placement')
    final_ranking = finalists['celebrity_name'].values.tolist()
    final_placements = finalists['placement'].values.tolist()
    
    return weekly_data, final_ranking, max_week, final_placements

# ============================================================================
# REGIME A: PERCENT SEASONS (3-27) - CONVEX OPTIMIZATION
# ============================================================================

def estimate_votes_percent_season(weekly_data, final_ranking, max_week, final_placements,
                                   alpha=0.1, beta=0.5, tau=20.0, n_ensemble=20):
    """
    Estimate fan votes for Percent seasons using MAP estimation with:
    - Soft elimination constraints
    - Finals ranking constraints
    - Entropy + temporal smoothness priors
    - No constraint for no-elimination weeks
    """
    
    # Build variable index mapping
    var_info = []
    for week in sorted(weekly_data.keys()):
        data = weekly_data[week]
        for i, name in enumerate(data['names']):
            var_info.append({'week': week, 'name': name, 'local_idx': i})
    
    n_vars = len(var_info)
    
    def get_week_vars(v_flat, week):
        """Extract variables for a specific week."""
        indices = [i for i, info in enumerate(var_info) if info['week'] == week]
        return v_flat[indices], indices, [var_info[i]['name'] for i in indices]
    
    def objective(v_flat):
        """Negative log-posterior."""
        loss = 0.0
        prev_v = None
        prev_names = None
        
        for week in sorted(weekly_data.keys()):
            data = weekly_data[week]
            v_week_raw, indices, names = get_week_vars(v_flat, week)
            
            # Simplex projection via softmax
            v = softmax(v_week_raw)
            n = len(v)
            
            # Judge shares
            J = data['J_scores']
            j_share = J / J.sum()
            
            # Combined scores
            c = j_share + v
            
            # --- Likelihood based on week type ---
            if data['week_type'] == 'normal' or data['week_type'] == 'multi':
                # Soft elimination: eliminated should have lowest combined score
                for elim_name in data['eliminated']:
                    if elim_name in names:
                        elim_idx = names.index(elim_name)
                        # Softmax likelihood
                        log_probs = -tau * c
                        log_probs = log_probs - log_probs.max()
                        log_sum_exp = np.log(np.sum(np.exp(log_probs)))
                        loss -= (log_probs[elim_idx] - log_sum_exp)
            
            elif data['week_type'] == 'finals':
                # Ranking constraint using Plackett-Luce
                if final_ranking:
                    for k in range(len(final_ranking) - 1):
                        if final_ranking[k] in names and final_ranking[k+1] in names:
                            idx_better = names.index(final_ranking[k])
                            idx_worse = names.index(final_ranking[k+1])
                            # Better should have higher combined score
                            diff = c[idx_better] - c[idx_worse]
                            # Sigmoid penalty
                            loss += np.log(1 + np.exp(-tau * diff))
            
            # No constraint for 'none' or 'withdrew_only' weeks
            
            # --- Priors ---
            # Entropy regularization
            entropy = -np.sum(v * np.log(v + 1e-10))
            loss -= alpha * entropy
            
            # Temporal smoothness
            if prev_v is not None and prev_names is not None:
                for i, name in enumerate(names):
                    if name in prev_names:
                        j = prev_names.index(name)
                        loss += beta * abs(v[i] - prev_v[j])
            
            prev_v = v
            prev_names = names
        
        return loss
    
    # Initial guess
    v0 = np.zeros(n_vars)
    
    # Optimize
    try:
        res = minimize(objective, v0, method='L-BFGS-B', options={'maxiter': 2000})
        v_opt = res.x
    except:
        v_opt = v0
    
    # Extract point estimates
    point_estimates = {}
    for week in sorted(weekly_data.keys()):
        v_week_raw, indices, names = get_week_vars(v_opt, week)
        v = softmax(v_week_raw)
        for i, name in enumerate(names):
            point_estimates[(name, week)] = v[i]
    
    # Ensemble for uncertainty
    ensemble_estimates = {key: [] for key in point_estimates.keys()}
    
    for k in range(n_ensemble):
        # Perturb hyperparameters
        alpha_k = alpha * (1 + np.random.uniform(-0.3, 0.3))
        beta_k = beta * (1 + np.random.uniform(-0.3, 0.3))
        tau_k = tau * (1 + np.random.uniform(-0.2, 0.2))
        
        # Perturb initial point
        v0_k = v_opt + np.random.normal(0, 0.1, n_vars)
        
        # Perturb judge scores
        weekly_data_k = {}
        for week, data in weekly_data.items():
            data_k = data.copy()
            noise = np.random.normal(0, 0.5, len(data['J_scores']))
            data_k['J_scores'] = np.maximum(1, data['J_scores'] + noise)
            weekly_data_k[week] = data_k
        
        # Create perturbed objective
        def objective_k(v_flat):
            loss = 0.0
            prev_v = None
            prev_names = None
            
            for week in sorted(weekly_data_k.keys()):
                data = weekly_data_k[week]
                v_week_raw, indices, names = get_week_vars(v_flat, week)
                v = softmax(v_week_raw)
                J = data['J_scores']
                j_share = J / J.sum()
                c = j_share + v
                
                if data['week_type'] in ['normal', 'multi']:
                    for elim_name in data['eliminated']:
                        if elim_name in names:
                            elim_idx = names.index(elim_name)
                            log_probs = -tau_k * c
                            log_probs = log_probs - log_probs.max()
                            log_sum_exp = np.log(np.sum(np.exp(log_probs)))
                            loss -= (log_probs[elim_idx] - log_sum_exp)
                
                elif data['week_type'] == 'finals' and final_ranking:
                    for m in range(len(final_ranking) - 1):
                        if final_ranking[m] in names and final_ranking[m+1] in names:
                            idx_better = names.index(final_ranking[m])
                            idx_worse = names.index(final_ranking[m+1])
                            diff = c[idx_better] - c[idx_worse]
                            loss += np.log(1 + np.exp(-tau_k * diff))
                
                entropy = -np.sum(v * np.log(v + 1e-10))
                loss -= alpha_k * entropy
                
                if prev_v is not None and prev_names is not None:
                    for i, name in enumerate(names):
                        if name in prev_names:
                            j = prev_names.index(name)
                            loss += beta_k * abs(v[i] - prev_v[j])
                
                prev_v = v
                prev_names = names
            
            return loss
        
        try:
            res_k = minimize(objective_k, v0_k, method='L-BFGS-B', options={'maxiter': 500})
            v_k = res_k.x
        except:
            v_k = v_opt
        
        for week in sorted(weekly_data.keys()):
            v_week_raw, indices, names = get_week_vars(v_k, week)
            v = softmax(v_week_raw)
            for i, name in enumerate(names):
                ensemble_estimates[(name, week)].append(v[i])
    
    return point_estimates, ensemble_estimates, weekly_data

# ============================================================================
# REGIME B: RANK SEASONS (1-2) - CONSTRAINT-BASED RANK INFERENCE
# ============================================================================

def estimate_votes_rank_season(weekly_data, final_ranking, max_week, final_placements,
                                lambda_param=0.5, n_ensemble=20):
    """
    Estimate fan votes for Rank seasons with proper constraint handling:
    - Eliminated contestant must have WORST combined rank (highest c_i)
    - Fan ranks are permutations
    - Finals use ranking constraints
    - Temporal smoothness for stability
    """
    point_estimates = {}
    ensemble_estimates = {}
    
    prev_fan_ranks = None
    prev_names = None
    
    for week in sorted(weekly_data.keys()):
        data = weekly_data[week]
        n = data['n_active']
        names = data['names']
        J = data['J_scores']
        
        # Judge ranks: 1 = best (highest score), n = worst (lowest score)
        j_order = np.argsort(-J)  # descending order of scores
        j_ranks = np.zeros(n)
        for rank, idx in enumerate(j_order, 1):
            j_ranks[idx] = rank
        
        eliminated = data['eliminated']
        week_type = data['week_type']
        
        # Find fan ranks that satisfy constraints
        best_fan_ranks = None
        best_score = float('inf')
        
        if week_type == 'finals':
            # For finals, use ranking constraint
            # final_ranking[0] should have best (lowest) combined rank
            fan_ranks = np.zeros(n)
            for i, name in enumerate(names):
                if name in final_ranking:
                    # Better placement -> lower fan rank
                    placement_idx = final_ranking.index(name)
                    fan_ranks[i] = placement_idx + 1
                else:
                    fan_ranks[i] = len(final_ranking) + 1
            best_fan_ranks = fan_ranks
            
        elif week_type == 'normal' and len(eliminated) == 1:
            elim_name = eliminated[0]
            elim_idx = names.index(elim_name)
            
            # Constraint: c_elim = j_ranks[elim] + fan_ranks[elim] must be MAXIMUM
            # We need fan_ranks[elim] such that combined is worst
            
            # Try to find fan ranks that work
            # Strategy: Give eliminated the worst fan rank (n), then optimize others
            for elim_fan_rank in range(n, 0, -1):  # Try worst first
                fan_ranks = np.zeros(n)
                fan_ranks[elim_idx] = elim_fan_rank
                c_elim = j_ranks[elim_idx] + elim_fan_rank
                
                # Assign fan ranks to others such that c_i < c_elim
                others = [i for i in range(n) if i != elim_idx]
                used_ranks = {elim_fan_rank}
                
                valid = True
                for i in others:
                    # We need j_ranks[i] + fan_ranks[i] < c_elim
                    max_fan_rank = c_elim - j_ranks[i] - 0.001
                    
                    # Find available rank <= max_fan_rank
                    available = [r for r in range(1, n+1) if r not in used_ranks and r <= max_fan_rank]
                    if not available:
                        valid = False
                        break
                    
                    # Prefer rank close to judge rank (temporal smoothness)
                    if prev_fan_ranks is not None and names[i] in prev_names:
                        prev_idx = prev_names.index(names[i])
                        target = prev_fan_ranks[prev_idx]
                        available.sort(key=lambda r: abs(r - target))
                    
                    fan_ranks[i] = available[0]
                    used_ranks.add(available[0])
                
                if valid:
                    # Check constraint satisfied
                    c = j_ranks + fan_ranks
                    if np.argmax(c) == elim_idx:
                        # Compute temporal smoothness score
                        smooth_score = 0
                        if prev_fan_ranks is not None:
                            for i, name in enumerate(names):
                                if name in prev_names:
                                    prev_idx = prev_names.index(name)
                                    smooth_score += abs(fan_ranks[i] - prev_fan_ranks[prev_idx])
                        
                        if smooth_score < best_score:
                            best_score = smooth_score
                            best_fan_ranks = fan_ranks.copy()
                    break
            
            # Fallback if no valid assignment found
            if best_fan_ranks is None:
                # Give eliminated the worst fan rank
                fan_ranks = np.zeros(n)
                fan_ranks[elim_idx] = n
                remaining = [i for i in range(n) if i != elim_idx]
                remaining.sort(key=lambda i: -J[i])  # Better dancers get better fan ranks
                for rank, i in enumerate(remaining, 1):
                    fan_ranks[i] = rank
                best_fan_ranks = fan_ranks
                
        elif week_type == 'multi' and len(eliminated) > 1:
            # Multiple eliminations: all eliminated should be in bottom |E|
            fan_ranks = np.zeros(n)
            elim_indices = [names.index(e) for e in eliminated]
            non_elim_indices = [i for i in range(n) if i not in elim_indices]
            
            # Give eliminated contestants the worst fan ranks
            for rank, idx in enumerate(elim_indices):
                fan_ranks[idx] = n - len(elim_indices) + rank + 1
            
            # Others get better ranks
            non_elim_indices.sort(key=lambda i: -J[i])
            for rank, idx in enumerate(non_elim_indices, 1):
                fan_ranks[idx] = rank
            
            best_fan_ranks = fan_ranks
            
        else:
            # No elimination week: use judge-based ordering or previous week
            if prev_fan_ranks is not None and prev_names is not None:
                # Try to maintain previous ranks
                fan_ranks = np.zeros(n)
                used_ranks = set()
                for i, name in enumerate(names):
                    if name in prev_names:
                        prev_idx = prev_names.index(name)
                        target = prev_fan_ranks[prev_idx]
                        # Find closest available rank
                        for offset in range(n):
                            for r in [target + offset, target - offset]:
                                if 1 <= r <= n and r not in used_ranks:
                                    fan_ranks[i] = r
                                    used_ranks.add(r)
                                    break
                            if fan_ranks[i] > 0:
                                break
                # Fill remaining
                for i in range(n):
                    if fan_ranks[i] == 0:
                        for r in range(1, n+1):
                            if r not in used_ranks:
                                fan_ranks[i] = r
                                used_ranks.add(r)
                                break
                best_fan_ranks = fan_ranks
            else:
                # Use judge order
                j_order = np.argsort(-J)
                fan_ranks = np.zeros(n)
                for rank, idx in enumerate(j_order, 1):
                    fan_ranks[idx] = rank
                best_fan_ranks = fan_ranks
        
        # Convert fan ranks to vote shares
        v = np.exp(-lambda_param * (best_fan_ranks - 1))
        v = v / v.sum()
        
        for i, name in enumerate(names):
            point_estimates[(name, week)] = v[i]
            ensemble_estimates[(name, week)] = []
        
        prev_fan_ranks = best_fan_ranks
        prev_names = names
    
    # Ensemble by perturbing
    for k in range(n_ensemble):
        lambda_k = lambda_param * (1 + np.random.uniform(-0.4, 0.4))
        prev_fan_ranks_k = None
        prev_names_k = None
        
        for week in sorted(weekly_data.keys()):
            data = weekly_data[week]
            n = data['n_active']
            names = data['names']
            
            # Add noise to judge scores
            J = data['J_scores'] + np.random.normal(0, 1.0, n)
            J = np.maximum(1, J)
            
            j_order = np.argsort(-J)
            j_ranks = np.zeros(n)
            for rank, idx in enumerate(j_order, 1):
                j_ranks[idx] = rank
            
            eliminated = data['eliminated']
            week_type = data['week_type']
            
            # Similar logic as above but with perturbations
            if week_type == 'finals':
                fan_ranks = np.zeros(n)
                for i, name in enumerate(names):
                    if name in final_ranking:
                        placement_idx = final_ranking.index(name)
                        fan_ranks[i] = placement_idx + 1 + np.random.choice([-1, 0, 0, 1])
                        fan_ranks[i] = max(1, min(n, fan_ranks[i]))
                    else:
                        fan_ranks[i] = len(final_ranking) + 1
                        
            elif week_type == 'normal' and len(eliminated) == 1:
                elim_name = eliminated[0]
                elim_idx = names.index(elim_name)
                
                fan_ranks = np.zeros(n)
                fan_ranks[elim_idx] = n  # Worst rank
                remaining = [i for i in range(n) if i != elim_idx]
                np.random.shuffle(remaining)  # Add randomness
                remaining.sort(key=lambda i: -J[i] + np.random.normal(0, 2))
                for rank, i in enumerate(remaining, 1):
                    fan_ranks[i] = rank
                    
            elif week_type == 'multi':
                fan_ranks = np.zeros(n)
                elim_indices = [names.index(e) for e in eliminated if e in names]
                non_elim = [i for i in range(n) if i not in elim_indices]
                
                for rank, idx in enumerate(elim_indices):
                    fan_ranks[idx] = n - len(elim_indices) + rank + 1
                np.random.shuffle(non_elim)
                non_elim.sort(key=lambda i: -J[i] + np.random.normal(0, 2))
                for rank, idx in enumerate(non_elim, 1):
                    fan_ranks[idx] = rank
                    
            else:
                # No elimination week: use judge order with perturbation
                j_order = np.argsort(-J)
                fan_ranks = np.zeros(n)
                for rank, idx in enumerate(j_order, 1):
                    fan_ranks[idx] = rank
                fan_ranks = fan_ranks + np.random.uniform(-0.3, 0.3, size=n)
                fan_ranks = np.clip(fan_ranks, 1, n)
            
            v = np.exp(-lambda_k * (fan_ranks - 1))
            v = v / v.sum()
            
            for i, name in enumerate(names):
                if (name, week) in ensemble_estimates:
                    ensemble_estimates[(name, week)].append(v[i])
            
            prev_fan_ranks_k = fan_ranks
            prev_names_k = names
    
    return point_estimates, ensemble_estimates, weekly_data

# ============================================================================
# REGIME C: SEASONS 28-34 - RANK + BOTTOM2 + JUDGES SAVE
# ============================================================================

def estimate_votes_bottom2_season(weekly_data, final_ranking, max_week, final_placements,
                                   lambda_param=0.5, n_ensemble=20):
    """
    Estimate fan votes for Bottom2 + Judges Save seasons.
    Key constraint: eliminated must be in bottom 2 by combined ranks,
    but judges choose which of the bottom 2 to eliminate.
    """
    point_estimates = {}
    ensemble_estimates = {}
    
    prev_fan_ranks = None
    prev_names = None
    
    for week in sorted(weekly_data.keys()):
        data = weekly_data[week]
        n = data['n_active']
        names = data['names']
        J = data['J_scores']
        
        # Judge ranks
        j_order = np.argsort(-J)
        j_ranks = np.zeros(n)
        for rank, idx in enumerate(j_order, 1):
            j_ranks[idx] = rank
        
        eliminated = data['eliminated']
        week_type = data['week_type']
        
        if week_type == 'finals':
            # Finals: ranking constraint
            fan_ranks = np.zeros(n)
            for i, name in enumerate(names):
                if name in final_ranking:
                    placement_idx = final_ranking.index(name)
                    fan_ranks[i] = placement_idx + 1
                else:
                    fan_ranks[i] = len(final_ranking) + 1
                    
        elif week_type == 'normal' and len(eliminated) == 1:
            elim_name = eliminated[0]
            elim_idx = names.index(elim_name)
            
            # Key difference from Rank regime:
            # Eliminated only needs to be in BOTTOM 2, not necessarily the worst
            # Judges then choose from bottom 2
            
            # Strategy: Find fan ranks such that:
            # 1. Eliminated is in bottom 2 by combined rank
            # 2. Prefer judges would choose to eliminate (lower judge score)
            
            # First, find who might be the "saved" bottom-2 contestant
            # Judges prefer to eliminate the one with lower judge score
            # So the saved one should have higher judge score than eliminated
            
            possible_partners = []
            for i, name in enumerate(names):
                if i != elim_idx:
                    if J[i] >= J[elim_idx]:  # Could be saved by judges
                        possible_partners.append(i)
            
            # If no one has higher judge score, anyone could be partner
            if not possible_partners:
                possible_partners = [i for i in range(n) if i != elim_idx]
            
            best_fan_ranks = None
            best_score = float('inf')
            
            for partner_idx in possible_partners:
                # Try to construct fan ranks where elim and partner are in bottom 2
                fan_ranks = np.zeros(n)
                
                # Give bottom 2 the worst fan ranks
                fan_ranks[elim_idx] = n
                fan_ranks[partner_idx] = n - 1
                
                # Others get ranks 1 to n-2
                others = [i for i in range(n) if i != elim_idx and i != partner_idx]
                others.sort(key=lambda i: -J[i])
                for rank, i in enumerate(others, 1):
                    fan_ranks[i] = rank
                
                # Check if bottom 2 constraint is satisfied
                c = j_ranks + fan_ranks
                sorted_c_indices = np.argsort(-c)  # Descending (worst first)
                bottom_2_indices = set(sorted_c_indices[:2])
                
                if elim_idx in bottom_2_indices:
                    # Compute score (temporal smoothness + judge preference)
                    score = 0
                    if prev_fan_ranks is not None:
                        for i, name in enumerate(names):
                            if name in prev_names:
                                prev_i = prev_names.index(name)
                                score += abs(fan_ranks[i] - prev_fan_ranks[prev_i])
                    
                    # Prefer partner with higher judge score
                    if J[partner_idx] >= J[elim_idx]:
                        score -= 1  # Bonus
                    
                    if score < best_score:
                        best_score = score
                        best_fan_ranks = fan_ranks.copy()
            
            if best_fan_ranks is None:
                # Fallback: give eliminated worst fan rank
                fan_ranks = np.zeros(n)
                fan_ranks[elim_idx] = n
                others = [i for i in range(n) if i != elim_idx]
                others.sort(key=lambda i: -J[i])
                for rank, i in enumerate(others, 1):
                    fan_ranks[i] = rank
                best_fan_ranks = fan_ranks
            
            fan_ranks = best_fan_ranks
            
        elif week_type == 'multi':
            # Multiple eliminations
            fan_ranks = np.zeros(n)
            elim_indices = [names.index(e) for e in eliminated if e in names]
            non_elim = [i for i in range(n) if i not in elim_indices]
            
            for rank, idx in enumerate(elim_indices):
                fan_ranks[idx] = n - len(elim_indices) + rank + 1
            
            non_elim.sort(key=lambda i: -J[i])
            for rank, idx in enumerate(non_elim, 1):
                fan_ranks[idx] = rank
                
        else:
            # No elimination: maintain previous or use judge order
            if prev_fan_ranks is not None and prev_names is not None:
                fan_ranks = np.zeros(n)
                used = set()
                for i, name in enumerate(names):
                    if name in prev_names:
                        prev_i = prev_names.index(name)
                        target = prev_fan_ranks[prev_i]
                        for offset in range(n):
                            for r in [target + offset, target - offset]:
                                if 1 <= r <= n and r not in used:
                                    fan_ranks[i] = r
                                    used.add(r)
                                    break
                            if fan_ranks[i] > 0:
                                break
                for i in range(n):
                    if fan_ranks[i] == 0:
                        for r in range(1, n+1):
                            if r not in used:
                                fan_ranks[i] = r
                                used.add(r)
                                break
            else:
                j_order = np.argsort(-J)
                fan_ranks = np.zeros(n)
                for rank, idx in enumerate(j_order, 1):
                    fan_ranks[idx] = rank
        
        # Convert to vote shares
        v = np.exp(-lambda_param * (fan_ranks - 1))
        v = v / v.sum()
        
        for i, name in enumerate(names):
            point_estimates[(name, week)] = v[i]
            ensemble_estimates[(name, week)] = []
        
        prev_fan_ranks = fan_ranks
        prev_names = names
    
    # Ensemble
    for k in range(n_ensemble):
        lambda_k = lambda_param * (1 + np.random.uniform(-0.4, 0.4))
        prev_fan_ranks_k = None
        prev_names_k = None
        
        for week in sorted(weekly_data.keys()):
            data = weekly_data[week]
            n = data['n_active']
            names = data['names']
            J = data['J_scores'] + np.random.normal(0, 1.0, n)
            J = np.maximum(1, J)
            
            j_order = np.argsort(-J)
            j_ranks = np.zeros(n)
            for rank, idx in enumerate(j_order, 1):
                j_ranks[idx] = rank
            
            eliminated = data['eliminated']
            week_type = data['week_type']
            
            if week_type == 'finals':
                fan_ranks = np.zeros(n)
                for i, name in enumerate(names):
                    if name in final_ranking:
                        idx_f = final_ranking.index(name)
                        fan_ranks[i] = idx_f + 1 + np.random.choice([-1, 0, 0, 1])
                        fan_ranks[i] = max(1, min(n, fan_ranks[i]))
                    else:
                        fan_ranks[i] = len(final_ranking) + 1
                        
            elif week_type == 'normal' and eliminated:
                elim_name = eliminated[0]
                if elim_name in names:
                    elim_idx = names.index(elim_name)
                    fan_ranks = np.zeros(n)
                    
                    # Random partner from those with higher judge scores
                    possible = [i for i in range(n) if i != elim_idx and J[i] >= J[elim_idx] - 2]
                    if not possible:
                        possible = [i for i in range(n) if i != elim_idx]
                    
                    partner_idx = np.random.choice(possible) if possible else (elim_idx + 1) % n
                    
                    fan_ranks[elim_idx] = n
                    fan_ranks[partner_idx] = n - 1
                    
                    others = [i for i in range(n) if i != elim_idx and i != partner_idx]
                    np.random.shuffle(others)
                    others.sort(key=lambda i: -J[i] + np.random.normal(0, 2))
                    for rank, i in enumerate(others, 1):
                        fan_ranks[i] = rank
                else:
                    j_order = np.argsort(-J)
                    fan_ranks = np.zeros(n)
                    for rank, idx in enumerate(j_order, 1):
                        fan_ranks[idx] = rank
                        
            elif week_type == 'multi':
                fan_ranks = np.zeros(n)
                elim_indices = [names.index(e) for e in eliminated if e in names]
                non_elim = [i for i in range(n) if i not in elim_indices]
                
                for rank, idx in enumerate(elim_indices):
                    fan_ranks[idx] = n - len(elim_indices) + rank + 1
                np.random.shuffle(non_elim)
                non_elim.sort(key=lambda i: -J[i] + np.random.normal(0, 2))
                for rank, idx in enumerate(non_elim, 1):
                    fan_ranks[idx] = rank
                    
            else:
                # No elimination week: use judge order with perturbation
                j_order = np.argsort(-J)
                fan_ranks = np.zeros(n)
                for rank, idx in enumerate(j_order, 1):
                    fan_ranks[idx] = rank
                fan_ranks = fan_ranks + np.random.uniform(-0.3, 0.3, size=n)
                fan_ranks = np.clip(fan_ranks, 1, n)
            
            v = np.exp(-lambda_k * (fan_ranks - 1))
            v = v / v.sum()
            
            for i, name in enumerate(names):
                if (name, week) in ensemble_estimates:
                    ensemble_estimates[(name, week)].append(v[i])
            
            prev_fan_ranks_k = fan_ranks
            prev_names_k = names
    
    return point_estimates, ensemble_estimates, weekly_data

# ============================================================================
# CONSISTENCY AND CERTAINTY METRICS
# ============================================================================

def compute_consistency_metrics(point_estimates, weekly_data, regime='percent'):
    """Compute consistency metrics with proper handling of different regimes."""
    correct = 0
    total_elim_weeks = 0
    jaccard_scores = []
    margins = []
    
    for week, data in weekly_data.items():
        if not data['eliminated'] or data['week_type'] == 'finals':
            continue
        
        total_elim_weeks += 1
        names = data['names']
        n = len(names)
        J = data['J_scores']
        
        v = np.array([point_estimates.get((name, week), 1/n) for name in names])
        
        if regime == 'percent':
            j_share = J / J.sum()
            c = j_share + v
            pred_elim_idx = np.argmin(c)
            sorted_c = np.sort(c)
            margin = sorted_c[1] - sorted_c[0] if len(sorted_c) > 1 else 0
            
        elif regime == 'rank':
            j_order = np.argsort(-J)
            j_ranks = np.zeros(n)
            for rank, idx in enumerate(j_order, 1):
                j_ranks[idx] = rank
            
            v_order = np.argsort(-v)
            v_ranks = np.zeros(n)
            for rank, idx in enumerate(v_order, 1):
                v_ranks[idx] = rank
            
            c = j_ranks + v_ranks
            pred_elim_idx = np.argmax(c)  # Worst combined rank
            sorted_c = np.sort(c)
            margin = sorted_c[-1] - sorted_c[-2] if len(sorted_c) > 1 else 0
            
        else:  # bottom2
            j_order = np.argsort(-J)
            j_ranks = np.zeros(n)
            for rank, idx in enumerate(j_order, 1):
                j_ranks[idx] = rank
            
            v_order = np.argsort(-v)
            v_ranks = np.zeros(n)
            for rank, idx in enumerate(v_order, 1):
                v_ranks[idx] = rank
            
            c = j_ranks + v_ranks
            
            # Bottom 2 by combined rank
            bottom2_indices = np.argsort(-c)[:2]
            
            # Within bottom 2, judges choose (prefer lower judge score)
            if len(bottom2_indices) >= 2:
                if J[bottom2_indices[0]] <= J[bottom2_indices[1]]:
                    pred_elim_idx = bottom2_indices[0]
                else:
                    pred_elim_idx = bottom2_indices[1]
            else:
                pred_elim_idx = bottom2_indices[0] if len(bottom2_indices) > 0 else 0
            
            sorted_c = np.sort(c)
            margin = sorted_c[-1] - sorted_c[-2] if len(sorted_c) > 1 else 0
        
        pred_elim = names[pred_elim_idx]
        actual_elim = data['eliminated']
        
        if pred_elim in actual_elim:
            correct += 1
        
        pred_set = {pred_elim}
        actual_set = set(actual_elim)
        jaccard = len(pred_set & actual_set) / len(pred_set | actual_set) if actual_set else 0
        jaccard_scores.append(jaccard)
        margins.append(margin)
    
    accuracy = correct / total_elim_weeks if total_elim_weeks > 0 else 1.0
    avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 1.0
    avg_margin = np.mean(margins) if margins else 0
    
    return {
        'accuracy': accuracy,
        'avg_jaccard': avg_jaccard,
        'avg_margin': avg_margin,
        'n_elim_weeks': total_elim_weeks
    }

def compute_certainty_metrics(ensemble_estimates):
    """Compute certainty metrics."""
    certainty = {}
    
    for key, samples in ensemble_estimates.items():
        if len(samples) < 2:
            certainty[key] = {'mean': np.nan, 'std': np.nan, 'cv': np.nan, 
                             'ci_low': np.nan, 'ci_high': np.nan, 'ci_width': np.nan}
            continue
        
        samples = np.array(samples)
        mean = np.mean(samples)
        std = np.std(samples)
        cv = std / mean if mean > 0 else np.nan
        
        ci_low = np.percentile(samples, 2.5)
        ci_high = np.percentile(samples, 97.5)
        ci_width = ci_high - ci_low
        
        certainty[key] = {
            'mean': mean,
            'std': std,
            'cv': cv,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'ci_width': ci_width
        }
    
    return certainty

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("Fan Vote Estimation Model v2 - Dancing with the Stars")
    print("=" * 70)
    
    # Load data
    print("\n[1/4] Loading and preprocessing data...")
    df = load_and_preprocess_data('d:/2026-repo/data/2026_MCM_Problem_C_Data.csv')
    print(f"      Loaded {len(df)} contestants across {df['season'].nunique()} seasons")
    
    all_results = []
    season_metrics = []
    
    print("\n[2/4] Estimating fan votes for each season...")
    
    for season in range(1, 35):
        print(f"      Season {season:2d}...", end=" ")
        
        try:
            weekly_data, final_ranking, max_week, final_placements = get_season_data(df, season)
            
            if not weekly_data:
                print("No valid data")
                continue
            
            # Determine regime
            if season <= 2:
                regime = 'rank'
                point_est, ensemble_est, weekly_data = estimate_votes_rank_season(
                    weekly_data, final_ranking, max_week, final_placements)
            elif season <= 27:
                regime = 'percent'
                point_est, ensemble_est, weekly_data = estimate_votes_percent_season(
                    weekly_data, final_ranking, max_week, final_placements)
            else:
                regime = 'bottom2'
                point_est, ensemble_est, weekly_data = estimate_votes_bottom2_season(
                    weekly_data, final_ranking, max_week, final_placements)
            
            # Compute metrics
            consistency = compute_consistency_metrics(point_est, weekly_data, regime)
            certainty = compute_certainty_metrics(ensemble_est)
            
            print(f"Regime: {regime:7s} | Accuracy: {consistency['accuracy']:.2%} | "
                  f"Jaccard: {consistency['avg_jaccard']:.3f}")
            
            # Store season-level metrics
            cv_values = [c['cv'] for c in certainty.values() if not np.isnan(c.get('cv', np.nan))]
            ci_widths = [c['ci_width'] for c in certainty.values() if not np.isnan(c.get('ci_width', np.nan))]
            
            season_metrics.append({
                'season': season,
                'regime': regime,
                'accuracy': consistency['accuracy'],
                'avg_jaccard': consistency['avg_jaccard'],
                'avg_margin': consistency['avg_margin'],
                'n_elim_weeks': consistency['n_elim_weeks'],
                'avg_cv': np.mean(cv_values) if cv_values else np.nan,
                'avg_ci_width': np.mean(ci_widths) if ci_widths else np.nan
            })
            
            # Store individual results
            for (name, week), v in point_est.items():
                cert = certainty.get((name, week), {})
                all_results.append({
                    'season': season,
                    'week': week,
                    'celebrity_name': name,
                    'fan_vote_share': v,
                    'fan_vote_share_mean': cert.get('mean', v),
                    'fan_vote_share_std': cert.get('std', 0),
                    'cv': cert.get('cv', 0),
                    'ci_low': cert.get('ci_low', v),
                    'ci_high': cert.get('ci_high', v),
                    'ci_width': cert.get('ci_width', 0),
                    'judge_total': weekly_data[week]['J_scores'][
                        weekly_data[week]['names'].index(name)] if name in weekly_data[week]['names'] else np.nan,
                    'eliminated_this_week': name in weekly_data[week]['eliminated'],
                    'regime': regime
                })
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    print("\n[3/4] Saving results...")
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('d:/2026-repo/data/fan_vote_results.csv', index=False)
    print(f"      Saved {len(results_df)} rows to fan_vote_results.csv")
    
    metrics_df = pd.DataFrame(season_metrics)
    metrics_df.to_csv('d:/2026-repo/data/consistency_metrics.csv', index=False)
    print(f"      Saved {len(metrics_df)} seasons to consistency_metrics.csv")
    
    # Summary statistics
    print("\n[4/4] Summary Statistics")
    print("=" * 70)
    
    for regime in ['rank', 'percent', 'bottom2']:
        regime_df = metrics_df[metrics_df['regime'] == regime]
        if len(regime_df) > 0:
            print(f"\n{regime.upper()} Regime (Seasons {regime_df['season'].min()}-{regime_df['season'].max()}):")
            print(f"  Accuracy:     {regime_df['accuracy'].mean():.2%} (std: {regime_df['accuracy'].std():.2%})")
            print(f"  Avg Jaccard:  {regime_df['avg_jaccard'].mean():.3f}")
            print(f"  Avg CV:       {regime_df['avg_cv'].mean():.3f}")
            print(f"  Avg CI Width: {regime_df['avg_ci_width'].mean():.4f}")
    
    print("\n" + "=" * 70)
    print("Overall Accuracy: {:.2%}".format(metrics_df['accuracy'].mean()))
    print("=" * 70)

if __name__ == "__main__":
    main()
