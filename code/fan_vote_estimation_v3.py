"""
Fan Vote Estimation Model for Dancing with the Stars (Version 3 - Simplified)
===============================================================================
Uses analytical/heuristic solutions for faster execution while maintaining accuracy.

Key improvements:
- Faster analytical solutions for Percent regime
- Proper constraint handling for all week types
- Correct Bottom2 logic with judges' save

Output:
- fan_vote_results.csv
- consistency_metrics.csv
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(filepath):
    """Load data and compute weekly judge totals."""
    df = pd.read_csv(filepath)
    
    for week in range(1, 12):
        judge_cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
        cols = [c for c in judge_cols if c in df.columns]
        if cols:
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df[f'J_week{week}'] = df[cols].sum(axis=1, skipna=True)
    
    return df

def get_season_data(df, season):
    """Extract season-specific data structures."""
    import re
    season_df = df[df['season'] == season].copy()
    
    def parse_result(result):
        result = str(result)
        info = {'is_withdrew': 'Withdrew' in result, 'is_finalist': False, 
                'elim_week': None, 'placement': None}
        
        match = re.search(r'Eliminated Week (\d+)', result)
        if match:
            info['elim_week'] = int(match.group(1))
        
        if 'Place' in result:
            info['is_finalist'] = True
            place_match = re.search(r'(\d+)', result)
            if place_match:
                info['placement'] = int(place_match.group(1))
        
        return info
    
    for idx, row in season_df.iterrows():
        parsed = parse_result(row['results'])
        for k, v in parsed.items():
            season_df.loc[idx, k] = v
    
    def get_last_active_week(row):
        for week in range(11, 0, -1):
            col = f'J_week{week}'
            if col in row and pd.notna(row[col]) and row[col] > 0:
                return week
        return 0
    
    season_df['last_active_week'] = season_df.apply(get_last_active_week, axis=1)
    
    mask = season_df['is_withdrew'] & season_df['elim_week'].isna()
    season_df.loc[mask, 'elim_week'] = season_df.loc[mask, 'last_active_week']
    
    max_week = int(season_df['last_active_week'].max())
    
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
        
        eliminated, withdrew = [], []
        for idx, row in active.iterrows():
            if row['elim_week'] == week:
                if row['is_withdrew']:
                    withdrew.append(row['celebrity_name'])
                else:
                    eliminated.append(row['celebrity_name'])
        
        survivors = [n for n in names if n not in eliminated and n not in withdrew]
        
        if week == max_week:
            week_type = 'finals'
        elif len(eliminated) == 0:
            week_type = 'none' if len(withdrew) == 0 else 'withdrew_only'
        elif len(eliminated) == 1:
            week_type = 'normal'
        else:
            week_type = 'multi'
        
        weekly_data[week] = {
            'names': names, 'J_scores': J_scores, 'eliminated': eliminated,
            'withdrew': withdrew, 'survivors': survivors, 'n_active': len(names),
            'week_type': week_type
        }
    
    finalists = season_df[season_df['is_finalist'] == True].sort_values('placement')
    final_ranking = finalists['celebrity_name'].values.tolist()
    
    return weekly_data, final_ranking, max_week

# ============================================================================
# REGIME A: PERCENT SEASONS (3-27) - ANALYTICAL SOLUTION
# ============================================================================

def estimate_votes_percent_season(weekly_data, final_ranking, max_week, n_ensemble=20):
    """
    Analytical solution for Percent regime:
    - combined score c_i = j_i + v_i
    - Eliminated has lowest c_i
    - We solve for v_i such that eliminated is lowest
    """
    point_estimates = {}
    ensemble_estimates = {}
    
    prev_v = None
    prev_names = None
    
    for week in sorted(weekly_data.keys()):
        data = weekly_data[week]
        n = data['n_active']
        names = data['names']
        J = data['J_scores']
        
        j_share = J / J.sum()
        eliminated = data['eliminated']
        week_type = data['week_type']
        
        if week_type == 'finals' and final_ranking:
            # Finals: winner should have highest combined score
            # Assign vote shares proportional to inverse placement
            v = np.zeros(n)
            for i, name in enumerate(names):
                if name in final_ranking:
                    place = final_ranking.index(name) + 1
                    v[i] = 1.0 / place
                else:
                    v[i] = 0.1 / n
            v = v / v.sum()
            
        elif week_type == 'normal' and len(eliminated) == 1:
            elim_name = eliminated[0]
            elim_idx = names.index(elim_name)
            
            # Eliminated must have lowest combined score
            # c_elim = j_elim + v_elim <= c_i = j_i + v_i for all i
            # This means: v_elim - v_i <= j_i - j_elim
            
            # Strategy: give eliminated the minimum vote share needed
            # Others get shares based on judge scores + temporal smoothness
            
            v = np.zeros(n)
            
            # Base vote shares from judge scores (positive correlation assumed)
            base_v = np.exp(j_share * 2)  # Higher judge -> higher fan vote
            base_v = base_v / base_v.sum()
            
            # Ensure eliminated has lowest combined score
            # Find min margin needed
            min_other_c = min(j_share[i] + base_v[i] for i in range(n) if i != elim_idx)
            
            # Eliminated vote share should make c_elim < min_other_c
            v_elim_max = min_other_c - j_share[elim_idx] - 0.001
            v[elim_idx] = max(0.01 / n, min(v_elim_max, base_v[elim_idx]))
            
            # Distribute remaining to others proportionally
            remaining = 1 - v[elim_idx]
            other_weights = [base_v[i] for i in range(n) if i != elim_idx]
            other_sum = sum(other_weights)
            
            j = 0
            for i in range(n):
                if i != elim_idx:
                    v[i] = remaining * other_weights[j] / other_sum
                    j += 1
            
            # Verify constraint
            c = j_share + v
            if np.argmin(c) != elim_idx:
                # Fallback: force eliminated to have minimum
                v = base_v.copy()
                v[elim_idx] = 0.001
                v = v / v.sum()
                
        elif week_type == 'multi' and len(eliminated) > 0:
            elim_indices = [names.index(e) for e in eliminated if e in names]
            
            v = np.exp(j_share * 2)
            for idx in elim_indices:
                v[idx] = 0.001
            v = v / v.sum()
            
        else:
            # No elimination: use judge-based with smoothness
            if prev_v is not None and prev_names is not None:
                v = np.zeros(n)
                for i, name in enumerate(names):
                    if name in prev_names:
                        v[i] = prev_v[prev_names.index(name)]
                    else:
                        v[i] = 1.0 / n
                v = v / v.sum()
            else:
                v = np.exp(j_share * 2)
                v = v / v.sum()
        
        for i, name in enumerate(names):
            point_estimates[(name, week)] = v[i]
            ensemble_estimates[(name, week)] = []
        
        prev_v = v
        prev_names = names
    
    # Ensemble with perturbations
    for k in range(n_ensemble):
        prev_v_k = None
        prev_names_k = None
        
        for week in sorted(weekly_data.keys()):
            data = weekly_data[week]
            n = data['n_active']
            names = data['names']
            J = data['J_scores'] + np.random.normal(0, 0.5, n)
            J = np.maximum(1, J)
            j_share = J / J.sum()
            
            eliminated = data['eliminated']
            week_type = data['week_type']
            
            if week_type == 'finals' and final_ranking:
                v = np.zeros(n)
                for i, name in enumerate(names):
                    if name in final_ranking:
                        place = final_ranking.index(name) + 1
                        v[i] = 1.0 / (place + np.random.uniform(-0.3, 0.3))
                    else:
                        v[i] = 0.1 / n
                v = np.maximum(0.001, v)
                v = v / v.sum()
                
            elif week_type == 'normal' and eliminated:
                elim_name = eliminated[0]
                if elim_name in names:
                    elim_idx = names.index(elim_name)
                    v = np.exp(j_share * (2 + np.random.uniform(-0.5, 0.5)))
                    v[elim_idx] = 0.001 + np.random.uniform(0, 0.01)
                    v = v / v.sum()
                else:
                    v = np.ones(n) / n
                    
            elif week_type == 'multi':
                elim_indices = [names.index(e) for e in eliminated if e in names]
                v = np.exp(j_share * 2)
                for idx in elim_indices:
                    v[idx] = 0.001
                v = v / v.sum()
                
            else:
                v = np.exp(j_share * 2) + np.random.uniform(-0.1, 0.1, n)
                v = np.maximum(0.001, v)
                v = v / v.sum()
            
            for i, name in enumerate(names):
                if (name, week) in ensemble_estimates:
                    ensemble_estimates[(name, week)].append(v[i])
            
            prev_v_k = v
            prev_names_k = names
    
    return point_estimates, ensemble_estimates, weekly_data

# ============================================================================
# REGIME B: RANK SEASONS (1-2)
# ============================================================================

def estimate_votes_rank_season(weekly_data, final_ranking, max_week, lambda_param=0.5, n_ensemble=20):
    """Rank-based fan vote estimation with proper constraint handling."""
    point_estimates = {}
    ensemble_estimates = {}
    
    for week in sorted(weekly_data.keys()):
        data = weekly_data[week]
        n = data['n_active']
        names = data['names']
        J = data['J_scores']
        
        # Judge ranks (1=best, n=worst)
        j_order = np.argsort(-J)
        j_ranks = np.zeros(n)
        for rank, idx in enumerate(j_order, 1):
            j_ranks[idx] = rank
        
        eliminated = data['eliminated']
        week_type = data['week_type']
        
        if week_type == 'finals' and final_ranking:
            fan_ranks = np.zeros(n)
            for i, name in enumerate(names):
                if name in final_ranking:
                    fan_ranks[i] = final_ranking.index(name) + 1
                else:
                    fan_ranks[i] = len(final_ranking) + 1
                    
        elif week_type == 'normal' and len(eliminated) == 1:
            elim_name = eliminated[0]
            elim_idx = names.index(elim_name)
            
            # Eliminated must have worst combined rank (highest value)
            fan_ranks = np.zeros(n)
            fan_ranks[elim_idx] = n  # Worst fan rank
            
            remaining = [i for i in range(n) if i != elim_idx]
            remaining.sort(key=lambda i: -J[i])  # Better judge -> better fan rank
            for rank, i in enumerate(remaining, 1):
                fan_ranks[i] = rank
                
        elif week_type == 'multi':
            elim_indices = [names.index(e) for e in eliminated if e in names]
            non_elim = [i for i in range(n) if i not in elim_indices]
            
            fan_ranks = np.zeros(n)
            for rank, idx in enumerate(elim_indices):
                fan_ranks[idx] = n - len(elim_indices) + rank + 1
            non_elim.sort(key=lambda i: -J[i])
            for rank, idx in enumerate(non_elim, 1):
                fan_ranks[idx] = rank
                
        else:
            # No elimination: use judge order
            fan_ranks = j_ranks.copy()
        
        # Convert to vote shares
        v = np.exp(-lambda_param * (fan_ranks - 1))
        v = v / v.sum()
        
        for i, name in enumerate(names):
            point_estimates[(name, week)] = v[i]
            ensemble_estimates[(name, week)] = []
    
    # Ensemble
    for k in range(n_ensemble):
        lambda_k = lambda_param * (1 + np.random.uniform(-0.3, 0.3))
        
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
            
            if week_type == 'finals' and final_ranking:
                fan_ranks = np.zeros(n)
                for i, name in enumerate(names):
                    if name in final_ranking:
                        fan_ranks[i] = final_ranking.index(name) + 1
                    else:
                        fan_ranks[i] = len(final_ranking) + 1
                        
            elif week_type == 'normal' and eliminated:
                elim_idx = names.index(eliminated[0]) if eliminated[0] in names else 0
                fan_ranks = np.zeros(n)
                fan_ranks[elim_idx] = n
                remaining = [i for i in range(n) if i != elim_idx]
                np.random.shuffle(remaining)
                remaining.sort(key=lambda i: -J[i] + np.random.normal(0, 2))
                for rank, i in enumerate(remaining, 1):
                    fan_ranks[i] = rank
                    
            elif week_type == 'multi':
                elim_indices = [names.index(e) for e in eliminated if e in names]
                non_elim = [i for i in range(n) if i not in elim_indices]
                fan_ranks = np.zeros(n)
                for rank, idx in enumerate(elim_indices):
                    fan_ranks[idx] = n - len(elim_indices) + rank + 1
                np.random.shuffle(non_elim)
                for rank, idx in enumerate(non_elim, 1):
                    fan_ranks[idx] = rank
                    
            else:
                fan_ranks = j_ranks + np.random.uniform(-0.3, 0.3, n)
                fan_ranks = np.clip(fan_ranks, 1, n)
            
            v = np.exp(-lambda_k * (fan_ranks - 1))
            v = v / v.sum()
            
            for i, name in enumerate(names):
                if (name, week) in ensemble_estimates:
                    ensemble_estimates[(name, week)].append(v[i])
    
    return point_estimates, ensemble_estimates, weekly_data

# ============================================================================
# REGIME C: SEASONS 28-34 - RANK + BOTTOM2 + JUDGES SAVE
# ============================================================================

def estimate_votes_bottom2_season(weekly_data, final_ranking, max_week, lambda_param=0.5, n_ensemble=20):
    """
    Bottom2 + Judges Save: 
    - Bottom 2 by combined rank
    - Judges choose to eliminate (typically lower judge score)
    """
    point_estimates = {}
    ensemble_estimates = {}
    
    for week in sorted(weekly_data.keys()):
        data = weekly_data[week]
        n = data['n_active']
        names = data['names']
        J = data['J_scores']
        
        j_order = np.argsort(-J)
        j_ranks = np.zeros(n)
        for rank, idx in enumerate(j_order, 1):
            j_ranks[idx] = rank
        
        eliminated = data['eliminated']
        week_type = data['week_type']
        
        if week_type == 'finals' and final_ranking:
            fan_ranks = np.zeros(n)
            for i, name in enumerate(names):
                if name in final_ranking:
                    fan_ranks[i] = final_ranking.index(name) + 1
                else:
                    fan_ranks[i] = len(final_ranking) + 1
                    
        elif week_type == 'normal' and len(eliminated) == 1:
            elim_name = eliminated[0]
            elim_idx = names.index(elim_name)
            
            # Key: eliminated must be in bottom 2 (not necessarily worst)
            # Find a plausible bottom-2 partner
            # Partner typically has higher judge score (saved by judges)
            
            fan_ranks = np.zeros(n)
            
            # Give eliminated worst or second-worst fan rank
            fan_ranks[elim_idx] = n
            
            # Find partner: someone with similar or higher judge score
            candidates = [(i, J[i]) for i in range(n) if i != elim_idx]
            candidates.sort(key=lambda x: -x[1])  # Sort by judge score descending
            
            # Partner is someone judges might save (higher judge score than eliminated)
            partner_candidates = [i for i, js in candidates if js >= J[elim_idx]]
            
            if partner_candidates:
                partner_idx = partner_candidates[-1]  # Lowest among higher scorers
            else:
                partner_idx = candidates[-1][0] if candidates else (elim_idx + 1) % n
            
            fan_ranks[partner_idx] = n - 1
            
            # Others get remaining ranks
            others = [i for i in range(n) if i != elim_idx and i != partner_idx]
            others.sort(key=lambda i: -J[i])
            for rank, i in enumerate(others, 1):
                fan_ranks[i] = rank
                
        elif week_type == 'multi':
            elim_indices = [names.index(e) for e in eliminated if e in names]
            non_elim = [i for i in range(n) if i not in elim_indices]
            
            fan_ranks = np.zeros(n)
            for rank, idx in enumerate(elim_indices):
                fan_ranks[idx] = n - len(elim_indices) + rank + 1
            non_elim.sort(key=lambda i: -J[i])
            for rank, idx in enumerate(non_elim, 1):
                fan_ranks[idx] = rank
                
        else:
            fan_ranks = j_ranks.copy()
        
        v = np.exp(-lambda_param * (fan_ranks - 1))
        v = v / v.sum()
        
        for i, name in enumerate(names):
            point_estimates[(name, week)] = v[i]
            ensemble_estimates[(name, week)] = []
    
    # Ensemble
    for k in range(n_ensemble):
        lambda_k = lambda_param * (1 + np.random.uniform(-0.3, 0.3))
        
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
            
            if week_type == 'finals' and final_ranking:
                fan_ranks = np.zeros(n)
                for i, name in enumerate(names):
                    if name in final_ranking:
                        fan_ranks[i] = final_ranking.index(name) + 1
                    else:
                        fan_ranks[i] = len(final_ranking) + 1
                        
            elif week_type == 'normal' and eliminated:
                elim_idx = names.index(eliminated[0]) if eliminated[0] in names else 0
                fan_ranks = np.zeros(n)
                fan_ranks[elim_idx] = n
                
                # Random partner
                candidates = [i for i in range(n) if i != elim_idx and J[i] >= J[elim_idx] - 3]
                if not candidates:
                    candidates = [i for i in range(n) if i != elim_idx]
                partner_idx = np.random.choice(candidates) if candidates else (elim_idx + 1) % n
                fan_ranks[partner_idx] = n - 1
                
                others = [i for i in range(n) if i != elim_idx and i != partner_idx]
                np.random.shuffle(others)
                for rank, i in enumerate(others, 1):
                    fan_ranks[i] = rank
                    
            elif week_type == 'multi':
                elim_indices = [names.index(e) for e in eliminated if e in names]
                non_elim = [i for i in range(n) if i not in elim_indices]
                fan_ranks = np.zeros(n)
                for rank, idx in enumerate(elim_indices):
                    fan_ranks[idx] = n - len(elim_indices) + rank + 1
                np.random.shuffle(non_elim)
                for rank, idx in enumerate(non_elim, 1):
                    fan_ranks[idx] = rank
                    
            else:
                fan_ranks = j_ranks + np.random.uniform(-0.3, 0.3, n)
                fan_ranks = np.clip(fan_ranks, 1, n)
            
            v = np.exp(-lambda_k * (fan_ranks - 1))
            v = v / v.sum()
            
            for i, name in enumerate(names):
                if (name, week) in ensemble_estimates:
                    ensemble_estimates[(name, week)].append(v[i])
    
    return point_estimates, ensemble_estimates, weekly_data

# ============================================================================
# METRICS
# ============================================================================

def compute_consistency_metrics(point_estimates, weekly_data, regime='percent'):
    """Compute accuracy, Jaccard, margin."""
    correct = 0
    total = 0
    jaccard_scores = []
    margins = []
    
    for week, data in weekly_data.items():
        if not data['eliminated'] or data['week_type'] == 'finals':
            continue
        
        total += 1
        names = data['names']
        n = len(names)
        J = data['J_scores']
        
        v = np.array([point_estimates.get((name, week), 1/n) for name in names])
        
        if regime == 'percent':
            j_share = J / J.sum()
            c = j_share + v
            pred_idx = np.argmin(c)
            margin = np.sort(c)[1] - np.sort(c)[0] if n > 1 else 0
            
        elif regime == 'rank':
            j_order = np.argsort(-J)
            j_ranks = np.zeros(n)
            for r, idx in enumerate(j_order, 1):
                j_ranks[idx] = r
            
            v_order = np.argsort(-v)
            v_ranks = np.zeros(n)
            for r, idx in enumerate(v_order, 1):
                v_ranks[idx] = r
            
            c = j_ranks + v_ranks
            pred_idx = np.argmax(c)
            margin = np.sort(c)[-1] - np.sort(c)[-2] if n > 1 else 0
            
        else:  # bottom2
            j_order = np.argsort(-J)
            j_ranks = np.zeros(n)
            for r, idx in enumerate(j_order, 1):
                j_ranks[idx] = r
            
            v_order = np.argsort(-v)
            v_ranks = np.zeros(n)
            for r, idx in enumerate(v_order, 1):
                v_ranks[idx] = r
            
            c = j_ranks + v_ranks
            bottom2 = np.argsort(-c)[:2]
            
            # Judges choose from bottom 2 (lower judge score eliminated)
            if len(bottom2) >= 2:
                pred_idx = bottom2[0] if J[bottom2[0]] <= J[bottom2[1]] else bottom2[1]
            else:
                pred_idx = bottom2[0] if len(bottom2) > 0 else 0
            
            margin = np.sort(c)[-1] - np.sort(c)[-2] if n > 1 else 0
        
        pred = names[pred_idx]
        actual = set(data['eliminated'])
        
        if pred in actual:
            correct += 1
        
        jaccard = 1.0 if pred in actual else 0.0
        jaccard_scores.append(jaccard)
        margins.append(margin)
    
    return {
        'accuracy': correct / total if total > 0 else 1.0,
        'avg_jaccard': np.mean(jaccard_scores) if jaccard_scores else 1.0,
        'avg_margin': np.mean(margins) if margins else 0,
        'n_elim_weeks': total
    }

def compute_certainty_metrics(ensemble_estimates):
    """Compute CV, CI width."""
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
        
        certainty[key] = {
            'mean': mean, 'std': std, 'cv': cv,
            'ci_low': ci_low, 'ci_high': ci_high, 'ci_width': ci_high - ci_low
        }
    
    return certainty

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Fan Vote Estimation Model v3 - Dancing with the Stars")
    print("=" * 70)
    
    print("\n[1/4] Loading data...")
    df = load_and_preprocess_data('d:/2026-repo/data/2026_MCM_Problem_C_Data.csv')
    print(f"      Loaded {len(df)} contestants across {df['season'].nunique()} seasons")
    
    all_results = []
    season_metrics = []
    
    print("\n[2/4] Estimating fan votes...")
    
    for season in range(1, 35):
        try:
            weekly_data, final_ranking, max_week = get_season_data(df, season)
            
            if not weekly_data:
                print(f"      Season {season:2d}: No data")
                continue
            
            if season <= 2:
                regime = 'rank'
                point_est, ensemble_est, weekly_data = estimate_votes_rank_season(
                    weekly_data, final_ranking, max_week)
            elif season <= 27:
                regime = 'percent'
                point_est, ensemble_est, weekly_data = estimate_votes_percent_season(
                    weekly_data, final_ranking, max_week)
            else:
                regime = 'bottom2'
                point_est, ensemble_est, weekly_data = estimate_votes_bottom2_season(
                    weekly_data, final_ranking, max_week)
            
            consistency = compute_consistency_metrics(point_est, weekly_data, regime)
            certainty = compute_certainty_metrics(ensemble_est)
            
            print(f"      Season {season:2d}: {regime:7s} | Acc: {consistency['accuracy']:6.1%} | "
                  f"Jac: {consistency['avg_jaccard']:.3f}")
            
            cv_values = [c['cv'] for c in certainty.values() if not np.isnan(c.get('cv', np.nan))]
            ci_widths = [c['ci_width'] for c in certainty.values() if not np.isnan(c.get('ci_width', np.nan))]
            
            season_metrics.append({
                'season': season, 'regime': regime,
                'accuracy': consistency['accuracy'],
                'avg_jaccard': consistency['avg_jaccard'],
                'avg_margin': consistency['avg_margin'],
                'n_elim_weeks': consistency['n_elim_weeks'],
                'avg_cv': np.mean(cv_values) if cv_values else np.nan,
                'avg_ci_width': np.mean(ci_widths) if ci_widths else np.nan
            })
            
            for (name, week), v in point_est.items():
                cert = certainty.get((name, week), {})
                all_results.append({
                    'season': season, 'week': week, 'celebrity_name': name,
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
            print(f"      Season {season:2d}: Error - {e}")
            continue
    
    print("\n[3/4] Saving results...")
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('d:/2026-repo/data/fan_vote_results.csv', index=False)
    print(f"      Saved {len(results_df)} rows to fan_vote_results.csv")
    
    metrics_df = pd.DataFrame(season_metrics)
    metrics_df.to_csv('d:/2026-repo/data/consistency_metrics.csv', index=False)
    print(f"      Saved {len(metrics_df)} seasons to consistency_metrics.csv")
    
    print("\n[4/4] Summary")
    print("=" * 70)
    
    for regime in ['rank', 'percent', 'bottom2']:
        regime_df = metrics_df[metrics_df['regime'] == regime]
        if len(regime_df) > 0:
            print(f"\n{regime.upper()} (S{regime_df['season'].min()}-{regime_df['season'].max()}):")
            print(f"  Accuracy:  {regime_df['accuracy'].mean():.1%} (std: {regime_df['accuracy'].std():.1%})")
            print(f"  Jaccard:   {regime_df['avg_jaccard'].mean():.3f}")
            print(f"  Avg CV:    {regime_df['avg_cv'].mean():.3f}")
    
    print(f"\nOverall Accuracy: {metrics_df['accuracy'].mean():.1%}")
    print("=" * 70)

if __name__ == "__main__":
    main()
