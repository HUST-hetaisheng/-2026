"""
Fan Vote Estimation - Honest Version
=====================================

关键认识：
1. 这是一个逆问题，存在多个解（欠定）
2. 我们能保证的是"约束满足"，但这不等于"预测准确"
3. 真正的验证应该是：给定估计的投票，能否用独立方法验证其合理性

本版本诚实地报告：
- 约束满足率（我们构造的解是否满足规则）
- 解的唯一性/不确定性（同一约束下有多少种可能的解）
- 各种特殊情况的处理
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================
DATA_PATH = 'd:/2026-repo/data/2026_MCM_Problem_C_Data.csv'
OUTPUT_PATH = 'd:/2026-repo/data/fan_vote_results_v4.csv'
METRICS_PATH = 'd:/2026-repo/data/consistency_metrics_v4.csv'

N_ENSEMBLE = 30
LAMBDA_RANK = 0.5
np.random.seed(42)

# ============================================================================
# 数据加载
# ============================================================================
def load_data(filepath):
    df = pd.read_csv(filepath)
    
    for week in range(1, 12):
        judge_cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
        cols = [c for c in judge_cols if c in df.columns]
        if cols:
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df[f'J_week{week}'] = df[cols].sum(axis=1, skipna=True)
    
    return df


def parse_result(result):
    import re
    result = str(result)
    info = {'is_withdrew': 'Withdrew' in result, 'is_finalist': False, 
            'elim_week': None, 'placement': None}
    
    match = re.search(r'Eliminated Week (\d+)', result)
    if match:
        info['elim_week'] = int(match.group(1))
    
    match = re.search(r'(\d+)(st|nd|rd|th) Place', result)
    if match:
        info['placement'] = int(match.group(1))
        info['is_finalist'] = True
    
    return info


def get_season_info(df, season):
    season_df = df[df['season'] == season].copy()
    
    for idx, row in season_df.iterrows():
        info = parse_result(row['results'])
        season_df.loc[idx, 'elim_week'] = info['elim_week']
        season_df.loc[idx, 'is_withdrew'] = info['is_withdrew']
        season_df.loc[idx, 'is_finalist'] = info['is_finalist']
        season_df.loc[idx, 'placement'] = info['placement']
    
    valid_weeks = []
    for w in range(1, 12):
        col = f'J_week{w}'
        if col in season_df.columns and (season_df[col] > 0).any():
            valid_weeks.append(w)
    
    finalists = season_df[season_df['is_finalist'] == True].copy()
    finalists = finalists.sort_values('placement')
    final_ranking = finalists['celebrity_name'].tolist()
    
    # 统计特殊情况
    withdrew_count = season_df['is_withdrew'].sum()
    
    return season_df, valid_weeks, final_ranking, withdrew_count


def classify_week(season_df, week, valid_weeks, final_ranking):
    """分类每周的类型"""
    col = f'J_week{week}'
    active = season_df[season_df[col] > 0]
    
    eliminated = active[active['elim_week'] == week]['celebrity_name'].tolist()
    withdrew = active[active['is_withdrew'] == True]
    
    # 检查是否有 withdrew 在这周退出（该周有分数，下周没有）
    withdrew_this_week = []
    next_col = f'J_week{week+1}'
    for _, row in withdrew.iterrows():
        if next_col in season_df.columns:
            if row.get(next_col, 0) == 0 or pd.isna(row.get(next_col, 0)):
                withdrew_this_week.append(row['celebrity_name'])
        elif week == max(valid_weeks):
            pass  # 最后一周
    
    is_finals = (week == max(valid_weeks)) and len(final_ranking) > 0
    
    if is_finals:
        return 'finals', eliminated, withdrew_this_week
    elif len(eliminated) == 0 and len(withdrew_this_week) == 0:
        return 'no_elim', [], []
    elif len(eliminated) == 0 and len(withdrew_this_week) > 0:
        return 'withdrew_only', [], withdrew_this_week
    elif len(eliminated) == 1:
        return 'single_elim', eliminated, withdrew_this_week
    else:
        return 'multi_elim', eliminated, withdrew_this_week


# ============================================================================
# Percent 制度 (S3-27)
# ============================================================================
def solve_percent_season(season_df, valid_weeks, final_ranking):
    """
    Percent 制度的解析解
    
    约束：c_e = j_e + v_e < c_p = j_p + v_p 对所有非淘汰者 p
    
    这个约束很强，通常能唯一确定解的大致范围
    """
    results = {}
    fan_ranks = {}
    week_info = {}
    
    for week in valid_weeks:
        col = f'J_week{week}'
        active = season_df[season_df[col] > 0].copy()
        n = len(active)
        if n == 0:
            continue
        
        names = active['celebrity_name'].tolist()
        J = active[col].values.astype(float)
        j_share = J / J.sum() if J.sum() > 0 else np.ones(n) / n
        
        week_type, eliminated, withdrew = classify_week(season_df, week, valid_weeks, final_ranking)
        week_info[week] = {'type': week_type, 'n_elim': len(eliminated), 'n_withdrew': len(withdrew)}
        
        if week_type == 'finals':
            # 决赛：根据最终排名反推
            # 排名越好，combined score 越高
            v = np.zeros(n)
            for i, name in enumerate(names):
                if name in final_ranking:
                    place = final_ranking.index(name) + 1
                    # 第1名需要最高分
                    v[i] = 1.0 / place
                else:
                    v[i] = 0.01
            v = v / v.sum()
            
        elif week_type == 'no_elim':
            # 无淘汰周：无约束，使用均匀分布（高不确定性）
            v = np.ones(n) / n
            
        elif week_type == 'withdrew_only':
            # 只有退出，无投票淘汰：无约束
            v = np.ones(n) / n
            
        elif week_type == 'single_elim':
            # 单人淘汰：淘汰者 combined score 最低
            elim_idx = names.index(eliminated[0])
            
            # 策略：给淘汰者足够低的 vote share
            v = np.exp(j_share * 3)  # 基础：与 judge 正相关
            v = v / v.sum()
            
            # 找到需要满足的约束
            # v_e + j_e < v_p + j_p for all p
            # v_e < min_p(v_p + j_p - j_e)
            
            min_gap = float('inf')
            for i in range(n):
                if i != elim_idx:
                    gap = v[i] + j_share[i] - j_share[elim_idx]
                    min_gap = min(min_gap, gap)
            
            # 设置淘汰者的 vote share
            v[elim_idx] = max(0.001, min(min_gap - 0.001, v[elim_idx]))
            v = v / v.sum()
            
        elif week_type == 'multi_elim':
            # 多人淘汰：所有被淘汰者都在底部
            elim_indices = [names.index(e) for e in eliminated if e in names]
            
            v = np.exp(j_share * 3)
            v = v / v.sum()
            
            # 所有淘汰者的 combined score 都应该低于存活者
            non_elim = [i for i in range(n) if i not in elim_indices]
            if non_elim:
                min_survivor_c = min(j_share[i] + v[i] for i in non_elim)
                
                for e_idx in elim_indices:
                    max_ve = min_survivor_c - j_share[e_idx] - 0.001
                    v[e_idx] = max(0.001, max_ve)
            
            v = v / v.sum()
        
        for i, name in enumerate(names):
            results[(name, week)] = v[i]
            v_order = np.argsort(-v)
            rank = np.where(v_order == i)[0][0] + 1
            fan_ranks[(name, week)] = rank
    
    return results, fan_ranks, week_info


# ============================================================================
# Rank 制度 (S1-2)
# ============================================================================
def solve_rank_season(season_df, valid_weeks, final_ranking):
    """
    Rank 制度的解析解
    
    约束：c_e = j_rank_e + f_rank_e > c_p = j_rank_p + f_rank_p
    
    这个约束较弱，因为只涉及排名而非具体数值
    """
    results = {}
    fan_ranks = {}
    week_info = {}
    
    for week in valid_weeks:
        col = f'J_week{week}'
        active = season_df[season_df[col] > 0].copy()
        n = len(active)
        if n == 0:
            continue
        
        names = active['celebrity_name'].tolist()
        J = active[col].values.astype(float)
        
        # Judge ranks (1=best)
        j_order = np.argsort(-J)
        j_ranks = np.zeros(n)
        for r, idx in enumerate(j_order, 1):
            j_ranks[idx] = r
        
        week_type, eliminated, withdrew = classify_week(season_df, week, valid_weeks, final_ranking)
        week_info[week] = {'type': week_type, 'n_elim': len(eliminated), 'n_withdrew': len(withdrew)}
        
        if week_type == 'finals':
            f_ranks = np.zeros(n)
            for i, name in enumerate(names):
                if name in final_ranking:
                    f_ranks[i] = final_ranking.index(name) + 1
                else:
                    f_ranks[i] = len(final_ranking) + 1
                    
        elif week_type in ['no_elim', 'withdrew_only']:
            # 无约束，假设 fan rank 与 judge rank 相关
            f_ranks = j_ranks.copy()
            
        elif week_type == 'single_elim':
            elim_idx = names.index(eliminated[0])
            
            # 约束：淘汰者 combined rank 最大
            # 策略：给淘汰者最差的 fan rank
            f_ranks = np.zeros(n)
            
            # 非淘汰者按 judge rank 分配
            non_elim = [i for i in range(n) if i != elim_idx]
            non_elim_sorted = sorted(non_elim, key=lambda i: j_ranks[i])
            for rank, i in enumerate(non_elim_sorted, 1):
                f_ranks[i] = rank
            
            # 淘汰者得最差 rank
            f_ranks[elim_idx] = n
            
            # 验证约束
            c = j_ranks + f_ranks
            max_non_elim = max(c[i] for i in non_elim) if non_elim else 0
            
            if c[elim_idx] <= max_non_elim:
                # 约束不满足，需要进一步调整
                # 这意味着即使给最差 fan rank 也不够
                # 说明这个淘汰结果与 judge score 矛盾（争议性淘汰）
                needed = max_non_elim - j_ranks[elim_idx] + 1
                f_ranks[elim_idx] = max(n, needed)
                
        elif week_type == 'multi_elim':
            elim_indices = [names.index(e) for e in eliminated if e in names]
            non_elim = [i for i in range(n) if i not in elim_indices]
            
            f_ranks = np.zeros(n)
            
            # 非淘汰者
            non_elim_sorted = sorted(non_elim, key=lambda i: j_ranks[i])
            for rank, i in enumerate(non_elim_sorted, 1):
                f_ranks[i] = rank
            
            # 淘汰者（按 judge rank 排序，较差的得更差 fan rank）
            elim_sorted = sorted(elim_indices, key=lambda i: j_ranks[i])
            for rank, i in enumerate(elim_sorted):
                f_ranks[i] = len(non_elim) + rank + 1
        
        # 转换为 vote share
        v = np.exp(-LAMBDA_RANK * (f_ranks - 1))
        v = v / v.sum()
        
        for i, name in enumerate(names):
            results[(name, week)] = v[i]
            fan_ranks[(name, week)] = f_ranks[i]
    
    return results, fan_ranks, week_info


# ============================================================================
# Bottom2 制度 (S28-34)
# ============================================================================
def solve_bottom2_season(season_df, valid_weeks, final_ranking):
    """
    Bottom2 制度的解析解
    
    约束：
    1. 淘汰者在 combined rank 最差的两人中
    2. 评委从这两人中选择一人淘汰（决策过程未知）
    
    这个约束最弱，因为评委选择增加了不确定性
    """
    results = {}
    fan_ranks = {}
    week_info = {}
    
    for week in valid_weeks:
        col = f'J_week{week}'
        active = season_df[season_df[col] > 0].copy()
        n = len(active)
        if n == 0:
            continue
        
        names = active['celebrity_name'].tolist()
        J = active[col].values.astype(float)
        
        j_order = np.argsort(-J)
        j_ranks = np.zeros(n)
        for r, idx in enumerate(j_order, 1):
            j_ranks[idx] = r
        
        week_type, eliminated, withdrew = classify_week(season_df, week, valid_weeks, final_ranking)
        week_info[week] = {'type': week_type, 'n_elim': len(eliminated), 'n_withdrew': len(withdrew)}
        
        if week_type == 'finals':
            f_ranks = np.zeros(n)
            for i, name in enumerate(names):
                if name in final_ranking:
                    f_ranks[i] = final_ranking.index(name) + 1
                else:
                    f_ranks[i] = len(final_ranking) + 1
                    
        elif week_type in ['no_elim', 'withdrew_only']:
            f_ranks = j_ranks.copy()
            
        elif week_type in ['single_elim', 'multi_elim']:
            elim_indices = [names.index(e) for e in eliminated if e in names]
            non_elim = [i for i in range(n) if i not in elim_indices]
            
            f_ranks = np.zeros(n)
            
            # 策略：淘汰者在 bottom2 中
            # 需要找一个合适的 "partner"（被救的人）
            
            # 非淘汰者（除了 partner）
            # Partner 选择：judge rank 较差的非淘汰者
            non_elim_by_j = sorted(non_elim, key=lambda i: -j_ranks[i])  # 按 j_rank 降序
            
            if len(non_elim_by_j) > 0:
                partner_idx = non_elim_by_j[0]  # 最差的非淘汰者
                others = [i for i in non_elim if i != partner_idx]
            else:
                partner_idx = None
                others = non_elim
            
            # 分配 fan ranks
            others_sorted = sorted(others, key=lambda i: j_ranks[i])
            for rank, i in enumerate(others_sorted, 1):
                f_ranks[i] = rank
            
            if partner_idx is not None:
                f_ranks[partner_idx] = len(others) + 1
            
            for rank, i in enumerate(elim_indices):
                f_ranks[i] = len(others) + 2 + rank
            
            # 验证 bottom2 约束
            c = j_ranks + f_ranks
            sorted_by_c = sorted(range(n), key=lambda i: -c[i])
            bottom2 = set(sorted_by_c[:2])
            
            # 确保所有淘汰者在 bottom2 中
            elim_in_b2 = all(i in bottom2 for i in elim_indices)
            
            if not elim_in_b2:
                # 需要调整
                for i in elim_indices:
                    c_target = c[sorted_by_c[1]] if len(sorted_by_c) > 1 else c[sorted_by_c[0]]
                    needed_f = c_target - j_ranks[i] + 0.5
                    f_ranks[i] = max(f_ranks[i], needed_f)
        
        # 转换为 vote share
        v = np.exp(-LAMBDA_RANK * (f_ranks - 1))
        v = v / v.sum()
        
        for i, name in enumerate(names):
            results[(name, week)] = v[i]
            fan_ranks[(name, week)] = f_ranks[i]
    
    return results, fan_ranks, week_info


# ============================================================================
# 验证函数（诚实版）
# ============================================================================
def validate_percent(results, fan_ranks, season_df, valid_weeks):
    """验证 Percent 制度：淘汰者 combined score 是否最低"""
    correct = 0
    total = 0
    details = []
    
    for week in valid_weeks:
        col = f'J_week{week}'
        active = season_df[season_df[col] > 0].copy()
        eliminated = active[active['elim_week'] == week]['celebrity_name'].tolist()
        
        if not eliminated:
            continue
        
        total += 1
        names = active['celebrity_name'].tolist()
        n = len(names)
        J = active[col].values.astype(float)
        j_share = J / J.sum() if J.sum() > 0 else np.ones(n) / n
        
        v = np.array([results.get((name, week), 1/n) for name in names])
        c = j_share + v
        
        # 检查淘汰者是否有最低分
        elim_indices = [names.index(e) for e in eliminated if e in names]
        non_elim_indices = [i for i in range(n) if i not in elim_indices]
        
        if not elim_indices or not non_elim_indices:
            continue
        
        max_elim_c = max(c[i] for i in elim_indices)
        min_non_elim_c = min(c[i] for i in non_elim_indices)
        
        if max_elim_c < min_non_elim_c:
            correct += 1
            details.append({'week': week, 'ok': True, 'margin': min_non_elim_c - max_elim_c})
        else:
            details.append({'week': week, 'ok': False, 'margin': max_elim_c - min_non_elim_c})
    
    return correct / total if total > 0 else 1.0, details


def validate_rank(results, fan_ranks, season_df, valid_weeks):
    """验证 Rank 制度：淘汰者 combined rank 是否最大"""
    correct = 0
    total = 0
    details = []
    
    for week in valid_weeks:
        col = f'J_week{week}'
        active = season_df[season_df[col] > 0].copy()
        eliminated = active[active['elim_week'] == week]['celebrity_name'].tolist()
        
        if not eliminated:
            continue
        
        total += 1
        names = active['celebrity_name'].tolist()
        n = len(names)
        J = active[col].values.astype(float)
        
        j_order = np.argsort(-J)
        j_ranks = np.zeros(n)
        for r, idx in enumerate(j_order, 1):
            j_ranks[idx] = r
        
        f_ranks = np.array([fan_ranks.get((name, week), n/2) for name in names])
        c = j_ranks + f_ranks
        
        elim_indices = [names.index(e) for e in eliminated if e in names]
        non_elim_indices = [i for i in range(n) if i not in elim_indices]
        
        if not elim_indices or not non_elim_indices:
            continue
        
        min_elim_c = min(c[i] for i in elim_indices)
        max_non_elim_c = max(c[i] for i in non_elim_indices)
        
        if min_elim_c > max_non_elim_c:
            correct += 1
            details.append({'week': week, 'ok': True})
        else:
            details.append({'week': week, 'ok': False})
    
    return correct / total if total > 0 else 1.0, details


def validate_bottom2(results, fan_ranks, season_df, valid_weeks):
    """验证 Bottom2 制度：淘汰者是否在 bottom2 中"""
    in_bottom2 = 0
    total = 0
    details = []
    
    for week in valid_weeks:
        col = f'J_week{week}'
        active = season_df[season_df[col] > 0].copy()
        eliminated = active[active['elim_week'] == week]['celebrity_name'].tolist()
        
        if not eliminated:
            continue
        
        total += 1
        names = active['celebrity_name'].tolist()
        n = len(names)
        J = active[col].values.astype(float)
        
        j_order = np.argsort(-J)
        j_ranks = np.zeros(n)
        for r, idx in enumerate(j_order, 1):
            j_ranks[idx] = r
        
        f_ranks = np.array([fan_ranks.get((name, week), n/2) for name in names])
        c = j_ranks + f_ranks
        
        sorted_by_c = sorted(range(n), key=lambda i: -c[i])
        bottom2 = set(sorted_by_c[:2])
        bottom2_names = [names[i] for i in bottom2]
        
        elim_in_b2 = any(e in bottom2_names for e in eliminated)
        
        if elim_in_b2:
            in_bottom2 += 1
            details.append({'week': week, 'in_b2': True, 'bottom2': bottom2_names})
        else:
            details.append({'week': week, 'in_b2': False, 'bottom2': bottom2_names, 'elim': eliminated})
    
    return in_bottom2 / total if total > 0 else 1.0, details


# ============================================================================
# Ensemble（不确定性量化）
# ============================================================================
def ensemble_season(solve_func, season_df, valid_weeks, final_ranking, n_ensemble):
    all_results = []
    
    for k in range(n_ensemble):
        perturbed_df = season_df.copy()
        for week in valid_weeks:
            col = f'J_week{week}'
            if col in perturbed_df.columns:
                noise = np.random.normal(0, 0.5, len(perturbed_df))
                perturbed_df[col] = perturbed_df[col] + noise
                perturbed_df[col] = perturbed_df[col].clip(lower=1)
        
        results, _, _ = solve_func(perturbed_df, valid_weeks, final_ranking)
        all_results.append(results)
    
    ensemble_stats = {}
    all_keys = set()
    for r in all_results:
        all_keys.update(r.keys())
    
    for key in all_keys:
        samples = [r.get(key, np.nan) for r in all_results if key in r]
        
        if samples:
            mean = np.mean(samples)
            std = np.std(samples)
            cv = std / mean if mean > 0 else 0
            ci_low = np.percentile(samples, 2.5) if len(samples) > 1 else mean
            ci_high = np.percentile(samples, 97.5) if len(samples) > 1 else mean
        else:
            mean = std = cv = ci_low = ci_high = np.nan
        
        ensemble_stats[key] = {
            'mean': mean, 'std': std, 'cv': cv,
            'ci_low': ci_low, 'ci_high': ci_high, 'ci_width': ci_high - ci_low
        }
    
    return ensemble_stats


# ============================================================================
# 主函数
# ============================================================================
def main():
    print("=" * 70)
    print("Fan Vote Estimation - Honest Version")
    print("=" * 70)
    print("\n注意：这是一个逆问题。100% 约束满足率意味着我们构造了满足规则的解，")
    print("      但不意味着这就是真实的投票分布（因为存在多个可能的解）。")
    print("      CV（变异系数）反映了解的不确定性。\n")
    
    print("[1/5] Loading data...")
    df = load_data(DATA_PATH)
    print(f"      Loaded {len(df)} contestants, {df['season'].nunique()} seasons")
    
    all_rows = []
    all_metrics = []
    
    print("\n[2/5] Estimating fan votes...")
    print("-" * 70)
    print(f"{'Season':>8} | {'Regime':>8} | {'Constraint':>10} | {'CV':>6} | Week Types")
    print("-" * 70)
    
    for season in sorted(df['season'].unique()):
        season_df, valid_weeks, final_ranking, withdrew_count = get_season_info(df, season)
        
        if not valid_weeks:
            continue
        
        if season <= 2:
            regime = 'rank'
            solve_func = solve_rank_season
            validate_func = validate_rank
        elif season <= 27:
            regime = 'percent'
            solve_func = solve_percent_season
            validate_func = validate_percent
        else:
            regime = 'bottom2'
            solve_func = solve_bottom2_season
            validate_func = validate_bottom2
        
        # 求解
        results, fan_ranks, week_info = solve_func(season_df, valid_weeks, final_ranking)
        
        # 验证
        accuracy, details = validate_func(results, fan_ranks, season_df, valid_weeks)
        
        # Ensemble
        ensemble_stats = ensemble_season(solve_func, season_df, valid_weeks, final_ranking, N_ENSEMBLE)
        
        # 统计周类型
        week_types = {}
        for w, info in week_info.items():
            t = info['type']
            week_types[t] = week_types.get(t, 0) + 1
        week_types_str = ', '.join([f"{k}:{v}" for k, v in week_types.items()])
        
        # 收集结果
        for week in valid_weeks:
            col = f'J_week{week}'
            active = season_df[season_df[col] > 0].copy()
            eliminated = active[active['elim_week'] == week]['celebrity_name'].tolist()
            
            for _, row in active.iterrows():
                name = row['celebrity_name']
                key = (name, week)
                
                vote_share = results.get(key, np.nan)
                fan_rank = fan_ranks.get(key, np.nan)
                stats = ensemble_stats.get(key, {})
                winfo = week_info.get(week, {})
                
                all_rows.append({
                    'season': season,
                    'week': week,
                    'celebrity_name': name,
                    'fan_vote_share': vote_share,
                    'fan_rank': fan_rank,
                    'mean': stats.get('mean', vote_share),
                    'std': stats.get('std', 0),
                    'cv': stats.get('cv', 0),
                    'ci_low': stats.get('ci_low', vote_share),
                    'ci_high': stats.get('ci_high', vote_share),
                    'ci_width': stats.get('ci_width', 0),
                    'judge_total': row[col],
                    'eliminated': name in eliminated,
                    'week_type': winfo.get('type', 'unknown'),
                    'regime': regime
                })
        
        season_cv = np.nanmean([r['cv'] for r in all_rows if r['season'] == season])
        season_ci = np.nanmean([r['ci_width'] for r in all_rows if r['season'] == season])
        
        all_metrics.append({
            'season': season,
            'regime': regime,
            'constraint_satisfied': accuracy,
            'avg_cv': season_cv,
            'avg_ci_width': season_ci,
            'n_weeks': len(valid_weeks),
            'n_withdrew': withdrew_count
        })
        
        print(f"{season:>8} | {regime:>8} | {accuracy:>9.1%} | {season_cv:>5.3f} | {week_types_str}")
    
    print("-" * 70)
    
    print("\n[3/5] Saving results...")
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"      Saved {len(results_df)} rows to {OUTPUT_PATH}")
    
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(METRICS_PATH, index=False)
    print(f"      Saved {len(metrics_df)} seasons to {METRICS_PATH}")
    
    print("\n[4/5] Summary by Regime")
    print("=" * 70)
    
    for regime in ['rank', 'percent', 'bottom2']:
        regime_metrics = metrics_df[metrics_df['regime'] == regime]
        if len(regime_metrics) > 0:
            avg_acc = regime_metrics['constraint_satisfied'].mean()
            std_acc = regime_metrics['constraint_satisfied'].std()
            avg_cv = regime_metrics['avg_cv'].mean()
            n_seasons = len(regime_metrics)
            
            print(f"\n{regime.upper()} (n={n_seasons} seasons):")
            print(f"  Constraint Satisfaction: {avg_acc:.1%} (std: {std_acc:.1%})")
            print(f"  Average CV (uncertainty): {avg_cv:.3f}")
            
            if regime == 'percent':
                print("  → 约束最强：combined score 决定淘汰，解几乎唯一")
            elif regime == 'rank':
                print("  → 约束较强：combined rank 决定淘汰，但多种 fan rank 可产生相同结果")
            else:
                print("  → 约束最弱：只需在 bottom2 中，评委选择增加不确定性")
    
    print("\n[5/5] Special Cases Summary")
    print("=" * 70)
    
    # 统计各种特殊情况
    week_type_counts = results_df.groupby('week_type').size()
    print("\nWeek types across all seasons:")
    for wt, count in week_type_counts.items():
        print(f"  {wt}: {count} contestant-weeks")
    
    print("\n" + "=" * 70)
    print("完成！")
    print("\n重要提醒：")
    print("  - 'Constraint Satisfied' 是我们构造的解满足规则的比例")
    print("  - 这不等于'预测准确率'，因为真实投票是未知的")
    print("  - CV 越高说明该估计的不确定性越大")
    print("=" * 70)


if __name__ == "__main__":
    main()
