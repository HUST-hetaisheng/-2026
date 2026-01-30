"""
Fan Vote Estimation v4 - Fixed Constraints
===========================================
修复 Rank 和 Bottom2 制度的约束验证逻辑
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
    """加载并预处理数据"""
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
    """解析 results 字段"""
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
    """获取赛季信息"""
    season_df = df[df['season'] == season].copy()
    
    # 解析结果
    for idx, row in season_df.iterrows():
        info = parse_result(row['results'])
        season_df.loc[idx, 'elim_week'] = info['elim_week']
        season_df.loc[idx, 'is_withdrew'] = info['is_withdrew']
        season_df.loc[idx, 'is_finalist'] = info['is_finalist']
        season_df.loc[idx, 'placement'] = info['placement']
    
    # 找有效周
    valid_weeks = []
    for w in range(1, 12):
        col = f'J_week{w}'
        if col in season_df.columns and (season_df[col] > 0).any():
            valid_weeks.append(w)
    
    # 决赛排名
    finalists = season_df[season_df['is_finalist'] == True].copy()
    finalists = finalists.sort_values('placement')
    final_ranking = finalists['celebrity_name'].tolist()
    
    return season_df, valid_weeks, final_ranking


# ============================================================================
# Percent 制度 (S3-27) - 解析解
# ============================================================================
def solve_percent_season(season_df, valid_weeks, final_ranking):
    """Percent 制度：最低 combined share 淘汰"""
    results = {}  # (name, week) -> vote_share
    fan_ranks = {}  # (name, week) -> fan_rank (用于输出)
    
    for week in valid_weeks:
        col = f'J_week{week}'
        active = season_df[season_df[col] > 0].copy()
        n = len(active)
        if n == 0:
            continue
        
        names = active['celebrity_name'].tolist()
        J = active[col].values.astype(float)
        j_share = J / J.sum() if J.sum() > 0 else np.ones(n) / n
        
        # 淘汰者
        eliminated = active[active['elim_week'] == week]['celebrity_name'].tolist()
        is_finals = (week == max(valid_weeks)) and final_ranking
        
        # 求解 vote share
        if is_finals:
            # 决赛：按排名分配
            v = np.zeros(n)
            for i, name in enumerate(names):
                if name in final_ranking:
                    rank = final_ranking.index(name) + 1
                    v[i] = 1.0 / rank
                else:
                    v[i] = 0.01
            v = v / v.sum()
            
        elif eliminated:
            # 正常淘汰：淘汰者 combined share 最低
            v = np.zeros(n)
            
            # 基础分配：按 judge share 正相关
            base = np.exp(j_share * 3)
            base = base / base.sum()
            
            for i, name in enumerate(names):
                if name in eliminated:
                    # 淘汰者需要最低 combined score
                    # c_e = j_e + v_e < c_p = j_p + v_p 对所有 p
                    # 找到非淘汰者中最低的 combined (假设他们用 base)
                    min_other = float('inf')
                    for j, n2 in enumerate(names):
                        if n2 not in eliminated:
                            min_other = min(min_other, j_share[j] + base[j])
                    
                    # v_e < min_other - j_e
                    max_ve = min_other - j_share[i] - 0.001
                    v[i] = max(0.001, min(max_ve, base[i] * 0.1))
                else:
                    v[i] = base[i]
            
            v = v / v.sum()
            
            # 验证约束
            c = j_share + v
            elim_indices = [i for i, name in enumerate(names) if name in eliminated]
            non_elim_indices = [i for i in range(n) if i not in elim_indices]
            
            # 检查所有淘汰者是否比所有非淘汰者分数低
            constraint_ok = True
            for e_idx in elim_indices:
                for ne_idx in non_elim_indices:
                    if c[e_idx] >= c[ne_idx]:
                        constraint_ok = False
                        break
            
            if not constraint_ok:
                # 强制调整
                for e_idx in elim_indices:
                    v[e_idx] = 0.001
                v = v / v.sum()
        else:
            # 无淘汰周
            v = np.ones(n) / n
        
        # 保存结果
        for i, name in enumerate(names):
            results[(name, week)] = v[i]
            # 计算 fan rank
            v_order = np.argsort(-v)
            rank = np.where(v_order == i)[0][0] + 1
            fan_ranks[(name, week)] = rank
    
    return results, fan_ranks


# ============================================================================
# Rank 制度 (S1-2) - 解析解
# ============================================================================
def solve_rank_season(season_df, valid_weeks, final_ranking):
    """Rank 制度：最大 combined rank 淘汰"""
    results = {}  # (name, week) -> vote_share
    fan_ranks = {}  # (name, week) -> fan_rank
    
    for week in valid_weeks:
        col = f'J_week{week}'
        active = season_df[season_df[col] > 0].copy()
        n = len(active)
        if n == 0:
            continue
        
        names = active['celebrity_name'].tolist()
        J = active[col].values.astype(float)
        
        # Judge ranks (1=best=highest score)
        j_order = np.argsort(-J)
        j_ranks = np.zeros(n)
        for r, idx in enumerate(j_order, 1):
            j_ranks[idx] = r
        
        eliminated = active[active['elim_week'] == week]['celebrity_name'].tolist()
        is_finals = (week == max(valid_weeks)) and final_ranking
        
        # 求解 fan ranks
        if is_finals:
            f_ranks = np.zeros(n)
            for i, name in enumerate(names):
                if name in final_ranking:
                    # 最终排名越好，combined rank 越小
                    final_place = final_ranking.index(name) + 1
                    # 需要调整 fan rank 使得 c = j_rank + f_rank 符合最终排名
                    # 直接给第1名最好的 fan rank
                    f_ranks[i] = final_place
                else:
                    f_ranks[i] = len(final_ranking) + 1
        
        elif eliminated:
            # 淘汰者需要最大 combined rank
            f_ranks = np.zeros(n)
            elim_indices = [i for i, name in enumerate(names) if name in eliminated]
            non_elim_indices = [i for i in range(n) if i not in elim_indices]
            
            # 计算：淘汰者需要多大的 fan rank 才能使 combined rank 最大
            # c_e = j_e + f_e >= c_p = j_p + f_p 对所有 p
            
            # 策略：
            # 1. 非淘汰者按 judge rank 分配 fan rank（正相关假设）
            non_elim_sorted = sorted(non_elim_indices, key=lambda i: j_ranks[i])
            for rank, i in enumerate(non_elim_sorted, 1):
                f_ranks[i] = rank
            
            # 2. 淘汰者分配最差 fan ranks
            for rank, i in enumerate(elim_indices, len(non_elim_indices) + 1):
                f_ranks[i] = rank
            
            # 验证约束
            c = j_ranks + f_ranks
            max_non_elim_c = max(c[i] for i in non_elim_indices) if non_elim_indices else 0
            
            # 如果约束不满足，增加淘汰者的 fan rank
            for e_idx in elim_indices:
                if c[e_idx] <= max_non_elim_c:
                    needed = max_non_elim_c - j_ranks[e_idx] + 1
                    f_ranks[e_idx] = max(f_ranks[e_idx], needed)
        
        else:
            # 无淘汰：fan rank = judge rank（相关假设）
            f_ranks = j_ranks.copy()
        
        # 转换为 vote share
        v = np.exp(-LAMBDA_RANK * (f_ranks - 1))
        v = v / v.sum()
        
        for i, name in enumerate(names):
            results[(name, week)] = v[i]
            fan_ranks[(name, week)] = f_ranks[i]
    
    return results, fan_ranks


# ============================================================================
# Bottom2 制度 (S28-34) - 解析解
# ============================================================================
def solve_bottom2_season(season_df, valid_weeks, final_ranking):
    """Bottom2 制度：垫底2人由 combined rank 决定，评委从中选1人淘汰"""
    results = {}
    fan_ranks = {}
    
    for week in valid_weeks:
        col = f'J_week{week}'
        active = season_df[season_df[col] > 0].copy()
        n = len(active)
        if n == 0:
            continue
        
        names = active['celebrity_name'].tolist()
        J = active[col].values.astype(float)
        
        # Judge ranks
        j_order = np.argsort(-J)
        j_ranks = np.zeros(n)
        for r, idx in enumerate(j_order, 1):
            j_ranks[idx] = r
        
        eliminated = active[active['elim_week'] == week]['celebrity_name'].tolist()
        is_finals = (week == max(valid_weeks)) and final_ranking
        
        if is_finals:
            f_ranks = np.zeros(n)
            for i, name in enumerate(names):
                if name in final_ranking:
                    f_ranks[i] = final_ranking.index(name) + 1
                else:
                    f_ranks[i] = len(final_ranking) + 1
        
        elif eliminated:
            f_ranks = np.zeros(n)
            elim_indices = [i for i, name in enumerate(names) if name in eliminated]
            non_elim_indices = [i for i in range(n) if i not in elim_indices]
            
            # Bottom2 约束：淘汰者和另一个人组成 bottom 2
            # 评委从中选择淘汰（通常选 judge score 低的）
            
            # 找可能的 bottom2 partner（被评委救的人）
            # 假设：partner 的 judge score >= 淘汰者
            
            elim_j_scores = [J[i] for i in elim_indices]
            min_elim_j = min(elim_j_scores) if elim_j_scores else 0
            
            # Partner 候选：judge score 比淘汰者高或相近
            partner_candidates = [i for i in non_elim_indices if J[i] >= min_elim_j - 5]
            if not partner_candidates:
                partner_candidates = non_elim_indices.copy()
            
            # 选择一个 partner（judge rank 最差的非淘汰者之一）
            if partner_candidates:
                partner_idx = max(partner_candidates, key=lambda i: j_ranks[i])
            else:
                partner_idx = None
            
            # 分配 fan ranks
            # 策略：确保淘汰者在 Bottom2 中
            # 1. 淘汰者得到最差 fan rank
            # 2. Partner 得到第二差
            # 3. 其他人得到更好的 ranks
            
            # 非淘汰者（除 partner）：1 到 n-2
            others = [i for i in non_elim_indices if i != partner_idx]
            others_sorted = sorted(others, key=lambda i: j_ranks[i])
            for rank, i in enumerate(others_sorted, 1):
                f_ranks[i] = rank
            
            # Partner: n-1
            if partner_idx is not None:
                f_ranks[partner_idx] = len(others) + 1
            
            # 淘汰者: n
            for i in elim_indices:
                f_ranks[i] = n
            
            # 验证并强化 bottom2 约束
            c = j_ranks + f_ranks
            
            # 确保淘汰者在 bottom 2 中
            sorted_c = sorted(range(n), key=lambda i: -c[i])  # 降序，最差在前
            bottom2_set = set(sorted_c[:2])
            
            elim_in_bottom2 = all(i in bottom2_set for i in elim_indices)
            
            if not elim_in_bottom2:
                # 需要调整：增加淘汰者的 fan rank 使其进入 bottom2
                # 找出当前 bottom2 中非淘汰者的 combined rank
                current_bottom2_c = [c[i] for i in sorted_c[:2]]
                threshold = min(current_bottom2_c) if current_bottom2_c else n
                
                for i in elim_indices:
                    # 淘汰者的 combined rank 需要 >= threshold
                    needed_f = threshold - j_ranks[i] + 0.5
                    f_ranks[i] = max(f_ranks[i], needed_f, n - 0.5)
                
                # 重新计算并验证
                c = j_ranks + f_ranks
        
        else:
            f_ranks = j_ranks.copy()
        
        # 转换为 vote share
        v = np.exp(-LAMBDA_RANK * (f_ranks - 1))
        v = v / v.sum()
        
        for i, name in enumerate(names):
            results[(name, week)] = v[i]
            fan_ranks[(name, week)] = f_ranks[i]
    
    return results, fan_ranks


# ============================================================================
# 一致性验证（分制度）
# ============================================================================
def validate_percent(results, fan_ranks, season_df, valid_weeks):
    """验证 Percent 制度的一致性"""
    correct = 0
    total = 0
    
    for week in valid_weeks:
        col = f'J_week{week}'
        active = season_df[season_df[col] > 0].copy()
        eliminated = active[active['elim_week'] == week]['celebrity_name'].tolist()
        
        if not eliminated:
            continue
        
        total += 1
        names = active['celebrity_name'].tolist()
        J = active[col].values.astype(float)
        j_share = J / J.sum() if J.sum() > 0 else np.ones(len(J)) / len(J)
        
        v = np.array([results.get((name, week), 1/len(names)) for name in names])
        c = j_share + v
        
        pred_idx = np.argmin(c)
        pred = names[pred_idx]
        
        if pred in eliminated:
            correct += 1
    
    return correct / total if total > 0 else 1.0


def validate_rank(results, fan_ranks, season_df, valid_weeks):
    """验证 Rank 制度的一致性"""
    correct = 0
    total = 0
    
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
        
        # Judge ranks
        j_order = np.argsort(-J)
        j_ranks = np.zeros(n)
        for r, idx in enumerate(j_order, 1):
            j_ranks[idx] = r
        
        # Fan ranks（从保存的数据获取）
        f_ranks = np.array([fan_ranks.get((name, week), n/2) for name in names])
        
        c = j_ranks + f_ranks
        pred_idx = np.argmax(c)  # 最大 combined rank 淘汰
        pred = names[pred_idx]
        
        if pred in eliminated:
            correct += 1
    
    return correct / total if total > 0 else 1.0


def validate_bottom2(results, fan_ranks, season_df, valid_weeks):
    """验证 Bottom2 制度的一致性
    
    对于 Bottom2 制度，我们验证两个指标：
    1. 淘汰者是否在预测的 Bottom2 中 (bottom2_accuracy)
    2. 评委选择是否符合预期 (judge_choice_accuracy) - 较弱的约束
    """
    bottom2_correct = 0
    judge_choice_correct = 0
    total = 0
    
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
        
        # Bottom 2
        bottom2_indices = np.argsort(-c)[:2]
        bottom2_names = [names[i] for i in bottom2_indices]
        
        # 验证1：淘汰者是否在 Bottom2 中
        elim_in_bottom2 = any(e in bottom2_names for e in eliminated)
        if elim_in_bottom2:
            bottom2_correct += 1
        
        # 验证2：评委选择（淘汰者是 Bottom2 中 judge score 较低的）
        if len(bottom2_indices) >= 2 and elim_in_bottom2:
            if J[bottom2_indices[0]] <= J[bottom2_indices[1]]:
                pred_elim_idx = bottom2_indices[0]
            else:
                pred_elim_idx = bottom2_indices[1]
            
            if names[pred_elim_idx] in eliminated:
                judge_choice_correct += 1
    
    # 返回 bottom2 准确率（这是模型应该保证的）
    return bottom2_correct / total if total > 0 else 1.0


# ============================================================================
# Ensemble
# ============================================================================
def ensemble_season(solve_func, season_df, valid_weeks, final_ranking, n_ensemble):
    """运行 ensemble 获取不确定性"""
    all_results = []
    
    for k in range(n_ensemble):
        # 扰动数据
        perturbed_df = season_df.copy()
        for week in valid_weeks:
            col = f'J_week{week}'
            if col in perturbed_df.columns:
                noise = np.random.normal(0, 0.5, len(perturbed_df))
                perturbed_df[col] = perturbed_df[col] + noise
                perturbed_df[col] = perturbed_df[col].clip(lower=1)
        
        results, _ = solve_func(perturbed_df, valid_weeks, final_ranking)
        all_results.append(results)
    
    # 汇总
    ensemble_stats = {}
    all_keys = set()
    for r in all_results:
        all_keys.update(r.keys())
    
    for key in all_keys:
        samples = [r.get(key, np.nan) for r in all_results]
        samples = [s for s in samples if not np.isnan(s)]
        
        if samples:
            mean = np.mean(samples)
            std = np.std(samples)
            cv = std / mean if mean > 0 else 0
            ci_low = np.percentile(samples, 2.5)
            ci_high = np.percentile(samples, 97.5)
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
    print("Fan Vote Estimation v4 - Fixed Constraints")
    print("=" * 70)
    
    print("\n[1/4] Loading data...")
    df = load_data(DATA_PATH)
    print(f"      Loaded {len(df)} contestants, {df['season'].nunique()} seasons")
    
    all_rows = []
    all_metrics = []
    
    print("\n[2/4] Estimating fan votes...")
    
    for season in sorted(df['season'].unique()):
        season_df, valid_weeks, final_ranking = get_season_info(df, season)
        
        if not valid_weeks:
            continue
        
        # 确定制度
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
        
        # 点估计
        results, fan_ranks = solve_func(season_df, valid_weeks, final_ranking)
        
        # 验证一致性
        accuracy = validate_func(results, fan_ranks, season_df, valid_weeks)
        
        # Ensemble
        ensemble_stats = ensemble_season(solve_func, season_df, valid_weeks, final_ranking, N_ENSEMBLE)
        
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
                    'regime': regime
                })
        
        # 计算赛季指标
        season_cv = np.mean([r['cv'] for r in all_rows if r['season'] == season and not np.isnan(r['cv'])])
        season_ci = np.mean([r['ci_width'] for r in all_rows if r['season'] == season and not np.isnan(r['ci_width'])])
        
        all_metrics.append({
            'season': season,
            'regime': regime,
            'accuracy': accuracy,
            'avg_cv': season_cv,
            'avg_ci_width': season_ci
        })
        
        print(f"      Season {season:2d}: {regime:8s} | Acc: {accuracy:5.1%} | CV: {season_cv:.3f}")
    
    print("\n[3/4] Saving results...")
    
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"      Saved {len(results_df)} rows to {OUTPUT_PATH}")
    
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(METRICS_PATH, index=False)
    print(f"      Saved {len(metrics_df)} seasons to {METRICS_PATH}")
    
    print("\n[4/4] Summary")
    print("=" * 70)
    
    for regime in ['rank', 'percent', 'bottom2']:
        regime_metrics = metrics_df[metrics_df['regime'] == regime]
        if len(regime_metrics) > 0:
            avg_acc = regime_metrics['accuracy'].mean()
            std_acc = regime_metrics['accuracy'].std()
            avg_cv = regime_metrics['avg_cv'].mean()
            print(f"  {regime.upper():8s}: Acc = {avg_acc:.1%} (std: {std_acc:.1%}), CV = {avg_cv:.3f}")
    
    overall_acc = metrics_df['accuracy'].mean()
    print(f"\n  Overall Accuracy: {overall_acc:.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
