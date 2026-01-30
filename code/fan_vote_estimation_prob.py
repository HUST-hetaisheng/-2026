"""
Fan Vote Estimation - Probabilistic Version
=============================================

改进点（仅针对 Bottom2 制度 S28-34）：
- Rank 和 Percent 制度：保持确定性求解（因为规则直接决定淘汰）
- Bottom2 制度：评委在两人中选择时，使用概率模型
  P(elim=e | Bottom2={e, p}) ∝ exp(-β * J_e)
  评委更倾向于淘汰当周评分较低的选手
- 通过 MCMC 采样获得不同 partner 配对的后验分布
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================
DATA_PATH = 'd:/2026-repo/data/2026_MCM_Problem_C_Data.csv'
OUTPUT_PATH = 'd:/2026-repo/data/fan_vote_results_prob.csv'
METRICS_PATH = 'd:/2026-repo/data/consistency_metrics_prob.csv'

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
# Bottom2 制度 (S28-34) - 概率模型
# ============================================================================
# 评委选择参数
BETA_JUDGE = 1.0  # 评委倾向于淘汰低分者的强度
N_MCMC_SAMPLES = 100  # MCMC 采样数

def judge_choice_log_prob(elim_idx, partner_idx, J):
    """
    评委选择的对数概率
    
    P(elim=e | Bottom2={e, p}) ∝ exp(-β * J_e)
    
    评委更倾向于淘汰当周评分较低的选手
    """
    J_elim = J[elim_idx]
    J_partner = J[partner_idx]
    
    # softmax: exp(-β*J_e) / (exp(-β*J_e) + exp(-β*J_p))
    log_p_elim = -BETA_JUDGE * J_elim
    log_p_partner = -BETA_JUDGE * J_partner
    log_Z = np.logaddexp(log_p_elim, log_p_partner)
    
    return log_p_elim - log_Z


def sample_bottom2_fan_ranks(j_ranks, J, elim_idx, n, n_samples):
    """
    MCMC 采样 fan ranks 的后验分布
    
    约束：淘汰者必须在 Bottom2 中
    概率模型：评委选择用 softmax
    """
    samples = []
    
    # 初始化：满足约束的解
    f_ranks = np.zeros(n)
    non_elim = [i for i in range(n) if i != elim_idx]
    
    # 随机选择一个 partner（也在 Bottom2 中）
    partner_idx = np.random.choice(non_elim)
    others = [i for i in non_elim if i != partner_idx]
    
    # 分配初始 ranks
    np.random.shuffle(others)
    for rank, i in enumerate(others, 1):
        f_ranks[i] = rank
    f_ranks[partner_idx] = len(others) + 1
    f_ranks[elim_idx] = n
    
    # 当前对数概率
    log_p = judge_choice_log_prob(elim_idx, partner_idx, J)
    
    for step in range(n_samples * 10):
        # 提议：随机交换两个非淘汰者的 rank
        if np.random.random() < 0.5 and len(non_elim) >= 2:
            # 交换两个非淘汰者
            i1, i2 = np.random.choice(non_elim, 2, replace=False)
            f_ranks_prop = f_ranks.copy()
            f_ranks_prop[i1], f_ranks_prop[i2] = f_ranks_prop[i2], f_ranks_prop[i1]
            
            # 检查约束
            c_prop = j_ranks + f_ranks_prop
            sorted_c = sorted(range(n), key=lambda i: -c_prop[i])
            bottom2_prop = set(sorted_c[:2])
            
            if elim_idx in bottom2_prop:
                # 找到新的 partner
                partner_prop = [i for i in bottom2_prop if i != elim_idx][0] if len(bottom2_prop) > 1 else partner_idx
                log_p_prop = judge_choice_log_prob(elim_idx, partner_prop, J)
                
                # 接受概率
                if np.log(np.random.random()) < log_p_prop - log_p:
                    f_ranks = f_ranks_prop
                    partner_idx = partner_prop
                    log_p = log_p_prop
        else:
            # 提议新的 partner
            new_partner = np.random.choice(non_elim)
            if new_partner == partner_idx:
                continue
            
            # 构造新的 f_ranks 使得新 partner 进入 Bottom2
            f_ranks_prop = f_ranks.copy()
            
            # 让新 partner 的 fan rank 足够差
            old_others = [i for i in non_elim if i != new_partner]
            np.random.shuffle(old_others)
            for rank, i in enumerate(old_others, 1):
                f_ranks_prop[i] = rank
            f_ranks_prop[new_partner] = len(old_others) + 1
            f_ranks_prop[elim_idx] = n
            
            # 验证约束
            c_prop = j_ranks + f_ranks_prop
            sorted_c = sorted(range(n), key=lambda i: -c_prop[i])
            bottom2_prop = set(sorted_c[:2])
            
            if elim_idx in bottom2_prop and new_partner in bottom2_prop:
                log_p_prop = judge_choice_log_prob(elim_idx, new_partner, J)
                
                if np.log(np.random.random()) < log_p_prop - log_p:
                    f_ranks = f_ranks_prop
                    partner_idx = new_partner
                    log_p = log_p_prop
        
        if step >= n_samples * 5:  # burn-in
            samples.append(f_ranks.copy())
    
    return np.array(samples[::5])  # thinning


def solve_bottom2_season(season_df, valid_weeks, final_ranking):
    """
    Bottom2 制度的概率求解
    
    改进：使用 MCMC 对评委选择建模
    P(elim=e | Bottom2={e, p}) ∝ exp(-β * J_e)
    """
    results = {}
    fan_ranks = {}
    week_info = {}
    samples_by_week = {}
    
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
            # 决赛周：确定性
            f_ranks = np.zeros(n)
            for i, name in enumerate(names):
                if name in final_ranking:
                    f_ranks[i] = final_ranking.index(name) + 1
                else:
                    f_ranks[i] = len(final_ranking) + 1
            
            v = np.exp(-LAMBDA_RANK * (f_ranks - 1))
            v = v / v.sum()
            for i, name in enumerate(names):
                results[(name, week)] = v[i]
                fan_ranks[(name, week)] = f_ranks[i]
                    
        elif week_type in ['no_elim', 'withdrew_only']:
            # 无淘汰：无约束
            f_ranks = j_ranks.copy()
            v = np.exp(-LAMBDA_RANK * (f_ranks - 1))
            v = v / v.sum()
            for i, name in enumerate(names):
                results[(name, week)] = v[i]
                fan_ranks[(name, week)] = f_ranks[i]
            
        elif week_type == 'single_elim':
            # 单人淘汰：MCMC 采样
            elim_idx = names.index(eliminated[0])
            
            # 采样
            rank_samples = sample_bottom2_fan_ranks(j_ranks, J, elim_idx, n, N_MCMC_SAMPLES)
            
            # 转换为 vote share
            v_samples = np.exp(-LAMBDA_RANK * (rank_samples - 1))
            v_samples = v_samples / v_samples.sum(axis=1, keepdims=True)
            
            # 取均值
            v_mean = np.mean(v_samples, axis=0)
            f_ranks_mean = np.mean(rank_samples, axis=0)
            
            for i, name in enumerate(names):
                results[(name, week)] = v_mean[i]
                fan_ranks[(name, week)] = f_ranks_mean[i]
            
            samples_by_week[week] = {'names': names, 'v_samples': v_samples, 
                                     'rank_samples': rank_samples}
                
        elif week_type == 'multi_elim':
            # 多人淘汰：简化处理
            elim_indices = [names.index(e) for e in eliminated if e in names]
            non_elim = [i for i in range(n) if i not in elim_indices]
            
            f_ranks = np.zeros(n)
            non_elim_sorted = sorted(non_elim, key=lambda i: j_ranks[i])
            for rank, i in enumerate(non_elim_sorted, 1):
                f_ranks[i] = rank
            for rank, i in enumerate(elim_indices):
                f_ranks[i] = len(non_elim) + rank + 1
            
            v = np.exp(-LAMBDA_RANK * (f_ranks - 1))
            v = v / v.sum()
            for i, name in enumerate(names):
                results[(name, week)] = v[i]
                fan_ranks[(name, week)] = f_ranks[i]
    
    return results, fan_ranks, week_info, samples_by_week


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
def ensemble_season(solve_func, season_df, valid_weeks, final_ranking, n_ensemble, is_bottom2=False):
    all_results = []
    
    for k in range(n_ensemble):
        perturbed_df = season_df.copy()
        for week in valid_weeks:
            col = f'J_week{week}'
            if col in perturbed_df.columns:
                noise = np.random.normal(0, 0.5, len(perturbed_df))
                perturbed_df[col] = perturbed_df[col] + noise
                perturbed_df[col] = perturbed_df[col].clip(lower=1)
        
        if is_bottom2:
            results, _, _, _ = solve_func(perturbed_df, valid_weeks, final_ranking)
        else:
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
    print("Fan Vote Estimation - Probabilistic Version")
    print("=" * 70)
    print("\n改进点（仅 Bottom2 制度 S28-34）：")
    print("  - Rank/Percent: 确定性求解")
    print("  - Bottom2: 评委选择用概率模型 P(elim) ∝ exp(-β*J)")
    print(f"  - β = {BETA_JUDGE}, MCMC 采样数 = {N_MCMC_SAMPLES}\n")
    
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
        if regime == 'bottom2':
            results, fan_ranks, week_info, samples_by_week = solve_func(season_df, valid_weeks, final_ranking)
        else:
            results, fan_ranks, week_info = solve_func(season_df, valid_weeks, final_ranking)
            samples_by_week = {}
        
        # 验证
        accuracy, details = validate_func(results, fan_ranks, season_df, valid_weeks)
        
        # Ensemble
        is_bottom2 = (regime == 'bottom2')
        ensemble_stats = ensemble_season(solve_func, season_df, valid_weeks, final_ranking, N_ENSEMBLE, is_bottom2)
        
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
                
                cv_val = stats.get('cv', 0)
                certainty = 1 / (1 + cv_val) if cv_val >= 0 else 1.0
                
                all_rows.append({
                    'season': season,
                    'week': week,
                    'celebrity_name': name,
                    'fan_vote_share': vote_share,
                    'fan_rank': fan_rank,
                    'mean': stats.get('mean', vote_share),
                    'std': stats.get('std', 0),
                    'cv': cv_val,
                    'certainty': certainty,
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
            avg_certainty = 1 / (1 + avg_cv)  # Certainty = 1/(1+CV)
            n_seasons = len(regime_metrics)
            
            print(f"\n{regime.upper()} (n={n_seasons} seasons):")
            print(f"  Constraint Satisfaction: {avg_acc:.1%} (std: {std_acc:.1%})")
            print(f"  Average CV: {avg_cv:.3f}")
            print(f"  Average Certainty: {avg_certainty:.3f}")
            
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
