"""
Fan Vote Estimation - Convex Optimization (Fixed Version)
==========================================================
修复以下问题：
1. Ensemble 噪声会把已淘汰选手"复活"
2. 多淘汰周被当成单淘汰处理
3. 决赛约束在归一化后可能被破坏
4. 时间平滑未实现（简化为移除声明）
5. Bottom2 MCMC 未实际调用
6. CSR 验证用严格相等判断
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# 配置
DATA_PATH = 'd:/2026-repo/data/2026_MCM_Problem_C_Data.csv'
OUTPUT_PATH = 'd:/2026-repo/data/fan_vote_results.csv'
METRICS_PATH = 'd:/2026-repo/data/consistency_metrics.csv'

# 超参数
TAU = 15.0       # softmax 温度
ALPHA = 0.05     # 熵正则化
LAMBDA_RANK = 0.5
N_ENSEMBLE = 20
FLOAT_TOL = 1e-6  # 浮点容差
np.random.seed(42)


def load_data():
    df = pd.read_csv(DATA_PATH)
    for w in range(1, 12):
        cols = [f'week{w}_judge{j}_score' for j in range(1, 5)]
        cols = [c for c in cols if c in df.columns]
        if cols:
            df[f'J{w}'] = df[cols].apply(pd.to_numeric, errors='coerce').sum(axis=1, skipna=True)
    return df


def parse_results(df):
    import re
    df = df.copy()
    df['elim_week'] = None
    df['placement'] = None
    df['withdrew'] = False
    
    for idx, row in df.iterrows():
        r = str(row.get('results', ''))
        if 'Withdrew' in r:
            df.loc[idx, 'withdrew'] = True
        m = re.search(r'Eliminated Week (\d+)', r)
        if m:
            df.loc[idx, 'elim_week'] = int(m.group(1))
        m = re.search(r'(\d+)(st|nd|rd|th) Place', r)
        if m:
            df.loc[idx, 'placement'] = int(m.group(1))
    return df


def get_season_weeks(sdf):
    weeks = []
    for w in range(1, 12):
        col = f'J{w}'
        if col in sdf.columns and (sdf[col] > 0).any():
            weeks.append(w)
    return weeks


# =============================================================================
# Percent 制度 (S3-27): 凸优化
# =============================================================================

def neg_log_likelihood_multi(v, j_share, elim_idxs, tau):
    """
    多淘汰的负对数似然
    对于每个淘汰者 e，P(e in bottom |E|) ∝ exp(-τ * c_e)
    """
    c = j_share + v
    n = len(c)
    k = len(elim_idxs)  # 淘汰人数
    
    # 简化：要求所有淘汰者都是 combined score 最小的 k 个
    # 对数似然 = Σ_e (-τ*c_e) - log(Σ_all exp(-τ*c_i))
    log_p = 0
    for e in elim_idxs:
        log_p += -tau * c[e]
    log_p -= k * np.log(np.sum(np.exp(-tau * c)))
    
    return -log_p


def entropy_reg(v):
    v_safe = np.clip(v, 1e-10, 1)
    return -np.sum(v_safe * np.log(v_safe))


def solve_percent_week_multi(J, elim_idxs, tau=TAU, alpha=ALPHA):
    """
    凸优化求解单周 fan vote share (支持多淘汰)
    """
    n = len(J)
    j_share = J / J.sum() if J.sum() > 0 else np.ones(n) / n
    
    if not elim_idxs:
        # 无淘汰周：均匀分布
        return np.ones(n) / n
    
    v0 = np.ones(n) / n
    constraints = [{'type': 'eq', 'fun': lambda v: np.sum(v) - 1}]
    bounds = [(1e-6, 1)] * n
    
    def objective(v):
        return neg_log_likelihood_multi(v, j_share, elim_idxs, tau) - alpha * entropy_reg(v)
    
    result = minimize(objective, v0, method='SLSQP', bounds=bounds, 
                     constraints=constraints, options={'maxiter': 100, 'ftol': 1e-8})
    
    # 检查优化是否成功
    if not result.success:
        # 回退到简单构造
        v = np.ones(n) / n
        for e in elim_idxs:
            v[e] = 1e-4
        v = v / v.sum()
        return v
    
    v = result.x
    v = np.clip(v, 1e-6, 1)
    v = v / v.sum()
    return v


def solve_finals_robust(active, names, J):
    """
    决赛周：确保归一化后仍满足排名约束
    使用迭代修正
    """
    n = len(names)
    placements = []
    for i, name in enumerate(names):
        row = active[active['celebrity_name'] == name].iloc[0]
        p = row.get('placement', n)
        placements.append(p if pd.notna(p) else n)
    
    j_share = J / J.sum() if J.sum() > 0 else np.ones(n) / n
    sorted_idx = np.argsort(placements)  # 按排名排序 (1st, 2nd, 3rd...)
    
    # 迭代修正直到约束满足
    for _ in range(10):  # 最多10次迭代
        v = np.array([1.0 / max(p, 0.5) for p in placements])
        
        # 调整使 c_1 > c_2 > c_3 ...
        for k in range(1, len(sorted_idx)):
            i, j = sorted_idx[k-1], sorted_idx[k]  # i 排名更好
            c_i, c_j = j_share[i] + v[i], j_share[j] + v[j]
            if c_i <= c_j + FLOAT_TOL:
                v[i] = c_j - j_share[i] + 0.02 * (k + 1)  # 增量调整
        
        v = np.clip(v, 1e-6, 1)
        v = v / v.sum()
        
        # 验证
        c = j_share + v
        valid = True
        for k in range(1, len(sorted_idx)):
            i, j = sorted_idx[k-1], sorted_idx[k]
            if c[i] <= c[j] + FLOAT_TOL:
                valid = False
                break
        
        if valid:
            return v
    
    # 如果仍失败，直接分配使约束满足
    v = np.zeros(n)
    base = 1.0
    for k, idx in enumerate(sorted_idx):
        v[idx] = base
        base *= 0.8  # 递减
    v = v / v.sum()
    return v


def solve_percent_season_convex(sdf, weeks):
    """凸优化求解整季 (Percent 制度) - 支持多淘汰"""
    results = {}
    week_info = {}
    
    for w in weeks:
        col = f'J{w}'
        active = sdf[sdf[col] > 0].copy()
        n = len(active)
        if n == 0:
            continue
        
        names = active['celebrity_name'].tolist()
        J = active[col].values.astype(float)
        
        # 找所有淘汰者
        elim = active[(active['elim_week'] == w) & (~active['withdrew'])]
        elim_names = elim['celebrity_name'].tolist()
        elim_idxs = [names.index(e) for e in elim_names if e in names]
        
        is_finals = (w == max(weeks)) and (active['placement'].notna().any())
        
        if is_finals:
            v = solve_finals_robust(active, names, J)
            week_info[w] = 'finals'
        elif len(elim_idxs) == 0:
            v = np.ones(n) / n
            week_info[w] = 'no_elim'
        else:
            v = solve_percent_week_multi(J, elim_idxs)
            week_info[w] = 'single_elim' if len(elim_idxs) == 1 else f'multi_elim_{len(elim_idxs)}'
        
        for i, name in enumerate(names):
            results[(name, w)] = v[i]
    
    return results, week_info


# =============================================================================
# Rank 制度 (S1-2): 排名约束
# =============================================================================

def solve_rank_season(sdf, weeks):
    """Rank 制度：combined rank = j_rank + f_rank，最大者被淘汰"""
    results = {}
    week_info = {}
    
    for w in weeks:
        col = f'J{w}'
        active = sdf[sdf[col] > 0].copy()
        n = len(active)
        if n == 0:
            continue
        
        names = active['celebrity_name'].tolist()
        J = active[col].values.astype(float)
        
        j_order = np.argsort(-J)
        j_ranks = np.zeros(n)
        for r, idx in enumerate(j_order, 1):
            j_ranks[idx] = r
        
        elim = active[(active['elim_week'] == w) & (~active['withdrew'])]
        elim_names = elim['celebrity_name'].tolist()
        is_finals = (w == max(weeks)) and (active['placement'].notna().any())
        
        if is_finals:
            f_ranks = np.zeros(n)
            for i, name in enumerate(names):
                row = active[active['celebrity_name'] == name].iloc[0]
                p = row.get('placement', n)
                f_ranks[i] = p if pd.notna(p) else n
            week_info[w] = 'finals'
        elif len(elim_names) == 0:
            f_ranks = j_ranks.copy()
            week_info[w] = 'no_elim'
        else:
            elim_idxs = [names.index(e) for e in elim_names if e in names]
            non_elim = [i for i in range(n) if i not in elim_idxs]
            
            f_ranks = np.zeros(n)
            
            # 非淘汰者按 judge rank 排序分配 fan rank
            sorted_non = sorted(non_elim, key=lambda i: j_ranks[i])
            for r, i in enumerate(sorted_non, 1):
                f_ranks[i] = r
            
            # 淘汰者给最差的 fan rank
            for r, i in enumerate(elim_idxs):
                max_non_elim_c = max(j_ranks[p] + f_ranks[p] for p in non_elim) if non_elim else 0
                required_f = max_non_elim_c - j_ranks[i] + 1.5 + r
                f_ranks[i] = max(n, required_f)
            
            week_info[w] = 'single_elim' if len(elim_idxs) == 1 else f'multi_elim_{len(elim_idxs)}'
        
        v = np.exp(-LAMBDA_RANK * (f_ranks - 1))
        v = v / v.sum()
        
        for i, name in enumerate(names):
            results[(name, w)] = v[i]
    
    return results, week_info


# =============================================================================
# Bottom2 制度 (S28-34): 确定性模型
# =============================================================================

def solve_bottom2_season(sdf, weeks):
    """Bottom2 + 评委选择：确定性构造"""
    results = {}
    week_info = {}
    
    for w in weeks:
        col = f'J{w}'
        active = sdf[sdf[col] > 0].copy()
        n = len(active)
        if n == 0:
            continue
        
        names = active['celebrity_name'].tolist()
        J = active[col].values.astype(float)
        
        j_order = np.argsort(-J)
        j_ranks = np.zeros(n)
        for r, idx in enumerate(j_order, 1):
            j_ranks[idx] = r
        
        elim = active[(active['elim_week'] == w) & (~active['withdrew'])]
        elim_names = elim['celebrity_name'].tolist()
        is_finals = (w == max(weeks)) and (active['placement'].notna().any())
        
        if is_finals:
            f_ranks = np.zeros(n)
            for i, name in enumerate(names):
                row = active[active['celebrity_name'] == name].iloc[0]
                p = row.get('placement', n)
                f_ranks[i] = p if pd.notna(p) else n
            week_info[w] = 'finals'
        elif len(elim_names) == 0:
            f_ranks = j_ranks.copy()
            week_info[w] = 'no_elim'
        else:
            elim_idxs = [names.index(e) for e in elim_names if e in names]
            non_elim = [i for i in range(n) if i not in elim_idxs]
            
            # Partner: judge rank 最差的非淘汰者
            if non_elim:
                partner_idx = max(non_elim, key=lambda i: j_ranks[i])
                others = [i for i in non_elim if i != partner_idx]
            else:
                partner_idx = None
                others = non_elim
            
            f_ranks = np.zeros(n)
            
            # 好的选手给好的 fan rank
            sorted_others = sorted(others, key=lambda i: j_ranks[i])
            for r, i in enumerate(sorted_others, 1):
                f_ranks[i] = r
            
            # Partner 和淘汰者进入 bottom2
            if partner_idx is not None:
                f_ranks[partner_idx] = len(others) + 1
            
            for r, i in enumerate(elim_idxs):
                f_ranks[i] = len(others) + 2 + r
            
            week_info[w] = 'single_elim' if len(elim_idxs) == 1 else f'multi_elim_{len(elim_idxs)}'
        
        v = np.exp(-LAMBDA_RANK * (f_ranks - 1))
        v = v / v.sum()
        
        for i, name in enumerate(names):
            results[(name, w)] = v[i]
    
    return results, week_info


# =============================================================================
# Ensemble (修复：不复活已淘汰选手)
# =============================================================================

def ensemble_solve(solve_func, sdf, weeks, n_ens=N_ENSEMBLE):
    """扰动 + 重求解 (修复版：只对活跃选手加噪)"""
    all_results = []
    
    for _ in range(n_ens):
        sdf_p = sdf.copy()
        for w in weeks:
            col = f'J{w}'
            if col in sdf_p.columns:
                # 只对原本 >0 的分数加噪
                mask = sdf_p[col] > 0
                noise = np.random.normal(0, 0.5, len(sdf_p))
                sdf_p.loc[mask, col] = (sdf_p.loc[mask, col] + noise[mask]).clip(lower=1)
                # 保持原本为 0 的不变
        
        res, _ = solve_func(sdf_p, weeks)
        all_results.append(res)
    
    stats = {}
    all_keys = set()
    for r in all_results:
        all_keys.update(r.keys())
    
    for key in all_keys:
        vals = [r.get(key, np.nan) for r in all_results if key in r]
        if vals:
            mu, sigma = np.mean(vals), np.std(vals)
            cv = sigma / mu if mu > 0 else 0
            stats[key] = {
                'mean': mu, 'std': sigma, 'cv': cv,
                'certainty': 1 / (1 + cv),
                'ci_low': np.percentile(vals, 2.5),
                'ci_high': np.percentile(vals, 97.5)
            }
    
    return stats


# =============================================================================
# 验证 (修复：支持多淘汰，用容差比较)
# =============================================================================

def validate_percent(results, sdf, weeks):
    """验证 Percent 制度约束 (支持多淘汰)"""
    correct = 0
    total = 0
    
    for w in weeks:
        col = f'J{w}'
        active = sdf[sdf[col] > 0].copy()
        elim = active[(active['elim_week'] == w) & (~active['withdrew'])]
        
        if len(elim) == 0:
            continue
        
        names = active['celebrity_name'].tolist()
        J = active[col].values.astype(float)
        j_share = J / J.sum()
        
        v = np.array([results.get((name, w), 1/len(names)) for name in names])
        c = j_share + v
        
        # 检查所有淘汰者是否都在最差的 k 个
        elim_names = elim['celebrity_name'].tolist()
        k = len(elim_names)
        total += 1
        
        sorted_idx = np.argsort(c)  # 从小到大
        bottom_k = set(sorted_idx[:k])
        
        all_in_bottom = all(
            names.index(e) in bottom_k 
            for e in elim_names if e in names
        )
        
        if all_in_bottom:
            correct += 1
    
    return correct / total if total > 0 else 1.0


def validate_rank(results, sdf, weeks):
    """验证 Rank 制度约束 (支持多淘汰，用容差)"""
    correct = 0
    total = 0
    
    for w in weeks:
        col = f'J{w}'
        active = sdf[sdf[col] > 0].copy()
        elim = active[(active['elim_week'] == w) & (~active['withdrew'])]
        is_finals = (w == max(weeks)) and (active['placement'].notna().any())
        
        if len(elim) == 0 or is_finals:
            continue
        
        total += 1
        names = active['celebrity_name'].tolist()
        n = len(names)
        J = active[col].values.astype(float)
        
        j_order = np.argsort(-J)
        j_ranks = np.zeros(n)
        for r, idx in enumerate(j_order, 1):
            j_ranks[idx] = r
        
        v = np.array([results.get((name, w), 1/n) for name in names])
        f_ranks = np.argsort(np.argsort(-v)) + 1
        c = j_ranks + f_ranks
        
        elim_names = elim['celebrity_name'].tolist()
        elim_idxs = [names.index(e) for e in elim_names if e in names]
        k = len(elim_idxs)
        
        sorted_idx = np.argsort(-c)  # 从大到小
        top_k = set(sorted_idx[:k])
        
        all_in_top = all(i in top_k for i in elim_idxs)
        
        if all_in_top:
            correct += 1
    
    return correct / total if total > 0 else 1.0


def validate_bottom2(results, sdf, weeks):
    """验证 Bottom2 制度约束"""
    correct = 0
    total = 0
    
    for w in weeks:
        col = f'J{w}'
        active = sdf[sdf[col] > 0].copy()
        elim = active[(active['elim_week'] == w) & (~active['withdrew'])]
        is_finals = (w == max(weeks)) and (active['placement'].notna().any())
        
        if len(elim) == 0 or is_finals:
            continue
        
        total += 1
        names = active['celebrity_name'].tolist()
        n = len(names)
        J = active[col].values.astype(float)
        
        j_order = np.argsort(-J)
        j_ranks = np.zeros(n)
        for r, idx in enumerate(j_order, 1):
            j_ranks[idx] = r
        
        v = np.array([results.get((name, w), 1/n) for name in names])
        f_ranks = np.argsort(np.argsort(-v)) + 1
        c = j_ranks + f_ranks
        
        sorted_idx = np.argsort(-c)
        bottom2 = set(sorted_idx[:2])
        
        elim_names = elim['celebrity_name'].tolist()
        elim_idxs = [names.index(e) for e in elim_names if e in names]
        
        # 至少一个淘汰者在 bottom2
        any_in_bottom2 = any(i in bottom2 for i in elim_idxs)
        
        if any_in_bottom2:
            correct += 1
    
    return correct / total if total > 0 else 1.0


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("=" * 60)
    print("Fan Vote Estimation - Convex Optimization (Fixed)")
    print("=" * 60)
    
    df = load_data()
    df = parse_results(df)
    print(f"Loaded {len(df)} contestants, {df['season'].nunique()} seasons\n")
    
    all_rows = []
    metrics = []
    
    print(f"{'Season':>6} | {'Regime':>7} | {'CSR':>6} | {'Certainty':>9}")
    print("-" * 40)
    
    for season in sorted(df['season'].unique()):
        sdf = df[df['season'] == season].copy()
        weeks = get_season_weeks(sdf)
        if not weeks:
            continue
        
        if season <= 2:
            regime = 'rank'
            solve_func = solve_rank_season
            validate_func = validate_rank
        elif season <= 27:
            regime = 'percent'
            solve_func = solve_percent_season_convex
            validate_func = validate_percent
        else:
            regime = 'bottom2'
            solve_func = solve_bottom2_season
            validate_func = validate_bottom2
        
        results, week_info = solve_func(sdf, weeks)
        csr = validate_func(results, sdf, weeks)
        stats = ensemble_solve(solve_func, sdf, weeks)
        
        avg_cert = np.mean([s['certainty'] for s in stats.values()])
        
        for w in weeks:
            col = f'J{w}'
            active = sdf[sdf[col] > 0]
            elim_names = active[(active['elim_week'] == w) & (~active['withdrew'])]['celebrity_name'].tolist()
            
            for _, row in active.iterrows():
                name = row['celebrity_name']
                key = (name, w)
                s = stats.get(key, {})
                
                all_rows.append({
                    'season': season, 'week': w, 'celebrity_name': name,
                    'fan_vote_share': results.get(key, np.nan),
                    'mean': s.get('mean', results.get(key, np.nan)),
                    'std': s.get('std', 0),
                    'cv': s.get('cv', 0),
                    'certainty': s.get('certainty', 1),
                    'ci_low': s.get('ci_low', results.get(key, np.nan)),
                    'ci_high': s.get('ci_high', results.get(key, np.nan)),
                    'judge_total': row[col],
                    'eliminated': name in elim_names,
                    'week_type': week_info.get(w, 'unknown'),
                    'regime': regime
                })
        
        metrics.append({
            'season': season, 'regime': regime,
            'csr': csr, 'avg_certainty': avg_cert
        })
        
        print(f"{season:>6} | {regime:>7} | {csr:>5.1%} | {avg_cert:>8.3f}")
    
    print("-" * 40)
    
    pd.DataFrame(all_rows).to_csv(OUTPUT_PATH, index=False)
    pd.DataFrame(metrics).to_csv(METRICS_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    
    mdf = pd.DataFrame(metrics)
    print("\nSummary by Regime:")
    for regime in ['rank', 'percent', 'bottom2']:
        rm = mdf[mdf['regime'] == regime]
        if len(rm) > 0:
            print(f"  {regime.upper()}: CSR={rm['csr'].mean():.1%}, Certainty={rm['avg_certainty'].mean():.3f}")
    
    if mdf[mdf['regime'] == 'rank']['csr'].mean() < 1.0:
        print("\nNote: Rank CSR < 100% indicates 'upset' eliminations.")
    if mdf[mdf['regime'] == 'bottom2']['csr'].mean() < 1.0:
        print("\nNote: Bottom2 CSR < 100% reflects judge save decisions.")


if __name__ == '__main__':
    main()
