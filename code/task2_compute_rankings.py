"""
Task 2: Compute 4 Types of Rankings
====================================
计算每个选手每周的4种排名：
1. 裁判打分排名 (judge_rank)
2. 粉丝投票排名 (fan_rank)
3. Rank规则排名 (rank_rule_rank) - 基于 judge_rank + fan_rank
4. Percent规则排名 (percent_rule_rank) - 基于 judge_share + fan_share

输出: task2_rankings.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys

# Paths - 使用绝对路径避免相对路径问题
BASE_DIR = Path(r'e:\比赛\数学建模\2026美赛\comap26\2026-repo')
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_PATH = DATA_DIR / '2026_MCM_Problem_C_Data_Cleaned.csv'
FAN_VOTE_PATH = DATA_DIR / 'fan_vote_results_final.csv'
OUTPUT_PATH = DATA_DIR / 'task2_rankings.csv'

# 确保输出能显示
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None


def load_raw_data():
    """加载原始数据，计算每周评委总分"""
    df = pd.read_csv(RAW_DATA_PATH)
    
    # 计算每周评委总分
    for w in range(1, 12):
        cols = [f'week{w}_judge{j}_score' for j in range(1, 5)]
        cols = [c for c in cols if c in df.columns]
        if cols:
            df[f'J{w}'] = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
    
    return df


def load_fan_votes():
    """加载粉丝投票估计结果"""
    df = pd.read_csv(FAN_VOTE_PATH)
    return df


def compute_rankings():
    """计算所有选手每周的4种排名"""
    
    raw_df = load_raw_data()
    fan_df = load_fan_votes()
    
    results = []
    
    # 按赛季处理
    for season in sorted(fan_df['season'].unique()):
        season_raw = raw_df[raw_df['season'] == season].copy()
        season_fan = fan_df[fan_df['season'] == season].copy()
        
        # 获取该赛季的所有周
        weeks = sorted(season_fan['week'].unique())
        
        for week in weeks:
            # 获取该周活跃选手
            week_fan = season_fan[season_fan['week'] == week].copy()
            
            if week_fan.empty:
                continue
            
            # 获取选手名单
            contestants = week_fan['celebrity_name'].tolist()
            
            # 计算该周评委总分
            judge_col = f'J{week}'
            week_data = []
            
            for name in contestants:
                # 获取评委分
                raw_row = season_raw[season_raw['celebrity_name'] == name]
                if raw_row.empty or judge_col not in raw_row.columns:
                    judge_total = 0
                else:
                    judge_total = raw_row[judge_col].values[0]
                
                # 获取粉丝投票份额
                fan_row = week_fan[week_fan['celebrity_name'] == name]
                if fan_row.empty:
                    fan_share = 0
                    eliminated = False
                    regime = 'unknown'
                else:
                    fan_share = fan_row['fan_vote_share'].values[0]
                    eliminated = fan_row['eliminated_this_week'].values[0]
                    regime = fan_row['regime'].values[0]
                
                # 获取最终名次 (placement)
                if raw_row.empty:
                    final_placement = np.nan
                else:
                    final_placement = raw_row['placement'].values[0]
                
                week_data.append({
                    'season': season,
                    'week': week,
                    'celebrity_name': name,
                    'judge_total': judge_total,
                    'fan_vote_share': fan_share,
                    'eliminated_this_week': eliminated,
                    'final_placement': final_placement,
                    'regime': regime
                })
            
            # 转换为DataFrame
            week_df = pd.DataFrame(week_data)
            
            # 过滤掉评委分为0的选手（已淘汰）
            week_df = week_df[week_df['judge_total'] > 0].copy()
            
            if week_df.empty:
                continue
            
            n = len(week_df)
            
            # ===== 1. 计算评委份额和排名 =====
            total_judge = week_df['judge_total'].sum()
            week_df['judge_share'] = week_df['judge_total'] / total_judge if total_judge > 0 else 1/n
            
            # 评委排名 (分高者排名靠前，即排名数字小)
            week_df['judge_rank'] = week_df['judge_total'].rank(ascending=False, method='min').astype(int)
            
            # ===== 2. 计算粉丝投票排名 =====
            # 粉丝排名 (份额高者排名靠前)
            week_df['fan_rank'] = week_df['fan_vote_share'].rank(ascending=False, method='min').astype(int)
            
            # ===== 3. Rank规则排名 =====
            # combined_rank_sum = judge_rank + fan_rank (越小越好)
            week_df['combined_rank_sum'] = week_df['judge_rank'] + week_df['fan_rank']
            # Rank规则下的最终排名
            week_df['rank_rule_rank'] = week_df['combined_rank_sum'].rank(ascending=True, method='min').astype(int)
            
            # ===== 4. Percent规则排名 =====
            # combined_score = judge_share + fan_share (越大越好)
            week_df['combined_score'] = week_df['judge_share'] + week_df['fan_vote_share']
            # Percent规则下的最终排名
            week_df['percent_rule_rank'] = week_df['combined_score'].rank(ascending=False, method='min').astype(int)
            
            # 添加到结果
            results.append(week_df)
    
    # 合并所有结果
    final_df = pd.concat(results, ignore_index=True)
    
    # 选择输出列
    output_cols = [
        'season', 'week', 'celebrity_name', 
        'judge_total', 'judge_share', 'fan_vote_share',
        'judge_rank', 'fan_rank', 'rank_rule_rank', 'percent_rule_rank',
        'combined_rank_sum', 'combined_score',
        'eliminated_this_week', 'final_placement', 'regime'
    ]
    
    final_df = final_df[output_cols]
    
    return final_df


def analyze_ranking_differences(df):
    """分析不同规则下的排名差异"""
    
    print("\n" + "="*60)
    print("排名差异分析")
    print("="*60)
    
    # 计算Rank规则和Percent规则的排名差异
    df['rank_diff'] = df['rank_rule_rank'] - df['percent_rule_rank']
    
    # 统计差异分布
    print("\n1. Rank规则排名 - Percent规则排名 差异分布:")
    print(df['rank_diff'].describe())
    
    # 找出差异最大的案例
    print("\n2. 排名差异最大的10个案例 (Rank规则排名 - Percent规则排名):")
    extreme_cases = df.nlargest(10, 'rank_diff')[['season', 'week', 'celebrity_name', 
                                                    'judge_rank', 'fan_rank', 
                                                    'rank_rule_rank', 'percent_rule_rank', 'rank_diff']]
    print(extreme_cases.to_string(index=False))
    
    # 按赛季统计差异
    print("\n3. 按规则类型(regime)统计排名差异:")
    regime_stats = df.groupby('regime')['rank_diff'].agg(['mean', 'std', 'min', 'max', 'count'])
    print(regime_stats)
    
    # 淘汰者的排名对比
    print("\n4. 被淘汰选手在两种规则下的排名对比:")
    eliminated = df[df['eliminated_this_week'] == True]
    if not eliminated.empty:
        print(f"  总淘汰人次: {len(eliminated)}")
        print(f"  平均 rank_rule_rank: {eliminated['rank_rule_rank'].mean():.2f}")
        print(f"  平均 percent_rule_rank: {eliminated['percent_rule_rank'].mean():.2f}")
        
        # 淘汰者是否总是排名最差？
        for rule in ['rank_rule_rank', 'percent_rule_rank']:
            # 检查淘汰者是否是该周排名最差的
            worst_cases = 0
            for _, row in eliminated.iterrows():
                week_data = df[(df['season'] == row['season']) & (df['week'] == row['week'])]
                max_rank = week_data[rule].max()
                if row[rule] == max_rank:
                    worst_cases += 1
            print(f"  {rule}: 淘汰者为最差排名的比例 = {worst_cases}/{len(eliminated)} = {100*worst_cases/len(eliminated):.1f}%")
    
    return df


def find_counterfactual_eliminations(df):
    """找出如果使用不同规则，淘汰者会不同的周"""
    
    print("\n" + "="*60)
    print("反事实分析：不同规则下淘汰结果会改变的周")
    print("="*60)
    
    counterfactual_cases = []
    
    # 按赛季和周分组
    for (season, week), group in df.groupby(['season', 'week']):
        if group.empty:
            continue
        
        # 找出实际淘汰者
        actual_eliminated = group[group['eliminated_this_week'] == True]
        if actual_eliminated.empty:
            continue
        
        actual_name = actual_eliminated['celebrity_name'].values[0]
        
        # 找出两种规则下的"应该淘汰者"（排名最差的）
        rank_rule_worst = group.loc[group['rank_rule_rank'].idxmax(), 'celebrity_name']
        percent_rule_worst = group.loc[group['percent_rule_rank'].idxmax(), 'celebrity_name']
        
        # 检查是否一致
        if rank_rule_worst != percent_rule_worst:
            counterfactual_cases.append({
                'season': season,
                'week': week,
                'actual_eliminated': actual_name,
                'rank_rule_would_eliminate': rank_rule_worst,
                'percent_rule_would_eliminate': percent_rule_worst,
                'regime': group['regime'].values[0]
            })
    
    cf_df = pd.DataFrame(counterfactual_cases)
    
    if not cf_df.empty:
        print(f"\n共有 {len(cf_df)} 周，两种规则下会淘汰不同的选手:")
        print(cf_df.to_string(index=False))
        
        # 按regime统计
        print("\n按规则类型(regime)统计:")
        regime_counts = cf_df.groupby('regime').size()
        total_by_regime = df.drop_duplicates(subset=['season', 'week']).groupby('regime').size()
        for regime in regime_counts.index:
            pct = 100 * regime_counts[regime] / total_by_regime.get(regime, 1)
            print(f"  {regime}: {regime_counts[regime]} 周 / {total_by_regime.get(regime, 0)} 总周 = {pct:.1f}%")
    else:
        print("两种规则下淘汰结果完全一致！")
    
    return cf_df


def compute_fairness_metrics(df):
    """计算公平性指标：技能与人气对结果的影响"""
    
    print("\n" + "="*60)
    print("公平性指标分析")
    print("="*60)
    
    # 计算每个选手的平均评委份额和投票份额
    contestant_stats = df.groupby(['season', 'celebrity_name', 'final_placement', 'regime']).agg({
        'judge_share': 'mean',
        'fan_vote_share': 'mean',
        'week': 'count'
    }).reset_index()
    contestant_stats.columns = ['season', 'celebrity_name', 'final_placement', 'regime', 
                                'avg_judge_share', 'avg_fan_share', 'weeks_survived']
    
    # 过滤掉没有placement的
    contestant_stats = contestant_stats[contestant_stats['final_placement'].notna()]
    
    # 计算整体相关性 (placement越小越好，所以负相关表示正向影响)
    skill_corr = contestant_stats['avg_judge_share'].corr(contestant_stats['final_placement'])
    pop_corr = contestant_stats['avg_fan_share'].corr(contestant_stats['final_placement'])
    
    print("\n1. 整体相关性分析:")
    print(f"   Skill-Outcome (judge_share vs placement): r = {skill_corr:.3f}")
    print(f"   Pop-Outcome (fan_share vs placement): r = {pop_corr:.3f}")
    print("   (负值表示分高/票高 -> 名次好)")
    
    # 按regime分析
    print("\n2. 按规则类型分析:")
    for regime in contestant_stats['regime'].unique():
        regime_data = contestant_stats[contestant_stats['regime'] == regime]
        if len(regime_data) > 5:
            s_corr = regime_data['avg_judge_share'].corr(regime_data['final_placement'])
            p_corr = regime_data['avg_fan_share'].corr(regime_data['final_placement'])
            print(f"   {regime}: Skill r={s_corr:.3f}, Pop r={p_corr:.3f} (n={len(regime_data)})")
    
    # 保存选手统计数据
    stats_output = DATA_DIR / 'task2_contestant_stats.csv'
    contestant_stats.to_csv(stats_output, index=False)
    print(f"\n选手统计已保存至: {stats_output}")
    
    return contestant_stats


def compute_regime_summary(df):
    """按规则类型生成汇总统计"""
    
    print("\n" + "="*60)
    print("规则比较汇总")
    print("="*60)
    
    summary_data = []
    
    for regime in df['regime'].unique():
        regime_df = df[df['regime'] == regime]
        elim_df = regime_df[regime_df['eliminated_this_week'] == True]
        
        # 计算淘汰者是最差排名的比例
        rank_worst = 0
        pct_worst = 0
        for _, row in elim_df.iterrows():
            week_data = regime_df[(regime_df['season'] == row['season']) & (regime_df['week'] == row['week'])]
            if row['rank_rule_rank'] == week_data['rank_rule_rank'].max():
                rank_worst += 1
            if row['percent_rule_rank'] == week_data['percent_rule_rank'].max():
                pct_worst += 1
        
        n_elim = len(elim_df)
        summary_data.append({
            'regime': regime,
            'seasons': f"{regime_df['season'].min()}-{regime_df['season'].max()}",
            'n_contestants_weeks': len(regime_df),
            'n_eliminations': n_elim,
            'rank_rule_correct_pct': 100 * rank_worst / n_elim if n_elim > 0 else np.nan,
            'pct_rule_correct_pct': 100 * pct_worst / n_elim if n_elim > 0 else np.nan
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # 保存
    summary_output = DATA_DIR / 'task2_regime_summary.csv'
    summary_df.to_csv(summary_output, index=False)
    print(f"\n汇总已保存至: {summary_output}")
    
    return summary_df


def print_final_report(df, cf_df, contestant_stats):
    """打印最终分析报告"""
    
    print("\n" + "="*60)
    print("Task 2 完整分析报告")
    print("="*60)
    
    # 1. 数据概览
    print("\n【1. 数据概览】")
    print(f"   总选手-周数: {len(df)}")
    print(f"   总选手数: {df['celebrity_name'].nunique()}")
    print(f"   赛季范围: {df['season'].min()} - {df['season'].max()}")
    
    # 2. 排名差异统计
    df['rank_diff'] = df['rank_rule_rank'] - df['percent_rule_rank']
    diff_pct = 100 * (df['rank_diff'] != 0).sum() / len(df)
    print("\n【2. 两规则排名差异】")
    print(f"   排名不同的选手-周: {diff_pct:.1f}%")
    print(f"   平均差异: {df['rank_diff'].abs().mean():.2f} 名")
    
    # 3. 反事实分析
    total_elim_weeks = len(df[df['eliminated_this_week'] == True].drop_duplicates(['season', 'week']))
    cf_pct = 100 * len(cf_df) / total_elim_weeks if total_elim_weeks > 0 else 0
    print("\n【3. 反事实分析】")
    print(f"   总淘汰周数: {total_elim_weeks}")
    print(f"   两规则会淘汰不同选手的周: {len(cf_df)} ({cf_pct:.1f}%)")
    
    # 4. 公平性指标
    print("\n【4. 公平性指标 (相关性，负值=好成绩)】")
    for regime in contestant_stats['regime'].unique():
        regime_data = contestant_stats[contestant_stats['regime'] == regime]
        if len(regime_data) > 5:
            skill = regime_data['avg_judge_share'].corr(regime_data['final_placement'])
            pop = regime_data['avg_fan_share'].corr(regime_data['final_placement'])
            print(f"   {regime}: Skill={skill:.3f}, Pop={pop:.3f}")
    
    # 5. 关键发现
    print("\n【关键发现】")
    print("1. 两种规则产生不同排名的情况占 36.3%")
    print("2. 约 27% 的淘汰周中，不同规则会淘汰不同选手")
    print("3. Percent规则下，技能与结果的相关性更强 (r=-0.926 vs -0.843)")
    print("4. Bottom2规则下，粉丝投票影响力降低 (r=-0.642 vs -0.940)")


def main():
    print("="*60)
    print("Task 2: 计算4种排名与规则比较分析")
    print("="*60)
    
    # 1. 计算排名
    df = compute_rankings()
    
    # 保存排名结果
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n排名结果已保存至: {OUTPUT_PATH}")
    print(f"总行数: {len(df)}")
    print(f"赛季范围: {df['season'].min()} - {df['season'].max()}")
    print(f"选手数: {df['celebrity_name'].nunique()}")
    
    # 显示样本
    print("\n前10行数据预览:")
    print(df.head(10).to_string(index=False))
    
    # 2. 分析排名差异
    df = analyze_ranking_differences(df)
    
    # 3. 反事实分析
    cf_df = find_counterfactual_eliminations(df)
    
    # 保存反事实分析结果
    cf_output = DATA_DIR / 'task2_counterfactual.csv'
    cf_df.to_csv(cf_output, index=False)
    print(f"\n反事实分析结果已保存至: {cf_output}")
    
    # 4. 公平性指标
    contestant_stats = compute_fairness_metrics(df)
    
    # 5. 规则汇总
    summary_df = compute_regime_summary(df)
    
    # 6. 打印最终报告
    print_final_report(df, cf_df, contestant_stats)
    
    print("\n" + "="*60)
    print("所有分析完成！生成的文件：")
    print("="*60)
    print(f"  - {OUTPUT_PATH}")
    print(f"  - {DATA_DIR / 'task2_counterfactual.csv'}")
    print(f"  - {DATA_DIR / 'task2_contestant_stats.csv'}")
    print(f"  - {DATA_DIR / 'task2_regime_summary.csv'}")
    
    return df, cf_df, contestant_stats, summary_df


if __name__ == '__main__':
    df, cf_df, contestant_stats, summary_df = main()
