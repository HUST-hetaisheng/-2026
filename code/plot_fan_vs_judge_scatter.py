import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


DATA_FAN = Path(r"D:\2026-repo\data\fan_vote_results_final.csv")
DATA_RAW = Path(r"D:\2026-repo\data\2026_MCM_Problem_C_Data.csv")
OUT_DIR = Path(r"D:\2026-repo\figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

weeks = list(range(1, 12))


def build_judge_weekly(raw):
    frames = []
    for w in weeks:
        cols = [c for c in raw.columns if c.startswith(f"week{w}_judge")]
        if not cols:
            continue
        scores = raw[cols]
        count = scores.notna().sum(axis=1)
        total = scores.sum(axis=1, skipna=True)
        mean = total / count.replace(0, np.nan)
        df_w = raw[["season", "celebrity_name"]].copy()
        df_w["week"] = w
        df_w["judge_total"] = total
        df_w["judge_mean"] = mean
        frames.append(df_w)
    all_w = pd.concat(frames, ignore_index=True)
    all_w = all_w[all_w["judge_total"] > 0]
    return all_w


def main():
    fan = pd.read_csv(DATA_FAN)
    raw = pd.read_csv(DATA_RAW)

    judge_weekly = build_judge_weekly(raw)
    df = fan.merge(judge_weekly, on=["season", "week", "celebrity_name"], how="left", suffixes=("_fan", "_raw"))

    # compute per-week ranks and shares
    df["fan_share"] = df["fan_vote_share"]
    # choose judge_total from raw if available; fallback to fan table
    if "judge_total_raw" in df.columns:
        df["judge_total_use"] = df["judge_total_raw"].fillna(df.get("judge_total_fan"))
    else:
        df["judge_total_use"] = df.get("judge_total")

    df["judge_share"] = df["judge_total_use"] / df.groupby(["season", "week"])["judge_total_use"].transform("sum")
    df["fan_rank"] = df.groupby(["season", "week"])["fan_share"].rank(method="dense", ascending=False)
    df["judge_rank"] = df.groupby(["season", "week"])["judge_mean"].rank(method="dense", ascending=False)

    # style
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.edgecolor": "#333333",
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    alive = df[df["eliminated_this_week"] == False].copy()
    elim = df[df["eliminated_this_week"] == True].copy()

    # Remove points that overlap across groups (same coordinates)
    # Separate copies for rank plot (we may filter/jitter there only)
    alive_rank = alive.copy()
    elim_rank = elim.copy()

    alive_coords = list(zip(alive_rank["judge_rank"], alive_rank["fan_rank"]))
    elim_coords = list(zip(elim_rank["judge_rank"], elim_rank["fan_rank"]))
    overlap = set(alive_coords) & set(elim_coords)
    if overlap:
        # keep survived points, drop eliminated ones that overlap in rank space
        elim_rank = elim_rank[~pd.Series(elim_coords).isin(overlap).values]

    # jitter points near diagonal to reduce overplotting (visual only)
    def jitter_near_diag(df_in, col_x, col_y, eps=0.5, jitter=0.15):
        df_out = df_in.copy()
        near = (df_out[col_y] - df_out[col_x]).abs() <= eps
        # alternate jitter directions to avoid systematic shift
        signs = np.where((df_out.index % 2) == 0, 1.0, -1.0)
        df_out.loc[near, col_y] = df_out.loc[near, col_y] + signs[near] * jitter
        return df_out

    alive_plot = jitter_near_diag(alive_rank, "judge_rank", "fan_rank")
    elim_plot = jitter_near_diag(elim_rank, "judge_rank", "fan_rank")

    # Figure 1: rank vs rank
    fig, ax = plt.subplots(figsize=(8.6, 5.8), dpi=200)
    ax.scatter(
        alive_plot["judge_rank"], alive_plot["fan_rank"],
        s=16, facecolors="none", edgecolors="#2b6cb0",
        alpha=0.35, linewidths=0.8, marker="o", label="Survived"
    )
    ax.scatter(
        elim_plot["judge_rank"], elim_plot["fan_rank"],
        s=36, c="#d62728", alpha=0.9, marker="X",
        linewidths=0.8, label="Eliminated", zorder=3
    )
    # diagonal y=x reference
    max_rank = int(max(df["judge_rank"].max(), df["fan_rank"].max()))
    ax.plot([1, max_rank], [1, max_rank], color="#888888", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("Judge Mean Score Rank (1 = Best)")
    ax.set_ylabel("Fan Vote Rank (1 = Best)")
    ax.set_title("Fan Rank vs. Judge Mean Rank")
    ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0.5, max_rank + 0.5)
    ax.set_ylim(0.5, max_rank + 0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fan_vs_judge_rank_scatter.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fan_vs_judge_rank_scatter.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: share vs share (full scatter, no filtering)
    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=200)
    # Use full (unfiltered) sets for share plot to avoid dropping any eliminated points
    ax.scatter(
        alive["judge_share"], alive["fan_share"],
        s=16, facecolors="none", edgecolors="#2b6cb0",
        alpha=0.30, linewidths=0.8, marker="o", label="Survived"
    )
    ax.scatter(
        elim["judge_share"], elim["fan_share"],
        s=36, c="#d62728", alpha=0.9, marker="X",
        linewidths=0.8, label="Eliminated", zorder=3
    )
    ax.plot([0, 0.5], [0, 0.5], color="#999999", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Judge Score Share")
    ax.set_ylabel("Fan Vote Share")
    ax.set_title("Fan Share vs. Judge Share")
    ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fan_vs_judge_share_scatter.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fan_vs_judge_share_scatter.pdf", bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(OUT_DIR / "fan_vs_judge_rank_scatter.png")
    print(OUT_DIR / "fan_vs_judge_share_scatter.png")


if __name__ == "__main__":
    main()
