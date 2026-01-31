import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
FIG_DIR = ROOT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


COLORS = {
    "rank": "#1f77b4",     # blue
    "percent": "#2ca02c",  # green
    "bottom2": "#ff7f0e",  # orange
}

REGIME_LABELS = {
    "rank": "Rank Regime",
    "percent": "Percent Regime",
    "bottom2": "Bottom-2 Regime",
}


def apply_style():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 100,
    })


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)


def add_regime_legend(ax, include_bottom2=True):
    handles = [
        Patch(facecolor=COLORS["rank"], label=REGIME_LABELS["rank"]),
        Patch(facecolor=COLORS["percent"], label=REGIME_LABELS["percent"]),
    ]
    if include_bottom2:
        handles.append(Patch(facecolor=COLORS["bottom2"], label=REGIME_LABELS["bottom2"]))
    ncol = 3 if include_bottom2 else 2
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=ncol)


def catmull_rom_spline(x, y, points_per_segment=30):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return x, y
    if len(x) < 3:
        return x, y

    xs = []
    ys = []
    for i in range(len(x) - 1):
        p0x, p0y = (x[i - 1], y[i - 1]) if i > 0 else (x[i], y[i])
        p1x, p1y = x[i], y[i]
        p2x, p2y = x[i + 1], y[i + 1]
        p3x, p3y = (x[i + 2], y[i + 2]) if i + 2 < len(x) else (x[i + 1], y[i + 1])

        ts = np.linspace(0, 1, points_per_segment, endpoint=False)
        for t in ts:
            t2 = t * t
            t3 = t2 * t
            cx = 0.5 * (
                2 * p1x
                + (-p0x + p2x) * t
                + (2 * p0x - 5 * p1x + 4 * p2x - p3x) * t2
                + (-p0x + 3 * p1x - 3 * p2x + p3x) * t3
            )
            cy = 0.5 * (
                2 * p1y
                + (-p0y + p2y) * t
                + (2 * p0y - 5 * p1y + 4 * p2y - p3y) * t2
                + (-p0y + 3 * p1y - 3 * p2y + p3y) * t3
            )
            xs.append(cx)
            ys.append(cy)

    xs.append(x[-1])
    ys.append(y[-1])

    ys = np.clip(ys, np.nanmin(y), np.nanmax(y))
    return np.asarray(xs), np.asarray(ys)


def plot_smoothed_series(ax, x, y, color, label=None, marker=True, zorder=3):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return

    segments = []
    start = 0
    for i in range(1, len(x)):
        if x[i] - x[i - 1] > 1.01:
            segments.append((x[start:i], y[start:i]))
            start = i
    segments.append((x[start:], y[start:]))

    first = True
    for xs, ys in segments:
        if len(xs) < 2:
            continue
        smooth_x, smooth_y = catmull_rom_spline(xs, ys, points_per_segment=30)
        ax.plot(
            smooth_x,
            smooth_y,
            color=color,
            linewidth=1.8,
            label=label if first else None,
            zorder=zorder,
        )
        first = False

    if marker:
        ax.scatter(x, y, color=color, s=22, zorder=zorder + 1, edgecolors="none")


def plot_consistency_by_season():
    df = pd.read_csv(DATA_DIR / "table_consistency_by_season.csv")
    df = df.sort_values("season")
    df["season"] = df["season"].astype(int)
    df["consistency_prob"] = pd.to_numeric(df["consistency_prob"], errors="coerce")

    colors = df["regime"].map(COLORS)

    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    ax.bar(df["season"], df["consistency_prob"], color=colors, edgecolor="none", zorder=1)
    plot_smoothed_series(
        ax,
        df["season"],
        df["consistency_prob"],
        color="#222222",
        label="Consistency",
        marker=True,
        zorder=3,
    )
    ax.set_title("Consistency of Fan Vote Inference Across Seasons")
    ax.set_xlabel("Season")
    ax.set_ylabel("Consistency Probability")
    ax.set_xlim(0.5, 34.5)
    ax.set_ylim(0, max(1.08, df["consistency_prob"].max() + 0.08))
    ax.set_xticks(list(range(1, 35, 2)))

    ax.axvline(2.5, color="#aaaaaa", linestyle=":", linewidth=1)
    ax.axvline(27.5, color="#aaaaaa", linestyle=":", linewidth=1)
    y_top = ax.get_ylim()[1]
    ax.text(1.5, y_top * 0.97, "Rank", ha="center", va="top", fontsize=10)
    ax.text(15, y_top * 0.97, "Percent", ha="center", va="top", fontsize=10)
    ax.text(31, y_top * 0.97, "Bottom-2", ha="center", va="top", fontsize=10)

    style_axes(ax)
    add_regime_legend(ax)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "consistency_by_season.png", dpi=600)


def plot_consistency_by_week():
    df = pd.read_csv(DATA_DIR / "table_consistency_by_week.csv")
    for col in ["rank", "percent", "bottom2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for key in ["rank", "percent", "bottom2"]:
        series = df[key]
        mask = series.notna()
        plot_smoothed_series(
            ax,
            df.loc[mask, "week"],
            series[mask],
            color=COLORS[key],
            label=REGIME_LABELS[key],
            marker=True,
            zorder=3,
        )

    ax.set_title("Weekly Consistency Trends by Regime")
    ax.set_xlabel("Week Number")
    ax.set_ylabel("Average Consistency Probability")
    ax.set_xlim(0.5, 11.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(1, 12))

    style_axes(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "consistency_by_week.png", dpi=600)


def plot_uncertainty_by_season():
    df = pd.read_csv(DATA_DIR / "table_uncertainty_by_season.csv")
    df = df.sort_values("season")
    df["season"] = df["season"].astype(int)
    df["uncertainty_avg_cv"] = pd.to_numeric(df["uncertainty_avg_cv"], errors="coerce")
    df = df[df["regime"] != "bottom2"]
    colors = df["regime"].map(COLORS)

    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    ax.bar(df["season"], df["uncertainty_avg_cv"], color=colors, edgecolor="none", zorder=1)
    plot_smoothed_series(
        ax,
        df["season"],
        df["uncertainty_avg_cv"],
        color="#222222",
        label="Uncertainty",
        marker=True,
        zorder=3,
    )
    ax.set_title("Model Uncertainty (CV) Across Seasons")
    ax.set_xlabel("Season")
    ax.set_ylabel("Average Coefficient of Variation (CV)")
    ax.set_xlim(0.5, 27.5)
    ax.set_xticks(list(range(1, 28, 2)))

    # No numeric arrow annotations for this figure (per request).

    ax.axvline(2.5, color="#aaaaaa", linestyle=":", linewidth=1)
    y_top = ax.get_ylim()[1]
    ax.text(1.5, y_top * 0.96, "Rank", ha="center", va="top", fontsize=10)
    ax.text(15, y_top * 0.96, "Percent", ha="center", va="top", fontsize=10)

    style_axes(ax)
    add_regime_legend(ax, include_bottom2=False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "uncertainty_by_season.png", dpi=600)


def plot_uncertainty_by_week():
    df = pd.read_csv(DATA_DIR / "table_uncertainty_by_week.csv")
    for col in ["rank", "percent", "bottom2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for key in ["rank", "percent", "bottom2"]:
        series = df[key]
        mask = series.notna()
        plot_smoothed_series(
            ax,
            df.loc[mask, "week"],
            series[mask],
            color=COLORS[key],
            label=REGIME_LABELS[key],
            marker=True,
            zorder=3,
        )

    ax.set_title("Evolution of Inference Uncertainty Over Weeks")
    ax.set_xlabel("Week Number")
    ax.set_ylabel("Average Coefficient of Variation (CV)")
    ax.set_xlim(0.5, 11.5)
    ax.set_xticks(range(1, 12))

    max_y = max(df[["rank", "percent", "bottom2"]].max())
    ax.set_ylim(0, max_y * 1.2)

    style_axes(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "uncertainty_by_week.png", dpi=600)


def main():
    apply_style()
    plot_consistency_by_season()
    plot_consistency_by_week()
    plot_uncertainty_by_season()
    plot_uncertainty_by_week()


if __name__ == "__main__":
    main()
