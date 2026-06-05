"""
IntelliCrash — Exploratory Data Analysis (EDA) Script.

Generates comprehensive analysis and plots for the IMU dataset:
1. Dataset overview & statistics
2. Label distribution analysis
3. Time-series signal plots (normal vs crash)
4. Acceleration distribution histograms
5. Feature correlation analysis
6. Class imbalance visualization

Outputs all plots to outputs/plots/ directory.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.config import get_config
from src.data.preprocess_imu import (
    load_imu_data, clean_imu_data,
    map_label_to_binary, get_event_category
)

# ── Style Setup ──────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams.update({
    "figure.figsize": (14, 8),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

COLORS = {
    "normal": "#2ecc71",
    "maneuver": "#3498db",
    "crash": "#e74c3c",
    "primary": "#2c3e50",
    "accent": "#e67e22",
}


def ensure_output_dir():
    """Create output directory if needed."""
    cfg = get_config()
    out = Path(cfg["paths"]["plots_dir"])
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_dataset_overview(df: pd.DataFrame, out_dir: Path):
    """Plot 1: Dataset overview — shape, dtypes, basic stats."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("IntelliCrash — Dataset Overview (df.csv)", fontsize=16, fontweight="bold")

    # 1a. Data types
    dtype_counts = df.dtypes.value_counts()
    axes[0].barh(dtype_counts.index.astype(str), dtype_counts.values,
                 color=COLORS["primary"])
    axes[0].set_title("Column Data Types")
    axes[0].set_xlabel("Count")

    # 1b. Missing values
    nan_per_col = df.isna().sum()
    axes[1].barh(nan_per_col.index, nan_per_col.values, color=COLORS["accent"])
    axes[1].set_title("Missing Values per Column")
    axes[1].set_xlabel("NaN Count")

    # 1c. Basic statistics text
    stats_text = (
        f"Total Rows: {len(df):,}\n"
        f"Total Columns: {len(df.columns)}\n"
        f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB\n"
        f"Time Range: {df['seconds_elapsed'].min():.1f}s "
        f"to {df['seconds_elapsed'].max():.1f}s\n"
        f"Duration: {(df['seconds_elapsed'].max() - df['seconds_elapsed'].min()) / 3600:.2f} hrs\n"
        f"Unique Labels: {df['label'].nunique()}\n"
        f"Sampling Rate: ~{1.0 / df['seconds_elapsed'].diff().median():.0f} Hz"
    )
    axes[2].text(0.1, 0.5, stats_text, transform=axes[2].transAxes,
                 fontsize=13, verticalalignment="center", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.8))
    axes[2].set_title("Dataset Statistics")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "01_dataset_overview.png", bbox_inches="tight")
    plt.close()
    print("  [1/7] Dataset overview")


def plot_label_distribution(df: pd.DataFrame, out_dir: Path):
    """Plot 2: Label distribution — all 74 labels + binary + category."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("IntelliCrash — Label Distribution Analysis",
                 fontsize=16, fontweight="bold")

    # 2a. Top 20 raw labels
    top_labels = df["label"].value_counts().head(20)
    colors = [COLORS[get_event_category(l)] for l in top_labels.index]
    axes[0].barh(top_labels.index[::-1], top_labels.values[::-1],
                 color=colors[::-1])
    axes[0].set_title("Top 20 Event Labels")
    axes[0].set_xlabel("Count")

    # 2b. Binary crash vs non-crash
    binary_counts = df["crash_label"].value_counts().reindex([0, 1], fill_value=0)
    pie_labels = []
    pie_values = []
    pie_colors = []
    pie_explode = []
    if binary_counts[0] > 0:
        pie_labels.append("Non-Crash")
        pie_values.append(binary_counts[0])
        pie_colors.append(COLORS["normal"])
        pie_explode.append(0)
    if binary_counts[1] > 0:
        pie_labels.append("Crash")
        pie_values.append(binary_counts[1])
        pie_colors.append(COLORS["crash"])
        pie_explode.append(0.05)

    if len(pie_values) > 0:
        wedges, texts, autotexts = axes[1].pie(
            pie_values, labels=pie_labels, colors=pie_colors,
            autopct="%1.1f%%", startangle=90, explode=pie_explode,
            textprops={"fontsize": 12}
        )
    else:
        axes[1].text(0.5, 0.5, "No data", transform=axes[1].transAxes, ha="center")
    axes[1].set_title("Binary Classification Split")

    # 2c. Event category distribution
    cat_counts = df["event_category"].value_counts()
    cat_colors = [COLORS.get(c, "#95a5a6") for c in cat_counts.index]
    axes[2].bar(cat_counts.index, cat_counts.values, color=cat_colors)
    axes[2].set_title("Event Categories")
    axes[2].set_ylabel("Count")
    for i, (cat, count) in enumerate(cat_counts.items()):
        axes[2].text(i, count + len(df) * 0.01, f"{count:,}",
                     ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_dir / "02_label_distribution.png", bbox_inches="tight")
    plt.close()
    print("  [2/7] Label distribution")


def plot_signal_timeseries(df: pd.DataFrame, out_dir: Path):
    """Plot 3: Time-series signals — accele_x, accele_y, gyro_z for different events."""
    fig, axes = plt.subplots(3, 3, figsize=(20, 12))
    fig.suptitle("IntelliCrash — Sensor Signals by Event Type",
                 fontsize=16, fontweight="bold")

    channels = ["accele_x", "accele_y", "gyro_z"]
    categories = ["normal", "maneuver", "crash"]

    for col_idx, category in enumerate(categories):
        cat_df = df[df["event_category"] == category]

        if len(cat_df) == 0:
            for row_idx in range(3):
                axes[row_idx, col_idx].text(0.5, 0.5, "No data",
                    transform=axes[row_idx, col_idx].transAxes, ha="center")
            continue

        # Take a representative 5-second segment
        sample_start = cat_df.index[min(1000, len(cat_df) // 4)]
        sample_end = sample_start + 500  # 5 sec @ 100Hz
        segment = df.iloc[sample_start:sample_end]

        for row_idx, channel in enumerate(channels):
            ax = axes[row_idx, col_idx]
            time = segment["seconds_elapsed"] - segment["seconds_elapsed"].iloc[0]
            ax.plot(time, segment[channel],
                    color=COLORS[category], linewidth=0.8, alpha=0.8)
            ax.fill_between(time, segment[channel], alpha=0.15,
                           color=COLORS[category])

            if col_idx == 0:
                ax.set_ylabel(channel, fontsize=11)
            if row_idx == 0:
                ax.set_title(f"{category.upper()}", fontsize=13,
                           color=COLORS[category], fontweight="bold")
            if row_idx == 2:
                ax.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(out_dir / "03_signal_timeseries.png", bbox_inches="tight")
    plt.close()
    print("  [3/7] Signal time-series")


def plot_acceleration_distributions(df: pd.DataFrame, out_dir: Path):
    """Plot 4: Acceleration/gyro distributions, normal vs crash."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("IntelliCrash — Sensor Value Distributions (Normal vs Crash)",
                 fontsize=16, fontweight="bold")

    channels = ["accele_x", "accele_y", "gyro_z",
                "accele_x_filtered", "accele_y_filtered", "gyro_z_filtered"]

    for idx, channel in enumerate(channels):
        ax = axes[idx // 3, idx % 3]

        normal = df[df["crash_label"] == 0][channel]
        crash = df[df["crash_label"] == 1][channel]

        # Subsample for KDE performance
        n_sample = min(50000, len(normal))
        normal_sample = normal.sample(n=n_sample, random_state=42) if len(normal) > n_sample else normal

        ax.hist(normal_sample, bins=80, alpha=0.5, density=True,
                color=COLORS["normal"], label="Non-Crash")
        if len(crash) > 0:
            ax.hist(crash, bins=80, alpha=0.5, density=True,
                    color=COLORS["crash"], label="Crash")

        ax.set_title(channel)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "04_sensor_distributions.png", bbox_inches="tight")
    plt.close()
    print("  [4/7] Sensor distributions")


def plot_correlation_matrix(df: pd.DataFrame, out_dir: Path):
    """Plot 5: Correlation matrix of sensor channels."""
    fig, ax = plt.subplots(figsize=(10, 8))

    sensor_cols = ["accele_x", "accele_y", "gyro_z",
                   "accele_x_filtered", "accele_y_filtered", "gyro_z_filtered"]

    corr = df[sensor_cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("IntelliCrash — Sensor Channel Correlation Matrix",
                 fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_dir / "05_correlation_matrix.png", bbox_inches="tight")
    plt.close()
    print("  [5/7] Correlation matrix")


def plot_class_imbalance(df: pd.DataFrame, out_dir: Path):
    """Plot 6: Class imbalance analysis with window-level projections."""
    cfg = get_config()
    window_size = cfg["imu"]["window_size_samples"]
    stride = cfg["imu"]["window_stride_samples"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("IntelliCrash — Class Imbalance Analysis",
                 fontsize=16, fontweight="bold")

    # 6a. Sample-level imbalance
    counts = df["crash_label"].value_counts().reindex([0, 1], fill_value=0)
    ratio_str = f"{counts[0] / counts[1]:.0f}:1" if counts[1] > 0 else "All Non-Crash"

    bar_vals = [counts[0], counts[1]]
    bars = axes[0].bar(["Non-Crash", "Crash"], bar_vals,
                       color=[COLORS["normal"], COLORS["crash"]])
    axes[0].set_title(f"Sample-Level Imbalance\n(Ratio: {ratio_str})")
    axes[0].set_ylabel("Count")
    for bar, val in zip(bars, bar_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:,}", ha="center", va="bottom", fontsize=11)

    # 6b. Projected window-level counts
    est_total = (len(df) - window_size) // stride + 1
    crash_samples = counts.get(1, 0)
    est_crash = int((crash_samples / len(df)) * est_total * 1.5)  # overlap boost
    est_normal = est_total - est_crash

    bars2 = axes[1].bar(["Non-Crash Windows", "Crash Windows"],
                        [est_normal, est_crash],
                        color=[COLORS["normal"], COLORS["crash"]])
    axes[1].set_title(f"Estimated Window-Level Split\n"
                      f"({est_total:,} total windows, "
                      f"window={window_size}, stride={stride})")
    axes[1].set_ylabel("Estimated Count")
    for bar, val in zip(bars2, [est_normal, est_crash]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:,}", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_dir / "06_class_imbalance.png", bbox_inches="tight")
    plt.close()
    print("  [6/7] Class imbalance")


def plot_temporal_overview(df: pd.DataFrame, out_dir: Path):
    """Plot 7: Temporal overview — event distribution over time."""
    fig, axes = plt.subplots(2, 1, figsize=(18, 8))
    fig.suptitle("IntelliCrash — Temporal Distribution of Events",
                 fontsize=16, fontweight="bold")

    time = df["seconds_elapsed"]

    # 7a. Acceleration magnitude over time (subsampled)
    subsample = max(1, len(df) // 10000)
    t_sub = time.iloc[::subsample]
    mag = np.sqrt(df["accele_x"].iloc[::subsample]**2 +
                  df["accele_y"].iloc[::subsample]**2)

    axes[0].plot(t_sub, mag, linewidth=0.3, color=COLORS["primary"], alpha=0.6)
    axes[0].set_ylabel("Resultant Accel (g)")
    axes[0].set_title("Acceleration Magnitude Over Time")

    # Highlight crash regions
    crash_mask = df["crash_label"].iloc[::subsample] == 1
    if crash_mask.any():
        axes[0].scatter(t_sub[crash_mask], mag[crash_mask],
                       s=2, color=COLORS["crash"], alpha=0.5, label="Crash")
        axes[0].legend()

    # 7b. Event category timeline
    categories = df["event_category"].iloc[::subsample]
    cat_map = {"normal": 0, "maneuver": 1, "crash": 2}
    cat_numeric = categories.map(cat_map)

    scatter_colors = [COLORS.get(c, "#95a5a6") for c in categories]
    axes[1].scatter(t_sub, cat_numeric, c=scatter_colors, s=0.5, alpha=0.3)
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(["Normal", "Maneuver", "Crash"])
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_title("Event Category Over Time")

    plt.tight_layout()
    plt.savefig(out_dir / "07_temporal_overview.png", bbox_inches="tight")
    plt.close()
    print("  [7/7] Temporal overview")


# ── Main ─────────────────────────────────────────────────────────────────────

def run_eda(nrows: int = None):
    """Run the full EDA pipeline.

    Args:
        nrows: Optional row limit for faster testing.
    """
    out_dir = ensure_output_dir()
    print(f"Output directory: {out_dir}")

    # Load & clean
    print("\nLoading data...")
    df = load_imu_data(nrows=nrows)
    df = clean_imu_data(df)

    print(f"\nGenerating EDA plots to {out_dir}/...")
    plot_dataset_overview(df, out_dir)
    plot_label_distribution(df, out_dir)
    plot_signal_timeseries(df, out_dir)
    plot_acceleration_distributions(df, out_dir)
    plot_correlation_matrix(df, out_dir)
    plot_class_imbalance(df, out_dir)
    plot_temporal_overview(df, out_dir)

    print(f"\nEDA complete! 7 plots saved to {out_dir}/")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run IntelliCrash EDA")
    parser.add_argument("--nrows", type=int, default=None,
                        help="Limit rows for faster testing")
    args = parser.parse_args()

    run_eda(nrows=args.nrows)
