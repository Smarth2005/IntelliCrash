"""
IntelliCrash — Exploratory Data Analysis (EDA) Script.

Generates comprehensive, publication-ready analysis and plots for the IMU dataset:
1. Time-Series Waveforms (Normal vs Hard Braking vs Crash)
2. Class Distribution Bar Chart (Raw vs Augmented)
3. Feature Distribution Boxplots (Delta-V, Peak G-Force)
4. Correlation Heatmap (18 engineered features)

Outputs all plots to outputs/plots/ directory.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import get_config
from src.data.preprocess_imu import load_imu_data, clean_imu_data
from src.features.feature_engineering import FEATURE_NAMES

# ── Premium Academic Style Setup ──────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
# A professional, high-contrast palette for academic papers
COLORS = {
    "normal": "#2ecc71",       # Green
    "lane_weave": "#3498db",   # Blue
    "lane_swerve": "#9b59b6",  # Purple
    "hard_brake": "#f1c40f",   # Yellow
    "hard_corner": "#e67e22",  # Orange
    "u_turn": "#1abc9c",       # Teal
    "crash": "#e74c3c",        # Red
    "primary": "#2c3e50",      # Dark Blue
}

plt.rcParams.update({
    "figure.figsize": (12, 8),
    "figure.dpi": 300,  # High resolution for papers
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "axes.titleweight": "bold",
    "legend.fontsize": 11,
})

def ensure_output_dir():
    cfg = get_config()
    out = Path(cfg["paths"]["plots_dir"]) / "eda"
    out.mkdir(parents=True, exist_ok=True)
    return out

def get_augmented_df():
    """Load the fully augmented dataset (windows & features) to plot real class distributions."""
    cfg = get_config()
    data_dir = Path(cfg["paths"]["windows_dir"])
    try:
        meta_df = pd.read_parquet(data_dir / "imu_augmented_metadata.parquet")
        # Load features if needed
        # features = np.load(Path(cfg["paths"]["processed_dir"]) / "imu_augmented_features.npy")
        return meta_df
    except Exception as e:
        print(f"Could not load augmented data: {e}")
        return None

def plot_waveform_comparison(df: pd.DataFrame, out_dir: Path):
    """Plot 1: Side-by-side Time-Series Waveforms (Normal vs Maneuver vs Crash)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle("IMU Sensor Waveform Comparison: Acceleration Profiles", fontsize=18, y=1.05)
    
    # We will pick representative 2-second windows from the raw time-series df.
    # To do this robustly, we just find segments where normal, maneuver, and crash happen.
    categories = {"Normal Driving": "normal", "Hard Braking / Maneuver": "maneuver", "Crash Impact": "crash"}
    
    from src.data.synthetic_crashes import generate_impact_pulse
    
    for ax, (title, cat) in zip(axes, categories.items()):
        cat_df = df[df["event_category"] == cat]
        
        # Determine if we plot from raw DF or generate synthetic for missing crash data
        if len(cat_df) > 200:
            # Pick a 2-second window (200 samples)
            start_idx = len(cat_df) // 2
            segment_x = cat_df.iloc[start_idx : start_idx + 200]["accele_x"].values
            segment_y = cat_df.iloc[start_idx : start_idx + 200]["accele_y"].values
        elif cat == "crash":
            # Dataset has 0 real crashes, so we visualize the synthetic physics-based crash pulse
            synthetic_crash = generate_impact_pulse(window_size=200, peak_g=8.0)
            segment_x = synthetic_crash[:, 0]  # Accel X
            segment_y = synthetic_crash[:, 1]  # Accel Y
        else:
            continue # Skip if no data and not crash
            
        time = np.linspace(0, 2.0, 200)
        
        ax.plot(time, segment_x, label="Accel X (Forward)", color=COLORS["primary"], linewidth=2, alpha=0.8)
        ax.plot(time, segment_y, label="Accel Y (Lateral)", color=COLORS["lane_weave"], linewidth=2, alpha=0.8)
        
        # Highlight area under curve
        ax.fill_between(time, segment_x, alpha=0.1, color=COLORS["primary"])
        ax.fill_between(time, segment_y, alpha=0.1, color=COLORS["lane_weave"])
        
        ax.set_title(title, color=COLORS["crash"] if cat == "crash" else COLORS["primary"])
        ax.set_xlabel("Time (seconds)")
        if ax == axes[0]:
            ax.set_ylabel("Acceleration (g)")
            ax.legend(loc="upper right")
        
        # Add gridlines
            ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_dir / "01_waveform_comparison.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  [1/4] Waveform Comparison generated.")

def plot_class_distribution(raw_df: pd.DataFrame, out_dir: Path):
    """Plot 2: Class Distribution Bar Chart showing raw vs. augmented counts."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = ["Normal", "Lane Weaving", "Lane Swerving", "Hard Braking", "Hard Cornering", "Quick U-turn", "Crash"]
    class_ids = [0, 1, 2, 3, 4, 5, 6]
    
    # Raw counts from labels (heuristic mapping)
    label_map = {
        "normal": 0, "sshape": 1, "weave": 1, "lane": 2, "swerve": 2,
        "brakes": 3, "corner": 4, "u-turn": 5, "uturn": 5,
        "crash": 6, "aggressive": 6, "collision": 6
    }
    
    raw_counts = [0] * 7
    if "label" in raw_df.columns:
        for lbl in raw_df["label"].dropna():
            lbl_lower = str(lbl).lower()
            for key, class_id in label_map.items():
                if key in lbl_lower:
                    raw_counts[class_id] += 1
                    break
            else:
                raw_counts[0] += 1
    
    # Augmented counts: The synthetic crash generator outputs a binary y.npy array
    # where 1 = Crash. Since we only synthetically augment crashes, the other 
    # maneuver classes remain identical to the raw dataset.
    aug_counts = list(raw_counts)
    cfg = get_config()
    try:
        y_aug = np.load(Path(cfg["paths"]["windows_dir"]) / "imu_augmented_y_train.npy")
        # Summing the binary array gives the exact number of augmented crashes
        aug_counts[6] = int(y_aug.sum())
    except FileNotFoundError:
        try:
            # Fallback if train split wasn't explicitly saved
            y_aug = np.load(Path(cfg["paths"]["windows_dir"]) / "imu_augmented_y.npy")
            aug_counts[6] = int(y_aug.sum())
        except FileNotFoundError:
            pass # Keep as raw counts if augmented data isn't generated yet
    
    x = np.arange(len(classes))
    width = 0.35

    rects1 = ax.bar(x - width/2, raw_counts, width, label='Raw Dataset', color="#95a5a6", alpha=0.8)
    rects2 = ax.bar(x + width/2, aug_counts, width, label='After Synthetic Augmentation', color=COLORS["primary"])

    ax.set_ylabel('Number of 2-Second Windows', fontweight="bold")
    ax.set_title('Class Imbalance Resolution: Raw vs Augmented Dataset', fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    # Use log scale only if max count > 0
    if max(raw_counts + aug_counts) > 0:
        ax.set_yscale("log")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "02_class_distribution.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  [2/4] Class Distribution generated.")

def plot_feature_boxplots(out_dir: Path):
    """Plot 3: Feature Distribution Boxplots for Delta-V and Peak G-Force."""
    # We need the unified features csv or augmented metadata.
    cfg = get_config()
    try:
        # Load the final unified dataset
        df = pd.read_csv("IntelliCrash_Dataset.csv")
    except Exception as e:
        print(f"  [3/4] Skipping boxplots: IntelliCrash_Dataset.csv not found. ({e})")
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Physics Feature Separability Across Driving Maneuvers", fontsize=18, fontweight="bold")
    
    if "event_type" in df.columns:
        x_col = "event_type"
    else:
        x_col = "crash_label"
        
    sns.boxplot(ax=axes[0], x=x_col, y="delta_v", data=df, palette="husl", showfliers=False)
    axes[0].set_title("Delta-V (Change in Velocity) Distribution", fontsize=14)
    axes[0].set_xlabel("Driving Pattern Class")
    axes[0].set_ylabel("Delta-V (km/h)")
    
    sns.boxplot(ax=axes[1], x=x_col, y="peak_accel", data=df, palette="husl", showfliers=False)
    axes[1].set_title("Peak Acceleration (G-Force) Distribution", fontsize=14)
    axes[1].set_xlabel("Driving Pattern Class")
    axes[1].set_ylabel("Peak G-Force (g)")
    
    plt.tight_layout()
    plt.savefig(out_dir / "03_feature_boxplots.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  [3/4] Feature Boxplots generated.")

def plot_correlation_heatmap(out_dir: Path):
    """Plot 4: Correlation Heatmap for the 26 engineered physics/temporal features."""
    cfg = get_config()
    try:
        df = pd.read_csv("IntelliCrash_Dataset.csv")
    except Exception as e:
        print(f"  [4/4] Skipping heatmap: IntelliCrash_Dataset.csv not found. ({e})")
        return
        
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get just the 18 features
    feature_cols = [c for c in df.columns if c in FEATURE_NAMES]
    if not feature_cols:
        return
        
    corr = df[feature_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .7}, annot=False)
                
    ax.set_title("Pearson Correlation of Engineered Physics Features", fontsize=16, fontweight="bold", pad=20)
    
    # Fix x-axis and y-axis label clipping
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment='center', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(out_dir / "04_correlation_heatmap.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  [4/4] Correlation Heatmap generated.")

def run_eda(nrows: int = None):
    out_dir = ensure_output_dir()
    print(f"Output directory: {out_dir}")

    print("\nLoading raw data for time-series plots...")
    raw_df = load_imu_data(nrows=nrows)
    raw_df = clean_imu_data(raw_df)

    print(f"\nGenerating Publication-Ready EDA plots to {out_dir}/...")
    plot_waveform_comparison(raw_df, out_dir)
    plot_class_distribution(raw_df, out_dir)
    plot_feature_boxplots(out_dir)
    plot_correlation_heatmap(out_dir)

    print(f"\nEDA complete! All plots saved to {out_dir}/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run IntelliCrash EDA")
    parser.add_argument("--nrows", type=int, default=None, help="Limit rows for testing")
    args = parser.parse_args()
    run_eda(nrows=args.nrows)
