"""
IntelliCrash Hardware — Data Preprocessing Pipeline
=====================================================
Merges MPU6050 sensor CSVs, applies Butterworth low-pass filter,
creates sliding windows, extracts 26 physics features (same as software),
and computes CSI scores using rc_buggy mode.

Run locally:   python hardware/data_pipeline.py
Run in Colab:  !python hardware/data_pipeline.py

Output saved to hardware/data/
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from pathlib import Path
from tqdm import tqdm
import sys
import os
import random

# ── Path Setup (works both locally, in Colab, and as standalone folder) ──
HARDWARE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = HARDWARE_DIR.parent
sys.path.insert(0, str(HARDWARE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# Import feature engineering (local or project-level)
try:
    from feature_engineering import (
        extract_features_batch,
        FEATURE_NAMES,
        NUM_ENGINEERED_FEATURES,
    )
except ImportError:
    from hardware.feature_engineering import (
        extract_features_batch,
        FEATURE_NAMES,
        NUM_ENGINEERED_FEATURES,
    )

# ── Configuration ──
SEED = 42
SAMPLING_RATE = 100   # Hz (MPU6050 configured at 100Hz)
WINDOW_SIZE = 200     # 2 seconds at 100Hz
STRIDE = 50           # 0.5 seconds (75% overlap)
BUTTER_ORDER = 5      # 5th-order Butterworth (per reference paper)
BUTTER_CUTOFF = 1.3   # Hz cutoff frequency (per reference paper)
CRASH_RATIO_THRESHOLD = 0.1  # Window labeled crash if >10% samples are crash

# CSV file names
CSV_FILES = [
    "labeled crash non crash sensor_data_2026-08-21_11-11-36.csv",
    "sensor_data_2026-08-21_11-36-48.csv",
]

# CSI parameters — calibrated for MPU6050 hardware dynamics
CSI_G_NORM = 28.0
CSI_DV_NORM = 150.0
CSI_WEIGHTS = {"peak_g": 0.35, "delta_v": 0.30, "jerk": 0.20, "duration": 0.15}

# ── Reproducibility ──
random.seed(SEED)
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: Butterworth Low-Pass Filter
# ══════════════════════════════════════════════════════════════════════════════

def butterworth_lowpass(data, cutoff=BUTTER_CUTOFF, fs=SAMPLING_RATE, order=BUTTER_ORDER):
    """Apply Butterworth low-pass filter (matching reference paper Figure 4).

    Steps:
        1. Remove DC bias (subtract mean)
        2. Apply 5th-order Butterworth low-pass at 1.3Hz
        3. Add mean back to restore baseline

    Args:
        data: 1D numpy array (raw sensor channel)
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order

    Returns:
        Filtered 1D array (same length as input)
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    # DC bias removal
    mean_val = np.mean(data)
    data_centered = data - mean_val
    filtered = filtfilt(b, a, data_centered)
    return filtered + mean_val


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: Load and Merge CSVs
# ══════════════════════════════════════════════════════════════════════════════

def load_and_merge():
    """Load both hardware CSV files and merge into a list of per-session DataFrames.

    Returns separate DataFrames per session to avoid windowing across file boundaries.
    """
    sessions = []

    for csv_name in CSV_FILES:
        candidates = [
            HARDWARE_DIR / "data" / csv_name,
            HARDWARE_DIR / csv_name,
            PROJECT_ROOT / csv_name,
        ]
        csv_path = next((p for p in candidates if p.exists()), None)
        if csv_path is None:
            print(f"[ERROR] File not found: {csv_name} in {[str(p) for p in candidates]}")
            sys.exit(1)

        print(f"\nLoading: {csv_path.name} from {csv_path.parent}")
        df = pd.read_csv(csv_path)

        n_crash = (df["Label"] == "CRASH").sum()
        n_non = (df["Label"] == "NON_CRASH").sum()
        print(f"  Rows: {len(df):,}  |  CRASH: {n_crash:,}  |  NON_CRASH: {n_non:,}")

        sessions.append(df)

    total_rows = sum(len(s) for s in sessions)
    total_crash = sum((s["Label"] == "CRASH").sum() for s in sessions)
    print(f"\n{'='*60}")
    print(f"MERGED TOTAL: {total_rows:,} rows  |  CRASH: {total_crash:,}  |  NON_CRASH: {total_rows - total_crash:,}")
    print(f"{'='*60}")

    return sessions


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: Apply Butterworth Filter to Each Session
# ══════════════════════════════════════════════════════════════════════════════

def filter_session(df):
    """Apply Butterworth filter to Accel_X, Accel_Y, Gyro_Z of a session."""
    df = df.copy()
    df["Accel_X_filtered"] = butterworth_lowpass(df["Accel_X"].values)
    df["Accel_Y_filtered"] = butterworth_lowpass(df["Accel_Y"].values)
    df["Gyro_Z_filtered"] = butterworth_lowpass(df["Gyro_Z"].values)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: Create Sliding Windows from a Session
# ══════════════════════════════════════════════════════════════════════════════

def create_windows(df):
    """Create sliding windows from one recording session.

    Builds 6-channel windows matching the software pipeline format:
        [Accel_X, Accel_Y, Gyro_Z, Accel_X_filt, Accel_Y_filt, Gyro_Z_filt]

    Returns:
        X: (num_windows, 200, 6) float32
        y: (num_windows,) int32 — binary crash label
    """
    # Build 6-channel array (same order as software pipeline)
    data = np.column_stack([
        df["Accel_X"].values,
        df["Accel_Y"].values,
        df["Gyro_Z"].values,
        df["Accel_X_filtered"].values,
        df["Accel_Y_filtered"].values,
        df["Gyro_Z_filtered"].values,
    ]).astype(np.float32)

    labels = (df["Label"] == "CRASH").astype(int).values

    num_windows = (len(data) - WINDOW_SIZE) // STRIDE + 1

    X = np.zeros((num_windows, WINDOW_SIZE, 6), dtype=np.float32)
    y = np.zeros(num_windows, dtype=np.int32)

    for i in range(num_windows):
        start = i * STRIDE
        end = start + WINDOW_SIZE
        X[i] = data[start:end]
        crash_ratio = labels[start:end].mean()
        y[i] = 1 if crash_ratio > CRASH_RATIO_THRESHOLD else 0

    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5: CSI Computation (rc_buggy mode — no config.yaml dependency)
# ══════════════════════════════════════════════════════════════════════════════

def compute_csi_rcbuggy(features):
    """Compute Crash Severity Index using rc_buggy normalizers.

    Same formula as src/features/feature_engineering.py compute_csi(),
    but with hardcoded rc_buggy parameters for portability.

    Args:
        features: (N, 26) feature matrix

    Returns:
        (N,) CSI scores clipped to [0, 1]
    """
    peak_g = features[:, 0]     # peak_accel
    dv = features[:, 1]         # delta_v
    jerk = features[:, 2]       # jerk_rms
    accel_std = features[:, 5]  # accel_mag_std

    peak_g_norm = np.clip(peak_g / CSI_G_NORM, 0, 1)
    dv_norm = np.clip(dv / CSI_DV_NORM, 0, 1)
    jerk_norm = np.clip(jerk / (CSI_G_NORM * 10), 0, 1)
    duration_factor = np.clip(accel_std / 2.0, 0, 1)

    csi = (CSI_WEIGHTS["peak_g"] * peak_g_norm +
           CSI_WEIGHTS["delta_v"] * dv_norm +
           CSI_WEIGHTS["jerk"] * jerk_norm +
           CSI_WEIGHTS["duration"] * duration_factor)

    return np.clip(csi, 0, 1).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6: Generate Filter Comparison Plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_filter_comparison(df, output_path):
    """Generate raw vs Butterworth-filtered signal plot (like paper Figure 4)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed — skipping plot")
        return

    # Find a crash segment for visualization
    crash_mask = (df["Label"] == "CRASH").values
    transitions = np.diff(crash_mask.astype(int))
    crash_starts = np.where(transitions == 1)[0]

    if len(crash_starts) > 0:
        # Take the first crash event, show 2 seconds before to 3 seconds after
        cs = crash_starts[0]
        start = max(0, cs - 200)
        end = min(len(df), cs + 300)
    else:
        start, end = 0, min(1000, len(df))

    t = np.arange(end - start) / SAMPLING_RATE

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Accel_X
    axes[0].plot(t, df["Accel_X"].values[start:end], alpha=0.6, label="Raw data", color="#1f77b4")
    axes[0].plot(t, df["Accel_X_filtered"].values[start:end], label="Low pass Butterworth filter data", color="red", linewidth=2)
    axes[0].set_ylabel("X axis acceleration (m/s²)")
    axes[0].legend()
    axes[0].set_title("Raw and Filtered Low-Pass Accelerometer Data (Hardware MPU6050)")

    # Accel_Y
    axes[1].plot(t, df["Accel_Y"].values[start:end], alpha=0.6, label="Raw data", color="#1f77b4")
    axes[1].plot(t, df["Accel_Y_filtered"].values[start:end], label="Low pass Butterworth filter data", color="red", linewidth=2)
    axes[1].set_ylabel("Y axis acceleration (m/s²)")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFilter comparison plot saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  IntelliCrash Hardware Data Pipeline")
    print("  Sensor: MPU6050 (6-DoF IMU, 100Hz)")
    print("=" * 60)

    out_dir = HARDWARE_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load CSVs ──
    sessions = load_and_merge()

    # ── Apply Butterworth Filter per session ──
    print(f"\nApplying 5th-order Butterworth LPF (cutoff={BUTTER_CUTOFF}Hz, fs={SAMPLING_RATE}Hz)...")
    filtered_sessions = []
    for i, df in enumerate(sessions):
        df_filtered = filter_session(df)
        filtered_sessions.append(df_filtered)
        print(f"  Session {i+1}: {len(df):,} rows filtered")

    # ── Generate filter plot from first session ──
    plot_filter_comparison(filtered_sessions[0], out_dir / "filter_comparison.png")

    # ── Create windows per session (avoids cross-session windowing) ──
    print(f"\nCreating sliding windows (size={WINDOW_SIZE}, stride={STRIDE}, overlap=75%)...")
    all_X, all_y = [], []

    for i, df in enumerate(filtered_sessions):
        X_sess, y_sess = create_windows(df)
        all_X.append(X_sess)
        all_y.append(y_sess)
        print(f"  Session {i+1}: {len(X_sess)} windows")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    crash_wins = y.sum()
    noncrash_wins = len(y) - crash_wins
    print(f"\n{'='*60}")
    print(f"TOTAL WINDOWS: {len(y):,}")
    print(f"  CRASH:     {crash_wins:,}  ({100*crash_wins/len(y):.1f}%)")
    print(f"  NON_CRASH: {noncrash_wins:,}  ({100*noncrash_wins/len(y):.1f}%)")
    print(f"  Shape: X={X.shape}, y={y.shape}")
    print(f"{'='*60}")

    # ── Extract 26 Physics Features (SAME code as software pipeline) ──
    print(f"\nExtracting {NUM_ENGINEERED_FEATURES} physics features (same as software pipeline)...")
    print(f"  Features: {FEATURE_NAMES}")
    features = extract_features_batch(X, fs=SAMPLING_RATE)
    print(f"  Feature matrix shape: {features.shape}")

    # ── Compute CSI (rc_buggy mode) ──
    print("\nComputing CSI scores (rc_buggy mode)...")
    csi_scores = compute_csi_rcbuggy(features)
    print(f"  CSI: mean={csi_scores.mean():.4f}, std={csi_scores.std():.4f}, "
          f"min={csi_scores.min():.4f}, max={csi_scores.max():.4f}")

    # ── Feature Statistics ──
    print(f"\n{'='*60}")
    print("FEATURE STATISTICS (first 10 features):")
    print(f"{'='*60}")
    feat_df = pd.DataFrame(features, columns=FEATURE_NAMES)
    stats = feat_df.describe().T[["mean", "std", "min", "max"]]
    print(stats.head(10).to_string())

    # ── Save ──
    print(f"\nSaving to {out_dir}/...")
    np.save(out_dir / "hw_X.npy", X)
    np.save(out_dir / "hw_y.npy", y)
    np.save(out_dir / "hw_features.npy", features)
    np.save(out_dir / "hw_csi.npy", csi_scores)

    print(f"  hw_X.npy:        {X.shape} — raw 6-channel windows")
    print(f"  hw_y.npy:        {y.shape} — binary crash labels")
    print(f"  hw_features.npy: {features.shape} — 26 engineered features")
    print(f"  hw_csi.npy:      {csi_scores.shape} — CSI scores (rc_buggy)")

    print(f"\n{'='*60}")
    print("  Pipeline complete! Ready for training.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
