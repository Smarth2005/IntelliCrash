"""
IntelliCrash — VZCrash Dataset Downloader and Preprocessor.

This script connects to Hugging Face, streams the VZCrash dataset (ITSC 2026),
filters and transforms the telemetry data to match the IntelliCrash pipeline,
and saves a real-world validation set (raw windows, labels, features, and CSI).

It uses streaming=True to download only what is needed, avoiding the full 7.74 GB download.
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt

# Ensure dependencies are installed
def check_dependencies():
    required = ["datasets", "huggingface_hub", "pyarrow"]
    missing = []
    for lib in required:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    if missing:
        print(f"📦 Missing libraries found: {missing}. Installing them now...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
            print("✅ All dependencies installed successfully!")
        except Exception as e:
            print(f"❌ Failed to install dependencies: {e}. Please run: pip install {' '.join(missing)}")
            sys.exit(1)

check_dependencies()

from datasets import load_dataset

# Setup project paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import get_config
from src.features.feature_engineering import extract_features_batch, compute_csi

# ── Butterworth Filter ─────────────────────────────────────────────────────────
def butterworth_lpf(data, cutoff=1.3, fs=100.0, order=5):
    """Apply a 5th-order Butterworth low-pass filter with 1.3 Hz cutoff."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Handle small windows or pad if necessary
    if len(data) <= 15: # filtfilt requires at least padlen samples
        return data
    return filtfilt(b, a, data, axis=0)

# ── Process Single Event ──────────────────────────────────────────────────────
def process_vz_event(event):
    """Convert a single 16-second VZCrash event into 2-second IntelliCrash windows.
    
    Remaps axes, converts units, filters, and segments.
    """
    # Tri-axial raw sensors
    accel_data = event['gsensor']
    gyro_data = event['gyro']
    
    # Handle pandas/parquet list loading variations safely
    if hasattr(accel_data, 'tolist'):
        accel_data = accel_data.tolist()
    if hasattr(gyro_data, 'tolist'):
        gyro_data = gyro_data.tolist()
        
    accel = np.vstack(accel_data).astype(np.float32)  # shape: (1600, 3) in g
    gyro = np.vstack(gyro_data).astype(np.float32)      # shape: (1600, 3) in deg/s
    label = int(event['label'])  # 0: crash, 1: near_miss, 2: normal_driving
    
    # Map to binary crash classification:
    # VZCrash '0' is crash. '1' (near_miss) and '2' (normal_driving) are non-crash.
    binary_label = 1 if label == 0 else 0

    # ─── Coordinate Remap (VZCrash → IntelliCrash) ───
    # VZ: X=forward, Y=left, Z=up
    # IC: X=lateral, Y=longitudinal, Z=up
    # Remap: VZ.Y (left) -> IC.X (lateral); VZ.X (forward) -> IC.Y (longitudinal)
    # Note: Both datasets use g for acceleration. The feature engineering
    # pipeline converts g to m/s^2 internally, so we keep the raw signals in g.
    accele_x = accel[:, 1]          # VZ.Y to lateral (in g)
    accele_y = accel[:, 0]          # VZ.X to longitudinal (in g)
    gyro_z = gyro[:, 2] * (np.pi / 180.0)  # VZ.Gz (yaw) to rad/s
    
    # ─── Noise Filtering ───
    accele_x_filtered = butterworth_lpf(accele_x)
    accele_y_filtered = butterworth_lpf(accele_y)
    gyro_z_filtered = butterworth_lpf(gyro_z)
    
    # ─── Slicing into 2-second windows (200 samples) with 0.5s stride (50 samples)
    window_size = 200
    stride = 50
    windows = []
    
    # In VZCrash, the 16-second event is centered around the crash trigger at t = 8.0s (sample 800).
    # Therefore, the actual collision forces are concentrated around sample 800.
    # We only label a window as a crash (1) if it contains the impact point (sample 800).
    # Since window_size = 200, this means start <= 800 and end >= 800 (i.e., 600 <= start <= 800).
    # All other windows represent pre-crash normal driving or post-crash stationary state, which must be 0.
    for start in range(0, len(accele_x) - window_size + 1, stride):
        end = start + window_size
        win_data = np.column_stack([
            accele_x[start:end],
            accele_y[start:end],
            gyro_z[start:end],
            accele_x_filtered[start:end],
            accele_y_filtered[start:end],
            gyro_z_filtered[start:end]
        ])  # Shape: (200, 6)
        
        # Assign window-level label
        if binary_label == 1:
            if start >= 600 and start <= 800:
                win_label = 1
            else:
                win_label = 0
        else:
            win_label = 0
            
        windows.append((win_data, win_label))
        
    return windows

# ── Main downloader ───────────────────────────────────────────────────────────
def main():
    print("--- IntelliCrash VZCrash Validation Downloader ---")
    cfg = get_config()
    
    # Determine save directory
    processed_dir = Path(cfg["paths"]["processed_dir"])
    val_dir = processed_dir / "vz_validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse optional command-line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, help="Hugging Face Read Access Token")
    args = parser.parse_args()

    token = args.token
    
    # If not provided via CLI, check environment
    if not token:
        token = os.environ.get("HF_TOKEN")
        
    # If not in env, check cached tokens
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except ImportError:
            try:
                from huggingface_hub.utils import HfFolder
                token = HfFolder.get_token()
            except ImportError:
                try:
                    from huggingface_hub import HfFolder
                    token = HfFolder.get_token()
                except ImportError:
                    token = None

    # Configure validation subset size
    num_crash_events = 250     # Yields ~250 * 29 = 7,250 windows
    num_noncrash_events = 250  # Yields ~250 * 29 = 7,250 windows

    print("\n🌐 Connecting to Hugging Face...")
    print("📢 Downloading a single Parquet shard directly (much faster and more stable)...")
    
    try:
        from huggingface_hub import hf_hub_download
        # Download the first shard (contains ~17,000 events, more than enough)
        local_file_path = hf_hub_download(
            repo_id="vzc-research-chapter/VZCrash",
            filename="data/train-00000-of-00011.parquet",
            repo_type="dataset",
            token=token,
            local_dir=val_dir
        )
        print(f"✅ Shard downloaded successfully to: {local_file_path}")
    except Exception as e:
        print(f"\n⚠️ First connection attempt failed: {e}")
        print("This is likely due to an invalid or expired cached Hugging Face token.")
        print("\n🔑 Please paste a valid HF Read Access Token:")
        token = input("Token: ").strip()
        
        try:
            print("\n🔄 Retrying download with the new token...")
            from huggingface_hub import hf_hub_download
            local_file_path = hf_hub_download(
                repo_id="vzc-research-chapter/VZCrash",
                filename="data/train-00000-of-00011.parquet",
                repo_type="dataset",
                token=token,
                local_dir=val_dir
            )
            print(f"✅ Shard downloaded successfully to: {local_file_path}")
        except Exception as retry_err:
            print(f"\n❌ Error connecting to dataset: {retry_err}")
            print("Make sure you:")
            print("1. Accepted the agreement at: https://huggingface.co/datasets/vzc-research-chapter/VZCrash")
            print("2. Passed a valid Read Access Token.")
            sys.exit(1)
        
    print("\n📖 Loading Parquet file into memory with Pandas...")
    df = pd.read_parquet(local_file_path)
    print(f"✅ Loaded {len(df):,} events from shard.")
    
    X_windows = []
    y_labels = []
    
    crash_count = 0
    noncrash_count = 0
    total_events_read = 0
    
    print(f"\n⏳ Processing events. Gathering {num_crash_events} crash and {num_noncrash_events} non-crash events...")
    
    # Progress trackers
    from tqdm import tqdm
    pbar = tqdm(total=num_crash_events + num_noncrash_events, desc="Processing events")
    
    for idx, row in df.iterrows():
        total_events_read += 1
        label = int(row['label'])
        
        # Label mapping: 0=crash, others=non-crash
        is_crash = (label == 0)
        
        if is_crash and crash_count < num_crash_events:
            event_windows = process_vz_event(row)
            for win, blabel in event_windows:
                X_windows.append(win)
                y_labels.append(blabel)
            crash_count += 1
            pbar.update(1)
        elif not is_crash and noncrash_count < num_noncrash_events:
            event_windows = process_vz_event(row)
            for win, blabel in event_windows:
                X_windows.append(win)
                y_labels.append(blabel)
            noncrash_count += 1
            pbar.update(1)
            
        if crash_count >= num_crash_events and noncrash_count >= num_noncrash_events:
            break
    
    X_windows = np.array(X_windows, dtype=np.float32) # shape: (N, 200, 6)
    y_labels = np.array(y_labels, dtype=np.int64)     # shape: (N,)
    
    print(f"\n📊 Extracted {len(X_windows):,} windows from {total_events_read} events.")
    print(f"   Crash Windows: {np.sum(y_labels == 1):,}")
    print(f"   Non-Crash Windows: {np.sum(y_labels == 0):,}")
    
    # ─── Feature Engineering ───
    print("\n⚡ Running Physics-Informed Feature Engineering (26 features per window)...")
    # Features shape: (N, 26)
    features_2d = extract_features_batch(X_windows)
    
    print("📐 Calculating Physics-based Crash Severity Index (CSI) scores...")
    csi_scores = compute_csi(features_2d, mode="real_car")
    
    # Save the processed numpy arrays
    print(f"\n💾 Saving validation set to directory: {val_dir.resolve()}")
    np.save(val_dir / "vz_X.npy", X_windows)
    np.save(val_dir / "vz_y.npy", y_labels)
    np.save(val_dir / "vz_features.npy", features_2d)
    np.save(val_dir / "vz_csi.npy", csi_scores)
    
    # Save the 26 engineered features + labels + CSI as a clean CSV file
    from src.features.feature_engineering import FEATURE_NAMES
    features_df = pd.DataFrame(features_2d, columns=FEATURE_NAMES)
    features_df['label'] = y_labels
    features_df['csi_score'] = csi_scores
    csv_path = val_dir / "vz_features.csv"
    features_df.to_csv(csv_path, index=False)
    print(f"📊 Exported features and labels to CSV: {csv_path.resolve()}")
    
    # Create metadata frame to aid future analysis/dashboard
    meta_df = pd.DataFrame({
        "window_index": range(len(y_labels)),
        "label": y_labels,
        "csi_score": csi_scores
    })
    meta_df.to_parquet(val_dir / "vz_metadata.parquet")
    
    print("\n✨ Processing Complete! You are now ready to run validation tests using this real crash dataset.")
    print("🚀 Deployed arrays:")
    print(f"   - vz_X.npy        : {X_windows.shape}")
    print(f"   - vz_y.npy        : {y_labels.shape}")
    print(f"   - vz_features.npy : {features_2d.shape}")
    print(f"   - vz_csi.npy      : {csi_scores.shape}")

if __name__ == "__main__":
    main()
