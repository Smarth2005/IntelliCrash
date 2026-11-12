import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.feature_engineering import extract_features_batch, compute_csi, FEATURE_NAMES

def generate_unified_csv():
    print("Loading augmented IMU dataset (the data used for training)...")
    data_dir = Path("data/processed/windows")
    
    try:
        X = np.load(data_dir / "imu_augmented_X.npy")
        y = np.load(data_dir / "imu_augmented_y.npy")
    except FileNotFoundError:
        print("Error: Could not find imu_augmented_X.npy in data/processed/windows/")
        print("Please ensure the dataset has been processed and augmented first.")
        return

    print(f"Loaded {len(X):,} windows. Extracting the 18 engineered features...")
    # Extract features for all windows
    features = extract_features_batch(X)
    
    print("Computing Crash Severity Index (CSI)...")
    # Compute CSI severity scores
    severity = compute_csi(features, mode="real_car")
    
    print("Creating Unified DataFrame...")
    # Create a pandas DataFrame for the 18 features
    df = pd.DataFrame(features, columns=FEATURE_NAMES)
    
    # Add labels and severity
    df["crash_label"] = y
    df["event_type"] = df["crash_label"].map({0: "Normal/Maneuver", 1: "Crash"})
    df["csi_severity"] = severity
    
    # Save to CSV in the root directory
    out_path = Path("IntelliCrash_Dataset.csv")
    df.to_csv(out_path, index=False)
    
    print(f"\nSUCCESS! Unified dataset CSV saved to: {out_path.resolve()}")
    print("This file contains one row per 2-second window, showing all 18 engineered features, CSI severity score, and the crash label.")

if __name__ == "__main__":
    generate_unified_csv()
