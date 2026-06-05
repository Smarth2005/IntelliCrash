import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.feature_engineering import extract_features_batch, FEATURE_NAMES

def generate_panel_csv():
    print("Loading augmented dataset...")
    data_dir = Path("data/processed/windows")
    
    try:
        X = np.load(data_dir / "imu_augmented_X.npy")
        y = np.load(data_dir / "imu_augmented_y.npy")
    except FileNotFoundError:
        print("Error: Could not find imu_augmented_X.npy in data/processed/windows/")
        return

    print(f"Loaded {len(X)} windows. Extracting the 18 engineered features...")
    # Extract features for all windows
    features = extract_features_batch(X)
    
    print("Creating DataFrame...")
    # Create a pandas DataFrame
    df = pd.DataFrame(features, columns=FEATURE_NAMES)
    
    # Add the crash label
    df["crash_label"] = y
    
    # Map the binary label to a readable string for the panel
    df["event_type"] = df["crash_label"].map({0: "Normal/Maneuver", 1: "Crash"})
    
    # Save to CSV in the root directory so it's easy to find
    out_path = Path("Processed_Features_For_Panel.csv")
    df.to_csv(out_path, index=False)
    
    print(f"\nSUCCESS! CSV saved to: {out_path.resolve()}")
    print("This file contains one row per 2-second window, showing all 18 features (Peak Accel, Jerk, etc.) and whether it was a crash.")

if __name__ == "__main__":
    generate_panel_csv()
