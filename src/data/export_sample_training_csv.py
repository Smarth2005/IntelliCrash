import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.feature_engineering import extract_features_single, FEATURE_NAMES

def generate_sample_training_csv():
    data_dir = Path("data/processed/windows")
    
    try:
        X = np.load(data_dir / "imu_augmented_X.npy")
        y = np.load(data_dir / "imu_augmented_y.npy")
    except FileNotFoundError:
        print("Error: Could not find imu_augmented data.")
        return

    # Find 1 normal window and 1 crash window
    normal_idx = np.where(y == 0)[0][0]
    crash_idx = np.where(y == 1)[0][0]
    
    indices_to_export = [(normal_idx, "Normal"), (crash_idx, "Crash")]
    
    all_rows = []
    
    raw_channels = [
        "accele_x", "accele_y", "gyro_z", 
        "accele_x_filtered", "accele_y_filtered", "gyro_z_filtered"
    ]
    
    all_columns = ["window_type", "timestep"] + raw_channels + FEATURE_NAMES
    
    for idx, window_type in indices_to_export:
        window_data = X[idx] # Shape: (200, 6)
        
        # The LSTM training script extracts features and broadcasts them across all 200 timesteps
        features_1d = extract_features_single(window_data) # Shape: (18,)
        
        # Recreate exactly what goes into the LSTM: (200 timesteps, 24 channels)
        for t in range(200):
            row = [window_type, t]
            row.extend(window_data[t].tolist()) # The 6 raw/filtered channels at this timestep
            row.extend(features_1d.tolist())    # The 18 broadcasted features
            all_rows.append(row)
            
    df = pd.DataFrame(all_rows, columns=all_columns)
    
    out_path = Path("Sample_LSTM_Training_Data.csv")
    df.to_csv(out_path, index=False)
    
    print(f"SUCCESS! Sample training CSV saved to: {out_path.resolve()}")

if __name__ == "__main__":
    generate_sample_training_csv()
