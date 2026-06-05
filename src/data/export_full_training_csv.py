import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

def generate_full_csv():
    data_dir = Path("data/processed/windows")
    
    print("Loading the full .npy files...")
    try:
        X = np.load(data_dir / "imu_augmented_X.npy") # Shape: (37268, 200, 6)
        y = np.load(data_dir / "imu_augmented_y.npy") # Shape: (37268,)
    except FileNotFoundError:
        print("Error: Could not find imu_augmented data.")
        return
        
    num_windows, timesteps, channels = X.shape
    
    print(f"Flattening {num_windows} windows from 3D ({timesteps}x{channels}) to 2D...")
    # Reshape from (37268, 200, 6) to (37268, 1200)
    X_flat = X.reshape(num_windows, -1)
    
    # Generate column names (e.g. t0_accele_x, t0_accele_y, ... t199_gyro_z_filtered)
    channel_names = ["accele_x", "accele_y", "gyro_z", "accele_x_filtered", "accele_y_filtered", "gyro_z_filtered"]
    col_names = []
    for t in range(timesteps):
        for ch in channel_names:
            col_names.append(f"t{t}_{ch}")
            
    print("Creating DataFrame...")
    df = pd.DataFrame(X_flat, columns=col_names)
    
    # Add labels
    df["crash_label"] = y
    
    out_path = Path("Full_Training_Data.csv")
    print(f"Saving to {out_path}... (This might take a minute)")
    df.to_csv(out_path, index=False)
    
    print(f"SUCCESS! Full dataset saved to: {out_path.resolve()}")

if __name__ == "__main__":
    generate_full_csv()
