import numpy as np
import pandas as pd
from pathlib import Path

data_dir = Path("data/processed")

# Find all .npy files recursively
for npy_file in data_dir.rglob("*.npy"):
    print(f"\nProcessing {npy_file}...")
    try:
        data = np.load(npy_file)
        print(f"Shape: {data.shape}")
        
        # Handle 1D arrays
        if data.ndim == 1:
            df = pd.DataFrame(data, columns=["Value"])
            
        # Handle 2D arrays (like imu_features.npy)
        elif data.ndim == 2:
            df = pd.DataFrame(data)
            # If it's exactly 26 columns, we can add the feature names
            if data.shape[1] == 26:
                df.columns = [
                    "Peak_Accel", "Delta_V", "RMS_Jerk", "Peak_Jerk", "Resultant_Mag", "RMS_Accel",
                    "Accel_Std", "Axial_Var_X", "Axial_Var_Y", "Yaw_Var", "Spectral_Energy_X",
                    "Spectral_Energy_Y", "Spectral_Energy_Z", "Spectral_Centroid_X", "Spectral_Centroid_Y",
                    "ZCR_X", "ZCR_Y", "Cross_Corr_XY", "Gyro_Z_Peak", "Gyro_Z_Range", "Steering_Oscillations",
                    "Decel_Duration", "Lat_Accel_Peak", "Energy_Ratio", "Peak_to_Peak", "Autocorr"
                ]
                
        # Handle 3D arrays (like X_train.npy which is [samples, 200, 32])
        elif data.ndim == 3:
            print(f"Skipping {npy_file.name} - 3D tensors cannot be saved directly as a readable 2D CSV.")
            continue
        else:
            print(f"Skipping {npy_file.name} - Unsupported dimensions ({data.ndim}D).")
            continue
            
        csv_path = npy_file.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"Successfully saved {csv_path.name}")
        
    except Exception as e:
        print(f"Error processing {npy_file.name}: {e}")
