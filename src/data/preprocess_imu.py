"""
IntelliCrash — IMU Data Preprocessor for df.csv.

Handles:
1. Loading the 1.3M-row IMU dataset
2. Cleaning and validation
3. Binary crash label mapping (74 labels → crash vs. non-crash)
4. Time-series segmentation into sliding windows
5. Saving preprocessed windows for model training
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import get_config


# ── Label Mapping ─────────────────────────────────────────────────────────────
# df.csv has 74 event labels. We classify them into binary:
#   0 = normal driving (includes maneuvers like S-shape, lane changes, braking)
#   1 = crash / aggressive event

# Known crash/aggressive keywords
CRASH_KEYWORDS = [
    "crash", "collision", "aggressive", "accident", "impact", "hit"
]


def map_label_to_binary(label: str) -> int:
    """Map a driving event label to binary crash (1) / non-crash (0).

    The dataset labels include:
    - 'normal' → non-crash
    - 'Sshape*', 'f-lane*', 'brakes*' → driving maneuvers → non-crash
    - Any label containing crash keywords → crash

    Args:
        label: Raw string label from df.csv 'label' column.

    Returns:
        1 if crash event, 0 if non-crash.
    """
    label_lower = label.strip().lower()

    # Check for crash keywords
    for keyword in CRASH_KEYWORDS:
        if keyword in label_lower:
            return 1

    # Everything else is non-crash (normal driving + maneuvers)
    return 0


def get_event_category(label: str) -> str:
    """Get human-readable event category for EDA analysis.

    Returns one of: 'normal', 'maneuver', 'crash'
    """
    label_lower = label.strip().lower()

    if label_lower == "normal":
        return "normal"

    for keyword in CRASH_KEYWORDS:
        if keyword in label_lower:
            return "crash"

    return "maneuver"


def map_label_to_multiclass(label: str) -> int:
    """Map a driving event label to a 7-class system for predictive safety.
    
    Classes:
    0: Normal
    1: Lane Weaving (Sshape, weave)
    2: Lane Swerving (f-lane, swerve)
    3: Hard Braking (brakes)
    4: Hard Cornering (corners, turn)
    5: Quick U-turn (uturn)
    6: Crash (any crash keyword)
    """
    label_lower = str(label).strip().lower()
    
    # Check for crash keywords first (highest priority)
    for keyword in CRASH_KEYWORDS:
        if keyword in label_lower:
            return 6
            
    # Catch "brake", "braking", "hard braking", "brakes"
    if "brake" in label_lower or "braking" in label_lower:
        return 3
    # Catch "u-turn", "uturn", "quick u-turn" (must be before "turn")
    elif "u-turn" in label_lower or "uturn" in label_lower or "u turn" in label_lower:
        return 5
    # Catch "corner", "cornering", "turn", "turning"
    elif "corner" in label_lower or "turn" in label_lower:
        return 4
    # Catch "weave", "weaving", "sshape", "s-shape", "lane weaving"
    elif "weave" in label_lower or "weaving" in label_lower or "sshape" in label_lower or "s-shape" in label_lower:
        return 1
    # Catch "swerve", "swerving", "lane swerving", "lane"
    elif "swerve" in label_lower or "swerving" in label_lower or "lane" in label_lower:
        return 2
        
    return 0


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_imu_data(csv_path: str = None, nrows: int = None) -> pd.DataFrame:
    """Load the raw IMU dataset from df.csv.

    Args:
        csv_path: Path to df.csv. Uses config default if None.
        nrows: Optional row limit for testing.

    Returns:
        pd.DataFrame with columns: time, seconds_elapsed, accele_x, accele_y,
        accele_x_filtered, accele_y_filtered, label, gyro_z, gyro_z_filtered
    """
    cfg = get_config()
    path = csv_path or cfg["paths"]["df_csv"]

    print(f"Loading IMU data from: {path}")
    df = pd.read_csv(path, nrows=nrows)
    print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")

    return df


def clean_imu_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the IMU dataframe.

    Steps:
    1. Drop duplicates
    2. Handle NaN values (forward fill then drop remaining)
    3. Sort by time
    4. Add binary crash label
    5. Add event category

    Args:
        df: Raw dataframe from load_imu_data()

    Returns:
        Cleaned dataframe with added columns: 'crash_label', 'event_category'
    """
    print("Cleaning IMU data...")
    initial_len = len(df)

    # 1. Drop exact duplicates
    df = df.drop_duplicates()
    dup_removed = initial_len - len(df)
    if dup_removed > 0:
        print(f"  Removed {dup_removed:,} duplicate rows")

    # 2. Handle NaN
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"  Found {nan_count:,} NaN values — forward-filling")
        df = df.ffill()
        # Drop any remaining NaN (e.g., at the very start)
        remaining = df.isna().sum().sum()
        if remaining > 0:
            df = df.dropna()
            print(f"  Dropped {remaining:,} rows with unfillable NaN")

    # 3. Sort by time
    df = df.sort_values("seconds_elapsed").reset_index(drop=True)

    # 4. Map labels to binary and multiclass
    df["crash_label"] = df["label"].apply(map_label_to_binary)
    df["rash_class"] = df["label"].apply(map_label_to_multiclass)
    df["event_category"] = df["label"].apply(get_event_category)

    crash_count = df["crash_label"].sum()
    print(f"  Label distribution: {len(df) - crash_count:,} non-crash, "
          f"{crash_count:,} crash ({100 * crash_count / len(df):.2f}%)")

    print(f"  Final: {len(df):,} rows")
    return df


# ── Sliding Window Segmentation ──────────────────────────────────────────────

def create_sliding_windows(
    df: pd.DataFrame,
    window_size: int = 200,
    stride: int = 50,
    feature_cols: list = None,
    label_col: str = "crash_label",
) -> tuple:
    """Segment time-series data into overlapping sliding windows.

    Each window gets the MAJORITY label (>50% of samples are crash → crash window).

    Args:
        df: Cleaned IMU dataframe
        window_size: Number of samples per window (200 = 2 sec @ 100Hz)
        stride: Step size between windows (50 = 0.5 sec, gives 75% overlap)
        feature_cols: Columns to include as features. Defaults to raw sensor channels.
        label_col: Column to use for window labeling.

    Returns:
        Tuple of:
            X: np.ndarray of shape (num_windows, window_size, num_features)
            y: np.ndarray of shape (num_windows,) — binary labels
            metadata: list of dicts with window start/end times
    """
    cfg = get_config()

    if feature_cols is None:
        feature_cols = [
            "accele_x", "accele_y", "gyro_z",
            "accele_x_filtered", "accele_y_filtered", "gyro_z_filtered",
        ]

    data = df[feature_cols].values
    labels = df[label_col].values
    rash_labels = df["rash_class"].values if "rash_class" in df.columns else np.zeros(len(df))
    times = df["seconds_elapsed"].values

    num_windows = (len(data) - window_size) // stride + 1
    print(f"Creating {num_windows:,} sliding windows "
          f"(size={window_size}, stride={stride})")

    X = np.zeros((num_windows, window_size, len(feature_cols)), dtype=np.float32)
    y = np.zeros(num_windows, dtype=np.int32)
    metadata = []

    for i in tqdm(range(num_windows), desc="Windowing", unit="win"):
        start = i * stride
        end = start + window_size

        X[i] = data[start:end]

        # Crash ratio threshold lowered from 0.5 to 0.1 so short crashes aren't erased
        crash_ratio = labels[start:end].mean()
        y[i] = 1 if crash_ratio > 0.1 else 0

        # Prioritize rash driving over 'Normal' (0) if it takes up at least 10% of the window (0.2s)
        # This prevents noisy labels where a 2-second window is 99% normal driving
        window_rash = rash_labels[start:end]
        rash_only = window_rash[window_rash != 0]
        if len(rash_only) >= (0.1 * window_size):
            unique_rash, counts = np.unique(rash_only, return_counts=True)
            majority_rash = unique_rash[np.argmax(counts)]
        else:
            majority_rash = 0

        metadata.append({
            "window_idx": i,
            "start_sample": start,
            "end_sample": end,
            "start_time": float(times[start]),
            "end_time": float(times[end - 1]),
            "crash_ratio": float(crash_ratio),
            "rash_class": int(majority_rash),
        })

    crash_windows = y.sum()
    print(f"  Windows: {num_windows:,} total, "
          f"{crash_windows:,} crash ({100 * crash_windows / num_windows:.2f}%), "
          f"{num_windows - crash_windows:,} non-crash")

    return X, y, metadata


# ── Save/Load Preprocessed Data ──────────────────────────────────────────────

def save_windows(X: np.ndarray, y: np.ndarray, metadata: list,
                 output_dir: str = None, prefix: str = "imu"):
    """Save windowed data to disk.

    Args:
        X: Feature windows array
        y: Label array
        metadata: Window metadata list
        output_dir: Directory to save to. Uses config default if None.
        prefix: Filename prefix (e.g., 'imu', 'buggy')
    """
    cfg = get_config()
    out_dir = Path(output_dir or cfg["paths"]["windows_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"{prefix}_X.npy", X)
    np.save(out_dir / f"{prefix}_y.npy", y)
    pd.DataFrame(metadata).to_parquet(out_dir / f"{prefix}_metadata.parquet")

    print(f"Saved to {out_dir}/:")
    print(f"  {prefix}_X.npy: {X.shape}")
    print(f"  {prefix}_y.npy: {y.shape}")
    print(f"  {prefix}_metadata.parquet: {len(metadata)} records")


def load_windows(input_dir: str = None, prefix: str = "imu") -> tuple:
    """Load previously saved windowed data.

    Returns:
        Tuple of (X, y, metadata_df)
    """
    cfg = get_config()
    in_dir = Path(input_dir or cfg["paths"]["windows_dir"])

    X = np.load(in_dir / f"{prefix}_X.npy")
    y = np.load(in_dir / f"{prefix}_y.npy")
    meta = pd.read_parquet(in_dir / f"{prefix}_metadata.parquet")

    print(f"Loaded from {in_dir}/:")
    print(f"  X: {X.shape}, y: {y.shape}, metadata: {len(meta)} records")

    return X, y, meta


# ── Main Preprocessing Entry Point ───────────────────────────────────────────

def preprocess_pipeline(csv_path: str = None, nrows: int = None) -> tuple:
    """Run the full preprocessing pipeline on df.csv.

    Steps:
    1. Load raw data
    2. Clean & validate
    3. Create sliding windows
    4. Save to disk

    Args:
        csv_path: Optional path to df.csv
        nrows: Optional row limit for testing

    Returns:
        Tuple of (X, y, metadata)
    """
    cfg = get_config()

    # Load
    df = load_imu_data(csv_path, nrows)

    # Clean
    df = clean_imu_data(df)

    # Window
    window_size = cfg["imu"]["window_size_samples"]
    stride = cfg["imu"]["window_stride_samples"]

    X, y, metadata = create_sliding_windows(
        df,
        window_size=window_size,
        stride=stride,
    )

    # Save
    save_windows(X, y, metadata)

    return X, y, metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess df.csv for IntelliCrash")
    parser.add_argument("--nrows", type=int, default=None,
                        help="Limit rows for testing (default: all)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to df.csv (default: from config)")
    args = parser.parse_args()

    X, y, meta = preprocess_pipeline(csv_path=args.csv, nrows=args.nrows)
    print(f"\nDone! X shape: {X.shape}, y shape: {y.shape}")
    print(f"Crash windows: {y.sum():,} / {len(y):,}")
