"""
IntelliCrash — Synthetic Crash Data Generator.

Since the df.csv dataset has extreme class imbalance (very few crash events),
this module generates realistic synthetic crash windows to augment training data.

Techniques:
1. Physics-based crash simulation (sudden deceleration + noise)
2. Time-warping of existing maneuver windows
3. Magnitude scaling of non-crash windows
4. Jitter and noise injection on real crash windows (if any exist)
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import get_config


def generate_impact_pulse(
    window_size: int = 200,
    num_channels: int = 6,
    peak_g: float = 5.0,
    impact_duration_samples: int = 30,
    fs: float = 100.0,
) -> np.ndarray:
    """Generate a synthetic crash pulse based on vehicle crash physics.

    A typical crash pulse has:
    - Pre-impact: steady low acceleration
    - Impact: sharp half-sine pulse (duration ~100-200ms)
    - Post-impact: oscillating decay

    Args:
        window_size: Number of samples in the window
        num_channels: Number of sensor channels
        peak_g: Peak acceleration in g
        impact_duration_samples: How many samples the impact phase lasts
        fs: Sampling frequency

    Returns:
        np.ndarray of shape (window_size, num_channels) — synthetic crash window
    """
    window = np.zeros((window_size, num_channels))

    # Impact starts at a random position in the window (40-70% through)
    impact_start = np.random.randint(
        int(window_size * 0.4), int(window_size * 0.7)
    )
    impact_end = min(impact_start + impact_duration_samples, window_size)
    actual_duration = impact_end - impact_start

    # --- Pre-impact: normal driving noise ---
    pre_noise_level = 0.1
    window[:impact_start, :] = np.random.normal(0, pre_noise_level,
                                                 (impact_start, num_channels))

    # --- Impact pulse: half-sine shape ---
    t = np.linspace(0, np.pi, actual_duration)
    pulse = peak_g * np.sin(t)

    # Apply to primary acceleration axes (0=accele_x, 1=accele_y)
    # Random crash direction
    crash_angle = np.random.uniform(0, 2 * np.pi)
    window[impact_start:impact_end, 0] = pulse * np.cos(crash_angle)
    window[impact_start:impact_end, 1] = pulse * np.sin(crash_angle)

    # Gyro response (channel 2 = gyro_z): derivative of acceleration + noise
    gyro_pulse = np.diff(pulse, prepend=0) * 2.0
    window[impact_start:impact_end, 2] = gyro_pulse + np.random.normal(
        0, 0.3, actual_duration
    )

    # --- Post-impact: decaying oscillation ---
    post_start = impact_end
    post_samples = window_size - post_start
    if post_samples > 0:
        decay_t = np.linspace(0, 3 * np.pi, post_samples)
        decay = peak_g * 0.3 * np.exp(-decay_t / np.pi) * np.sin(decay_t * 3)

        window[post_start:, 0] = decay * np.cos(crash_angle) + np.random.normal(
            0, 0.1, post_samples
        )
        window[post_start:, 1] = decay * np.sin(crash_angle) + np.random.normal(
            0, 0.1, post_samples
        )
        window[post_start:, 2] = np.random.normal(0, 0.2, post_samples)

    # Filtered channels (3, 4, 5) = smoothed versions of raw (0, 1, 2)
    from scipy.signal import savgol_filter
    for raw_idx, filt_idx in [(0, 3), (1, 4), (2, 5)]:
        window[:, filt_idx] = savgol_filter(window[:, raw_idx], 11, 3)

    return window.astype(np.float32)


def augment_by_scaling(window: np.ndarray, scale_range: tuple = (1.5, 4.0)) -> np.ndarray:
    """Scale a maneuver window to crash-level magnitudes.

    Args:
        window: Original (non-crash) window
        scale_range: Random scale factor range

    Returns:
        Scaled window
    """
    scale = np.random.uniform(*scale_range)
    return (window * scale).astype(np.float32)


def augment_by_time_warp(window: np.ndarray, sigma: float = 0.2) -> np.ndarray:
    """Apply random time warping to a window.

    Stretches and compresses the time axis to create temporal variations.

    Args:
        window: Original window (window_size, channels)
        sigma: Std of the random warping

    Returns:
        Time-warped window
    """
    from scipy.interpolate import interp1d

    n_samples, n_channels = window.shape
    time_orig = np.arange(n_samples, dtype=np.float64)

    # Generate smooth random warping
    warp_steps = np.random.normal(1.0, sigma, n_samples)
    warp_steps = np.maximum(warp_steps, 0.5)  # prevent negatives
    time_warped = np.cumsum(warp_steps)
    time_warped = time_warped / time_warped[-1] * (n_samples - 1)

    warped = np.zeros_like(window)
    for ch in range(n_channels):
        f = interp1d(time_warped, window[:, ch], kind="linear",
                     fill_value="extrapolate")
        warped[:, ch] = f(time_orig)

    return warped.astype(np.float32)


def add_jitter(window: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    """Add Gaussian jitter noise to a window.

    Args:
        window: Input window
        noise_level: Std of added noise

    Returns:
        Jittered window
    """
    noise = np.random.normal(0, noise_level, window.shape)
    return (window + noise).astype(np.float32)


def generate_synthetic_crashes(
    X_existing: np.ndarray,
    y_existing: np.ndarray,
    target_crash_ratio: float = 0.3,
    window_size: int = 200,
    num_channels: int = 6,
) -> tuple:
    """Generate synthetic crash windows to balance the dataset.

    Strategy:
    1. Physics-based pulse generation (50% of synthetic data)
    2. Scaling existing maneuver windows (25%)
    3. Time-warping + jitter on maneuver windows (25%)

    Args:
        X_existing: (num_windows, window_size, channels) — existing windowed data
        y_existing: (num_windows,) — binary labels
        target_crash_ratio: Desired fraction of crash windows in final dataset
        window_size: Samples per window
        num_channels: Channels per sample

    Returns:
        Tuple of (X_augmented, y_augmented) — combined original + synthetic
    """
    n_total = len(y_existing)
    n_crash = int(y_existing.sum())
    n_normal = n_total - n_crash

    # Calculate how many synthetic crashes we need
    # target_ratio = (n_crash + n_synthetic) / (n_total + n_synthetic)
    # Solving: n_synthetic = (target_ratio * n_total - n_crash) / (1 - target_ratio)
    n_synthetic = int((target_crash_ratio * n_total - n_crash) / (1 - target_crash_ratio))
    n_synthetic = max(0, n_synthetic)

    if n_synthetic == 0:
        print("No synthetic data needed — crash ratio already sufficient.")
        return X_existing, y_existing

    print(f"Generating {n_synthetic:,} synthetic crash windows...")
    print(f"  Current: {n_crash:,} crash / {n_total:,} total "
          f"({100 * n_crash / n_total:.2f}%)")
    print(f"  Target: {target_crash_ratio * 100:.0f}% crash ratio")

    # Get maneuver windows (non-crash, non-zero activity) for augmentation
    normal_mask = y_existing == 0
    normal_windows = X_existing[normal_mask]

    # Compute activity level (std of acceleration) to find active windows
    activity = np.std(normal_windows[:, :, :2], axis=(1, 2))
    active_mask = activity > np.percentile(activity, 60)
    maneuver_windows = normal_windows[active_mask]

    if len(maneuver_windows) == 0:
        maneuver_windows = normal_windows

    # Allocate synthetic windows
    n_physics = n_synthetic // 2
    n_scaled = n_synthetic // 4
    n_warped = n_synthetic - n_physics - n_scaled

    synthetic_X = []

    # 1. Physics-based pulses
    for _ in tqdm(range(n_physics), desc="Physics pulses", unit="win"):
        peak = np.random.uniform(2.0, 10.0)  # Random crash severity
        duration = np.random.randint(15, 50)
        window = generate_impact_pulse(window_size, num_channels, peak, duration)
        synthetic_X.append(window)

    # 2. Scaled maneuver windows
    for _ in tqdm(range(n_scaled), desc="Scaled maneuvers", unit="win"):
        idx = np.random.randint(len(maneuver_windows))
        window = augment_by_scaling(maneuver_windows[idx])
        synthetic_X.append(window)

    # 3. Time-warped + jittered maneuver windows
    for _ in tqdm(range(n_warped), desc="Time-warped", unit="win"):
        idx = np.random.randint(len(maneuver_windows))
        window = augment_by_time_warp(maneuver_windows[idx])
        window = augment_by_scaling(window, scale_range=(1.2, 3.0))
        window = add_jitter(window, noise_level=0.1)
        synthetic_X.append(window)

    synthetic_X = np.array(synthetic_X, dtype=np.float32)
    synthetic_y = np.ones(n_synthetic, dtype=np.int32)

    # Combine
    X_aug = np.concatenate([X_existing, synthetic_X], axis=0)
    y_aug = np.concatenate([y_existing, synthetic_y], axis=0)

    # Shuffle
    perm = np.random.permutation(len(X_aug))
    X_aug = X_aug[perm]
    y_aug = y_aug[perm]

    final_crash = int(y_aug.sum())
    print(f"  Final: {final_crash:,} crash / {len(y_aug):,} total "
          f"({100 * final_crash / len(y_aug):.2f}%)")

    return X_aug, y_aug


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data.preprocess_imu import load_windows, save_windows

    print("Loading preprocessed windows...")
    X, y, meta = load_windows()

    print(f"\nOriginal: X={X.shape}, crash={y.sum():,}/{len(y):,}")

    X_aug, y_aug = generate_synthetic_crashes(X, y, target_crash_ratio=0.3)

    print(f"\nAugmented: X={X_aug.shape}, crash={y_aug.sum():,}/{len(y_aug):,}")

    # Save augmented version
    save_windows(X_aug, y_aug, [], prefix="imu_augmented")
    print("Done!")
