"""
IntelliCrash — Physics-Informed Feature Engineering.

Extracts 18 engineered features from raw IMU sliding windows:
- Peak acceleration, delta-V, jerk (RMS + max)
- Resultant magnitude, acceleration magnitude
- FFT spectral energy, spectral centroid
- RMS acceleration, zero-crossing rate
- Cross-correlation between axes
- Variance per channel

These features feed both the LSTM (as additional channels) and the
physics-based Crash Severity Index (CSI).
"""

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import get_config


# ── Individual Feature Functions ─────────────────────────────────────────────

def peak_acceleration(window: np.ndarray, ax_idx: int = 0,
                      ay_idx: int = 1) -> float:
    """Peak resultant acceleration in the window.

    Args:
        window: (window_size, num_channels) array
        ax_idx: Index of accele_x column
        ay_idx: Index of accele_y column

    Returns:
        Peak resultant acceleration magnitude (g)
    """
    ax = window[:, ax_idx]
    ay = window[:, ay_idx]
    resultant = np.sqrt(ax**2 + ay**2)
    return float(np.max(resultant))


def delta_v(window: np.ndarray, ax_idx: int = 0, ay_idx: int = 1,
            dt: float = 0.01) -> float:
    """Delta-V: change in velocity over the window.

    Computed by integrating resultant acceleration (trapezoidal rule).

    Args:
        window: (window_size, num_channels) array
        dt: Time step between samples (1/100Hz = 0.01s)

    Returns:
        Delta-V magnitude (m/s), converted from g*s
    """
    ax = window[:, ax_idx]
    ay = window[:, ay_idx]
    resultant = np.sqrt(ax**2 + ay**2) * 9.81  # Convert g to m/s²
    return float(np.trapezoid(resultant, dx=dt))


def jerk_rms(window: np.ndarray, ax_idx: int = 0, ay_idx: int = 1,
             dt: float = 0.01) -> float:
    """RMS jerk (rate of change of acceleration).

    Jerk = d(acceleration)/dt. High jerk indicates sudden impact.

    Args:
        window: (window_size, num_channels) array
        dt: Time step between samples

    Returns:
        RMS jerk magnitude (g/s)
    """
    ax = window[:, ax_idx]
    ay = window[:, ay_idx]
    resultant = np.sqrt(ax**2 + ay**2)
    jerk = np.diff(resultant) / dt
    return float(np.sqrt(np.mean(jerk**2)))


def max_jerk(window: np.ndarray, ax_idx: int = 0, ay_idx: int = 1,
             dt: float = 0.01) -> float:
    """Maximum absolute jerk in the window.

    Args:
        window: (window_size, num_channels) array
        dt: Time step

    Returns:
        Max absolute jerk (g/s)
    """
    ax = window[:, ax_idx]
    ay = window[:, ay_idx]
    resultant = np.sqrt(ax**2 + ay**2)
    jerk = np.diff(resultant) / dt
    return float(np.max(np.abs(jerk)))


def resultant_magnitude(window: np.ndarray, ax_idx: int = 0,
                        ay_idx: int = 1) -> float:
    """Mean resultant acceleration magnitude over the window.

    Args:
        window: (window_size, num_channels) array

    Returns:
        Mean resultant acceleration (g)
    """
    ax = window[:, ax_idx]
    ay = window[:, ay_idx]
    return float(np.mean(np.sqrt(ax**2 + ay**2)))


def accel_magnitude_stats(window: np.ndarray, ax_idx: int = 0,
                          ay_idx: int = 1) -> float:
    """Standard deviation of resultant acceleration.

    High std = dynamic event; low std = steady driving.
    """
    ax = window[:, ax_idx]
    ay = window[:, ay_idx]
    return float(np.std(np.sqrt(ax**2 + ay**2)))


def fft_energy(signal_1d: np.ndarray, fs: float = 100.0) -> float:
    """Spectral energy of a 1D signal via FFT.

    Energy = sum of squared FFT magnitudes (Parseval's theorem).

    Args:
        signal_1d: 1D time-series array
        fs: Sampling frequency

    Returns:
        Total spectral energy
    """
    n = len(signal_1d)
    yf = rfft(signal_1d)
    power = np.abs(yf)**2 / n
    return float(np.sum(power))


def spectral_centroid(signal_1d: np.ndarray, fs: float = 100.0) -> float:
    """Spectral centroid — the 'center of mass' of the frequency spectrum.

    Higher centroid = more high-frequency content (impacts, vibrations).

    Args:
        signal_1d: 1D time-series array
        fs: Sampling frequency

    Returns:
        Spectral centroid frequency (Hz)
    """
    n = len(signal_1d)
    yf = rfft(signal_1d)
    freqs = rfftfreq(n, d=1.0 / fs)
    power = np.abs(yf)**2
    total_power = np.sum(power)
    if total_power == 0:
        return 0.0
    return float(np.sum(freqs * power) / total_power)


def rms_value(signal_1d: np.ndarray) -> float:
    """Root mean square of a signal."""
    return float(np.sqrt(np.mean(signal_1d**2)))


def zero_crossing_rate(signal_1d: np.ndarray) -> float:
    """Zero-crossing rate — fraction of successive samples that change sign.

    High ZCR indicates oscillation/vibration (crash signature).
    """
    mean_centered = signal_1d - np.mean(signal_1d)
    crossings = np.sum(np.diff(np.sign(mean_centered)) != 0)
    return float(crossings / len(signal_1d))


def cross_correlation(signal_a: np.ndarray, signal_b: np.ndarray) -> float:
    """Normalized cross-correlation between two signals at zero lag.

    High correlation = coupled motion; low = independent axes.
    """
    a = signal_a - np.mean(signal_a)
    b = signal_b - np.mean(signal_b)
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if denom == 0:
        return 0.0
    return float(np.sum(a * b) / denom)


# ── Feature Extraction Pipeline ──────────────────────────────────────────────

# Column indices in the raw window array (based on preprocess_imu.py output)
# [accele_x, accele_y, gyro_z, accele_x_filtered, accele_y_filtered, gyro_z_filtered]
AX_IDX = 0   # accele_x
AY_IDX = 1   # accele_y
GZ_IDX = 2   # gyro_z
AX_F_IDX = 3 # accele_x_filtered
AY_F_IDX = 4 # accele_y_filtered
GZ_F_IDX = 5 # gyro_z_filtered

FEATURE_NAMES = [
    "peak_accel",
    "delta_v",
    "jerk_rms",
    "max_jerk",
    "resultant_mag",
    "accel_mag_std",
    "fft_energy_ax",
    "fft_energy_ay",
    "fft_energy_gz",
    "spectral_centroid_ax",
    "spectral_centroid_ay",
    "rms_accel",
    "zero_crossing_rate_ax",
    "zero_crossing_rate_ay",
    "cross_corr_xy",
    "variance_accel_x",
    "variance_accel_y",
    "variance_gyro_z",
]

NUM_ENGINEERED_FEATURES = len(FEATURE_NAMES)


def extract_features_single(window: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """Extract 18 engineered features from a single window.

    Args:
        window: (window_size, 6) raw sensor window
        fs: Sampling frequency in Hz

    Returns:
        np.ndarray of shape (18,) — one value per feature
    """
    dt = 1.0 / fs

    features = np.array([
        peak_acceleration(window, AX_IDX, AY_IDX),
        delta_v(window, AX_IDX, AY_IDX, dt),
        jerk_rms(window, AX_IDX, AY_IDX, dt),
        max_jerk(window, AX_IDX, AY_IDX, dt),
        resultant_magnitude(window, AX_IDX, AY_IDX),
        accel_magnitude_stats(window, AX_IDX, AY_IDX),
        fft_energy(window[:, AX_IDX], fs),
        fft_energy(window[:, AY_IDX], fs),
        fft_energy(window[:, GZ_IDX], fs),
        spectral_centroid(window[:, AX_IDX], fs),
        spectral_centroid(window[:, AY_IDX], fs),
        rms_value(np.sqrt(window[:, AX_IDX]**2 + window[:, AY_IDX]**2)),
        zero_crossing_rate(window[:, AX_IDX]),
        zero_crossing_rate(window[:, AY_IDX]),
        cross_correlation(window[:, AX_IDX], window[:, AY_IDX]),
        np.var(window[:, AX_IDX]),
        np.var(window[:, AY_IDX]),
        np.var(window[:, GZ_IDX]),
    ], dtype=np.float32)

    return features


def extract_features_batch(X: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """Extract features for an entire batch of windows.

    Args:
        X: (num_windows, window_size, num_raw_channels) array
        fs: Sampling frequency

    Returns:
        np.ndarray of shape (num_windows, 18) — feature matrix
    """
    num_windows = X.shape[0]
    features = np.zeros((num_windows, NUM_ENGINEERED_FEATURES), dtype=np.float32)

    for i in tqdm(range(num_windows), desc="Extracting features", unit="win"):
        features[i] = extract_features_single(X[i], fs)

    return features


def augment_windows_with_features(X: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """Create the full 18-feature input for the LSTM.

    Instead of appending features as extra timesteps, we broadcast each
    window-level feature across all timesteps so the LSTM input is
    (batch, 200, 6 + 18) = (batch, 200, 24).

    But per the implementation plan, the LSTM expects (batch, 200, 18).
    So we use the engineered features replicated across timesteps,
    giving the LSTM access to both temporal patterns and window-level stats.

    ALTERNATIVE: Just use the 18 engineered features as a tabular vector
    and have the LSTM work on raw (batch, 200, 6) while a separate FC
    head processes the 18 features. This is a design decision.

    For now: return the feature matrix (num_windows, 18) separately.

    Args:
        X: (num_windows, window_size, 6) raw windows

    Returns:
        features: (num_windows, 18) feature matrix
    """
    return extract_features_batch(X, fs)


# ── Crash Severity Index (Physics-Based) ─────────────────────────────────────

def compute_csi(features: np.ndarray, mode: str = "real_car") -> np.ndarray:
    """Compute physics-based Crash Severity Index for each window.

    CSI = w1 * (peak_g / g_norm) + w2 * (delta_v / dv_norm)
        + w3 * (jerk / jerk_norm) + w4 * duration_factor

    Args:
        features: (num_windows, 18) feature matrix
        mode: 'real_car' or 'rc_buggy' — selects normalization thresholds

    Returns:
        np.ndarray of shape (num_windows,) — CSI scores in [0, 1]
    """
    cfg = get_config()
    csi_cfg = cfg["csi"][mode]

    g_norm = csi_cfg["g_normalizer"]
    dv_norm = csi_cfg["delta_v_normalizer"]
    weights = csi_cfg["weights"]

    # Feature indices (from FEATURE_NAMES)
    peak_g = features[:, 0]    # peak_accel
    dv = features[:, 1]        # delta_v
    jerk = features[:, 2]      # jerk_rms

    # Normalize to [0, 1]
    peak_g_norm = np.clip(peak_g / g_norm, 0, 1)
    dv_norm_val = np.clip(dv / dv_norm, 0, 1)
    jerk_norm = np.clip(jerk / (g_norm * 10), 0, 1)  # jerk normalizer

    # Duration factor: high if the event is sustained (use accel_mag_std as proxy)
    accel_std = features[:, 5]  # accel_mag_std
    duration_factor = np.clip(accel_std / 2.0, 0, 1)

    csi = (weights["peak_g"] * peak_g_norm +
           weights["delta_v"] * dv_norm_val +
           weights["jerk"] * jerk_norm +
           weights["duration"] * duration_factor)

    return np.clip(csi, 0, 1).astype(np.float32)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data.preprocess_imu import load_windows

    print("Loading preprocessed windows...")
    X, y, meta = load_windows()

    print(f"\nExtracting {NUM_ENGINEERED_FEATURES} features from {X.shape[0]:,} windows...")
    features = extract_features_batch(X)

    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Feature names: {FEATURE_NAMES}")

    # Basic stats
    import pandas as pd
    feat_df = pd.DataFrame(features, columns=FEATURE_NAMES)
    print(f"\nFeature statistics:")
    print(feat_df.describe().T[["mean", "std", "min", "max"]].to_string())

    # Compute CSI
    csi_scores = compute_csi(features, mode="real_car")
    print(f"\nCSI scores: mean={csi_scores.mean():.4f}, "
          f"std={csi_scores.std():.4f}, "
          f"max={csi_scores.max():.4f}")

    # Save features
    cfg = get_config()
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "imu_features.npy", features)
    np.save(out_dir / "imu_csi_scores.npy", csi_scores)
    print(f"\nSaved features to {out_dir}/")
