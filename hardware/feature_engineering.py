"""
IntelliCrash — Physics-Informed Feature Engineering.

Extracts 26 engineered features from raw IMU sliding windows:
- Peak acceleration, delta-V, jerk (RMS + max)
- Resultant magnitude, acceleration magnitude
- FFT spectral energy, spectral centroid
- RMS acceleration, zero-crossing rate
- Cross-correlation between axes
- Variance per channel
- Gyro-Z peak and range (rotational dynamics)
- Steering oscillation count
- Deceleration duration, lateral accel peak
- Energy ratio (longitudinal vs lateral)
- Peak-to-peak amplitude, autocorrelation

These features feed both the LSTM (as additional channels) and the
physics-based Crash Severity Index (CSI).
"""

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from tqdm import tqdm

import sys
from pathlib import Path

# Default CSI configurations (self-contained, no external YAML dependency)
DEFAULT_CSI_CONFIGS = {
    "real_car": {
        "g_normalizer": 60.0,
        "delta_v_normalizer": 50.0,
        "weights": {"peak_g": 0.35, "delta_v": 0.30, "jerk": 0.20, "duration": 0.15},
    },
    "rc_buggy": {
        "g_normalizer": 28.0,
        "delta_v_normalizer": 150.0,
        "weights": {"peak_g": 0.35, "delta_v": 0.30, "jerk": 0.20, "duration": 0.15},
    },
    "mpu6050_hardware": {
        "g_normalizer": 28.0,
        "delta_v_normalizer": 150.0,
        "weights": {"peak_g": 0.35, "delta_v": 0.30, "jerk": 0.20, "duration": 0.15},
    },
}


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


# ── Temporal / Pattern Features (for Rash Driving Discrimination) ────────────

def gyro_z_peak(window: np.ndarray, gz_idx: int = 2) -> float:
    """Peak absolute gyro-Z value — extreme for Quick U-turn, moderate for Hard Cornering."""
    return float(np.max(np.abs(window[:, gz_idx])))


def gyro_z_range(window: np.ndarray, gz_idx: int = 2) -> float:
    """Range (max - min) of gyro-Z — captures total rotational sweep."""
    gz = window[:, gz_idx]
    return float(np.max(gz) - np.min(gz))


def steering_oscillation_count(window: np.ndarray, gz_idx: int = 2) -> float:
    """Count of direction changes in gyro-Z (sign changes).

    Lane Weaving has many oscillations (high count);
    Quick U-turn has few (1-2); Normal driving is low.
    """
    gz = window[:, gz_idx]
    gz_centered = gz - np.mean(gz)
    sign_changes = np.sum(np.diff(np.sign(gz_centered)) != 0)
    return float(sign_changes)


def deceleration_duration(window: np.ndarray, ax_idx: int = 0,
                          threshold: float = -0.3) -> float:
    """Fraction of the window with sustained negative longitudinal acceleration.

    Hard Braking has long deceleration duration (>50% of window);
    Other maneuvers have brief or no sustained deceleration.
    """
    ax = window[:, ax_idx]
    decel_samples = np.sum(ax < threshold)
    return float(decel_samples / len(ax))


def lateral_accel_peak(window: np.ndarray, ay_idx: int = 1) -> float:
    """Peak absolute lateral (Y-axis) acceleration.

    Hard Cornering has high lateral accel; Hard Braking has low.
    """
    return float(np.max(np.abs(window[:, ay_idx])))


def energy_ratio_ax_ay(window: np.ndarray, ax_idx: int = 0,
                       ay_idx: int = 1) -> float:
    """Ratio of energy in longitudinal vs lateral axis.

    Braking is ax-dominant (ratio > 1); Cornering is ay-dominant (ratio < 1).
    Returns log ratio to center around 0.
    """
    energy_ax = np.sum(window[:, ax_idx] ** 2) + 1e-8
    energy_ay = np.sum(window[:, ay_idx] ** 2) + 1e-8
    return float(np.log(energy_ax / energy_ay))


def peak_to_peak_amplitude(signal_1d: np.ndarray) -> float:
    """Peak-to-peak amplitude of a signal — captures total dynamic range."""
    return float(np.max(signal_1d) - np.min(signal_1d))


def autocorrelation_lag1(signal_1d: np.ndarray) -> float:
    """Autocorrelation at lag 1 — captures periodicity/smoothness.

    Weaving (periodic) has high autocorrelation;
    Braking (abrupt) has lower autocorrelation.
    """
    n = len(signal_1d)
    if n < 2:
        return 0.0
    mean = np.mean(signal_1d)
    centered = signal_1d - mean
    var = np.sum(centered ** 2)
    if var == 0:
        return 0.0
    autocorr = np.sum(centered[:-1] * centered[1:]) / var
    return float(autocorr)


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
    # Original 18 physics features
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
    # 8 NEW temporal/pattern features for rash driving discrimination
    "gyro_z_peak",
    "gyro_z_range",
    "steering_oscillation_count",
    "deceleration_duration",
    "lateral_accel_peak",
    "energy_ratio_ax_ay",
    "peak_to_peak_ay",
    "autocorrelation_lag1_ax",
]

NUM_ENGINEERED_FEATURES = len(FEATURE_NAMES)  # Now 26


def extract_features_single(window: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """Extract 26 engineered features from a single window.

    18 original physics features + 8 temporal/pattern features for
    rash driving pattern discrimination.

    Args:
        window: (window_size, 6) raw sensor window
        fs: Sampling frequency in Hz

    Returns:
        np.ndarray of shape (26,) — one value per feature
    """
    dt = 1.0 / fs

    features = np.array([
        # ── Original 18 physics features ──
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
        # ── 8 NEW temporal/pattern features ──
        gyro_z_peak(window, GZ_IDX),
        gyro_z_range(window, GZ_IDX),
        steering_oscillation_count(window, GZ_IDX),
        deceleration_duration(window, AX_IDX),
        lateral_accel_peak(window, AY_IDX),
        energy_ratio_ax_ay(window, AX_IDX, AY_IDX),
        peak_to_peak_amplitude(window[:, AY_IDX]),
        autocorrelation_lag1(window[:, AX_IDX]),
    ], dtype=np.float32)

    return features


def extract_features_batch(X: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """Extract features for an entire batch of windows.

    Args:
        X: (num_windows, window_size, num_raw_channels) array
        fs: Sampling frequency

    Returns:
        np.ndarray of shape (num_windows, NUM_ENGINEERED_FEATURES) — feature matrix
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
        features: (num_windows, NUM_ENGINEERED_FEATURES) feature matrix
    """
    return extract_features_batch(X, fs)


# ── Crash Severity Index (Physics-Based) ─────────────────────────────────────

def compute_csi(features: np.ndarray, mode: str = "real_car") -> np.ndarray:
    """Compute physics-based Crash Severity Index for each window.

    CSI = w1 * (peak_g / g_norm) + w2 * (delta_v / dv_norm)
        + w3 * (jerk / jerk_norm) + w4 * duration_factor

    Args:
        features: (num_windows, NUM_ENGINEERED_FEATURES) feature matrix
        mode: 'real_car' or 'rc_buggy' — selects normalization thresholds

    Returns:
        np.ndarray of shape (num_windows,) — CSI scores in [0, 1]
    """
    csi_cfg = DEFAULT_CSI_CONFIGS.get(mode, DEFAULT_CSI_CONFIGS["mpu6050_hardware"])

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
    duration_factor = np.clip(accel_std / 4.0, 0, 1)

    csi = (weights["peak_g"] * peak_g_norm +
           weights["delta_v"] * dv_norm_val +
           weights["jerk"] * jerk_norm +
           weights["duration"] * duration_factor)

    return np.clip(csi, 0, 1).astype(np.float32)


# ── Standalone Self-Test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    dummy_window = np.random.randn(200, 6).astype(np.float32)
    feats = extract_features_single(dummy_window)
    print(f"Extracted {len(feats)} features successfully from 200x6 dummy window.")
    csi_val = compute_csi(feats[np.newaxis, :], mode="mpu6050_hardware")
    print(f"Computed CSI score: {csi_val[0]:.4f}")
