import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features.feature_engineering import (
    peak_acceleration, delta_v, jerk_rms, compute_csi, 
    extract_features_single, NUM_ENGINEERED_FEATURES
)

def test_peak_acceleration():
    # Mock window (200, 6)
    window = np.zeros((200, 6))
    window[0, 0] = 3.0 # ax
    window[0, 1] = 4.0 # ay
    
    # Resultant should be sqrt(3^2 + 4^2) = 5.0
    val = peak_acceleration(window)
    assert np.isclose(val, 5.0)

def test_delta_v():
    window = np.zeros((200, 6))
    # Constant 1g resultant for 200 samples (2 seconds at dt=0.01)
    window[:, 0] = 1.0 
    window[:, 1] = 0.0
    
    # dv = integral of 1g over 1.99s (200 samples - 1 interval) = 1g * 9.81 * 1.99s = 19.5219
    val = delta_v(window, dt=0.01)
    assert np.isclose(val, 19.5219)

def test_jerk_rms():
    window = np.zeros((200, 6))
    # Linearly increasing accel 0 to 200g -> diff is 1g per step -> 100g/s jerk
    window[:, 0] = np.arange(200) * 1.0
    val = jerk_rms(window, dt=0.01)
    assert np.isclose(val, 100.0)

def test_extract_features_single():
    window = np.random.normal(0, 1, (200, 6))
    features = extract_features_single(window)
    assert features.shape == (NUM_ENGINEERED_FEATURES,)
    assert not np.isnan(features).any()

def test_compute_csi_fatal():
    # Mock feature array with huge values (Fatal crash)
    features = np.zeros((1, NUM_ENGINEERED_FEATURES))
    features[0, 0] = 65.0  # peak_accel > 60g
    features[0, 1] = 55.0  # delta_v > 50 m/s
    features[0, 2] = 700.0 # jerk
    features[0, 5] = 3.0   # accel_std
    
    csi = compute_csi(features)
    assert csi[0] >= 0.75 # Fatal threshold

def test_compute_csi_normal():
    # Mock feature array with tiny values (Normal driving)
    features = np.zeros((1, NUM_ENGINEERED_FEATURES))
    features[0, 0] = 0.5   # 0.5g
    features[0, 1] = 2.0   # small delta_v
    features[0, 2] = 10.0  # small jerk
    features[0, 5] = 0.1   # low std
    
    csi = compute_csi(features)
    assert csi[0] < 0.40 # Minor / No Crash threshold
