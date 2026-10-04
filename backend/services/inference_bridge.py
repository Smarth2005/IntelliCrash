"""
IntelliCrash — Inference Bridge Service.

Bridges the FastAPI backend to the existing ML inference pipeline.
Provides a clean API for running crash detection without importing
heavy ML dependencies at import time.

This module wraps:
    1. Feature extraction (26 physics features)
    2. ONNX Runtime Bi-LSTM inference
    3. Physics-based CSI computation
    4. Fusion Gate scoring
"""

import time
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger("intellicrash.inference")

# Lazy-loaded globals
_ort_session = None
_onnx_available = False


def _init_onnx():
    """Lazy-initialize the ONNX Runtime session."""
    global _ort_session, _onnx_available

    try:
        import onnxruntime as ort
        _onnx_available = True
    except ImportError:
        logger.warning("[Inference] onnxruntime not installed. Using mock inference.")
        return

    from src.backend.config import get_settings
    settings = get_settings()
    model_path = settings.ONNX_MODEL_PATH

    if Path(model_path).exists():
        try:
            _ort_session = ort.InferenceSession(model_path)
            logger.info(f"[Inference] ONNX model loaded from {model_path}")
        except Exception as e:
            logger.error(f"[Inference] Failed to load ONNX model: {e}")
    else:
        logger.warning(f"[Inference] ONNX model not found at {model_path}")


def run_inference(window: np.ndarray) -> dict:
    """
    Run the full crash detection pipeline on a 2-second IMU window.

    Args:
        window: np.ndarray of shape (200, 6) — raw IMU data
                Columns: [accel_x, accel_y, gyro_z, accel_x_filt, accel_y_filt, gyro_z_filt]

    Returns:
        dict with keys:
            - ml_probability: float (0-1) from Bi-LSTM
            - csi_score: float (0-1) from physics engine
            - fused_score: float (0-1) from Fusion Gate
            - severity: str (MINOR/SEVERE/FATAL or NONE)
            - is_crash: bool
            - latency_ms: float
            - features: list[float] (26 engineered features)
    """
    global _ort_session
    if _ort_session is None:
        _init_onnx()

    start_time = time.time()

    # 1. Feature extraction — import lazily to avoid heavy deps at startup
    try:
        from src.features.feature_engineering import extract_features_single, compute_csi
    except ImportError:
        logger.error("[Inference] Could not import feature_engineering module")
        return _mock_inference()

    features = extract_features_single(window)
    features_batch = np.expand_dims(features, axis=0)

    # 2. Physics CSI
    csi_score = float(compute_csi(features_batch, mode="real_car")[0])

    # 3. ML Inference
    ml_prob = 0.0
    if _ort_session:
        try:
            features_expanded = np.repeat(
                features_batch[:, np.newaxis, :], 200, axis=1
            )
            window_batch = np.expand_dims(window, axis=0)
            input_tensor = np.concatenate(
                [window_batch, features_expanded], axis=2
            ).astype(np.float32)

            outputs = _ort_session.run(None, {"input": input_tensor})
            ml_prob = float(outputs[0][0][0])
        except Exception as e:
            logger.error(f"[Inference] ONNX runtime error: {e}")
            ml_prob = 0.0
    else:
        # Mock inference when ONNX is unavailable
        peak_g = float(np.max(np.abs(window[:, :2])))
        ml_prob = min(1.0, peak_g / 30.0)

    # 4. Fusion Gate
    fused_score = 0.5 * ml_prob + 0.5 * csi_score
    is_crash = fused_score > 0.50

    # 5. Severity mapping
    if is_crash:
        if csi_score < 0.40:
            severity = "MINOR"
        elif csi_score < 0.75:
            severity = "SEVERE"
        else:
            severity = "FATAL"
    else:
        severity = "NONE"

    latency_ms = (time.time() - start_time) * 1000

    return {
        "ml_probability": round(ml_prob, 4),
        "csi_score": round(csi_score, 4),
        "fused_score": round(fused_score, 4),
        "severity": severity,
        "is_crash": is_crash,
        "latency_ms": round(latency_ms, 2),
        "features": features.tolist(),
    }


def _mock_inference() -> dict:
    """Fallback mock inference for testing without ML dependencies."""
    return {
        "ml_probability": 0.02,
        "csi_score": 0.01,
        "fused_score": 0.015,
        "severity": "NONE",
        "is_crash": False,
        "latency_ms": 0.1,
        "features": [0.0] * 26,
    }
