"""
IntelliCrash Edge — Phase 1 Pre-Crash Rash Driving Inference Engine
===================================================================
Runs real-time inference on the 26 engineered physics features using the
trained Phase-1 classifier (HistGradientBoosting / LightGBM baseline).
On detecting unsafe driving behavior (rash driving maneuver):
1. Returns prediction and confidence probability.
2. Identifies key physical triggering parameters (fast attribution).
3. Optionally triggers background SHAP XAI forensic report generation.
"""

import sys
import time
import threading
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
import joblib

HARDWARE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HARDWARE_DIR))

from feature_engineering import FEATURE_NAMES

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


class PreCrashRashDetector:
    """Phase-1 Real-Time Rash Driving Detector with Feature Attribution & XAI."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        encoder_path: Optional[str] = None,
        probability_threshold: float = 0.50,
        enable_xai_reports: bool = False,
    ):
        self.model_dir = HARDWARE_DIR / "models"
        self.model_path = Path(model_path or self.model_dir / "best_sensor_rash_classifier.pkl")
        self.encoder_path = Path(encoder_path or self.model_dir / "sensor_rash_label_encoder.pkl")
        self.threshold = probability_threshold
        self.enable_xai_reports = enable_xai_reports

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.encoder_path.exists():
            raise FileNotFoundError(f"Label encoder not found: {self.encoder_path}")

        self.model = joblib.load(str(self.model_path))
        self.label_encoder = joblib.load(str(self.encoder_path))
        self.classes = list(self.label_encoder.classes_)
        self.rash_class_idx = 1 if "Rash Driving Maneuver" in self.classes else 0

        # Precompute baseline reference feature means for quick attribution
        self.baseline_means = np.zeros(26, dtype=np.float32)
        # Typical top feature indices for rash driving:
        # spectral_centroid_ay (10), gyro_z_peak (18), steering_oscillation_count (20),
        # deceleration_duration (21), fft_energy_ay (7), delta_v (1)
        self.top_feature_indices = [10, 18, 20, 21, 7, 1]

        print(
            f"{GREEN}[OK] Phase-1 Pre-Crash Detector Loaded | Classes: {self.classes} | Threshold: {self.threshold:.2f}{RESET}"
        )

    def predict(self, features: np.ndarray) -> Tuple[bool, str, float, List[Tuple[str, float]]]:
        """Runs inference on a single 26-feature vector.

        Args:
            features: 1D numpy array of length 26.

        Returns:
            is_rash: True if rash driving detected, False otherwise.
            label_name: String label ('Normal Driving' or 'Rash Driving Maneuver').
            prob_rash: Probability of rash driving in [0, 1].
            key_factors: List of top (feature_name, value) contributing factors.
        """
        feats_2d = features.reshape(1, -1)
        probs = self.model.predict_proba(feats_2d)[0]
        prob_rash = float(probs[self.rash_class_idx])

        is_rash = prob_rash >= self.threshold
        label_name = self.classes[self.rash_class_idx] if is_rash else self.classes[1 - self.rash_class_idx]

        key_factors = []
        if is_rash:
            # Highlight top physical parameters responsible
            sorted_indices = sorted(
                self.top_feature_indices,
                key=lambda idx: abs(float(features[idx])),
                reverse=True,
            )
            for idx in sorted_indices[:3]:
                fname = FEATURE_NAMES[idx]
                fval = float(features[idx])
                key_factors.append((fname, fval))

            if self.enable_xai_reports:
                self._dispatch_background_xai_report(features, prob_rash)

        return is_rash, label_name, prob_rash, key_factors

    def _dispatch_background_xai_report(self, features: np.ndarray, prob_rash: float):
        """Asynchronously saves SHAP forensic snapshot without stalling the loop."""
        def _worker():
            try:
                import shap
                explainer = shap.TreeExplainer(self.model)
                shap_vals = explainer.shap_values(features.reshape(1, -1))
                # Save snapshot to reports
                out_dir = HARDWARE_DIR / "reports" / "xai"
                out_dir.mkdir(parents=True, exist_ok=True)
                ts = int(time.time())
                np.save(out_dir / f"shap_rash_{ts}.npy", shap_vals)
            except Exception:
                pass

        t = threading.Thread(target=_worker, daemon=True, name="SHAPWorker")
        t.start()


if __name__ == "__main__":
    print("Testing PreCrashRashDetector standalone...")
    detector = PreCrashRashDetector()
    dummy_feats = np.random.randn(26).astype(np.float32)
    is_rash, label, prob, factors = detector.predict(dummy_feats)
    print(f"Prediction: {label} | Is Rash: {is_rash} | Prob: {prob:.4f}")
    print(f"Key factors: {factors}")

