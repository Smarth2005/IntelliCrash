"""
IntelliCrash Hardware — Phase 2 Edge Inference Loop
=====================================================
Runs on Raspberry Pi 4 (or local PC for testing).
Reads MPU6050 via I2C or replays CSV for testing.

Two modes:
    Live:   python hardware/edge_inference.py --live
    Replay: python hardware/edge_inference.py --replay "sensor_data.csv"

Requirements (RPi): pip install smbus2 onnxruntime numpy scipy
Requirements (PC):  pip install onnxruntime numpy scipy
"""

import numpy as np
import time
import sys
import argparse
import threading
from pathlib import Path
from collections import deque
from scipy.signal import butter, filtfilt

# ── Path Setup ──
HARDWARE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = HARDWARE_DIR.parent
sys.path.insert(0, str(HARDWARE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from feature_engineering import extract_features_single, NUM_ENGINEERED_FEATURES
except ImportError:
    from hardware.feature_engineering import extract_features_single, NUM_ENGINEERED_FEATURES

# ── Configuration ──
SAMPLING_RATE = 100     # Hz
BUFFER_SIZE = 1000      # 10 seconds ring buffer
WINDOW_SIZE = 200       # 2 seconds
INFERENCE_INTERVAL = 0.5  # Run inference every 0.5 seconds
BUTTER_ORDER = 5
BUTTER_CUTOFF = 1.3     # Hz

# CSI parameters (calibrated for MPU6050 hardware dynamics)
CSI_G_NORM = 28.0
CSI_DV_NORM = 150.0
CSI_WEIGHTS = {"peak_g": 0.35, "delta_v": 0.30, "jerk": 0.20, "duration": 0.15}

# Fusion (Validation-Optimized Adaptive Weights)
LSTM_WEIGHT = 0.90
CSI_WEIGHT = 0.10
CRASH_THRESHOLD = 0.50

# ── ANSI Colors for Console ──
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ══════════════════════════════════════════════════════════════════════════════
#  BUTTERWORTH FILTER
# ══════════════════════════════════════════════════════════════════════════════

def butterworth_lowpass(data, cutoff=BUTTER_CUTOFF, fs=SAMPLING_RATE, order=BUTTER_ORDER):
    """Apply Butterworth low-pass filter to 1D array."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    mean_val = np.mean(data)
    filtered = filtfilt(b, a, data - mean_val)
    return filtered + mean_val


# ══════════════════════════════════════════════════════════════════════════════
#  CSI COMPUTATION (rc_buggy)
# ══════════════════════════════════════════════════════════════════════════════

def compute_csi_single(features):
    """Compute CSI for a single 26-feature vector."""
    peak_g = features[0]
    dv = features[1]
    jerk = features[2]
    accel_std = features[5]

    csi = (CSI_WEIGHTS["peak_g"] * min(peak_g / CSI_G_NORM, 1.0) +
           CSI_WEIGHTS["delta_v"] * min(dv / CSI_DV_NORM, 1.0) +
           CSI_WEIGHTS["jerk"] * min(jerk / (CSI_G_NORM * 10), 1.0) +
           CSI_WEIGHTS["duration"] * min(accel_std / 2.0, 1.0))
    return max(0.0, min(1.0, csi))


# ══════════════════════════════════════════════════════════════════════════════
#  MPU6050 HARDWARE READER
# ══════════════════════════════════════════════════════════════════════════════

class MPU6050Reader:
    """Reads MPU6050 via I2C on Raspberry Pi."""

    def __init__(self, bus=1, address=0x68):
        import smbus2
        self.bus = smbus2.SMBus(bus)
        self.address = address

        # Wake up MPU6050
        self.bus.write_byte_data(self.address, 0x6B, 0x00)
        time.sleep(0.1)
        # Sample rate: 100Hz
        self.bus.write_byte_data(self.address, 0x19, 0x09)
        # Config
        self.bus.write_byte_data(self.address, 0x1A, 0x01)
        # Gyro: ±500 dps
        self.bus.write_byte_data(self.address, 0x1B, 0x08)
        # Accel: ±4g
        self.bus.write_byte_data(self.address, 0x1C, 0x08)

        print(f"{GREEN}[OK] MPU6050 initialized at 0x{address:02X}{RESET}")

    def read_sample(self):
        """Read one sample: (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)."""
        raw = self.bus.read_i2c_block_data(self.address, 0x3B, 14)

        def to_signed(high, low):
            val = (high << 8) | low
            return val - 65536 if val > 32767 else val

        # ±4g → 8192 LSB/g
        ax = to_signed(raw[0], raw[1]) / 8192.0
        ay = to_signed(raw[2], raw[3]) / 8192.0
        az = to_signed(raw[4], raw[5]) / 8192.0
        # ±500 dps → 65.5 LSB/dps
        gx = to_signed(raw[6], raw[7]) / 65.5
        gy = to_signed(raw[8], raw[9]) / 65.5
        gz = to_signed(raw[10], raw[11]) / 65.5

        return ax, ay, az, gx, gy, gz


# ══════════════════════════════════════════════════════════════════════════════
#  CSV REPLAY READER
# ══════════════════════════════════════════════════════════════════════════════

class CSVReplayReader:
    """Replays a hardware CSV file sample-by-sample at 100Hz."""

    def __init__(self, csv_path):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.idx = 0
        self.total = len(self.df)
        self.labels = self.df["Label"].values if "Label" in self.df.columns else None
        print(f"{GREEN}[OK] CSV loaded: {csv_path} ({self.total:,} samples){RESET}")

    def read_sample(self):
        if self.idx >= self.total:
            return None  # End of file

        row = self.df.iloc[self.idx]
        self.idx += 1

        return (
            row["Accel_X"], row["Accel_Y"], row["Accel_Z"],
            row["Gyro_X"], row["Gyro_Y"], row["Gyro_Z"],
        )

    def get_current_label(self):
        if self.labels is not None and self.idx > 0:
            return self.labels[self.idx - 1]
        return "UNKNOWN"


# ══════════════════════════════════════════════════════════════════════════════
#  ONNX MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_onnx_model():
    """Load the hardware-trained ONNX model."""
    try:
        import onnxruntime as ort
    except ImportError:
        print(f"{RED}[ERROR] onnxruntime not installed. pip install onnxruntime{RESET}")
        return None

    model_path = HARDWARE_DIR / "models" / "hw_bilstm.onnx"
    if not model_path.exists():
        print(f"{RED}[ERROR] ONNX model not found: {model_path}{RESET}")
        print("  Run train_phase2.py first to generate the model.")
        return None

    session = ort.InferenceSession(str(model_path))
    print(f"{GREEN}[OK] ONNX model loaded: {model_path}{RESET}")
    return session


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN INFERENCE LOOP
# ══════════════════════════════════════════════════════════════════════════════

def inference_loop(reader, onnx_session, alert_module=None):
    """Main real-time inference loop.

    Algorithm 3.1 from the report:
    1. Poll sensor at 100Hz → fill ring buffer
    2. Every 0.5s, extract 2-second window from buffer
    3. Apply Butterworth filter to buffer, extract filtered window
    4. Compute 26 physics features
    5. Run Bi-LSTM ONNX → P_LSTM
    6. Compute CSI → CSI score
    7. Fusion: P_final = 0.5 * P_LSTM + 0.5 * CSI
    8. If P_final >= 0.50 → CRASH DETECTED → dispatch alert
    """
    # Ring buffers for raw channels
    buf_ax = deque(maxlen=BUFFER_SIZE)
    buf_ay = deque(maxlen=BUFFER_SIZE)
    buf_az = deque(maxlen=BUFFER_SIZE)
    buf_gx = deque(maxlen=BUFFER_SIZE)
    buf_gy = deque(maxlen=BUFFER_SIZE)
    buf_gz = deque(maxlen=BUFFER_SIZE)

    sample_count = 0
    inference_count = 0
    crash_count = 0
    last_inference_time = time.time()

    print(f"\n{BOLD}{'='*70}")
    print("  IntelliCrash Phase 2 — Real-Time Edge Inference")
    print(f"  Threshold: P_final >= {CRASH_THRESHOLD}")
    print(f"  Buffer: {BUFFER_SIZE} samples ({BUFFER_SIZE/SAMPLING_RATE:.0f}s)")
    print(f"  Window: {WINDOW_SIZE} samples ({WINDOW_SIZE/SAMPLING_RATE:.0f}s)")
    print(f"  Inference interval: {INFERENCE_INTERVAL}s")
    print(f"{'='*70}{RESET}\n")

    try:
        while True:
            # ── Poll sensor ──
            sample = reader.read_sample()
            if sample is None:
                print(f"\n{YELLOW}[END] CSV replay complete.{RESET}")
                break

            ax, ay, az, gx, gy, gz = sample
            buf_ax.append(ax)
            buf_ay.append(ay)
            buf_az.append(az)
            buf_gx.append(gx)
            buf_gy.append(gy)
            buf_gz.append(gz)
            sample_count += 1

            # ── Run inference every INFERENCE_INTERVAL ──
            current_time = time.time()
            samples_since = sample_count % int(SAMPLING_RATE * INFERENCE_INTERVAL)

            if samples_since == 0 and len(buf_ax) >= WINDOW_SIZE:
                inference_count += 1

                # Extract last 2 seconds from buffer
                raw_ax = np.array(list(buf_ax))[-WINDOW_SIZE:]
                raw_ay = np.array(list(buf_ay))[-WINDOW_SIZE:]
                raw_gz = np.array(list(buf_gz))[-WINDOW_SIZE:]

                # Apply Butterworth to full buffer, then extract last 2s
                full_ax = np.array(list(buf_ax))
                full_ay = np.array(list(buf_ay))
                full_gz = np.array(list(buf_gz))

                if len(full_ax) >= 50:  # Need enough samples for filter
                    filt_ax = butterworth_lowpass(full_ax)[-WINDOW_SIZE:]
                    filt_ay = butterworth_lowpass(full_ay)[-WINDOW_SIZE:]
                    filt_gz = butterworth_lowpass(full_gz)[-WINDOW_SIZE:]
                else:
                    filt_ax, filt_ay, filt_gz = raw_ax, raw_ay, raw_gz

                # Build 6-channel window (same format as training)
                window = np.column_stack([
                    raw_ax, raw_ay, raw_gz,
                    filt_ax, filt_ay, filt_gz,
                ]).astype(np.float32)

                # ── 26 Features ──
                features = extract_features_single(window, fs=SAMPLING_RATE)

                # ── CSI ──
                csi_score = compute_csi_single(features)

                # ── Bi-LSTM ONNX Inference ──
                if onnx_session is not None:
                    # Broadcast features across timesteps
                    feat_broadcast = np.tile(features, (WINDOW_SIZE, 1))
                    lstm_input = np.concatenate([window, feat_broadcast], axis=1)
                    lstm_input = lstm_input[np.newaxis, :, :].astype(np.float32)

                    ort_input = {onnx_session.get_inputs()[0].name: lstm_input}
                    ort_crash, ort_sev = onnx_session.run(None, ort_input)
                    p_lstm = float(ort_crash[0][0])
                else:
                    p_lstm = 0.0  # No model loaded

                # ── Fusion ──
                p_final = LSTM_WEIGHT * p_lstm + CSI_WEIGHT * csi_score

                # ── Decision ──
                timestamp = time.strftime("%H:%M:%S")

                if p_final >= CRASH_THRESHOLD:
                    crash_count += 1
                    # Severity mapping
                    if csi_score >= 0.75:
                        severity = "FATAL"
                        color = RED
                    elif csi_score >= 0.40:
                        severity = "SEVERE"
                        color = RED
                    else:
                        severity = "MINOR"
                        color = YELLOW

                    print(f"  {color}{BOLD}[{timestamp}] 🚨 CRASH DETECTED | "
                          f"P_LSTM: {p_lstm:.3f} | CSI: {csi_score:.3f} | "
                          f"P_final: {p_final:.3f} | Severity: {severity}{RESET}")

                    # Dispatch alert
                    if alert_module is not None:
                        try:
                            alert_module.dispatch_crash_alert(
                                severity=severity,
                                p_final=p_final,
                                csi=csi_score,
                                p_lstm=p_lstm,
                            )
                        except Exception as e:
                            print(f"  {YELLOW}[ALERT ERROR] {e}{RESET}")
                else:
                    # Normal — print every 10th inference to avoid console flood
                    if inference_count % 10 == 0:
                        # Get actual label if in replay mode
                        actual = ""
                        if hasattr(reader, "get_current_label"):
                            actual = f" | Actual: {reader.get_current_label()}"

                        print(f"  {GREEN}[{timestamp}] ✓ NORMAL    | "
                              f"P_LSTM: {p_lstm:.3f} | CSI: {csi_score:.3f} | "
                              f"P_final: {p_final:.3f}{actual}{RESET}")

            # Simulate 100Hz sampling in replay mode
            if isinstance(reader, CSVReplayReader):
                time.sleep(0.001)  # Fast replay (1ms per sample instead of 10ms)

    except KeyboardInterrupt:
        print(f"\n{YELLOW}[STOP] Inference loop interrupted.{RESET}")

    # Summary
    print(f"\n{BOLD}{'='*70}")
    print(f"  INFERENCE SUMMARY")
    print(f"{'='*70}{RESET}")
    print(f"  Total samples processed: {sample_count:,}")
    print(f"  Total inferences run:    {inference_count:,}")
    print(f"  Crash detections:        {crash_count:,}")
    print(f"  Detection rate:          {100*crash_count/max(inference_count,1):.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="IntelliCrash Phase 2 Edge Inference")
    parser.add_argument("--live", action="store_true", help="Read MPU6050 via I2C (RPi only)")
    parser.add_argument("--replay", type=str, help="Replay a CSV file for testing")
    parser.add_argument("--no-alert", action="store_true", help="Disable Twilio alerts")
    args = parser.parse_args()

    if not args.live and not args.replay:
        print("Usage:")
        print("  Live mode (RPi):    python hardware/edge_inference.py --live")
        print("  Replay mode (PC):   python hardware/edge_inference.py --replay sensor_data.csv")
        sys.exit(1)

    # Load ONNX model
    onnx_session = load_onnx_model()

    # Load alert module
    alert_module = None
    if not args.no_alert:
        try:
            from hardware.alert_dispatch import AlertDispatcher
            alert_module = AlertDispatcher()
        except Exception as e:
            print(f"{YELLOW}[WARN] Alert module not loaded: {e}{RESET}")

    # Initialize reader
    if args.live:
        try:
            reader = MPU6050Reader()
        except Exception as e:
            print(f"{RED}[ERROR] Cannot initialize MPU6050: {e}{RESET}")
            sys.exit(1)
    else:
        csv_path = Path(args.replay)
        if not csv_path.is_absolute():
            csv_path = PROJECT_ROOT / csv_path
        if not csv_path.exists():
            print(f"{RED}[ERROR] CSV file not found: {csv_path}{RESET}")
            sys.exit(1)
        reader = CSVReplayReader(str(csv_path))

    # Run inference
    inference_loop(reader, onnx_session, alert_module)


if __name__ == "__main__":
    main()
