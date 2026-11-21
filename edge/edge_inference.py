"""
IntelliCrash Edge Node — Unified Dual-Phase Edge Inference Engine
==================================================================
Runs both ML phases concurrently on the incoming sensor stream:
- Phase 1 (Pre-Crash Monitoring): Rash Driving Classifier (26 physical features)
    -> Triggers Active Piezo Buzzer (GPIO 18) & logs physical triggers / XAI
- Phase 2 (Crash Detection & CSI Fusion): Bi-LSTM ONNX + Physics CSI + Kinematic Gate
    -> Triggers PiCam video evidence recording & Emergency Multi-Channel Dispatch
       (HTML Email Dossier with Video, Telegram Alert with GPS Pin, Twilio Call/SMS)
"""

import numpy as np
import time
import sys
import json
import argparse
from pathlib import Path
from collections import deque
from scipy.signal import butter, filtfilt

HARDWARE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = HARDWARE_DIR.parent
sys.path.insert(0, str(HARDWARE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from feature_engineering import extract_features_single
from buzzer import BuzzerController
from rash_inference import PreCrashRashDetector
from picam_recorder import PiCamCrashRecorder

SAMPLING_RATE = 100
BUFFER_SIZE = 1000
WINDOW_SIZE = 200
INFERENCE_INTERVAL = 0.5
BUTTER_ORDER = 5
BUTTER_CUTOFF = 1.3

CSI_G_NORM = 28.0
CSI_DV_NORM = 150.0
CSI_WEIGHTS = {"peak_g": 0.35, "delta_v": 0.30, "jerk": 0.20, "duration": 0.15}

RED, GREEN, YELLOW, CYAN, BOLD, RESET = (
    "\033[91m",
    "\033[92m",
    "\033[93m",
    "\033[96m",
    "\033[1m",
    "\033[0m",
)


def butterworth_lowpass(
    data, cutoff=BUTTER_CUTOFF, fs=SAMPLING_RATE, order=BUTTER_ORDER
):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    mean_val = np.mean(data)
    filtered = filtfilt(b, a, data - mean_val)
    return filtered + mean_val


def compute_csi_single(features):
    peak_g, dv, jerk, accel_std = features[0], features[1], features[2], features[5]
    csi = (
        CSI_WEIGHTS["peak_g"] * min(peak_g / CSI_G_NORM, 1.0)
        + CSI_WEIGHTS["delta_v"] * min(dv / CSI_DV_NORM, 1.0)
        + CSI_WEIGHTS["jerk"] * min(jerk / (CSI_G_NORM * 10), 1.0)
        + CSI_WEIGHTS["duration"] * min(accel_std / 2.0, 1.0)
    )
    return max(0.0, min(1.0, csi))


class CSVReplayReader:
    """Replays a saved sensor CSV for offline testing (no hardware needed)."""

    def __init__(self, csv_path: str):
        import csv
        self._rows = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                norm_row = {k.strip().lower(): v for k, v in row.items() if k}
                self._rows.append(norm_row)
        self._idx = 0
        print(f"{GREEN}[OK] CSV Replay: {csv_path} ({len(self._rows)} samples){RESET}")

    def read_sample(self):
        if self._idx >= len(self._rows):
            return None
        row = self._rows[self._idx]
        self._idx += 1

        def get_val(keys):
            for k in keys:
                if k in row and row[k] != "":
                    try:
                        return float(row[k])
                    except ValueError:
                        pass
            return 0.0

        ax = get_val(["accel_x_m_s2", "accel_x", "accele_x", "ax"])
        ay = get_val(["accel_y_m_s2", "accel_y", "accele_y", "ay"])
        az = get_val(["accel_z_m_s2", "accel_z", "accele_z", "az"])
        gx = get_val(["gyro_x_rad_s", "gyro_x", "gx"])
        gy = get_val(["gyro_y_rad_s", "gyro_y", "gy"])
        gz = get_val(["gyro_z_rad_s", "gyro_z", "gz"])

        if abs(gz) > 10.0:
            gx = gx * (np.pi / 180.0)
            gy = gy * (np.pi / 180.0)
            gz = gz * (np.pi / 180.0)

        return float(ax), float(ay), float(az), float(gx), float(gy), float(gz)


class ISM330DHCXReader:
    """Reads SmartElex Micro ISM330DHCX 6DoF IMU via I2C on Raspberry Pi."""

    def __init__(self, bus=1, address=0x6B):
        import smbus2

        self.bus = smbus2.SMBus(bus)
        self.address = address

        # CTRL1_XL (0x10): 0x68 -> ODR code 0110 (416 Hz per datasheet, verify), Full Scale = ±4g. The loop polls at ~100 Hz.
        self.bus.write_byte_data(self.address, 0x10, 0x68)
        # CTRL2_G (0x11): Gyro ODR = 104 Hz, Full Scale = ±500 dps
        self.bus.write_byte_data(self.address, 0x11, 0x64)
        # CTRL3_C (0x12): Enable Block Data Update & Auto-Increment
        self.bus.write_byte_data(self.address, 0x12, 0x44)

        time.sleep(0.1)
        print(f"{GREEN}[OK] SmartElex ISM330DHCX initialized at 0x{address:02X}{RESET}")

    def read_sample(self):
        try:
            raw = self.bus.read_i2c_block_data(self.address, 0x22, 12)
        except Exception as e:
            print(f"{RED}[ERROR] I2C read failed from ISM330DHCX: {e}{RESET}")
            return None

        def to_signed(high, low):
            val = (high << 8) | low
            return val - 65536 if val > 32767 else val

        gx_dps = to_signed(raw[1], raw[0]) * 17.5 / 1000.0
        gy_dps = to_signed(raw[3], raw[2]) * 17.5 / 1000.0
        gz_dps = to_signed(raw[5], raw[4]) * 17.5 / 1000.0

        gx = gx_dps * (np.pi / 180.0)
        gy = gy_dps * (np.pi / 180.0)
        gz = gz_dps * (np.pi / 180.0)

        ax_g = to_signed(raw[7], raw[6]) * 0.122 / 1000.0
        ay_g = to_signed(raw[9], raw[8]) * 0.122 / 1000.0
        az_g = to_signed(raw[11], raw[10]) * 0.122 / 1000.0

        ax = ax_g * 9.80665
        ay = ay_g * 9.80665
        az = az_g * 9.80665

        return float(ax), float(ay), float(az), float(gx), float(gy), float(gz)


class NEO8MGPSReader:
    """Non-blocking background thread reading u-blox NEO-8M GNSS module.

    Supports multi-GNSS NMEA talker sentences ($GNRMC, $GNGGA, $GPRMC, $GPGGA)
    across standard serial ports (/dev/ttyS0, /dev/serial0, /dev/ttyAMA0, /dev/ttyUSB0).
    """

    def __init__(self, port=None, baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.latest_coords = None  # no position until the receiver reports a real fix
        self.last_fix_t = 0.0
        self.has_fix = False
        self.nmea_ok = False
        self.speed_kmh = 0.0
        self._running = True

        import threading

        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        try:
            import serial
        except ImportError:
            return

        candidate_ports = (
            [self.port]
            if self.port
            else [
                "/dev/ttyS0",
                "/dev/serial0",
                "/dev/ttyAMA0",
                "/dev/ttyUSB0",
                "/dev/ttyACM0",
            ]
        )
        ser = None
        for p in candidate_ports:
            try:
                cand = serial.Serial(p, baudrate=self.baudrate, timeout=1)
            except Exception:
                continue
            # a port that opens is not necessarily the GPS: wait up to 3 s for real NMEA text
            seen = False
            t_end = time.time() + 3.0
            while time.time() < t_end and not seen:
                try:
                    if cand.readline().decode("ascii", errors="replace").startswith("$G"):
                        seen = True
                except Exception:
                    break
            if seen:
                ser = cand
                print(f"{GREEN}[OK] u-blox NEO-8M GNSS connected on {p} ({self.baudrate} baud, NMEA data received){RESET}")
                break
            cand.close()
            print(f"{YELLOW}[GPS] {p} opened but no NMEA data within 3 s (check TX/RX wiring, baud rate, serial port setup).{RESET}")

        if not ser:
            print(f"{YELLOW}[GPS] No working GPS serial port found. Crash locations will be empty unless --demo-location is set.{RESET}")
            return

        t_start = time.time()
        warned_nofix = False
        while self._running:
            try:
                line = ser.readline().decode("ascii", errors="replace").strip()
                if line.startswith("$G"):
                    self.nmea_ok = True
                if (not self.has_fix) and (not warned_nofix) and self.nmea_ok and time.time() - t_start > 20:
                    warned_nofix = True
                    print(f"{YELLOW}[GPS] Receiver is talking but has NO FIX yet: it needs a clear view of the sky (indoors it never locks).{RESET}")
                if line.startswith(("$GNRMC", "$GPRMC")):
                    parts = line.split(",")
                    if len(parts) >= 7 and parts[2] == "A":
                        raw_lat = parts[3]
                        lat_dir = parts[4]
                        raw_lon = parts[5]
                        lon_dir = parts[6]

                        lat = float(raw_lat[:2]) + float(raw_lat[2:]) / 60.0
                        if lat_dir == "S":
                            lat = -lat

                        lon = float(raw_lon[:3]) + float(raw_lon[3:]) / 60.0
                        if lon_dir == "W":
                            lon = -lon

                        self.latest_coords = (round(lat, 5), round(lon, 5))
                        try:
                            self.speed_kmh = float(parts[7]) * 1.852  # knots -> km/h
                        except (IndexError, ValueError):
                            pass
                        if not self.has_fix:
                            print(f"{GREEN}[GPS] First fix acquired: {self.latest_coords[0]}, {self.latest_coords[1]}{RESET}")
                        self.has_fix = True
                        self.last_fix_t = time.time()
            except Exception:
                pass

    def get_coordinates(self, max_age=10.0):
        """Last real GNSS fix, or None if there is no fix or it is older than max_age seconds.
        Never returns a made-up position."""
        if self.latest_coords is None or (time.time() - self.last_fix_t) > max_age:
            return None
        return self.latest_coords

    def get_speed_kmh(self, max_age=10.0):
        return self.speed_kmh if self.get_coordinates(max_age) is not None else None


def parse_demo_location():
    """DEMO_LOCATION='lat,lon' (env, set by --demo-location). Used ONLY when there is no real GPS fix and always
    stored/labelled as source=DEMO, never passed off as a GPS reading."""
    import os
    try:
        a, b = os.environ.get("DEMO_LOCATION", "").split(",")
        return (round(float(a), 5), round(float(b), 5))
    except Exception:
        return None


def inference_loop(
    reader,
    onnx_session,
    alert_module=None,
    gps_reader=None,
    precrash_detector: PreCrashRashDetector = None,
    buzzer: BuzzerController = None,
    camera_recorder: PiCamCrashRecorder = None,
    crash_threshold: float = None,  # None -> use the validation-tuned value from optimal_fusion_weights.json
    min_impact_accel: float = 16.0,
    auto_calibrate: bool = True,
    alert_cooldown: float = 5.0,
    buzzer_cooldown: float = 3.0,
    bridge=None,
    buzzer_on_crash: bool = True,
):
    models_dir = HARDWARE_DIR / "models"
    with open(models_dir / "scaler_params.json") as f:
        scaler_meta = json.load(f)
    with open(models_dir / "optimal_fusion_weights.json") as f:
        fusion_meta = json.load(f)

    lstm_weight = fusion_meta.get("w_ml", 0.6)
    csi_weight = fusion_meta.get("w_csi", 0.4)
    if crash_threshold is None:
        crash_threshold = float(fusion_meta.get("threshold", 0.50))

    raw_mean = np.array(scaler_meta["raw_mean"], dtype=np.float32)
    raw_std = np.array(scaler_meta["raw_std"], dtype=np.float32)

    # RobustScaler parameters for 26 engineered features
    feat_center = np.array(scaler_meta["center"], dtype=np.float32)
    feat_scale = np.array(scaler_meta["scale"], dtype=np.float32)

    # ── Startup Zero-Calibration (Tare) ──────────────────────────────────────
    bias_ax, bias_ay, bias_gz = 0.0, 0.0, 0.0
    if auto_calibrate:
        print(f"\n{CYAN}{BOLD}[CALIBRATING] Keep RC car stationary on the floor for 1 second...{RESET}")
        calib_samples = []
        calib_start = time.time()
        while len(calib_samples) < 80 and (time.time() - calib_start) < 2.5:
            s = reader.read_sample()
            if s is not None:
                calib_samples.append(s)
            time.sleep(0.01)
        if calib_samples:
            calib_arr = np.array(calib_samples)
            bias_ax = float(np.mean(calib_arr[:, 0]))
            bias_ay = float(np.mean(calib_arr[:, 1]))
            bias_gz = float(np.mean(calib_arr[:, 5]))
        print(
            f"{GREEN}[CALIBRATED] Offsets zeroed -> Ax: {bias_ax:+.2f} m/s^2, "
            f"Ay: {bias_ay:+.2f} m/s^2, Gz: {bias_gz:+.3f} rad/s{RESET}\n"
        )

    buf_ax, buf_ay, buf_gz = (
        deque(maxlen=BUFFER_SIZE),
        deque(maxlen=BUFFER_SIZE),
        deque(maxlen=BUFFER_SIZE),
    )
    sample_count = 0
    last_alert_time = 0.0
    last_buzzer_time = 0.0

    print(
        f"\n{BOLD}{'=' * 75}\n  IntelliCrash Edge Node — Unified Dual-Phase Inference Running\n"
        f"  Phase 1 (Pre-Crash): {'ENABLED (Buzzer + Physics Attribution)' if precrash_detector else 'DISABLED'}\n"
        f"  Phase 2 (Crash Det): ML Weight {lstm_weight:.2f} | CSI Weight {csi_weight:.2f} | Threshold {crash_threshold:.2f} | Gate >= {min_impact_accel:.1f} m/s^2\n"
        f"  Emergency Dispatch:  {'Twilio / Telegram / Hospital Email' if alert_module else 'OFF'}\n"
        f"  Video Blackbox:      {'PiCam Rolling Forensic Buffer' if camera_recorder else 'OFF'}\n"
        f"{'=' * 75}{RESET}\n"
    )

    try:
        while True:
            sample = reader.read_sample()
            if sample is None:
                break
            ax, ay, az, _, _, gz = sample
            ax = ax - bias_ax
            ay = ay - bias_ay
            gz = gz - bias_gz

            if bridge is not None:
                bridge.push_sample(ax, ay, az, gz,
                                   gps_reader.get_speed_kmh() if gps_reader else None,
                                   gps_reader.get_coordinates() if gps_reader else None)

            buf_ax.append(ax)
            buf_ay.append(ay)
            buf_gz.append(gz)
            sample_count += 1

            if (
                sample_count % int(SAMPLING_RATE * INFERENCE_INTERVAL) == 0
                and len(buf_ax) >= WINDOW_SIZE
            ):
                raw_ax = np.array(list(buf_ax))[-WINDOW_SIZE:]
                raw_ay = np.array(list(buf_ay))[-WINDOW_SIZE:]
                raw_gz = np.array(list(buf_gz))[-WINDOW_SIZE:]

                filt_ax = butterworth_lowpass(raw_ax)
                filt_ay = butterworth_lowpass(raw_ay)
                filt_gz = butterworth_lowpass(raw_gz)

                # 6 Channels: [ax, ay, gz, ax_filtered, ay_filtered, gz_filtered]
                raw_window = np.column_stack(
                    [raw_ax, raw_ay, raw_gz, filt_ax, filt_ay, filt_gz]
                ).astype(np.float32)

                # 1. Extract 26 physics features from the physical-unit window
                features = extract_features_single(raw_window, fs=SAMPLING_RATE)
                csi_score = compute_csi_single(features)
                peak_accel = float(features[0])

                # ── Phase 1: Pre-Crash Monitoring & Buzzer Actuation ─────────
                is_rash = False
                rash_prob = 0.0
                trigger_factors = []
                now = time.time()

                if precrash_detector is not None:
                    is_rash, rash_label, rash_prob, trigger_factors = precrash_detector.predict(features)
                    if is_rash:
                        factor_str = ", ".join([f"{k}={v:.2f}" for k, v in trigger_factors])
                        print(
                            f"  {YELLOW}{BOLD}[PHASE-1 PRE-CRASH] RASH DRIVING MANEUVER DETECTED! | Prob: {rash_prob:.2f} | Triggers: [{factor_str}]{RESET}",
                            flush=True,
                        )
                        if buzzer is not None and (now - last_buzzer_time >= buzzer_cooldown):
                            last_buzzer_time = now
                            buzzer.trigger_rash_alert(count=2, duration=0.15, interval=0.08)

                # ── Phase 2: Crash Detection & CSI Dual-Path Fusion ──────────
                norm_window = (raw_window - raw_mean) / raw_std
                scaled_features = (features - feat_center) / (feat_scale + 1e-8)
                feat_broadcast = np.tile(scaled_features, (WINDOW_SIZE, 1))

                lstm_input = np.concatenate([norm_window, feat_broadcast], axis=1)[
                    np.newaxis, :, :
                ].astype(np.float32)

                input_name = onnx_session.get_inputs()[0].name
                ort_crash, _ = onnx_session.run(None, {input_name: lstm_input})
                p_lstm = float(ort_crash[0][0])

                p_final = (lstm_weight * p_lstm) + (csi_weight * csi_score)
                is_ml_crash = p_final >= crash_threshold
                is_impact = peak_accel >= min_impact_accel
                is_crash = is_ml_crash and is_impact
                if bridge is not None:
                    bridge.update_scores(p_lstm, csi_score, p_final, "CRASH DETECTED" if is_crash else ("RASH DRIVING" if is_rash else "NORMAL"))

                if is_crash:
                    severity = (
                        "FATAL"
                        if csi_score >= 0.75
                        else ("SEVERE" if csi_score >= 0.40 else "MINOR")
                    )
                    print(
                        f"  {RED}{BOLD}[PHASE-2 CRASH DETECTED] | Impact: {peak_accel:.1f} m/s^2 | P_final: {p_final:.3f} | LSTM: {p_lstm:.3f} | CSI: {csi_score:.3f} | Severity: {severity}{RESET}",
                        flush=True,
                    )
                    if alert_module and (now - last_alert_time >= alert_cooldown):
                        last_alert_time = now
                        # Real GNSS position at the moment of the crash (last fix within 60 s, since the impact
                        # may knock the antenna). None if the receiver never had a fix: no made-up location.
                        current_gps = gps_reader.get_coordinates(max_age=60.0) if gps_reader else None
                        location_source = "GPS" if current_gps else "NONE"
                        if current_gps is None:
                            demo_loc = parse_demo_location()
                            if demo_loc:
                                current_gps, location_source = demo_loc, "DEMO"
                        if location_source == "GPS":
                            print(f"  {GREEN}[LOCATION] Real GPS fix saved: {current_gps[0]}, {current_gps[1]}{RESET}", flush=True)
                        elif location_source == "DEMO":
                            print(f"  {YELLOW}[LOCATION] No GPS fix -> saving the configured DEMO position {current_gps[0]}, {current_gps[1]} (marked DEMO){RESET}", flush=True)
                        else:
                            why = "GPS disabled or replay mode" if gps_reader is None else "GPS has no fix (needs open sky)"
                            print(f"  {YELLOW}[LOCATION] No location available: {why}. Use --demo-location LAT,LON for indoor demos.{RESET}", flush=True)

                        # In-cabin alarm while the driver is asked "Are you OK?"
                        if buzzer is not None and buzzer_on_crash:
                            buzzer.trigger_crash_alert()

                        # Capture video clip evidence via PiCam
                        video_clip = None
                        if camera_recorder is not None:
                            try:
                                video_clip = camera_recorder.save_crash_clip(
                                    severity=severity,
                                    p_final=p_final,
                                    gps_coords=current_gps,
                                )
                            except Exception as e:
                                print(f"{YELLOW}[WARN] Could not save crash video clip: {e}{RESET}")

                        # Crash row first (so the incident can be linked to it), then the closed-loop flow runs in
                        # its own thread: driver "I'm OK" check -> hospital request -> acknowledgement.
                        crash_id = bridge.log_crash(severity, peak_accel, p_lstm, csi_score, p_final, current_gps, video_clip, location_source) if bridge is not None else None
                        alert_module.start_incident(
                            crash_id=crash_id,
                            severity=severity,
                            peak_accel=peak_accel,
                            p_final=p_final,
                            csi=csi_score,
                            p_lstm=p_lstm,
                            gps=current_gps,
                            video_path=video_clip,
                            bridge=bridge,
                            location_source=location_source,
                        )
                else:
                    status_note = ""
                    if is_ml_crash and not is_impact:
                        status_note = f" (Maneuver Filtered: Peak {peak_accel:.1f} < {min_impact_accel:.1f} m/s^2)"
                    rash_tag = f" | PreCrash: [RASH {rash_prob:.2f}]" if is_rash else " | PreCrash: [OK]"
                    print(
                        f"  {GREEN}[NORMAL]{rash_tag} | Impact: {peak_accel:.1f} m/s^2 | P_final: {p_final:.3f} | CSI: {csi_score:.3f}{status_note}{RESET}",
                        flush=True,
                    )

            time.sleep(0.005)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}[STOP] Loop terminated.{RESET}")
    finally:
        if buzzer:
            buzzer.cleanup()
        if camera_recorder:
            camera_recorder.stop()
        if bridge is not None:
            bridge.close()


def main():
    parser = argparse.ArgumentParser(description="IntelliCrash Unified Edge Node")
    parser.add_argument("--live", action="store_true", help="Read ISM330DHCX via I2C")
    parser.add_argument("--replay", type=str, help="Replay a CSV file for testing")
    parser.add_argument("--no-alert", action="store_true", help="Disable remote alerts")
    parser.add_argument("--no-gps", action="store_true", help="Disable GPS reading")
    parser.add_argument("--no-calib", action="store_true", help="Disable startup zero-calibration")
    parser.add_argument("--no-buzzer", action="store_true", help="Disable audible driver buzzer")
    parser.add_argument("--no-cam", action="store_true", help="Disable camera video evidence recording")
    parser.add_argument("--no-precrash", action="store_true", help="Disable Phase-1 pre-crash monitoring")
    parser.add_argument("--enable-xai", action="store_true", help="Enable background SHAP forensic report generation")
    parser.add_argument("--buzzer-pin", type=int, default=18, help="GPIO BCM Pin for active buzzer (default: 18)")
    parser.add_argument("--gps-port", type=str, default=None, help="Explicit serial port for NEO-8M")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Crash threshold (default: tuned value from models/optimal_fusion_weights.json)",
    )
    parser.add_argument(
        "--min-impact",
        type=float,
        default=16.0,
        help="Minimum peak impact acceleration in m/s^2 to confirm crash against wall (default: 16.0 m/s^2)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=5.0,
        help="Alert cooldown duration in seconds (default: 5.0)",
    )
    args = parser.parse_args()

    if not args.live and not args.replay:
        print("Usage:")
        print("  Live mode (RPi):    python Raspberry/edge_inference.py --live")
        print("  Replay mode (PC):   python Raspberry/edge_inference.py --replay sensor_data.csv")
        sys.exit(1)

    try:
        import onnxruntime as ort

        model_path = HARDWARE_DIR / "models" / "hw_bilstm.onnx"
        onnx_session = ort.InferenceSession(str(model_path))
    except Exception as e:
        print(f"{RED}[ERROR] Could not load ONNX model: {e}{RESET}")
        sys.exit(1)

    alert_module = None
    if not args.no_alert:
        try:
            from alert_dispatch import AlertDispatcher

            alert_module = AlertDispatcher()
        except Exception as e:
            print(f"{YELLOW}[WARN] Alert module not loaded: {e}{RESET}")

    gps_reader = None
    if args.live and not args.no_gps:
        try:
            gps_reader = NEO8MGPSReader(port=args.gps_port)
        except Exception as e:
            print(f"{YELLOW}[WARN] NEO-8M GPS reader could not be started: {e}{RESET}")

    precrash_detector = None
    if not args.no_precrash:
        try:
            precrash_detector = PreCrashRashDetector(enable_xai_reports=args.enable_xai)
        except Exception as e:
            print(f"{YELLOW}[WARN] Pre-Crash Detector could not be initialized: {e}{RESET}")

    buzzer = None
    if not args.no_buzzer:
        try:
            buzzer = BuzzerController(pin=args.buzzer_pin, enabled=True)
        except Exception as e:
            print(f"{YELLOW}[WARN] Buzzer could not be initialized: {e}{RESET}")

    camera_recorder = None
    if not args.no_cam:
        try:
            camera_recorder = PiCamCrashRecorder(buffer_seconds=15, fps=15, enabled=True)
        except Exception as e:
            print(f"{YELLOW}[WARN] PiCam recorder could not be initialized: {e}{RESET}")

    if args.live:
        reader = ISM330DHCXReader(bus=1, address=0x6B)
    else:
        reader = CSVReplayReader(str(args.replay))

    inference_loop(
        reader,
        onnx_session,
        alert_module=alert_module,
        gps_reader=gps_reader,
        precrash_detector=precrash_detector,
        buzzer=buzzer,
        camera_recorder=camera_recorder,
        crash_threshold=args.threshold,
        min_impact_accel=args.min_impact,
        auto_calibrate=not args.no_calib,
        alert_cooldown=args.cooldown,
    )


if __name__ == "__main__":
    main()
