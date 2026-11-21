"""
IntelliCrash — Master Raspberry Pi Node Execution Script
==========================================================
Unified Edge Deployment:
- Phase 1 (Pre-Crash Monitoring): Real-time rash driving detection + Buzzer alert + SHAP attribution
- Phase 2 (Crash Detection): Bi-LSTM + CSI dual-path fusion + Kinematic impact gate
- Telematics & Evidence: NEO-8M GNSS + PiCam rolling forensic buffer
- Multi-Channel Dispatch: HTML Hospital Dossier + Telegram with GPS Pin + Twilio Call/SMS

Usage:
    python Raspberry/raspberry_pi_node.py --live
    python Raspberry/raspberry_pi_node.py --replay path/to/sensor.csv
"""

import argparse
import os
import sys
from pathlib import Path

HARDWARE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HARDWARE_DIR))

from edge_inference import (
    ISM330DHCXReader,
    CSVReplayReader,
    NEO8MGPSReader,
    inference_loop,
)
from alert_dispatch import AlertDispatcher
from rash_inference import PreCrashRashDetector
from buzzer import BuzzerController
from picam_recorder import PiCamCrashRecorder
from dashboard_bridge import DashboardBridge
import onnxruntime as ort

CYAN, GREEN, YELLOW, RED, BOLD, RESET = (
    "\033[96m",
    "\033[92m",
    "\033[93m",
    "\033[91m",
    "\033[1m",
    "\033[0m",
)


def main():
    parser = argparse.ArgumentParser(
        description="IntelliCrash Master Raspberry Pi Node"
    )
    parser.add_argument(
        "--live", action="store_true", help="Run live sensor streaming via I2C (SmartElex ISM330DHCX)"
    )
    parser.add_argument(
        "--replay", type=str, default=None, help="Replay a pre-recorded sensor CSV file"
    )
    parser.add_argument(
        "--no-gps", action="store_true", help="Disable GPS tracking"
    )
    parser.add_argument(
        "--no-calib", action="store_true", help="Disable startup zero-calibration"
    )
    parser.add_argument(
        "--no-buzzer", action="store_true", help="Disable audible driver buzzer"
    )
    parser.add_argument(
        "--no-cam", action="store_true", help="Disable camera video evidence recording"
    )
    parser.add_argument(
        "--no-precrash", action="store_true", help="Disable Phase-1 pre-crash monitoring"
    )
    parser.add_argument(
        "--enable-xai", action="store_true", help="Enable background SHAP forensic snapshots"
    )
    parser.add_argument(
        "--no-bridge", action="store_true", help="Disable dashboard WebSocket stream + SQLite logging"
    )
    parser.add_argument(
        "--ws-port", type=int, default=8765, help="Dashboard WebSocket port (default: 8765)"
    )
    parser.add_argument(
        "--dashboard-url", default=None,
        help="Dashboard server base URL for DB sync, e.g. http://192.168.1.10:3000 (or env INTELLICRASH_DASHBOARD_URL)"
    )
    parser.add_argument(
        "--ingest-key", default=None,
        help="Shared secret for /api/ingest (or env INTELLICRASH_INGEST_KEY; must match INGEST_KEY on the dashboard)"
    )
    parser.add_argument(
        "--vehicle-id", default="PB-11-AB-2027", help="Vehicle ID stored with every session/crash (default: PB-11-AB-2027)"
    )
    parser.add_argument(
        "--buzzer-pin", type=int, default=18, help="GPIO BCM Pin for active buzzer (default: 18)"
    )
    parser.add_argument(
        "--buzzer-active-low", action="store_true", help="Buzzer module sounds when the pin is LOW (many 3-pin modules)"
    )
    parser.add_argument(
        "--no-crash-beep", action="store_true", help="Do not sound the buzzer when a crash is confirmed (rash warning stays on)"
    )
    parser.add_argument(
        "--gps-port", default=None, help="GPS serial port, e.g. /dev/serial0 or /dev/ttyAMA0 (default: auto-detect)"
    )
    parser.add_argument(
        "--gps-baud", type=int, default=9600, help="GPS baud rate (NEO-8M default: 9600)"
    )
    parser.add_argument(
        "--demo-location", default=None, metavar="LAT,LON",
        help="Position saved (marked DEMO) when there is no GPS fix, e.g. 30.35487,76.36954 for indoor demos"
    )
    parser.add_argument(
        "--video-pre-seconds", type=int, default=15, help="Seconds of video kept BEFORE the crash (default: 15)"
    )
    parser.add_argument(
        "--video-post-seconds", type=int, default=15, help="Seconds of video recorded AFTER the crash (default: 15; clip = pre + post)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Crash probability threshold (default: tuned value in models/optimal_fusion_weights.json)",
    )
    parser.add_argument(
        "--min-impact",
        type=float,
        default=16.0,
        help="Minimum peak impact acceleration in m/s^2 to confirm wall crash (default: 16.0 m/s^2)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=5.0,
        help="Alert cooldown duration in seconds (default: 5.0)",
    )
    args = parser.parse_args()

    if not args.live and not args.replay:
        print(f"\n{YELLOW}[USAGE ERROR] Please specify either --live or --replay <csv_path>{RESET}")
        print("  On Raspberry Pi:   python Raspberry/raspberry_pi_node.py --live")
        print("  On Desktop/Laptop: python Raspberry/raspberry_pi_node.py --replay hardware/data/imu_session_0015.csv")
        sys.exit(1)

    print(f"\n{CYAN}{BOLD}{'=' * 70}")
    print("      INTELLICRASH UNIFIED DUAL-PHASE EDGE NODE INITIALIZATION")
    print(f"{'=' * 70}{RESET}")

    if args.demo_location:
        os.environ["DEMO_LOCATION"] = args.demo_location
        print(f"{YELLOW}[INFO] Demo location {args.demo_location} will be saved (marked DEMO) whenever there is no GPS fix.{RESET}")

    # 1. Phase-2 Bi-LSTM ONNX Crash Detection Engine
    print(f"{CYAN}[INIT 1/6] Loading Phase-2 Bi-LSTM Crash Model (ONNX Runtime)...{RESET}")
    model_path = HARDWARE_DIR / "models" / "hw_bilstm.onnx"
    onnx_session = ort.InferenceSession(str(model_path))
    print(f"{GREEN}          -> Bi-LSTM ONNX loaded successfully.{RESET}")

    # 2. Phase-1 Pre-Crash Rash Driving Classifier
    precrash_detector = None
    if not args.no_precrash:
        print(f"{CYAN}[INIT 2/6] Loading Phase-1 Pre-Crash Classifier (HistGradientBoosting / LightGBM)...{RESET}")
        try:
            precrash_detector = PreCrashRashDetector(enable_xai_reports=args.enable_xai)
            print(f"{GREEN}          -> Phase-1 Pre-Crash Detector ready.{RESET}")
        except Exception as e:
            print(f"{YELLOW}          -> [WARN] Phase-1 Pre-Crash init skipped: {e}{RESET}")
    else:
        print(f"{YELLOW}[INIT 2/6] Phase-1 Pre-Crash Classifier: DISABLED (--no-precrash){RESET}")

    # 3. Active Piezo Driver Buzzer (GPIO 18)
    buzzer = None
    if not args.no_buzzer:
        print(f"{CYAN}[INIT 3/6] Initializing Driver Warning Buzzer (BCM GPIO {args.buzzer_pin})...{RESET}")
        try:
            buzzer = BuzzerController(pin=args.buzzer_pin, enabled=True, active_low=args.buzzer_active_low)
        except Exception as e:
            print(f"{YELLOW}          -> [WARN] Buzzer init skipped: {e}{RESET}")
    else:
        print(f"{YELLOW}[INIT 3/6] Driver Warning Buzzer: DISABLED (--no-buzzer){RESET}")

    # 4. PiCam Rolling Crash Evidence Recorder
    camera_recorder = None
    if not args.no_cam:
        print(f"{CYAN}[INIT 4/6] Initializing PiCam Rolling Crash Witness Recorder...{RESET}")
        try:
            camera_recorder = PiCamCrashRecorder(buffer_seconds=args.video_pre_seconds, post_seconds=args.video_post_seconds, fps=15, enabled=True)
        except Exception as e:
            print(f"{YELLOW}          -> [WARN] PiCam recorder init skipped: {e}{RESET}")
    else:
        print(f"{YELLOW}[INIT 4/6] PiCam Rolling Recorder: DISABLED (--no-cam){RESET}")

    # 5. Multi-Channel Emergency Alert Dispatcher (Twilio, Telegram, Email)
    print(f"{CYAN}[INIT 5/6] Initializing Multi-Channel Emergency Alert Dispatcher...{RESET}")
    alert_module = AlertDispatcher()

    # 6. Sensor Ingestion & GNSS Reader
    print(f"{CYAN}[INIT 6/6] Initializing Sensor Hardware Readers...{RESET}")
    gps_reader = None
    if args.live and not args.no_gps:
        try:
            gps_reader = NEO8MGPSReader(port=args.gps_port, baudrate=args.gps_baud)
            print(f"{GREEN}          -> u-blox NEO-8M GNSS reader active.{RESET}")
        except Exception as e:
            print(f"{YELLOW}          -> [WARN] GPS reader skipped: {e}{RESET}")

    if args.live:
        reader = ISM330DHCXReader(bus=1, address=0x6B)
        print(f"{GREEN}          -> SmartElex ISM330DHCX 6DoF IMU online (I2C @ 0x6B).{RESET}")
    else:
        reader = CSVReplayReader(args.replay)
        print(f"{GREEN}          -> CSV Replay source online: {args.replay}{RESET}")

    print(f"\n{GREEN}{BOLD}[OK] All systems operational. Launching Dual-Phase Real-Time Loop...{RESET}\n")

    bridge = None
    if not args.no_bridge:
        bridge = DashboardBridge(
            port=args.ws_port,
            source="LIVE" if args.live else "REPLAY",
            vehicle_id=args.vehicle_id,
            dashboard_url=args.dashboard_url,
            ingest_key=args.ingest_key,
        )

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
        bridge=bridge,
        buzzer_on_crash=not args.no_crash_beep,
    )


if __name__ == "__main__":
    main()
