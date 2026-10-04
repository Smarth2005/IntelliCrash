"""
IntelliCrash - Real-Time Telemetry Server
========================================
Runs on the Raspberry Pi 4 (or local PC for testing).
Exposes a lightweight, zero-dependency HTTP server on port 5000.
Serves live IMU values and model states to the Next.js dashboard.

Supports:
1. Real Hardware Mode: Reads MPU6050/ISM330 sensors if connected.
2. Interactive Mock Mode: Fallback keyboard triggers for testing/demo backup.
"""

import http.server
import json
import threading
import time
import sys
import random
import numpy as np
from urllib.parse import urlparse, parse_qs
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.feature_engineering import extract_features_single, compute_csi, NUM_ENGINEERED_FEATURES

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("[WARNING] onnxruntime not found. Model inference will be mocked.")

# Global state
telemetry_state = {
    "status": "Normal",
    "accel_x": 0.0,
    "accel_y": 0.0,
    "accel_z": 9.81,
    "gyro_z": 0.0,
    "crash_prob": 1.2,
    "severity": 0.0,
    "behavior_score": 98,
    "call_status": "Standby",
    "mode": "Mock Mode (Keyboard Triggers Active)",
    "timestamp": ""
}
state_lock = threading.Lock()
sensor_device = None

# Try to initialize real hardware
try:
    import smbus2
    import struct
    
    I2C_BUS = 1
    MPU6050_ADDR = 0x68
    PWR_MGMT_1 = 0x6B
    SMPLRT_DIV = 0x19
    CONFIG = 0x1A
    GYRO_CONFIG = 0x1B
    ACCEL_CONFIG = 0x1C
    ACCEL_XOUT_H = 0x3B
    
    bus = smbus2.SMBus(I2C_BUS)
    # Wake up sensor
    bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0x00)
    time.sleep(0.1)
    bus.write_byte_data(MPU6050_ADDR, SMPLRT_DIV, 0x09) # 100 Hz
    bus.write_byte_data(MPU6050_ADDR, CONFIG, 0x01)
    bus.write_byte_data(MPU6050_ADDR, GYRO_CONFIG, 0x08) # ±500 dps
    bus.write_byte_data(MPU6050_ADDR, ACCEL_CONFIG, 0x08) # ±4g
    
    sensor_device = "MPU6050"
    telemetry_state["mode"] = "Hardware Mode (MPU6050 Connected)"
    print("[OK] MPU6050 IMU initialized successfully!")
except Exception as e:
    print(f"[!] Physical IMU sensor not found or I2C disabled ({e}).")
    print("[INFO] Initializing Interactive Keyboard Mock Mode for demo safety.")

# Load ONNX Model
ort_session = None
if ONNX_AVAILABLE:
    model_path = str(Path(__file__).resolve().parent.parent.parent / "outputs" / "models" / "onnx" / "intellicrash_lstm.onnx")
    if Path(model_path).exists():
        try:
            ort_session = ort.InferenceSession(model_path)
            print(f"[OK] ONNX model loaded from {model_path}")
        except Exception as e:
            print(f"[!] Failed to load ONNX model: {e}")
    else:
        print(f"[!] ONNX model not found at {model_path}. Please run train_bilstm.py first.")

# Background thread to poll sensors or run mock simulation
def sensor_loop():
    global telemetry_state
    
    while True:
        with state_lock:
            current_status = telemetry_state["status"]
        
        # If in a triggered crash/rash sequence, let it animate and decay
        if current_status == "CRASH DETECTED":
            # Simulate decaying crash pulse parameters
            time.sleep(0.1)
            continue
        elif current_status == "RASH DRIVING":
            # Simulate a temporary 2-second rash swerve duration
            time.sleep(0.1)
            continue
            
        if sensor_device == "MPU6050":
            try:
                # Read 14 bytes from MPU6050
                raw = bus.read_i2c_block_data(MPU6050_ADDR, ACCEL_XOUT_H, 14)
                
                def to_signed(high, low):
                    val = (high << 8) | low
                    return val - 65536 if val > 32767 else val
                
                # ±4g -> 8192 LSB/g -> multiply by 9.81 for m/s²
                ax = to_signed(raw[0], raw[1]) / 8192.0 * 9.81
                ay = to_signed(raw[2], raw[3]) / 8192.0 * 9.81
                az = to_signed(raw[4], raw[5]) / 8192.0 * 9.81
                
                # ±500 dps -> 65.5 LSB/dps
                gz = to_signed(raw[12], raw[13]) / 65.5
                
                # Dynamic model checks
                crash_prob = 1.2
                severity = 0.0
                status = "Normal"
                
                # Simple logic for physical test trigger (a tap)
                g_force = abs(ax) + abs(ay)
                if g_force > 25.0: # Crash impact (> 2.5G)
                    status = "CRASH DETECTED"
                    crash_prob = 98.4
                    severity = 0.76
                elif g_force > 12.0 or abs(gz) > 80.0: # Rash driving (Hard turn/brake)
                    status = "RASH DRIVING"
                    crash_prob = 15.6
                
                with state_lock:
                    telemetry_state["accel_x"] = round(ax, 3)
                    telemetry_state["accel_y"] = round(ay, 3)
                    telemetry_state["accel_z"] = round(az, 3)
                    telemetry_state["gyro_z"] = round(gz, 3)
                    telemetry_state["status"] = status
                    telemetry_state["crash_prob"] = crash_prob
                    telemetry_state["severity"] = severity
                    if status == "RASH DRIVING":
                        telemetry_state["behavior_score"] = max(40, telemetry_state["behavior_score"] - 8)
                    elif status == "Normal" and telemetry_state["behavior_score"] < 100:
                        telemetry_state["behavior_score"] = min(100, telemetry_state["behavior_score"] + 0.1)
                        
            except Exception as ex:
                print(f"Error reading IMU: {ex}")
                time.sleep(1.0)
        else:
            # Mock Mode: Add mild noise to static values
            with state_lock:
                telemetry_state["accel_x"] = round(random.uniform(-0.3, 0.3), 3)
                telemetry_state["accel_y"] = round(random.uniform(-0.2, 0.4), 3)
                telemetry_state["accel_z"] = round(9.81 + random.uniform(-0.1, 0.1), 3)
                telemetry_state["gyro_z"] = round(random.uniform(-2.0, 2.0), 3)
                telemetry_state["crash_prob"] = round(random.uniform(0.5, 2.5), 1)
                telemetry_state["severity"] = 0.0
                if telemetry_state["behavior_score"] < 100:
                    telemetry_state["behavior_score"] = min(100, telemetry_state["behavior_score"] + 0.2)
                    
        time.sleep(0.1) # 10 Hz telemetry loop for UI

# Keyboard handler to trigger mock events
def keyboard_handler():
    global telemetry_state
    
    print("\n----------------------------------------------------")
    print("[KEYBOARD] CONSOLE TRIGGER BOARD (Use keys to simulate events):")
    print("   [n] -> Normal Driving")
    print("   [r] -> Trigger Rash Driving (Braking/Swerving)")
    print("   [c] -> Trigger Fatal Crash Event (Twilio Alert)")
    print("   [q] -> Quit Server")
    print("----------------------------------------------------\n")
    
    while True:
        try:
            cmd = sys.stdin.readline().strip().lower()
            if cmd == 'n':
                with state_lock:
                    telemetry_state["status"] = "Normal"
                    telemetry_state["accel_x"] = 0.0
                    telemetry_state["accel_y"] = 0.0
                    telemetry_state["accel_z"] = 9.81
                    telemetry_state["gyro_z"] = 0.0
                    telemetry_state["crash_prob"] = 1.2
                    telemetry_state["severity"] = 0.0
                    telemetry_state["call_status"] = "Standby"
                print("[RESET] Resetting telemetry to: Normal Driving")
            elif cmd == 'r':
                print("[RASH] Simulating Rash Driving: Speed weave & hard cornering...")
                # Spawn a thread to animate a swerve
                def swerve():
                    global telemetry_state
                    with state_lock:
                        telemetry_state["status"] = "RASH DRIVING"
                        telemetry_state["accel_x"] = -3.8
                        telemetry_state["gyro_z"] = 115.0
                    time.sleep(0.8)
                    with state_lock:
                        telemetry_state["accel_x"] = 3.2
                        telemetry_state["gyro_z"] = -90.0
                    time.sleep(0.8)
                    with state_lock:
                        telemetry_state["status"] = "Normal"
                        telemetry_state["behavior_score"] = max(40, telemetry_state["behavior_score"] - 15)
                threading.Thread(target=swerve, daemon=True).start()
            elif cmd == 'c':
                print("[CRASH] Simulating CRASH IMPACT! Running real-time inference pipeline...")
                # Spawn a thread to run inference and animate crash
                def crash():
                    global telemetry_state
                    
                    # 1. Generate a mock severe crash window (200, 6)
                    window = np.zeros((200, 6), dtype=np.float32)
                    window[:, 2] = np.random.normal(0, 0.1, 200) # gyro_z
                    window[:, 0] = np.random.normal(0, 0.1, 200) # accel_x
                    window[:, 1] = np.random.normal(0, 0.1, 200) # accel_y
                    # Inject a massive spike simulating impact
                    window[100:110, 0] = np.random.normal(25.0, 5.0, 10)
                    window[100:110, 1] = np.random.normal(-50.0, 10.0, 10)
                    window[100:120, 2] = np.random.normal(150.0, 30.0, 20)
                    
                    # 2. Extract 26 physics features
                    start_time = time.time()
                    features = extract_features_single(window)
                    
                    # 3. Compute Physics CSI
                    features_batch = np.expand_dims(features, axis=0)
                    csi_score = compute_csi(features_batch, mode="real_car")[0]
                    
                    # 4. Prepare ONNX Input (1, 200, 32)
                    ml_prob = 0.0
                    severity_pred = 0.0
                    if ort_session:
                        features_expanded = np.repeat(features_batch[:, np.newaxis, :], 200, axis=1)
                        window_batch = np.expand_dims(window, axis=0)
                        input_tensor = np.concatenate([window_batch, features_expanded], axis=2).astype(np.float32)
                        
                        # 5. Run ONNX Inference
                        outputs = ort_session.run(None, {'input': input_tensor})
                        ml_prob = outputs[0][0][0]
                        severity_pred = outputs[1][0][0]
                    else:
                        ml_prob = 0.95 # Mock high probability if no ONNX
                    
                    # 6. Fusion Gate
                    fused_score = 0.5 * ml_prob + 0.5 * csi_score
                    latency = (time.time() - start_time) * 1000
                    
                    print(f"[PIPELINE] Inference complete in {latency:.1f}ms")
                    print(f"  -> ML Prob : {ml_prob:.3f}")
                    print(f"  -> CSI     : {csi_score:.3f}")
                    print(f"  -> FUSED   : {fused_score:.3f}")
                    
                    if fused_score > 0.50:
                        with state_lock:
                            telemetry_state["status"] = "CRASH DETECTED"
                            telemetry_state["accel_x"] = float(np.max(window[:, 0]))
                            telemetry_state["accel_y"] = float(np.min(window[:, 1]))
                            telemetry_state["accel_z"] = 14.8
                            telemetry_state["gyro_z"] = float(np.max(window[:, 2]))
                            telemetry_state["crash_prob"] = float(fused_score * 100)
                            telemetry_state["severity"] = float(csi_score)
                            telemetry_state["call_status"] = "Dispatching AI Call..."
                        
                        # Decaying oscillation for UI effect
                        for i in range(10):
                            time.sleep(0.1)
                            with state_lock:
                                telemetry_state["accel_x"] = round(telemetry_state["accel_x"] * -0.6, 3)
                                telemetry_state["accel_y"] = round(telemetry_state["accel_y"] * -0.6, 3)
                                telemetry_state["gyro_z"] = round(telemetry_state["gyro_z"] * -0.5, 3)
                        
                        time.sleep(2.0)
                        with state_lock:
                            telemetry_state["call_status"] = "Sent"
                            telemetry_state["behavior_score"] = 0
                    else:
                        print("[PIPELINE] Fusion gate rejected event.")
                
                threading.Thread(target=crash, daemon=True).start()
            elif cmd == 'q':
                print("Shutting down telemetry server.")
                sys.exit(0)
        except Exception as e:
            print(f"Keyboard input error: {e}")
            break

# CORS-Enabled HTTP Handler
class TelemetryHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress request spam logs to keep console clean for input
        return

    def end_headers(self):
        # Enable CORS
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed_url = urlparse(self.path)
        
        # Endpoint 1: GET /telemetry
        if parsed_url.path == '/telemetry':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            with state_lock:
                telemetry_state["timestamp"] = time.strftime('%H:%M:%S', time.localtime())
                payload = json.dumps(telemetry_state)
                
            self.wfile.write(payload.encode('utf-8'))
            
        # Endpoint 2: GET /trigger (Optional query parameter trigger)
        elif parsed_url.path == '/trigger':
            query_params = parse_qs(parsed_url.query)
            event_type = query_params.get('type', ['normal'])[0]
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            response = {"status": "ok", "triggered": event_type}
            
            # Change state based on trigger type
            if event_type == 'crash':
                def external_crash():
                    global telemetry_state
                    with state_lock:
                        telemetry_state["status"] = "CRASH DETECTED"
                        telemetry_state["crash_prob"] = 98.2
                        telemetry_state["severity"] = 0.84
                        telemetry_state["call_status"] = "Dispatching AI Call..."
                    time.sleep(3.0)
                    with state_lock:
                        telemetry_state["call_status"] = "Sent"
                threading.Thread(target=external_crash, daemon=True).start()
            elif event_type == 'rash':
                with state_lock:
                    telemetry_state["status"] = "RASH DRIVING"
                    telemetry_state["crash_prob"] = 18.0
            else:
                with state_lock:
                    telemetry_state["status"] = "Normal"
                    telemetry_state["crash_prob"] = 1.2
                    telemetry_state["severity"] = 0.0
                    telemetry_state["call_status"] = "Standby"
                    
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')

def run_server(port=5000):
    server = http.server.HTTPServer(('0.0.0.0', port), TelemetryHandler)
    print(f"[SERVER] Telemetry Server running on http://0.0.0.0:{port}")
    server.serve_forever()

if __name__ == '__main__':
    # Start background sensor reader
    t_sensor = threading.Thread(target=sensor_loop, daemon=True)
    t_sensor.start()
    
    # Start keyboard trigger listener
    t_kbd = threading.Thread(target=keyboard_handler, daemon=True)
    t_kbd.start()
    
    # Start server
    try:
        run_server(5000)
    except KeyboardInterrupt:
        print("\nStopping telemetry server.")
        sys.exit(0)
