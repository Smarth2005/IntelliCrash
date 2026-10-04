"""
IntelliCrash — Edge to FastAPI Backend Streamer.

Simulates an IoT Edge device (Raspberry Pi 4) sending real-time
telemetry to the FastAPI backend.

Features:
- Periodic telemetry streaming (10 Hz)
- Realistic physics simulation (normal driving, rash swerving, crash collision)
- Interactive triggers or automated sequence
"""

import time
import math
import random
import argparse
import httpx

BACKEND_URL = "http://127.0.0.1:8000"

def stream_telemetry(backend_url: str = BACKEND_URL, duration_sec: int = 30, scenario: str = "mixed"):
    print(f"==================================================")
    print(f"  IntelliCrash Edge Streamer")
    print(f"  Target: {backend_url}")
    print(f"  Duration: {duration_sec}s | Scenario: {scenario}")
    print(f"==================================================")

    client = httpx.Client(base_url=backend_url, timeout=2.0)
    start_time = time.time()
    t = 0.0
    behavior_score = 98.0
    crash_dispatched = False

    try:
        while time.time() - start_time < duration_sec:
            t += 0.1
            elapsed = time.time() - start_time

            # Determine mode based on scenario and elapsed time
            is_crash = False
            is_rash = False

            if scenario == "crash" or (scenario == "mixed" and 10.0 <= elapsed <= 13.0):
                is_crash = True
            elif scenario == "rash" or (scenario == "mixed" and 5.0 <= elapsed < 10.0):
                is_rash = True

            if is_crash:
                status = "CRASH DETECTED"
                accel_x = random.uniform(18.0, 26.0) * random.choice([1, -1])
                accel_y = random.uniform(-45.0, -35.0)  # Violent forward impact deceleration
                accel_z = 9.81 + random.uniform(6.0, 14.0)
                gyro_z = random.uniform(120.0, 210.0)
                crash_prob = round(random.uniform(92.0, 99.5), 1)
                severity_score = round(random.uniform(0.75, 0.95), 2)
                behavior_score = max(20.0, behavior_score - 10.0)
            elif is_rash:
                status = "RASH DRIVING"
                accel_x = 4.5 * math.sin(t * 3.0) + random.uniform(-0.8, 0.8)
                accel_y = 3.0 * math.cos(t * 2.0) + random.uniform(-0.5, 0.5)
                accel_z = 9.81 + random.uniform(-1.0, 1.0)
                gyro_z = 65.0 * math.sin(t * 2.5)
                crash_prob = round(random.uniform(15.0, 35.0), 1)
                severity_score = round(random.uniform(0.1, 0.25), 2)
                behavior_score = max(50.0, behavior_score - 0.5)
            else:
                status = "Normal"
                accel_x = 0.3 * math.sin(t * 0.5) + random.uniform(-0.1, 0.1)
                accel_y = 0.4 * math.cos(t * 0.4) + random.uniform(-0.1, 0.1)
                accel_z = 9.81 + random.uniform(-0.2, 0.2)
                gyro_z = 2.0 * math.sin(t * 0.3) + random.uniform(-0.5, 0.5)
                crash_prob = round(random.uniform(0.5, 2.5), 1)
                severity_score = 0.0
                behavior_score = min(100.0, behavior_score + 0.1)

            payload = {
                "vehicle_id": "vehicle-001",
                "accel_x": round(accel_x, 3),
                "accel_y": round(accel_y, 3),
                "accel_z": round(accel_z, 3),
                "gyro_z": round(gyro_z, 2),
                "crash_prob": crash_prob,
                "severity_score": severity_score,
                "status": status,
                "behavior_score": round(behavior_score, 1),
            }

            try:
                client.post("/api/v1/telemetry", json=payload)
            except Exception as e:
                print(f"[ERR] Failed to post telemetry: {e}")

            # If crash detected for the first time, log full crash event
            if is_crash and not crash_dispatched:
                crash_dispatched = True
                crash_payload = {
                    "vehicle_id": "vehicle-001",
                    "fused_score": round(crash_prob / 100.0, 3),
                    "ml_probability": 0.94,
                    "csi_score": severity_score,
                    "severity": "SEVERE" if severity_score > 0.7 else "MINOR",
                    "latitude": 30.3533,
                    "longitude": 76.3607,
                    "peak_accel_x": round(accel_x, 2),
                    "peak_accel_y": round(accel_y, 2),
                    "peak_gyro_z": round(gyro_z, 2),
                }
                try:
                    res = client.post("/api/v1/crashes", json=crash_payload)
                    crash_id = res.json().get("id")
                    print(f"  >>> [CRASH EVENT CREATED] Event ID #{crash_id} - Severity: {crash_payload['severity']} <<<")
                    # Trigger alert dispatch
                    alert_res = client.post("/api/v1/alerts/dispatch", json={
                        "crash_event_id": crash_id,
                        "channels": ["sms", "email"]
                    })
                    print(f"  >>> [ALERT DISPATCHED] {alert_res.status_code} <<<")
                except Exception as e:
                    print(f"[ERR] Failed to log crash event: {e}")

            print(f"[{elapsed:04.1f}s] {status:<15} | Ax: {accel_x:+05.2f} Ay: {accel_y:+05.2f} Gz: {gyro_z:+06.1f} | Crash%: {crash_prob:04.1f}%", end="\r")
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")
    finally:
        client.close()
        print(f"\nStream completed. Elapsed: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream telemetry to IntelliCrash FastAPI backend")
    parser.add_argument("--url", default=BACKEND_URL, help="FastAPI backend URL")
    parser.add_argument("--duration", type=int, default=15, help="Streaming duration in seconds")
    parser.add_argument("--scenario", default="mixed", choices=["normal", "rash", "crash", "mixed"], help="Simulation scenario")
    args = parser.parse_args()

    stream_telemetry(backend_url=args.url, duration_sec=args.duration, scenario=args.scenario)
