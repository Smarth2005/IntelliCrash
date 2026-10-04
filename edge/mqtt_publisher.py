"""
IntelliCrash — Edge MQTT Publisher.

Simulates an IoT Edge device (e.g., Raspberry Pi 4 equipped with IMU)
publishing telemetry and crash telemetry directly to an MQTT Broker (Mosquitto)
with QoS 1 (telemetry) and QoS 2 (crash safety events).
"""

import time
import math
import random
import json
import argparse
import sys

try:
    import paho.mqtt.client as mqtt
    from paho.mqtt.enums import CallbackAPIVersion
except ImportError:
    print("[ERROR] paho-mqtt is required. Run: pip install paho-mqtt")
    sys.exit(1)


TOPIC_TELEMETRY = "intellicrash/v1/telemetry"
TOPIC_CRASHES = "intellicrash/v1/crashes"


def run_mqtt_edge_node(host: str = "localhost", port: int = 1883, duration: int = 20, scenario: str = "mixed"):
    print("==================================================")
    print("  IntelliCrash Edge MQTT Publisher (IoT Client)")
    print(f"  Target Broker: {host}:{port}")
    print(f"  Duration: {duration}s | Scenario: {scenario}")
    print("==================================================")

    client = mqtt.Client(
        callback_api_version=CallbackAPIVersion.VERSION2,
        client_id="rpi4-edge-node-001"
    )

    try:
        client.connect(host, port, keepalive=60)
        client.loop_start()
        print("[MQTT] Connected to broker successfully.\n")
    except Exception as e:
        print(f"[ERROR] Could not connect to MQTT broker at {host}:{port}: {e}")
        print("Ensure Mosquitto or Docker Compose is running.")
        sys.exit(1)

    start_time = time.time()
    t = 0.0
    crash_dispatched = False

    try:
        while time.time() - start_time < duration:
            t += 0.1
            elapsed = time.time() - start_time

            is_crash = False
            is_rash = False

            if scenario == "crash" or (scenario == "mixed" and 8.0 <= elapsed <= 11.0):
                is_crash = True
            elif scenario == "rash" or (scenario == "mixed" and 4.0 <= elapsed < 8.0):
                is_rash = True

            if is_crash:
                status = "CRASH DETECTED"
                accel_x = random.uniform(19.0, 28.0) * random.choice([1, -1])
                accel_y = random.uniform(-50.0, -38.0)
                accel_z = 9.81 + random.uniform(8.0, 15.0)
                gyro_z = random.uniform(140.0, 230.0)
                crash_prob = round(random.uniform(94.0, 99.8), 1)
                severity_score = round(random.uniform(0.80, 0.96), 2)
            elif is_rash:
                status = "RASH DRIVING"
                accel_x = 4.2 * math.sin(t * 3.2)
                accel_y = 2.8 * math.cos(t * 2.1)
                accel_z = 9.81 + random.uniform(-1.0, 1.0)
                gyro_z = 70.0 * math.sin(t * 2.8)
                crash_prob = round(random.uniform(18.0, 38.0), 1)
                severity_score = round(random.uniform(0.12, 0.28), 2)
            else:
                status = "Normal"
                accel_x = 0.25 * math.sin(t * 0.4)
                accel_y = 0.35 * math.cos(t * 0.5)
                accel_z = 9.81 + random.uniform(-0.15, 0.15)
                gyro_z = 1.5 * math.sin(t * 0.3)
                crash_prob = round(random.uniform(0.4, 2.0), 1)
                severity_score = 0.0

            telemetry_payload = {
                "vehicle_id": "vehicle-001",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "accel_x": round(accel_x, 3),
                "accel_y": round(accel_y, 3),
                "accel_z": round(accel_z, 3),
                "gyro_z": round(gyro_z, 2),
                "crash_prob": crash_prob,
                "severity_score": severity_score,
                "status": status,
                "behavior_score": 96.0 if not is_crash else 25.0,
            }

            # Publish telemetry (QoS 1: At least once delivery)
            client.publish(TOPIC_TELEMETRY, json.dumps(telemetry_payload), qos=1)

            # If crash triggered, publish to crash topic (QoS 2: Exactly once delivery)
            if is_crash and not crash_dispatched:
                crash_dispatched = True
                crash_payload = {
                    "vehicle_id": "vehicle-001",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "fused_score": round(crash_prob / 100.0, 3),
                    "ml_probability": 0.96,
                    "csi_score": severity_score,
                    "severity": "FATAL" if severity_score > 0.85 else "SEVERE",
                    "latitude": 30.3533,
                    "longitude": 76.3607,
                    "peak_accel_x": round(accel_x, 2),
                    "peak_accel_y": round(accel_y, 2),
                    "peak_gyro_z": round(gyro_z, 2),
                }
                client.publish(TOPIC_CRASHES, json.dumps(crash_payload), qos=2)
                print(f"\n  >>> [MQTT QoS 2 PUBLISHED] Crash Event #{status} to {TOPIC_CRASHES} <<<")

            print(f"[{elapsed:04.1f}s] MQTT -> {TOPIC_TELEMETRY} | Status: {status:<15} | Ax: {accel_x:+05.2f} Ay: {accel_y:+05.2f}", end="\r")
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nPublisher stopped by user.")
    finally:
        client.loop_stop()
        client.disconnect()
        print(f"\nMQTT session completed. Total run time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge MQTT Publisher for IntelliCrash")
    parser.add_argument("--host", default="localhost", help="MQTT Broker Host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT Broker Port")
    parser.add_argument("--duration", type=int, default=15, help="Publish duration in seconds")
    parser.add_argument("--scenario", default="mixed", choices=["normal", "rash", "crash", "mixed"])
    args = parser.parse_args()

    run_mqtt_edge_node(host=args.host, port=args.port, duration=args.duration, scenario=args.scenario)
