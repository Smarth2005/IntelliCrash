"""
IntelliCrash — MQTT Message Broker Service.

Provides an enterprise IoT pub/sub bridge between edge nodes (RPi4)
and the central FastAPI server via MQTT protocol (Mosquitto/HiveMQ).

Features:
- Subscribes to high-frequency telemetry topic (`intellicrash/v1/telemetry`)
- Subscribes to mission-critical crash topic (`intellicrash/v1/crashes`) with QoS 2
- Persists payloads to SQLite and broadcasts to connected WebSocket dashboards
- Resilient background worker with automatic reconnect and graceful offline handling
"""

import json
import logging
import threading
import time
from typing import Optional

try:
    import paho.mqtt.client as mqtt
    from paho.mqtt.enums import CallbackAPIVersion
    PAHO_AVAILABLE = True
except ImportError:
    PAHO_AVAILABLE = False

from src.backend.config import get_settings
from src.backend.database import get_db
from src.backend.ws.manager import ws_manager

logger = logging.getLogger("intellicrash.mqtt")


class MQTTBridgeService:
    """
    Manages MQTT connection and event dispatching.
    """

    def __init__(self):
        self.settings = get_settings()
        self.client: Optional[mqtt.Client] = None
        self._is_connected = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the background MQTT client if enabled or available."""
        if not PAHO_AVAILABLE:
            logger.warning("[MQTT] paho-mqtt is not installed. MQTT service disabled.")
            return

        if not self.settings.MQTT_ENABLED:
            logger.info("[MQTT] MQTT integration disabled in settings (MQTT_ENABLED=false).")
            return

        self._start_client()

    def _start_client(self):
        try:
            # Use CallbackAPIVersion.VERSION2 for paho-mqtt 2.x
            self.client = mqtt.Client(
                callback_api_version=CallbackAPIVersion.VERSION2,
                client_id=self.settings.MQTT_CLIENT_ID,
            )

            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message

            logger.info(
                f"[MQTT] Connecting to broker at {self.settings.MQTT_BROKER_HOST}:{self.settings.MQTT_BROKER_PORT}..."
            )
            self.client.connect_async(
                self.settings.MQTT_BROKER_HOST,
                self.settings.MQTT_BROKER_PORT,
                self.settings.MQTT_KEEPALIVE,
            )
            self.client.loop_start()

        except Exception as e:
            logger.warning(f"[MQTT] Could not connect to MQTT broker: {e}. Operating in REST-only mode.")

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            self._is_connected = True
            logger.info("[MQTT] Successfully connected to MQTT Broker!")

            # Subscribe to topics
            # Telemetry: QoS 1 (at least once)
            client.subscribe(self.settings.MQTT_TOPIC_TELEMETRY, qos=1)
            # Crashes: QoS 2 (exactly once for mission-critical crash payloads)
            client.subscribe(self.settings.MQTT_TOPIC_CRASHES, qos=2)
            logger.info(
                f"[MQTT] Subscribed to:\n"
                f"       - {self.settings.MQTT_TOPIC_TELEMETRY} (QoS 1)\n"
                f"       - {self.settings.MQTT_TOPIC_CRASHES} (QoS 2)"
            )
        else:
            logger.warning(f"[MQTT] Broker connection failed with reason code: {reason_code}")

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties=None):
        self._is_connected = False
        logger.warning(f"[MQTT] Disconnected from MQTT Broker (code: {reason_code})")

    def _on_message(self, client, userdata, msg):
        """Route incoming MQTT messages to database and WebSocket."""
        topic = msg.topic
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except Exception as e:
            logger.error(f"[MQTT] Failed to decode JSON payload on topic '{topic}': {e}")
            return

        if topic == self.settings.MQTT_TOPIC_TELEMETRY:
            self._handle_telemetry_packet(payload)
        elif topic == self.settings.MQTT_TOPIC_CRASHES:
            self._handle_crash_packet(payload)

    def _handle_telemetry_packet(self, data: dict):
        """Persist MQTT telemetry and broadcast to dashboard."""
        db = get_db()
        timestamp = data.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%S")

        with db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO telemetry_log
                    (vehicle_id, timestamp, accel_x, accel_y, accel_z,
                     gyro_z, crash_prob, severity_score, status, behavior_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data.get("vehicle_id", "vehicle-001"),
                    timestamp,
                    data.get("accel_x", 0.0),
                    data.get("accel_y", 0.0),
                    data.get("accel_z", 9.81),
                    data.get("gyro_z", 0.0),
                    data.get("crash_prob", 0.0),
                    data.get("severity_score", 0.0),
                    data.get("status", "Normal"),
                    data.get("behavior_score", 100.0),
                ),
            )
            conn.commit()

        # Push directly to WebSocket dashboard
        ws_payload = {
            "type": "telemetry",
            "source": "mqtt",
            **data,
            "timestamp": timestamp,
        }
        ws_manager.broadcast_sync(ws_payload)

    def _handle_crash_packet(self, data: dict):
        """Persist mission-critical crash event received via MQTT."""
        db = get_db()
        timestamp = data.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%S")

        with db.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO crash_events
                    (vehicle_id, timestamp, fused_score, ml_probability, csi_score,
                     severity, latitude, longitude, peak_accel_x, peak_accel_y,
                     peak_gyro_z, shap_summary, video_clip_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data.get("vehicle_id", "vehicle-001"),
                    timestamp,
                    data.get("fused_score", 1.0),
                    data.get("ml_probability", 1.0),
                    data.get("csi_score", 1.0),
                    data.get("severity", "SEVERE"),
                    data.get("latitude", 28.6139),
                    data.get("longitude", 77.2090),
                    data.get("peak_accel_x"),
                    data.get("peak_accel_y"),
                    data.get("peak_gyro_z"),
                    data.get("shap_summary"),
                    data.get("video_clip_path"),
                ),
            )
            conn.commit()
            event_id = cursor.lastrowid

        logger.critical(f"[MQTT] 🔥 CRASH EVENT RECORDED via MQTT (ID #{event_id})")

        ws_payload = {
            "type": "crash_alert",
            "source": "mqtt",
            "crash_event_id": event_id,
            **data,
            "timestamp": timestamp,
        }
        ws_manager.broadcast_sync(ws_payload)

    def publish_telemetry(self, payload: dict) -> bool:
        """Helper to publish telemetry over MQTT."""
        if not self.client or not self._is_connected:
            return False
        res = self.client.publish(
            self.settings.MQTT_TOPIC_TELEMETRY,
            json.dumps(payload),
            qos=1,
        )
        return res.rc == 0

    def publish_crash(self, payload: dict) -> bool:
        """Helper to publish crash event over MQTT with QoS 2."""
        if not self.client or not self._is_connected:
            return False
        res = self.client.publish(
            self.settings.MQTT_TOPIC_CRASHES,
            json.dumps(payload),
            qos=2,
        )
        return res.rc == 0

    def stop(self):
        """Gracefully stop MQTT loop on server shutdown."""
        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
                logger.info("[MQTT] MQTT client disconnected cleanly.")
            except Exception:
                pass


# Global singleton instance
mqtt_service = MQTTBridgeService()
