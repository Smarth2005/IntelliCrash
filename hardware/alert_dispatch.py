"""
IntelliCrash Hardware — Alert Dispatch Module
==============================================
Sends crash alerts via Twilio SMS and AI Voice Call.

Setup:
    1. pip install twilio
    2. Set environment variables or edit TWILIO_CONFIG below
    3. Get a free Twilio account: https://twilio.com/try-twilio

Usage:
    from hardware.alert_dispatch import AlertDispatcher
    dispatcher = AlertDispatcher()
    dispatcher.dispatch_crash_alert(severity="SEVERE", p_final=0.82, csi=0.71, p_lstm=0.93)
"""

import os
import time
from datetime import datetime

# ── ANSI Colors ──
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# ══════════════════════════════════════════════════════════════════════════════
#  TWILIO CONFIGURATION
#  Set via environment variables or edit directly below
# ══════════════════════════════════════════════════════════════════════════════

TWILIO_CONFIG = {
    "account_sid": os.environ.get("TWILIO_ACCOUNT_SID", "YOUR_ACCOUNT_SID"),
    "auth_token": os.environ.get("TWILIO_AUTH_TOKEN", "YOUR_AUTH_TOKEN"),
    "from_number": os.environ.get("TWILIO_FROM_NUMBER", "+1234567890"),
    "emergency_contact": os.environ.get("TWILIO_EMERGENCY_CONTACT", "+919876543210"),
}


class AlertDispatcher:
    """Handles crash alert dispatch via SMS and Voice Call."""

    def __init__(self, config=None):
        self.config = config or TWILIO_CONFIG
        self.client = None
        self.enabled = False

        # Validate configuration
        if self.config["account_sid"].startswith("YOUR_"):
            print(f"{YELLOW}[ALERT] Twilio not configured. Alerts will be logged to console only.{RESET}")
            print(f"  Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN env vars or edit alert_dispatch.py")
            return

        try:
            from twilio.rest import Client
            self.client = Client(self.config["account_sid"], self.config["auth_token"])
            self.enabled = True
            print(f"{GREEN}[OK] Twilio client initialized. Alerts ACTIVE.{RESET}")
        except ImportError:
            print(f"{YELLOW}[WARN] twilio package not installed. pip install twilio{RESET}")
        except Exception as e:
            print(f"{RED}[ERROR] Twilio init failed: {e}{RESET}")

    def dispatch_crash_alert(self, severity, p_final, csi, p_lstm, gps_coords=None):
        """Dispatch crash alert via SMS + Voice Call.

        Args:
            severity: "MINOR", "SEVERE", or "FATAL"
            p_final: Final fusion probability
            csi: CSI score
            p_lstm: Bi-LSTM probability
            gps_coords: Optional (lat, lon) tuple
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        location = f"GPS: {gps_coords[0]:.6f}, {gps_coords[1]:.6f}" if gps_coords else "GPS: Not available"

        # Always log to console
        print(f"\n  {RED}{'='*50}")
        print(f"  🚨 CRASH ALERT DISPATCHED")
        print(f"  {'='*50}{RESET}")
        print(f"  Time:     {timestamp}")
        print(f"  Severity: {severity}")
        print(f"  P_final:  {p_final:.3f}")
        print(f"  P_LSTM:   {p_lstm:.3f}")
        print(f"  CSI:      {csi:.3f}")
        print(f"  {location}")
        print(f"  {RED}{'='*50}{RESET}\n")

        if not self.enabled:
            print(f"  {YELLOW}[DRY RUN] Twilio not configured — alert logged to console only.{RESET}")
            return

        # ── Send SMS ──
        self._send_sms(severity, p_final, csi, timestamp, location)

        # ── Make Voice Call ──
        if severity in ("SEVERE", "FATAL"):
            self._make_voice_call(severity, csi, timestamp, location)

    def _send_sms(self, severity, p_final, csi, timestamp, location):
        """Send SMS alert via Twilio."""
        message_body = (
            f"🚨 INTELLICRASH ALERT\n"
            f"Crash Detected: {severity}\n"
            f"Time: {timestamp}\n"
            f"{location}\n"
            f"Crash Probability: {p_final:.1%}\n"
            f"Severity Index: {csi:.2f}\n"
            f"— IntelliCrash Emergency System"
        )

        try:
            msg = self.client.messages.create(
                body=message_body,
                from_=self.config["from_number"],
                to=self.config["emergency_contact"],
            )
            print(f"  {GREEN}[SMS] Sent to {self.config['emergency_contact']} (SID: {msg.sid}){RESET}")
        except Exception as e:
            print(f"  {RED}[SMS ERROR] {e}{RESET}")

    def _make_voice_call(self, severity, csi, timestamp, location):
        """Make automated voice call with TTS crash details."""
        twiml_speech = (
            f"<Response><Say voice='alice'>"
            f"Emergency alert from IntelliCrash system. "
            f"A {severity.lower()} crash has been detected at {timestamp}. "
            f"The crash severity index is {csi:.1f}. "
            f"Please dispatch emergency services immediately. "
            f"Repeating: A {severity.lower()} crash has been detected. "
            f"Crash severity index is {csi:.1f}."
            f"</Say></Response>"
        )

        try:
            call = self.client.calls.create(
                twiml=twiml_speech,
                from_=self.config["from_number"],
                to=self.config["emergency_contact"],
            )
            print(f"  {GREEN}[CALL] Initiated to {self.config['emergency_contact']} (SID: {call.sid}){RESET}")
        except Exception as e:
            print(f"  {RED}[CALL ERROR] {e}{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing IntelliCrash Alert Dispatcher...\n")
    dispatcher = AlertDispatcher()

    # Test dispatch (will be dry-run if Twilio not configured)
    dispatcher.dispatch_crash_alert(
        severity="SEVERE",
        p_final=0.82,
        csi=0.71,
        p_lstm=0.93,
        gps_coords=(30.3515, 76.3526),  # Thapar University coordinates
    )
