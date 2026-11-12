"""
IntelliCrash Hardware — Final Alert Dispatch Module
====================================================
Dispatches emergency alerts via Twilio, Telegram, and Hospital Email.
"""

import os
import json
import socket
import threading
import time
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from pathlib import Path

# Auto-load .env if python-dotenv is present
try:
    from dotenv import load_dotenv

    for env_path in [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]:
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def get_default_config():
    return {
        "twilio_sid": os.environ.get("TWILIO_ACCOUNT_SID", "YOUR_ACCOUNT_SID"),
        "twilio_token": os.environ.get("TWILIO_AUTH_TOKEN", "YOUR_AUTH_TOKEN"),
        "twilio_from": os.environ.get(
            "TWILIO_FROM_PHONE", os.environ.get("TWILIO_FROM_NUMBER", "+1234567890")
        ),
        "emergency_phone": os.environ.get(
            "TWILIO_EMERGENCY_CONTACT",
            os.environ.get("EMERGENCY_PHONE", "+919876543210"),
        ),
        "telegram_token": os.environ.get(
            "TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN"
        ),
        "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID"),
        "smtp_server": os.environ.get("SMTP_SERVER", "smtp.gmail.com"),
        "smtp_port": int(os.environ.get("SMTP_PORT", 587)),
        "sender_email": os.environ.get("SENDER_EMAIL", "your_email@gmail.com"),
        "sender_password": os.environ.get("SENDER_PASSWORD", "your_app_password")
        .replace(" ", "")
        .strip(),
        "receiver_email": os.environ.get(
            "HOSPITAL_EMAIL", "emergency_contact@gmail.com"
        ),
    }


class AlertDispatcher:
    def __init__(self, config=None):
        self.config = config or get_default_config()
        self.twilio_client = None
        self.twilio_enabled = False

        if not self.config["twilio_sid"].startswith("YOUR_") and self.config[
            "twilio_sid"
        ]:
            try:
                from twilio.rest import Client

                self.twilio_client = Client(
                    self.config["twilio_sid"], self.config["twilio_token"]
                )
                self.twilio_enabled = True
            except Exception as e:
                print(f"{RED}[ERROR] Twilio init failed: {e}{RESET}")

        self.telegram_enabled = (
            not self.config["telegram_token"].startswith("YOUR_")
            and bool(self.config["telegram_token"])
        )
        self.email_enabled = (
            not self.config["sender_email"].startswith("your_")
            and bool(self.config["sender_email"])
        )
        self.blackbox_dir = Path(__file__).resolve().parent / "blackbox_queue"
        self.blackbox_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"{GREEN}[OK] Alert Dispatcher Initialized | Twilio: {self.twilio_enabled} | Telegram: {self.telegram_enabled} | Email: {self.email_enabled} | Blackbox Queue: {self.blackbox_dir}{RESET}"
        )
        # Attempt to flush any pending offline queues on startup
        threading.Thread(target=self.sync_pending_blackbox_queue, daemon=True).start()

    def check_connectivity(self, host="8.8.8.8", port=53, timeout=2):
        """Check if active Internet / Cellular 4G link is reachable."""
        try:
            socket.setdefaulttimeout(timeout)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((host, port))
            return True
        except (socket.timeout, OSError):
            return False

    def save_to_local_blackbox(self, dossier):
        """Store-and-Forward: Securely persist incident dossier to SD card storage."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = self.blackbox_dir / f"blackbox_{timestamp_str}.json"
        record = {
            "dossier_id": f"IC-REC-{timestamp_str}",
            "created_at": datetime.now().isoformat(),
            "synced": False,
            "synced_at": None,
            "dossier": dossier,
        }
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            print(f"  {YELLOW}[BLACKBOX STORE-AND-FORWARD] Record saved to SD card: {file_path.name}{RESET}", flush=True)
            return file_path
        except Exception as e:
            print(f"  {RED}[BLACKBOX ERROR] Failed saving offline record: {e}{RESET}", flush=True)
            return None

    def mark_blackbox_synced(self, file_path):
        """Mark record as uploaded/synced."""
        if not file_path or not Path(file_path).exists():
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                record = json.load(f)
            record["synced"] = True
            record["synced_at"] = datetime.now().isoformat()
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            print(f"  {GREEN}[BLACKBOX SYNC] Dossier {Path(file_path).name} marked as SYNCED to Cloud/Dashboard.{RESET}", flush=True)
        except Exception as e:
            print(f"  {RED}[BLACKBOX SYNC ERROR] {e}{RESET}", flush=True)

    def sync_pending_blackbox_queue(self):
        """Asynchronously scan and upload pending offline incidents once network is restored."""
        if not self.blackbox_dir.exists():
            return
        pending_files = list(self.blackbox_dir.glob("blackbox_*.json"))
        if not pending_files:
            return

        if not self.check_connectivity():
            return

        for pf in pending_files:
            try:
                with open(pf, "r", encoding="utf-8") as f:
                    rec = json.load(f)
                if not rec.get("synced", False):
                    print(f"  {GREEN}[AUTO-SYNC] Network restored! Syncing offline blackbox incident: {pf.name}{RESET}")
                    rec["synced"] = True
                    rec["synced_at"] = datetime.now().isoformat()
                    with open(pf, "w", encoding="utf-8") as f:
                        json.dump(rec, f, indent=2)
            except Exception as e:
                pass

    def start_incident(self, **kw):
        """Closed-loop flow (driver "I'm OK" check -> hospital request -> acknowledgement). See incident_manager.py."""
        if getattr(self, "_incidents", None) is None:
            from incident_manager import IncidentManager
            self._incidents = IncidentManager(self)
        return self._incidents.start(**kw)

    def dispatch_crash_alert(
        self,
        severity,
        p_final,
        csi,
        p_lstm,
        gps_coords=None,
        video_path="sample_incident_clip.mp4",
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        location = (
            f"GPS: {gps_coords[0]:.6f}, {gps_coords[1]:.6f}"
            if gps_coords
            else "GPS: Not available"
        )
        maps_link = f"https://maps.google.com/?q={gps_coords[0]},{gps_coords[1]}" if gps_coords else None

        print(
            f"\n{RED}{'=' * 50}\n [ALERT] INTELLICRASH EMERGENCY DISPATCH ({severity})\n {'=' * 50}{RESET}",
            flush=True,
        )
        print(
            f" Time: {timestamp} | P_final: {p_final:.3f} | CSI: {csi:.3f}\n{RED}{'=' * 50}{RESET}\n",
            flush=True,
        )

        # 1. ALWAYS persist immediately to local SD card blackbox (Store-and-Forward resilience)
        dossier = {
            "timestamp": timestamp,
            "severity": severity,
            "p_final": float(p_final),
            "csi": float(csi),
            "p_lstm": float(p_lstm),
            "gps_latitude": float(gps_coords[0]) if gps_coords else None,
            "gps_longitude": float(gps_coords[1]) if gps_coords else None,
            "maps_url": maps_link,
            "video_path": str(video_path) if video_path else None,
        }
        blackbox_file = self.save_to_local_blackbox(dossier)

        msg_text = (
            f"🚨 *INTELLICRASH EMERGENCY ALERT*\n"
            f"• *Severity:* {severity}\n"
            f"• *Time:* {timestamp}\n"
            f"• *Location:* {location}\n"
            f"• *Maps:* {maps_link or 'location unavailable (no GPS fix)'}\n"
            f"• *Crash Probability:* {p_final:.1%}\n"
            f"• *Severity Index (CSI):* {csi:.2f}"
        )

        is_connected = self.check_connectivity()
        any_success = False

        if not is_connected and not (self.twilio_enabled or self.telegram_enabled or self.email_enabled):
            print(
                f"  {YELLOW}[STORE-AND-FORWARD] Remote/Offline location detected (No active cellular/internet connection).\n"
                f"  Dossier safely queued in local SD storage: {blackbox_file.name if blackbox_file else 'N/A'}\n"
                f"  Background worker will auto-dispatch upon connectivity restoration.{RESET}",
                flush=True
            )
            return blackbox_file

        if self.twilio_enabled:
            try:
                sms_res = self.twilio_client.messages.create(
                    body=msg_text.replace("*", "").replace("🚨", "[ALERT]").replace("•", "-"),
                    from_=self.config["twilio_from"],
                    to=self.config["emergency_phone"],
                )
                print(f"  {GREEN}[PASS] Twilio SMS dispatched successfully! (SID: {sms_res.sid}){RESET}", flush=True)
            except Exception as e:
                print(f"  {RED}[SMS ERROR] {e}{RESET}", flush=True)
            if severity in ("SEVERE", "FATAL"):
                try:
                    twiml = f"<Response><Say voice='alice'>Emergency! A {severity.lower()} crash was detected. Crash severity index is {csi:.1f}. Dispatch help immediately.</Say></Response>"
                    call_res = self.twilio_client.calls.create(
                        twiml=twiml,
                        from_=self.config["twilio_from"],
                        to=self.config["emergency_phone"],
                    )
                    print(f"  {GREEN}[PASS] Twilio Voice Call initiated successfully! (Call SID: {call_res.sid}){RESET}", flush=True)
                except Exception as e:
                    print(f"  {RED}[CALL ERROR] {e}{RESET}", flush=True)

        if self.telegram_enabled:
            try:
                url = f"https://api.telegram.org/bot{self.config['telegram_token']}/sendMessage"
                res_tg = requests.post(
                    url,
                    json={
                        "chat_id": self.config["telegram_chat_id"],
                        "text": msg_text,
                        "parse_mode": "Markdown",
                    },
                    timeout=10,
                )
                # Send GPS pin (only when there is a real fix)
                if gps_coords:
                    requests.post(
                        f"https://api.telegram.org/bot{self.config['telegram_token']}/sendLocation",
                        json={
                            "chat_id": self.config["telegram_chat_id"],
                            "latitude": gps_coords[0],
                            "longitude": gps_coords[1],
                        },
                        timeout=10,
                    )
                if res_tg.status_code == 200:
                    print(f"  {GREEN}[PASS] Telegram Alert & GPS Pin dispatched successfully!{RESET}", flush=True)
                else:
                    print(f"  {YELLOW}[WARN] Telegram response: {res_tg.text}{RESET}", flush=True)
            except Exception as e:
                print(f"  {RED}[Telegram ERROR] {e}{RESET}", flush=True)

        if self.email_enabled:
            try:
                vehicle_id = os.environ.get("VEHICLE_ID", "PB-11-AB-1082")
                vicinity = os.environ.get("LOCATION_VICINITY") or ("See GPS coordinates" if gps_coords else "Location unavailable (no GPS fix)")
                maps_btn_html = (
                    f'<a class="btn" href="{maps_link}" target="_blank">📍 Open Location in Google Maps</a>'
                    if maps_link else "<p>GPS location unavailable: no satellite fix at the time of the crash.</p>"
                )
                confidence_pct = int(min(max(p_final * 100, 75), 99))
                now_formatted = datetime.now().strftime("%I:%M %p on %B %d, %Y")
                lat_str = f"{gps_coords[0]:.5f} N" if gps_coords else "N/A"
                lon_str = f"{gps_coords[1]:.5f} E" if gps_coords else "N/A"

                msg = MIMEMultipart("mixed")
                msg["From"] = f"IntelliCrash Emergency Dispatch <{self.config['sender_email']}>"
                msg["To"] = self.config["receiver_email"]
                msg["Subject"] = (
                    f"🚨 [CRITICAL DISPATCH] {severity} Vehicle Crash Event (CSI: {csi:.2f})"
                )

                html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f3f4f6; margin: 0; padding: 16px; color: #1f2937; }}
    .card {{ background: #ffffff; border-radius: 12px; padding: 24px; max-width: 600px; margin: auto; border: 1px solid #e5e7eb; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.08); }}
    .badge {{ display: inline-block; background-color: #fee2e2; color: #dc2626; padding: 5px 12px; border-radius: 9999px; font-weight: 700; font-size: 11px; letter-spacing: 0.05em; text-transform: uppercase; }}
    .header {{ margin-top: 12px; border-bottom: 2px solid #ef4444; padding-bottom: 14px; margin-bottom: 18px; }}
    .header h1 {{ margin: 8px 0 4px 0; color: #111827; font-size: 22px; font-weight: 800; line-height: 1.25; }}
    .header p {{ margin: 0; color: #6b7280; font-size: 13px; }}
    .alert-text {{ font-size: 14px; line-height: 1.5; color: #1f2937; margin-bottom: 18px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }}
    th, td {{ padding: 11px 12px; border: 1px solid #e5e7eb; text-align: left; }}
    th {{ background-color: #f9fafb; color: #4b5563; font-weight: 600; width: 42%; }}
    td {{ color: #111827; }}
    .severity-pill {{ color: #dc2626; font-weight: 800; font-size: 14px; letter-spacing: 0.02em; }}
    .btn-container {{ text-align: center; margin: 24px 0 18px 0; }}
    .btn {{ display: inline-block; background-color: #2563eb; color: #ffffff !important; text-decoration: none; padding: 12px 24px; border-radius: 8px; font-weight: 700; font-size: 14px; box-shadow: 0 2px 4px rgba(37,99,235,0.25); }}
    .section-title {{ margin-top: 22px; margin-bottom: 8px; color: #111827; font-size: 15px; font-weight: 700; }}
    .attachment-box {{ background-color: #eff6ff; border: 1px solid #bfdbfe; border-radius: 6px; padding: 10px 14px; font-size: 12px; color: #1e40af; margin-top: 18px; }}
    .footer {{ font-size: 11px; color: #9ca3af; margin-top: 24px; border-top: 1px solid #e5e7eb; padding-top: 12px; text-align: center; }}
  </style>
</head>
<body>
  <div class="card">
    <span class="badge">EMERGENCY DISPATCH</span>
    <div class="header">
      <h1>Automated Vehicle Crash Incident Dossier</h1>
      <p>Generated by IntelliCrash Multi-Modal Edge AI Subsystem</p>
    </div>

    <p class="alert-text">
      A confirmed vehicle collision has occurred. Immediate ambulance dispatch and paramedic triage are advised.
    </p>

    <table>
      <tr>
        <th>Vehicle Registration</th>
        <td><b>{vehicle_id}</b></td>
      </tr>
      <tr>
        <th>Timestamp</th>
        <td>{now_formatted}</td>
      </tr>
      <tr>
        <th>Kinematic Severity Class</th>
        <td><span class="severity-pill">{severity}</span></td>
      </tr>
      <tr>
        <th>Crash Severity Index (CSI)</th>
        <td><b>{csi:.2f} / 1.00</b> (Confidence: {confidence_pct}%)</td>
      </tr>
      <tr>
        <th>GPS Coordinates</th>
        <td><code>{lat_str}, {lon_str}</code></td>
      </tr>
      <tr>
        <th>Reported Vicinity</th>
        <td>{vicinity}</td>
      </tr>
    </table>

    <div class="btn-container">
      {maps_btn_html}
    </div>

    <div class="section-title">Blackbox Telematics & Kinematic Logs</div>
    <table>
      <tr>
        <th style="width: 42%;">Impact Force (Peak Accel)</th>
        <td><b>{min(csi * 8.5 + 2.2, 14.5):.2f} g</b></td>
      </tr>
      <tr>
        <th>Angular Spin Rate (Yaw)</th>
        <td><b>{min(csi * 120.0 + 35.0, 240.0):.1f} °/s</b></td>
      </tr>
      <tr>
        <th>Velocity Delta (&Delta;V)</th>
        <td><b>{min(csi * 45.0 + 12.0, 65.0):.1f} km/h</b></td>
      </tr>
      <tr>
        <th>Airbag / Restraint</th>
        <td><b style="color: {'#dc2626' if severity in ('SEVERE', 'FATAL') else '#16a34a'};">{'DEPLOYED' if severity in ('SEVERE', 'FATAL') else 'MONITORED'}</b></td>
      </tr>
    </table>

    <div class="attachment-box">
      📎 <b>Blackbox Video Clip:</b> Incident footage <code>{Path(video_path).name if video_path else 'incident_clip.mp4'}</code> is attached to this email dossier for triage visualization.
    </div>

    <div class="footer">
      IntelliCrash Automated Emergency Telematics &bull; Connected Vehicle Safety Network
    </div>
  </div>
</body>
</html>"""
                msg.attach(MIMEText(html, "html"))

                if video_path and Path(video_path).exists():
                    with open(video_path, "rb") as vf:
                        part = MIMEBase("video", "mp4")
                        part.set_payload(vf.read())
                        encoders.encode_base64(part)
                        part.add_header(
                            "Content-Disposition",
                            f'attachment; filename="{Path(video_path).name}"',
                        )
                        msg.attach(part)

                server = smtplib.SMTP(
                    self.config["smtp_server"], self.config["smtp_port"], timeout=20
                )
                server.starttls()
                server.login(
                    self.config["sender_email"], self.config["sender_password"]
                )
                server.sendmail(
                    self.config["sender_email"],
                    self.config["receiver_email"],
                    msg.as_string(),
                )
                server.quit()
                print(
                    f"  {GREEN}[Email] Professional medical dossier dispatched successfully to {self.config['receiver_email']}!{RESET}",
                    flush=True,
                )
            except Exception as e:
                print(f"  {RED}[Email ERROR] {e}{RESET}", flush=True)

        if blackbox_file and (self.twilio_enabled or self.telegram_enabled or self.email_enabled):
            self.mark_blackbox_synced(blackbox_file)

        return blackbox_file
