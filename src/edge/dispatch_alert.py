"""
IntelliCrash — AI Alert Dispatcher (SMS & Email)

This script is triggered by the edge inference system when a crash is detected.
It calculates the kinematic severity (Minor, Severe, Fatal) based on the CSI 
score and dispatches a silent alert via Twilio (SMS) and SMTP (Email).
"""

import os
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from twilio.rest import Client
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import get_config

def send_sms(twilio_cfg, severity, csi, lat, lon):
    account_sid = twilio_cfg.get("account_sid")
    auth_token = twilio_cfg.get("auth_token")
    from_number = twilio_cfg.get("from_number")
    to_number = twilio_cfg.get("emergency_contact")

    if not all([account_sid, auth_token, from_number, to_number]):
        print("Twilio credentials missing. SMS skipped.")
        return False

    try:
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=f"INTELLICRASH ALERT: {severity} crash detected (CSI: {csi:.2f}). Location: https://maps.google.com/?q={lat},{lon}",
            from_=from_number,
            to=to_number
        )
        print(f"SMS Alert dispatched! SID: {message.sid}")
        return True
    except Exception as e:
        print(f"Failed to send SMS: {e}")
        return False

def send_email(email_cfg, severity, csi, lat, lon):
    smtp_server = email_cfg.get("smtp_server", "smtp.gmail.com")
    smtp_port = email_cfg.get("smtp_port", 587)
    sender_email = email_cfg.get("sender_email")
    sender_password = email_cfg.get("sender_password")
    hospital_email = email_cfg.get("hospital_email")

    if not all([sender_email, sender_password, hospital_email]):
        print("Email credentials missing. Email skipped.")
        return False

    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = hospital_email
        msg['Subject'] = f"EMERGENCY DISPATCH: {severity} Vehicle Crash Detected"

        body = f"""
        AUTOMATED INTELLICRASH ALERT
        ============================
        A {severity} vehicle crash has been detected.
        
        Kinematic Severity Score (CSI): {csi:.2f}
        
        Location:
        Latitude: {lat}
        Longitude: {lon}
        Google Maps: https://maps.google.com/?q={lat},{lon}
        
        Please dispatch emergency services immediately.
        """
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        print(f"Hospital Email Alert dispatched to {hospital_email}!")
        return True
    except Exception as e:
        print(f"Failed to send Email: {e}")
        return False

def dispatch(csi, lat, lon):
    # Map CSI to discrete severity
    if csi < 0.40:
        severity = "MINOR"
    elif csi < 0.75:
        severity = "SEVERE"
    else:
        severity = "FATAL"
        
    print(f"Dispatching alerts for {severity} crash (CSI: {csi:.2f})...")
    
    cfg = get_config()
    twilio_cfg = cfg.get("twilio", {})
    email_cfg = cfg.get("email", {})
    
    send_sms(twilio_cfg, severity, csi, lat, lon)
    send_email(email_cfg, severity, csi, lat, lon)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csi", type=float, required=True, help="Crash Severity Index (0.0 to 1.0)")
    parser.add_argument("--lat", type=float, default=28.6139, help="Latitude")
    parser.add_argument("--lon", type=float, default=77.2090, help="Longitude")
    args = parser.parse_args()
    
    dispatch(args.csi, args.lat, args.lon)
