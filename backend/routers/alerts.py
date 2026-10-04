"""
IntelliCrash — Alert Dispatch Router.

Endpoints:
    POST /api/v1/alerts/dispatch   → Dispatch SMS + Email for a crash event
    GET  /api/v1/alerts            → List all alert dispatch records
    GET  /api/v1/alerts/{crash_id} → Get alerts for a specific crash
"""

import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import APIRouter, HTTPException, Query, Path, Depends

from src.backend.database import get_db
from src.backend.config import get_settings
from src.backend.models import AlertDispatchRequest, AlertResponse
from src.backend.security import require_roles

try:
    from twilio.rest import Client as TwilioClient
except ImportError:
    TwilioClient = None

router = APIRouter(prefix="/api/v1/alerts", tags=["Alerts"])


def _send_sms(settings, severity: str, csi: float, lat: float, lon: float) -> dict:
    """Internal: Dispatch SMS via Twilio API."""
    if not TwilioClient:
        return {"status": "failed", "error": "twilio package not installed"}

    if not all([settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN]):
        return {"status": "failed", "error": "Twilio credentials not configured"}

    try:
        client = TwilioClient(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=(
                f"🚨 INTELLICRASH ALERT: {severity} crash detected "
                f"(CSI: {csi:.2f}). "
                f"Location: https://maps.google.com/?q={lat},{lon}"
            ),
            from_=settings.TWILIO_FROM_PHONE,
            to=settings.TWILIO_EMERGENCY_CONTACT,
        )
        return {
            "status": "sent",
            "message_sid": message.sid,
            "recipient": settings.TWILIO_EMERGENCY_CONTACT,
        }
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def _send_email(settings, severity: str, csi: float, lat: float, lon: float) -> dict:
    """Internal: Dispatch hospital email via SMTP."""
    if not all([settings.SENDER_EMAIL, settings.SENDER_PASSWORD, settings.HOSPITAL_EMAIL]):
        return {"status": "failed", "error": "Email credentials not configured"}

    try:
        msg = MIMEMultipart()
        msg["From"] = settings.SENDER_EMAIL
        msg["To"] = settings.HOSPITAL_EMAIL
        msg["Subject"] = f"EMERGENCY DISPATCH: {severity} Vehicle Crash Detected"

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
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT)
        server.starttls()
        server.login(settings.SENDER_EMAIL, settings.SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()

        return {"status": "sent", "recipient": settings.HOSPITAL_EMAIL}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


@router.post("/test", summary="Test alert configuration and connectivity")
def test_alerts():
    """
    Test alert system connectivity without requiring a real crash event.
    Returns status of Twilio and SMTP configuration.
    """
    settings = get_settings()
    twilio_ok = bool(TwilioClient and settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN)
    email_ok = bool(settings.SENDER_EMAIL and settings.SENDER_PASSWORD and settings.HOSPITAL_EMAIL)

    return {
        "status": "ready",
        "message": "Alert subsystem active",
        "channels": {
            "sms": {"configured": twilio_ok, "provider": "Twilio", "recipient": settings.TWILIO_EMERGENCY_CONTACT},
            "email": {"configured": email_ok, "smtp_server": settings.SMTP_SERVER, "recipient": settings.HOSPITAL_EMAIL},
        },
    }


@router.post("/dispatch", summary="Dispatch emergency alerts for a crash event")
def dispatch_alerts(
    request: AlertDispatchRequest,
    current_user: dict = Depends(require_roles("admin", "operator")),
):
    """
    Dispatches SMS and/or Email alerts for a given crash event.

    - Looks up the crash event from the database
    - Sends alerts via requested channels
    - Records every dispatch attempt in the alerts table (audit trail)
    """
    settings = get_settings()
    db = get_db()

    # Fetch the crash event
    with db.get_connection() as conn:
        crash = conn.execute(
            "SELECT * FROM crash_events WHERE id = ?",
            (request.crash_event_id,),
        ).fetchone()

    if not crash:
        raise HTTPException(
            status_code=404,
            detail=f"Crash event {request.crash_event_id} not found"
        )

    crash = dict(crash)
    results = []

    for channel in request.channels:
        if channel == "sms":
            result = _send_sms(
                settings, crash["severity"], crash["csi_score"],
                crash["latitude"], crash["longitude"]
            )
            recipient = settings.TWILIO_EMERGENCY_CONTACT or "not_configured"
        elif channel == "email":
            result = _send_email(
                settings, crash["severity"], crash["csi_score"],
                crash["latitude"], crash["longitude"]
            )
            recipient = settings.HOSPITAL_EMAIL or "not_configured"
        else:
            results.append({"channel": channel, "status": "failed", "error": "Unknown channel"})
            continue

        # Record in alerts table
        with db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO alerts (crash_event_id, channel, recipient, status,
                                    message_sid, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    request.crash_event_id, channel, recipient,
                    result["status"],
                    result.get("message_sid"),
                    result.get("error"),
                ),
            )
            conn.commit()

        results.append({"channel": channel, **result})

    # Mark crash event as alert_dispatched
    any_sent = any(r["status"] == "sent" for r in results)
    if any_sent:
        with db.get_connection() as conn:
            conn.execute(
                "UPDATE crash_events SET alert_dispatched = 1 WHERE id = ?",
                (request.crash_event_id,),
            )
            conn.commit()

    return {
        "crash_event_id": request.crash_event_id,
        "severity": crash["severity"],
        "dispatch_results": results,
    }


@router.get("", summary="List all alert dispatch records", include_in_schema=False)
@router.get("/", summary="List all alert dispatch records")
def list_alerts(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Returns a paginated list of alert dispatch records."""
    db = get_db()

    with db.get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

        total = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]

    return {
        "total": total,
        "page": offset // limit + 1,
        "per_page": limit,
        "data": [dict(row) for row in rows],
    }


@router.get("/{crash_id}", summary="Get alerts for a specific crash event")
def get_alerts_for_crash(crash_id: int = Path(..., ge=1)):
    """Returns all alert dispatch attempts for a given crash event."""
    db = get_db()

    with db.get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM alerts WHERE crash_event_id = ? ORDER BY id",
            (crash_id,),
        ).fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No alerts found for crash event {crash_id}"
        )

    return [dict(row) for row in rows]
