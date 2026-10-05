"""
IntelliCrash — Firebase Firestore Sync Service.

Provides cloud synchronization for crash events and telemetry.
When a crash event is recorded locally, this service pushes it to
Firebase Firestore for cloud access by the Next.js dashboard.

Firestore Collections:
    - crash_events/{event_id}   → Full crash record
    - vehicles/{vehicle_id}     → Latest telemetry snapshot

Firebase Storage:
    - crash_clips/{event_id}.mp4 → Dashcam video clips (future)

Setup:
    1. Create a Firebase project at https://console.firebase.google.com
    2. Generate a service account key (Project Settings → Service Accounts)
    3. Save the JSON key to configs/firebase_admin_key.json
    4. Set FIREBASE_ENABLED=true in .env
"""

import logging
from pathlib import Path

logger = logging.getLogger("intellicrash.firebase")

# Lazy-loaded globals
_firestore_client = None
_firebase_initialized = False


def _init_firebase():
    """Initialize Firebase Admin SDK. Called once on first use."""
    global _firestore_client, _firebase_initialized

    if _firebase_initialized:
        return

    from src.backend.config import get_settings
    settings = get_settings()

    if not settings.FIREBASE_ENABLED:
        logger.info("[Firebase] Disabled via FIREBASE_ENABLED=false")
        _firebase_initialized = True
        return

    creds_path = Path(settings.FIREBASE_CREDENTIALS_PATH)
    if not creds_path.exists():
        logger.warning(
            f"[Firebase] Credentials not found at {creds_path}. "
            "Cloud sync disabled. To enable, add firebase_admin_key.json "
            "to configs/ and set FIREBASE_ENABLED=true."
        )
        _firebase_initialized = True
        return

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        cred = credentials.Certificate(str(creds_path))
        firebase_admin.initialize_app(cred)
        _firestore_client = firestore.client()
        logger.info("[Firebase] Initialized successfully — cloud sync active")
    except ImportError:
        logger.warning(
            "[Firebase] firebase-admin package not installed. "
            "Install with: pip install firebase-admin"
        )
    except Exception as e:
        logger.error(f"[Firebase] Initialization failed: {e}")

    _firebase_initialized = True


def sync_crash_event(crash_data: dict) -> bool:
    """
    Push a crash event to Firestore.

    Args:
        crash_data: Dict with crash event fields (from the database).

    Returns:
        True if sync succeeded, False otherwise.
    """
    _init_firebase()

    if _firestore_client is None:
        logger.debug("[Firebase] Skipping sync — client not initialized")
        return False

    try:
        event_id = str(crash_data.get("id", "unknown"))
        doc_ref = _firestore_client.collection("crash_events").document(event_id)
        doc_ref.set({
            "vehicle_id": crash_data.get("vehicle_id"),
            "timestamp": crash_data.get("timestamp"),
            "fused_score": crash_data.get("fused_score"),
            "ml_probability": crash_data.get("ml_probability"),
            "csi_score": crash_data.get("csi_score"),
            "severity": crash_data.get("severity"),
            "location": {
                "latitude": crash_data.get("latitude"),
                "longitude": crash_data.get("longitude"),
            },
            "peak_accel_x": crash_data.get("peak_accel_x"),
            "peak_accel_y": crash_data.get("peak_accel_y"),
            "peak_gyro_z": crash_data.get("peak_gyro_z"),
            "alert_dispatched": bool(crash_data.get("alert_dispatched")),
            "created_at": crash_data.get("created_at"),
        })
        logger.info(f"[Firebase] Crash event {event_id} synced to Firestore")
        return True

    except Exception as e:
        logger.error(f"[Firebase] Failed to sync crash event: {e}")
        return False


def update_vehicle_status(vehicle_id: str, telemetry: dict) -> bool:
    """
    Update the latest telemetry snapshot for a vehicle in Firestore.
    The dashboard can listen to this document for real-time updates.

    Args:
        vehicle_id: Vehicle identifier
        telemetry: Latest telemetry readings

    Returns:
        True if update succeeded, False otherwise.
    """
    _init_firebase()

    if _firestore_client is None:
        return False

    try:
        doc_ref = _firestore_client.collection("vehicles").document(vehicle_id)
        doc_ref.set({
            "status": telemetry.get("status", "Normal"),
            "accel_x": telemetry.get("accel_x"),
            "accel_y": telemetry.get("accel_y"),
            "gyro_z": telemetry.get("gyro_z"),
            "crash_prob": telemetry.get("crash_prob"),
            "severity_score": telemetry.get("severity_score"),
            "behavior_score": telemetry.get("behavior_score"),
            "last_updated": telemetry.get("timestamp"),
        }, merge=True)
        return True

    except Exception as e:
        logger.error(f"[Firebase] Failed to update vehicle status: {e}")
        return False


def sync_pending_events() -> int:
    """
    Batch-sync all crash events that haven't been synced to Firebase yet.
    Called on server startup or periodically to catch missed syncs.

    Returns:
        Number of events successfully synced.
    """
    _init_firebase()

    if _firestore_client is None:
        return 0

    from src.backend.database import get_db
    db = get_db()
    synced_count = 0

    with db.get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM crash_events WHERE firebase_synced = 0"
        ).fetchall()

    for row in rows:
        crash_data = dict(row)
        if sync_crash_event(crash_data):
            with db.get_connection() as conn:
                conn.execute(
                    "UPDATE crash_events SET firebase_synced = 1 WHERE id = ?",
                    (crash_data["id"],)
                )
                conn.commit()
            synced_count += 1

    if synced_count > 0:
        logger.info(f"[Firebase] Batch synced {synced_count} pending crash events")

    return synced_count
