"""
IntelliCrash — Crash Events Router.

Endpoints:
    POST /api/v1/crashes         → Record a new crash event (from Fusion Gate)
    GET  /api/v1/crashes         → List all crash events (paginated)
    GET  /api/v1/crashes/{id}    → Get a single crash event detail
    GET  /api/v1/crashes/stats   → Aggregate crash statistics
"""

import time
from fastapi import APIRouter, Query, HTTPException, Path

from src.backend.database import get_db
from src.backend.models import CrashEventCreate, CrashEventResponse
from src.backend.ws.manager import ws_manager

router = APIRouter(prefix="/api/v1/crashes", tags=["Crash Events"])


@router.post("", status_code=201, summary="Record a verified crash event", include_in_schema=False)
@router.post("/", status_code=201, summary="Record a verified crash event")
def create_crash_event(data: CrashEventCreate):
    """
    Called when the Fusion Gate confirms a crash (fused_score > 0.50).

    - Persists the crash event to SQLite
    - Broadcasts a crash alert to all WebSocket clients
    - Returns the created event ID for subsequent alert dispatch
    """
    db = get_db()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

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
                data.vehicle_id, timestamp, data.fused_score,
                data.ml_probability, data.csi_score, data.severity,
                data.latitude, data.longitude, data.peak_accel_x,
                data.peak_accel_y, data.peak_gyro_z, data.shap_summary,
                data.video_clip_path,
            ),
        )
        conn.commit()
        event_id = cursor.lastrowid

    # Broadcast crash alert to all connected dashboard clients
    ws_payload = {
        "type": "crash_alert",
        "crash_event_id": event_id,
        "vehicle_id": data.vehicle_id,
        "timestamp": timestamp,
        "severity": data.severity,
        "fused_score": data.fused_score,
        "latitude": data.latitude,
        "longitude": data.longitude,
    }
    ws_manager.broadcast_sync(ws_payload)

    return {
        "id": event_id,
        "status": "crash_recorded",
        "severity": data.severity,
        "timestamp": timestamp,
    }


@router.get("/stats", summary="Get aggregate crash statistics")
def get_crash_stats(
    vehicle_id: str = Query(default=None, description="Optional vehicle filter"),
):
    """
    Returns aggregate statistics: total crashes, severity distribution,
    average fused score, and the most recent crash.
    """
    db = get_db()

    with db.get_connection() as conn:
        where_clause = "WHERE vehicle_id = ?" if vehicle_id else ""
        params = (vehicle_id,) if vehicle_id else ()

        total = conn.execute(
            f"SELECT COUNT(*) FROM crash_events {where_clause}", params
        ).fetchone()[0]

        severity_counts = conn.execute(
            f"""
            SELECT severity, COUNT(*) as count
            FROM crash_events {where_clause}
            GROUP BY severity
            """,
            params,
        ).fetchall()

        avg_score = conn.execute(
            f"SELECT AVG(fused_score) FROM crash_events {where_clause}", params
        ).fetchone()[0]

        latest = conn.execute(
            f"""
            SELECT id, timestamp, severity, fused_score
            FROM crash_events {where_clause}
            ORDER BY id DESC LIMIT 1
            """,
            params,
        ).fetchone()

    return {
        "total_crashes": total,
        "severity_distribution": {
            row["severity"]: row["count"] for row in severity_counts
        },
        "average_fused_score": round(avg_score, 4) if avg_score else 0.0,
        "latest_crash": dict(latest) if latest else None,
    }


@router.get("", summary="List all crash events", include_in_schema=False)
@router.get("/", summary="List all crash events")
def list_crash_events(
    vehicle_id: str = Query(default=None, description="Filter by vehicle"),
    severity: str = Query(default=None, description="Filter by severity: MINOR, SEVERE, FATAL"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Returns a paginated list of crash events, newest first."""
    db = get_db()

    conditions = []
    params = []

    if vehicle_id:
        conditions.append("vehicle_id = ?")
        params.append(vehicle_id)
    if severity:
        conditions.append("severity = ?")
        params.append(severity.upper())

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

    with db.get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM crash_events
            {where_clause}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        ).fetchall()

        total = conn.execute(
            f"SELECT COUNT(*) FROM crash_events {where_clause}", params
        ).fetchone()[0]

    return {
        "total": total,
        "page": offset // limit + 1,
        "per_page": limit,
        "data": [dict(row) for row in rows],
    }


@router.get("/{crash_id}", summary="Get crash event detail")
def get_crash_event(crash_id: int = Path(..., ge=1)):
    """Returns full details of a single crash event by ID."""
    db = get_db()

    with db.get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM crash_events WHERE id = ?", (crash_id,)
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Crash event {crash_id} not found")

    # Also fetch associated alerts
    with db.get_connection() as conn:
        alerts = conn.execute(
            "SELECT * FROM alerts WHERE crash_event_id = ?", (crash_id,)
        ).fetchall()

    result = dict(row)
    result["alerts"] = [dict(a) for a in alerts]
    return result
