"""
IntelliCrash — Telemetry Router.

Endpoints:
    POST /api/v1/telemetry         → Ingest a telemetry snapshot from edge device
    GET  /api/v1/telemetry         → Query recent telemetry history
    GET  /api/v1/telemetry/latest  → Get the most recent telemetry reading
"""

import time
from fastapi import APIRouter, Query, HTTPException

from src.backend.database import get_db
from src.backend.models import TelemetryCreate, TelemetryResponse
from src.backend.ws.manager import ws_manager

router = APIRouter(prefix="/api/v1/telemetry", tags=["Telemetry"])


@router.post("", status_code=201, summary="Ingest telemetry from edge device", include_in_schema=False)
@router.post("/", status_code=201, summary="Ingest telemetry from edge device")
def ingest_telemetry(data: TelemetryCreate):
    """
    Receives a telemetry snapshot from the Raspberry Pi edge node.

    - Persists to SQLite for historical queries
    - Broadcasts to all connected WebSocket clients in real-time
    """
    db = get_db()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    with db.get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO telemetry_log
                (vehicle_id, timestamp, accel_x, accel_y, accel_z,
                 gyro_z, crash_prob, severity_score, status, behavior_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.vehicle_id, timestamp, data.accel_x, data.accel_y,
                data.accel_z, data.gyro_z, data.crash_prob,
                data.severity_score, data.status, data.behavior_score,
            ),
        )
        conn.commit()
        record_id = cursor.lastrowid

    # Broadcast to WebSocket clients
    ws_payload = {
        "type": "telemetry",
        "vehicle_id": data.vehicle_id,
        "timestamp": timestamp,
        "accel_x": data.accel_x,
        "accel_y": data.accel_y,
        "accel_z": data.accel_z,
        "gyro_z": data.gyro_z,
        "crash_prob": data.crash_prob,
        "severity_score": data.severity_score,
        "status": data.status,
        "behavior_score": data.behavior_score,
    }
    ws_manager.broadcast_sync(ws_payload)

    return {"id": record_id, "status": "ingested", "timestamp": timestamp}


@router.get("", summary="Query telemetry history", include_in_schema=False)
@router.get("/", summary="Query telemetry history")
@router.get("/history", summary="Query telemetry history", include_in_schema=False)
def get_telemetry(
    vehicle_id: str = Query(default="vehicle-001", description="Filter by vehicle"),
    limit: int = Query(default=100, ge=1, le=1000, description="Max records to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
):
    """
    Returns recent telemetry snapshots for a vehicle, ordered by newest first.
    """
    db = get_db()

    with db.get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM telemetry_log
            WHERE vehicle_id = ?
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (vehicle_id, limit, offset),
        ).fetchall()

        total = conn.execute(
            "SELECT COUNT(*) FROM telemetry_log WHERE vehicle_id = ?",
            (vehicle_id,),
        ).fetchone()[0]

    return {
        "total": total,
        "page": offset // limit + 1,
        "per_page": limit,
        "data": [dict(row) for row in rows],
    }


@router.get("/latest", summary="Get most recent telemetry reading")
def get_latest_telemetry(
    vehicle_id: str = Query(default="vehicle-001"),
):
    """Returns the single most recent telemetry snapshot for a vehicle."""
    db = get_db()

    with db.get_connection() as conn:
        row = conn.execute(
            """
            SELECT * FROM telemetry_log
            WHERE vehicle_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (vehicle_id,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="No telemetry data found for this vehicle")

    return dict(row)
