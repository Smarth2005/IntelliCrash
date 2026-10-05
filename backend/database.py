"""
IntelliCrash — SQLite Database Layer.

Manages database lifecycle:
- Creates tables on first run (crash_events, telemetry_log, alerts)
- Provides connection helper for route handlers
- Thread-safe via sqlite3 check_same_thread=False

Schema Design:
- crash_events: Stores every verified crash with ML scores, physics CSI,
  severity, GPS, and sync status.
- telemetry_log: Rolling log of telemetry snapshots (auto-pruned to last 24h).
- alerts: Audit trail for every SMS/email dispatch attempt.
"""

import sqlite3
import threading
from pathlib import Path
from contextlib import contextmanager

from src.backend.config import get_settings


# ── Schema Definitions ───────────────────────────────────────────────────────

SCHEMA_CRASH_EVENTS = """
CREATE TABLE IF NOT EXISTS crash_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id      TEXT    NOT NULL DEFAULT 'vehicle-001',
    timestamp       TEXT    NOT NULL,
    fused_score     REAL    NOT NULL,
    ml_probability  REAL    NOT NULL,
    csi_score       REAL    NOT NULL,
    severity        TEXT    NOT NULL CHECK(severity IN ('MINOR', 'SEVERE', 'FATAL')),
    latitude        REAL    DEFAULT 28.6139,
    longitude       REAL    DEFAULT 77.2090,
    peak_accel_x    REAL,
    peak_accel_y    REAL,
    peak_gyro_z     REAL,
    shap_summary    TEXT,
    video_clip_path TEXT,
    alert_dispatched INTEGER DEFAULT 0,
    firebase_synced  INTEGER DEFAULT 0,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

SCHEMA_TELEMETRY_LOG = """
CREATE TABLE IF NOT EXISTS telemetry_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id      TEXT    NOT NULL DEFAULT 'vehicle-001',
    timestamp       TEXT    NOT NULL,
    accel_x         REAL,
    accel_y         REAL,
    accel_z         REAL,
    gyro_z          REAL,
    crash_prob      REAL,
    severity_score  REAL,
    status          TEXT    DEFAULT 'Normal',
    behavior_score  REAL    DEFAULT 100.0,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

SCHEMA_ALERTS = """
CREATE TABLE IF NOT EXISTS alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    crash_event_id  INTEGER NOT NULL,
    channel         TEXT    NOT NULL CHECK(channel IN ('sms', 'email')),
    recipient       TEXT    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending', 'sent', 'failed')),
    message_sid     TEXT,
    error_message   TEXT,
    dispatched_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (crash_event_id) REFERENCES crash_events(id)
);
"""

SCHEMA_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT    NOT NULL UNIQUE,
    email           TEXT    NOT NULL UNIQUE,
    hashed_password TEXT    NOT NULL,
    role            TEXT    NOT NULL DEFAULT 'operator' CHECK(role IN ('admin', 'operator', 'viewer')),
    is_active       INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

# Index for fast time-range queries on telemetry
SCHEMA_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp
    ON telemetry_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_crash_events_timestamp
    ON crash_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_crash_events_vehicle
    ON crash_events(vehicle_id);
CREATE INDEX IF NOT EXISTS idx_users_username
    ON users(username);
"""


# ── Database Manager ─────────────────────────────────────────────────────────

class DatabaseManager:
    """Thread-safe SQLite database manager with connection pooling."""

    def __init__(self, db_path: str = None):
        settings = get_settings()
        self.db_path = db_path or settings.DATABASE_PATH

        # Ensure parent directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """Create tables and indexes if they don't exist."""
        with self.get_connection() as conn:
            conn.executescript(SCHEMA_CRASH_EVENTS)
            conn.executescript(SCHEMA_TELEMETRY_LOG)
            conn.executescript(SCHEMA_ALERTS)
            conn.executescript(SCHEMA_USERS)
            conn.executescript(SCHEMA_INDEXES)
            conn.commit()
            print(f"[DB] SQLite database initialized at {self.db_path}")

    @contextmanager
    def get_connection(self):
        """Context manager providing a thread-safe database connection.

        Usage:
            with db.get_connection() as conn:
                conn.execute("SELECT * FROM crash_events")
        """
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Dict-like row access
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent reads
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def prune_telemetry(self, keep_hours: int = 24):
        """Delete telemetry records older than `keep_hours` to prevent DB bloat."""
        with self.get_connection() as conn:
            conn.execute(
                "DELETE FROM telemetry_log WHERE created_at < datetime('now', ?)",
                (f"-{keep_hours} hours",)
            )
            conn.commit()


# ── Singleton Instance ───────────────────────────────────────────────────────

_db_instance = None
_db_lock = threading.Lock()


def get_db() -> DatabaseManager:
    """Get or create the singleton DatabaseManager instance."""
    global _db_instance
    if _db_instance is None:
        with _db_lock:
            if _db_instance is None:
                _db_instance = DatabaseManager()
    return _db_instance
