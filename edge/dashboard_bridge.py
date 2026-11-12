"""
IntelliCrash — Dashboard Bridge (runs on the Pi / laptop next to the edge node)
==============================================================================
Three jobs:
  1. LIVE STREAM  : WebSocket server (default ws://0.0.0.0:8765). Every connected dashboard
                    receives JSON ~20x/s:  {ax, ay, az, gz, speed, p_bilstm, csi, p_final, status}
  2. EDGE STORAGE: SQLite file  Raspberry/intellicrash.db  (edge copy of the shared schema)
                    driving_sessions, telemetry_log (1 Hz summary), crash_events, alerts.
                    Every row has is_synced (0 = not yet on the dashboard server).
  3. SYNC         : a background thread POSTs unsynced rows to  <dashboard-url>/api/ingest
                    (store-and-forward: if the network is down rows stay is_synced=0 and are
                    re-sent automatically when the dashboard is reachable again).
  + LIVE CSV LOG  : Raspberry/live_readings.csv  - every raw sample of a LIVE session is appended
                    (REPLAY sessions are not logged). Columns ax,ay,az,gz are compatible with
                    `--replay live_readings.csv`.

Install once:  pip install websockets
"""

import asyncio
import csv
import json
import os
import shutil
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS driving_sessions (
  session_id TEXT PRIMARY KEY,
  vehicle_id TEXT NOT NULL, source TEXT, start_time TEXT, end_time TEXT
);
CREATE TABLE IF NOT EXISTS telemetry_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL REFERENCES driving_sessions(session_id),
  ts TEXT, ax REAL, ay REAL, az REAL, gz REAL, speed REAL,
  latitude REAL, longitude REAL,
  p_bilstm REAL, csi REAL, p_final REAL, driving_class TEXT,
  is_synced INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS crash_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL REFERENCES driving_sessions(session_id),
  vehicle_id TEXT, ts TEXT,
  latitude REAL, longitude REAL,
  peak_accel REAL, ml_prob REAL, csi_score REAL, fused_score REAL, severity TEXT,
  video_clip_path TEXT, is_synced INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  crash_event_id INTEGER NOT NULL REFERENCES crash_events(id),
  channel TEXT, status TEXT, dispatched_at TEXT, error_msg TEXT,
  is_synced INTEGER NOT NULL DEFAULT 0
);
"""

CSV_HEADER = ["timestamp", "session_id", "ax", "ay", "az", "gz", "speed",
              "p_bilstm", "csi", "p_final", "status"]


def _open_edge_db(path: Path):
    """Open the edge DB. An older-schema file (pre-sync version) is renamed, not deleted."""
    if path.exists():
        probe = sqlite3.connect(str(path))
        ver = probe.execute("PRAGMA user_version").fetchone()[0]
        has_tables = probe.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
        probe.close()
        if has_tables and ver != SCHEMA_VERSION:
            backup = path.with_name(f"{path.stem}_old_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
            path.rename(backup)
            print(f"[BRIDGE] Old-schema DB kept as {backup.name}; creating a fresh edge DB.")
    db = sqlite3.connect(str(path), check_same_thread=False)
    db.executescript(SCHEMA)
    db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    db.commit()
    return db


class DashboardBridge:
    def __init__(self, port=8765, db_path=None, vehicle_id="PB-11-AB-2027", source="LIVE",
                 broadcast_hz=20, dashboard_url=None, ingest_key=None, csv_path=None, sync_interval=2.0):
        self.port = port
        self.vehicle_id = vehicle_id
        self.source = source
        here = Path(__file__).resolve().parent
        self.db_path = Path(db_path or here / "intellicrash.db")
        self._db = _open_edge_db(self.db_path)
        self._lock = threading.Lock()
        self.session_id = uuid.uuid4().hex[:12]
        self._db.execute(
            "INSERT INTO driving_sessions(session_id, vehicle_id, source, start_time) VALUES (?,?,?,?)",
            (self.session_id, vehicle_id, source, datetime.now().isoformat(timespec="seconds")),
        )
        self._db.commit()

        # live CSV log (LIVE sessions only)
        self._csv_file = self._csv = None
        if source == "LIVE":
            cp = Path(csv_path or here / "live_readings.csv")
            new_file = not cp.exists() or cp.stat().st_size == 0
            self._csv_file = open(cp, "a", newline="", encoding="utf-8")
            self._csv = csv.writer(self._csv_file)
            if new_file:
                self._csv.writerow(CSV_HEADER)
            self.csv_path = cp
        else:
            self.csv_path = None

        # sync to dashboard server
        self.dashboard_url = (dashboard_url or os.environ.get("INTELLICRASH_DASHBOARD_URL") or "").rstrip("/")
        self.ingest_key = ingest_key or os.environ.get("INTELLICRASH_INGEST_KEY") or "intellicrash-demo-key"
        self.sync_interval = sync_interval
        self.sync_ok = None
        self._session_dirty = True
        self._stop = threading.Event()

        self._scores = {"p_bilstm": None, "csi": None, "p_final": None}
        self._status = "NORMAL"
        self._n = 0
        self._every = max(1, int(100 / broadcast_hz))
        self._clients = set()
        self._health_t = 0.0
        self._health_cache = {}
        self._loop = None
        self.ws_ok = False
        threading.Thread(target=self._serve, daemon=True).start()
        if self.dashboard_url:
            print(f"[BRIDGE] DB sync -> {self.dashboard_url}/api/ingest every {sync_interval:g}s (unsynced rows are retried)")
            threading.Thread(target=self._sync_loop, daemon=True).start()
        else:
            print("[BRIDGE] No --dashboard-url set -> rows stay in the edge DB (is_synced=0) until you set one.")

    # ── WebSocket side ───────────────────────────────────────────────────
    def _serve(self):
        try:
            import websockets
        except ImportError:
            print("[BRIDGE] 'websockets' not installed -> live stream OFF (pip install websockets). SQLite logging still works.")
            return

        async def handler(ws, *_):
            self._clients.add(ws)
            try:
                await ws.wait_closed()
            finally:
                self._clients.discard(ws)

        async def main():
            async with websockets.serve(handler, "0.0.0.0", self.port):
                self.ws_ok = True
                print(f"[BRIDGE] Dashboard stream ready on ws://0.0.0.0:{self.port}  |  DB: {self.db_path.name}")
                await asyncio.Future()

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(main())
        except Exception as e:
            print(f"[BRIDGE] stream stopped: {e}")

    def _broadcast(self, msg):
        for ws in list(self._clients):
            asyncio.ensure_future(self._safe_send(ws, msg))

    @staticmethod
    async def _safe_send(ws, msg):
        try:
            await ws.send(msg)
        except Exception:
            pass

    # ── Called by the edge loop ──────────────────────────────────────────
    def push_sample(self, ax, ay, az, gz, speed=None, gps=None):
        self._n += 1
        if self._csv is not None:  # append every raw LIVE sample to the CSV log
            sc = self._scores
            self._csv.writerow([datetime.now().isoformat(timespec="milliseconds"), self.session_id,
                                round(ax, 4), round(ay, 4), round(az, 4), round(gz, 5), speed,
                                sc["p_bilstm"], sc["csi"], sc["p_final"], self._status])
            if self._n % 100 == 0:
                self._csv_file.flush()
        if self._n % self._every == 0 and self._loop and self._clients:
            msg = json.dumps({"ax": round(ax, 3), "ay": round(ay, 3), "az": round(az, 3), "gz": round(gz, 4),
                              "speed": speed, "lat": gps[0] if gps else None, "lon": gps[1] if gps else None,
                              "health": self._health(), **self._scores, "status": self._status})
            self._loop.call_soon_threadsafe(self._broadcast, msg)
        if self._n % 100 == 0:  # 1 Hz summary into SQLite
            with self._lock:
                self._db.execute(
                    "INSERT INTO telemetry_log(session_id, ts, ax, ay, az, gz, speed, latitude, longitude, p_bilstm, csi, p_final, driving_class) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (self.session_id, datetime.now().isoformat(timespec="seconds"), ax, ay, az, gz, speed,
                     gps[0] if gps else None, gps[1] if gps else None,   # NULL when the GPS has no fix
                     self._scores["p_bilstm"], self._scores["csi"], self._scores["p_final"], self._status),
                )
                self._db.commit()

    def _health(self):
        """Real Pi readings (CPU temperature, memory, free disk), refreshed every 2 s. A value the OS cannot
        report (e.g. when running on a laptop) is None, never a made-up number."""
        now = time.time()
        if now - self._health_t < 2.0:
            return self._health_cache
        h = {"cpu_temp_c": None, "mem_used_mb": None, "mem_total_mb": None, "disk_free_gb": None, "disk_total_gb": None}
        try:
            h["cpu_temp_c"] = round(int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0, 1)
        except Exception:
            pass
        try:
            mem = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                k, v = line.split(":", 1)
                mem[k] = int(v.split()[0])  # kB
            h["mem_total_mb"] = round(mem["MemTotal"] / 1024)
            h["mem_used_mb"] = round((mem["MemTotal"] - mem["MemAvailable"]) / 1024)
        except Exception:
            pass
        try:
            u = shutil.disk_usage(str(self.db_path.parent))
            h["disk_free_gb"] = round(u.free / 1e9, 1)
            h["disk_total_gb"] = round(u.total / 1e9, 1)
        except Exception:
            pass
        self._health_t, self._health_cache = now, h
        return h

    def update_scores(self, p_bilstm, csi, p_final, status="NORMAL"):
        self._scores = {"p_bilstm": round(float(p_bilstm), 4), "csi": round(float(csi), 4), "p_final": round(float(p_final), 4)}
        self._status = status

    def log_crash(self, severity, peak_accel, p_lstm, csi, p_final, gps, video_path):
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO crash_events(session_id, vehicle_id, ts, latitude, longitude, peak_accel, ml_prob, csi_score, fused_score, severity, video_clip_path) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (self.session_id, self.vehicle_id, datetime.now().isoformat(timespec="seconds"),
                 gps[0] if gps else None, gps[1] if gps else None,
                 peak_accel, p_lstm, csi, p_final, severity, str(video_path) if video_path else None),
            )
            self._db.commit()
            return cur.lastrowid

    def log_alert(self, crash_event_id, channel, status="DISPATCHED", error_msg=None):
        with self._lock:
            self._db.execute(
                "INSERT INTO alerts(crash_event_id, channel, status, dispatched_at, error_msg) VALUES (?,?,?,?,?)",
                (crash_event_id, channel, status, datetime.now().isoformat(timespec="seconds"), error_msg),
            )
            self._db.commit()

    # ── Sync to dashboard server (store-and-forward) ─────────────────────
    def _sync_loop(self):
        while not self._stop.wait(self.sync_interval):
            try:
                self.sync_now()
            except Exception as e:  # never let sync kill the edge node
                self._report(False, f"sync error: {e}")

    def _report(self, ok, msg=""):
        if ok != self.sync_ok:  # print only on state change (no log spam)
            self.sync_ok = ok
            print(f"[BRIDGE] dashboard sync {'OK' if ok else 'OFFLINE'} {msg}".rstrip(), flush=True)

    def sync_now(self):
        """Send unsynced rows to <dashboard-url>/api/ingest. Returns True if something was sent."""
        if not self.dashboard_url:
            return False
        with self._lock:
            if self._db is None:
                return False
            q = self._db.execute
            tel = q("SELECT id, session_id, ts, ax, ay, az, gz, speed, latitude, longitude, p_bilstm, csi, p_final, driving_class "
                    "FROM telemetry_log WHERE is_synced=0 ORDER BY id LIMIT 300").fetchall()
            cr = q("SELECT id, session_id, vehicle_id, ts, latitude, longitude, peak_accel, ml_prob, csi_score, "
                   "fused_score, severity, video_clip_path FROM crash_events WHERE is_synced=0 ORDER BY id LIMIT 50").fetchall()
            batch_ids = {r[0] for r in cr}
            al = [r for r in q("SELECT a.id, a.crash_event_id, a.channel, a.status, a.dispatched_at, a.error_msg, "
                               "c.session_id, c.is_synced FROM alerts a JOIN crash_events c ON c.id=a.crash_event_id "
                               "WHERE a.is_synced=0 ORDER BY a.id LIMIT 100").fetchall()
                  if r[7] == 1 or r[1] in batch_ids]
            dirty = self._session_dirty
            sids = {r[1] for r in tel} | {r[1] for r in cr} | {r[6] for r in al}
            if dirty:
                sids.add(self.session_id)
            if not sids:
                return False
            marks = ",".join("?" * len(sids))
            sess = q(f"SELECT session_id, vehicle_id, source, start_time, end_time FROM driving_sessions "
                     f"WHERE session_id IN ({marks})", tuple(sids)).fetchall()

        def rows(cols, data):
            return [dict(zip(cols, r)) for r in data]

        payload = {
            "sessions": rows(("session_id", "vehicle_id", "source", "start_time", "end_time"), sess),
            "telemetry": rows(("edge_id", "session_id", "ts", "ax", "ay", "az", "gz", "speed", "latitude", "longitude",
                               "p_bilstm", "csi", "p_final", "driving_class"), tel),
            "crashes": rows(("edge_id", "session_id", "vehicle_id", "ts", "latitude", "longitude", "peak_accel",
                             "ml_prob", "csi_score", "fused_score", "severity", "video_clip_path"), cr),
            "alerts": rows(("edge_id", "crash_edge_id", "channel", "status", "dispatched_at", "error_msg",
                            "session_id"), [r[:7] for r in al]),
        }
        req = urllib.request.Request(
            self.dashboard_url + "/api/ingest", data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "x-ingest-key": self.ingest_key})
        try:
            with urllib.request.urlopen(req, timeout=4) as resp:
                ack = json.load(resp).get("ack", {})
        except urllib.error.HTTPError as e:
            self._report(False, f"(HTTP {e.code}{' - wrong ingest key' if e.code == 401 else ''})")
            return False
        except Exception as e:
            self._report(False, f"({type(e).__name__}) - rows kept, will retry")
            return False

        with self._lock:
            if self._db is None:
                return True
            for table, key in (("telemetry_log", "telemetry"), ("crash_events", "crashes"), ("alerts", "alerts")):
                ids = [int(i) for i in ack.get(key, [])]
                if ids:
                    self._db.executemany(f"UPDATE {table} SET is_synced=1 WHERE id=?", [(i,) for i in ids])
            self._db.commit()
            if dirty:
                self._session_dirty = False
        self._report(True, f"(sent {len(tel)} telemetry, {len(cr)} crash, {len(al)} alert rows)")
        return True

    def close(self):
        self._stop.set()
        with self._lock:
            self._db.execute("UPDATE driving_sessions SET end_time=? WHERE session_id=?",
                             (datetime.now().isoformat(timespec="seconds"), self.session_id))
            self._db.commit()
            self._session_dirty = True
        try:
            self.sync_now()  # last best-effort flush
        except Exception:
            pass
        with self._lock:
            if self._csv_file:
                self._csv_file.close()
                self._csv_file = self._csv = None
            self._db.close()
            self._db = None
