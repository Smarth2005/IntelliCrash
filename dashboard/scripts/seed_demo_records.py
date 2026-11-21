"""
Seed the dashboard SQLite DB (data/intellicrash.db) with records built from the REAL recorded clips
(src/data/recorded_clips.json): sessions 0012 (normal), 0015 (rash) and 0006 (collision).
Nothing is invented: IMU values and model scores are the stored ones. The crash row uses the clip's real peak
(51.7 m/s^2), CSI and fused score. Position = the fixed demo point from src/data/demo_config.json (CSED).
Rows are tagged source='SEED' so they are easy to find / delete.

Run from the dashboard folder:   python scripts/seed_demo_records.py
Remove again:                    python scripts/seed_demo_records.py --clear
View:  sqlite3 data/intellicrash.db "select * from crash_events;"
"""
import json, sqlite3, sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "intellicrash.db"
DDL = """
CREATE TABLE IF NOT EXISTS driving_sessions (session_id TEXT PRIMARY KEY, vehicle_id TEXT NOT NULL, source TEXT, start_time TEXT, end_time TEXT);
CREATE TABLE IF NOT EXISTS telemetry_log (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES driving_sessions(session_id) ON DELETE CASCADE, edge_id INTEGER NOT NULL, ts TEXT, ax REAL, ay REAL, az REAL, gz REAL, speed REAL, latitude REAL, longitude REAL, p_bilstm REAL, csi REAL, p_final REAL, driving_class TEXT, UNIQUE (session_id, edge_id));
CREATE TABLE IF NOT EXISTS crash_events (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES driving_sessions(session_id) ON DELETE CASCADE, edge_id INTEGER NOT NULL, vehicle_id TEXT, ts TEXT, latitude REAL, longitude REAL, peak_accel REAL, ml_prob REAL, csi_score REAL, fused_score REAL, severity TEXT, video_clip_path TEXT, location_source TEXT, is_synced INTEGER NOT NULL DEFAULT 1, UNIQUE (session_id, edge_id));
CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, crash_event_id INTEGER NOT NULL REFERENCES crash_events(id) ON DELETE CASCADE, edge_id INTEGER NOT NULL, channel TEXT, status TEXT, dispatched_at TEXT, error_msg TEXT, UNIQUE (crash_event_id, edge_id));
"""
VEHICLE = "PB-11-AB-2027"
CLIPS = {"normal_cruising": ("SEED-0012", "NORMAL"), "rash_maneuver": ("SEED-0015", "RASH"), "crash_impact": ("SEED-0006", "COLLISION")}

DDL += """
CREATE TABLE IF NOT EXISTS hospital_dispatches (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, crash_edge_id INTEGER NOT NULL, rank INTEGER, hospital_name TEXT, hospital_email TEXT, lat REAL, lon REAL, distance_km REAL, token TEXT NOT NULL UNIQUE, status TEXT NOT NULL DEFAULT 'REQUESTED', eta_min INTEGER, requested_at TEXT, responded_at TEXT);
CREATE TABLE IF NOT EXISTS incident_events (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, crash_edge_id INTEGER NOT NULL, ts TEXT, actor TEXT, event TEXT, detail TEXT, UNIQUE (session_id, crash_edge_id, ts, event, detail));
"""

def dist_km(a, b, c, d):
    import math
    p1, p2 = math.radians(a), math.radians(c)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(d - b) / 2) ** 2
    return round(12742 * math.asin(math.sqrt(h)), 2)

def clear(db):
    db.execute("DELETE FROM hospital_dispatches WHERE session_id LIKE 'SEED-%'")
    db.execute("DELETE FROM incident_events WHERE session_id LIKE 'SEED-%'")
    db.execute("DELETE FROM driving_sessions WHERE source='SEED'")

def main():
    DB.parent.mkdir(exist_ok=True)
    db = sqlite3.connect(DB); db.execute("PRAGMA foreign_keys=ON"); db.executescript(DDL)
    if "--clear" in sys.argv:
        clear(db); db.commit(); print("Seed rows removed."); return
    clips = json.load(open(ROOT / "src/data/recorded_clips.json"))["clips"]
    cfg = json.load(open(ROOT / "src/data/demo_config.json"))["campus"]
    clear(db)  # idempotent re-seed
    t0 = datetime.now().replace(microsecond=0) - timedelta(minutes=30)
    for key, (sid, cls) in CLIPS.items():
        clip = clips[key]; n = len(clip["samples"]); start = t0; end = start + timedelta(seconds=n / 25)
        db.execute("INSERT INTO driving_sessions VALUES (?,?,?,?,?)", (sid, VEHICLE, "SEED", start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")))
        for i in range(0, n, 5):  # every 5th sample (5 Hz) keeps the table readable
            r = clip["samples"][i]; ts = (start + timedelta(seconds=i / 25)).isoformat(timespec="milliseconds")
            label = {"crash": "COLLISION", "rash_drive": "RASH"}.get(r.get("truth"), "NORMAL")
            db.execute("INSERT INTO telemetry_log (session_id,edge_id,ts,ax,ay,az,gz,speed,p_bilstm,csi,p_final,driving_class) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       (sid, i, ts, r["ax"], r["ay"], r["az"], r["gz"], None, r["pb"], r["csi"], r["pf"], label))
        t0 = end + timedelta(minutes=2)
        if key == "crash_impact":
            hit = max(range(n), key=lambda j: (clip["samples"][j]["ax"]**2 + clip["samples"][j]["ay"]**2 + clip["samples"][j]["az"]**2))
            imp_ts = (start + timedelta(seconds=hit / 25)).isoformat(timespec="seconds")
            pf = max(s["pf"] for s in clip["samples"]); csi = max(s["csi"] for s in clip["samples"]); pb = max(s["pb"] for s in clip["samples"])
            cur = db.execute("INSERT INTO crash_events (session_id,edge_id,vehicle_id,ts,latitude,longitude,peak_accel,ml_prob,csi_score,fused_score,severity,video_clip_path,location_source,is_synced) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
                             (sid, hit, VEHICLE, imp_ts, cfg["lat"], cfg["lon"], clip["peak_m_s2"], pb, csi, pf, "FATAL", "crash_clip_session0006.mp4", "DEMO"))
            cid = cur.lastrowid
            for k, ch in enumerate(("Telegram", "Email"), start=1):
                db.execute("INSERT INTO alerts (crash_event_id,edge_id,channel,status,dispatched_at,error_msg) VALUES (?,?,?,?,?,NULL)", (cid, k, ch, "DISPATCHED", imp_ts))
            # SAMPLE dispatch timeline so the new tables/dashboard panel are not empty. It is NOT a real hospital
            # response - every detail says "sample". Real ones are written by the Pi + the hospital's accept link.
            base = datetime.fromisoformat(imp_ts)
            at = lambda sec: (base + timedelta(seconds=sec)).isoformat(timespec="seconds")
            for sec, actor, ev, det in [(0, "system", "CRASH_DETECTED", "FATAL · peak 51.7 m/s² (sample)"), (1, "system", "DRIVER_PROMPTED", "Telegram prompt sent (sample)"),
                                        (61, "system", "NO_DRIVER_RESPONSE", "no answer within 60 s (sample)"), (62, "system", "HOSPITAL_REQUESTED", "#1 Rajindra Hospital (sample)"),
                                        (63, "system", "CONTACT_NOTIFIED", "emergency contact told Rajindra was requested (sample)"),
                                        (95, "hospital", "HOSPITAL_ACCEPTED", "Rajindra Hospital · ambulance ETA 9 min (sample)")]:
                db.execute("INSERT INTO incident_events (session_id,crash_edge_id,ts,actor,event,detail) VALUES (?,?,?,?,?,?)", (sid, hit, at(sec), actor, ev, det))
            db.execute("INSERT INTO hospital_dispatches (session_id,crash_edge_id,rank,hospital_name,hospital_email,lat,lon,distance_km,token,status,eta_min,requested_at,responded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (sid, hit, 1, "Rajindra Hospital", "", 30.3291738, 76.3838964, dist_km(cfg["lat"], cfg["lon"], 30.3291738, 76.3838964), "seed-sample-token", "ACCEPTED", 9, at(62), at(95)))
    db.commit()
    for t in ("driving_sessions", "telemetry_log", "crash_events", "alerts", "hospital_dispatches", "incident_events"):
        print(f"{t:18s}", db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0], "rows")

main()
