# IntelliCrash: Pi <-> Dashboard integration

## Run it
1. **Laptop (dashboard)**: copy `.env.local.example` to `.env.local`, set `NEXT_PUBLIC_PI_WS_URL=ws://<PI-IP>:8765`, then
   `npm install` and `npm run dev` (it now listens on 0.0.0.0 so the Pi can reach it; allow port 3000 in the firewall).
2. **Raspberry Pi**: `pip install websockets`, then
   `python raspberry_pi_node.py --live --dashboard-url http://<LAPTOP-IP>:3000`
   (use `--replay some.csv` instead of `--live` to test without hardware; `--ingest-key` must equal `INGEST_KEY` on the laptop).

## Data flow
IMU -> Pi inference -> `DashboardBridge`
- WebSocket `ws://<pi>:8765` (about 20 msgs/s) -> browser Live Cockpit
- edge DB `Raspberry/intellicrash.db` (every row has `is_synced`)
- sync thread -> `POST /api/ingest` (header `x-ingest-key`) every 2 s -> dashboard DB `data/intellicrash.db`
- every raw sample of a LIVE session -> `Raspberry/live_readings.csv` (not written in --replay mode)
- Incident Dossiers / Alerts / Map read `GET /api/incidents` (tables `crash_events`, `alerts`)

## Hardware Session drop-down (Live Cockpit)
Normal, rash and crash options all play the real recorded IMU clips from `src/data/recorded_clips.json` at their
native 25 Hz, with the model scores stored per sample (p_bilstm, CSI, p_final). Labels come from the file itself
(normal = session 0012, rash = session 0015, collision = session 0006). The recordings have no speed channel,
so speed shows "n/a" instead of an invented number.

## Incident Dossiers / Alerts / Map
- Pi connected (LIVE): only crashes the Pi detected and synced (`crash_events`, `alerts`), refreshed every 3 s.
- Replay / backup / recorded sessions: those saved records plus the recorded collision clip (session 0006, real values).
- No bundled demo incidents. Fields the recordings do not contain (GPS, video, delta-v, jerk) show as "not recorded".
- The SOS drill (fixed-value fake crash) has been removed.

## Map tab and hospitals
- Blue dot = the car, drawn only while the Pi is connected and the NEO-8M has a real fix (lat/lon are in the live stream).
- Red dot = a crash, at the GPS position stored with it. Click one to select it; the default is the newest.
- Hospital icons = the hospitals nearest to the selected crash. The dashboard server asks OpenStreetMap (Overpass API,
  no key) via `GET /api/hospitals?lat=&lon=`, so that server needs internet. Results are cached 24 h per ~100 m.
  Names/phones come straight from OSM tags; there are no bed counts or other invented fields.
- No GPS fix = no position: the Pi no longer substitutes a built-in location. Such a crash is stored without a map
  position and alerts say "location unavailable".

## Demo mode (classroom presentation)
Indoors the GPS gets no fix, so the Map tab has a "Demo mode" switch (default ON, set in `src/data/demo_config.json`).
- Map opens on the campus point; the car dot is drawn there (badged "demo position") whenever Live is selected and no real fix exists.
- Crashes stored without GPS are shown at the campus point, tagged "Demo position, not a GPS reading". The database is never changed.
- Hospitals are the pinned list in `demo_config.json` (works offline), ranked by distance from the selected crash.
- Real GPS positions always win. Switch Demo mode OFF for real outdoor tests: hospitals then come from OpenStreetMap.
- Edit `campus.lat/lon` to the exact classroom point and check the hospital names/coordinates before presenting.

## Fallback
If the WebSocket cannot connect or drops, the Live tab switches by itself to `src/data/recorded_clips.json`
(normal -> rash -> crash, real recorded IMU with model scores) and shows an amber BACKUP MODE banner.
It keeps retrying every 3 s and switches back to live data when the Pi returns.
If the network is down, crash rows stay `is_synced=0` on the Pi and are sent when it is back.

## One logical schema, two physical SQLite files
| Table | Edge DB (Pi) | Central DB (dashboard) | Filled by |
|---|---|---|---|
| users, sessions (login), owner_profiles | - | yes | sign-up / profile form |
| driving_sessions, telemetry_log, crash_events, alerts | yes (source of truth) | yes (synced copy, plus `edge_id`) | Pi bridge |

Relationships: users 1-1 owner_profiles (owners only); users 1-N sessions;
driving_sessions 1-N telemetry_log; driving_sessions 1-N crash_events; crash_events 1-N alerts.
Logical link (no FK): driving_sessions.vehicle_id = owner_profiles.plate.

## Status rule for recorded sessions and backup replay (v2)
CRASH DETECTED = fused score >= 0.46 AND peak resultant acceleration in the last 2 s >= 16 m/s^2: the same rule the Pi
uses (`edge_inference.py`), applied to the scores stored in `recorded_clips.json`. For the collision clip (session 0006)
this shows CRASH DETECTED from 8.8 s to 12.7 s, right after the impact peak. RASH DRIVING comes from the label stored
with each recorded sample, because the recordings do not contain the rash classifier's output.

## Pi health meters (v2)
CPU temperature, memory and free storage are read on the Pi (`/sys`, `/proc`, disk usage) and sent in the live stream.
While the Pi is not connected, or the OS cannot report a value, the panel shows a dash. Nothing is invented.

## Closed-loop dispatch and hospital acknowledgement (v3)
Flow (Pi: `Raspberry/incident_manager.py`):
1. Crash detected -> crash row saved -> the **driver's** Telegram chat gets "I'm OK" / "I need help" buttons (60 s, `DRIVER_RESPONSE_SECONDS`).
2. "I'm OK" -> no hospital is contacted (event `DRIVER_OK`). "I need help" or no answer -> escalate. If the driver prompt cannot be
   delivered (no driver chat id, Telegram down) it also escalates (fail-safe).
3. Escalation: nearest hospital (from `Raspberry/hospitals.json`, ranked by distance from the crash) gets an e-mail with an
   Accept / Cannot-serve link; the **emergency contact** (TELEGRAM_CHAT_ID) is told which hospital was requested.
4. The hospital opens the link `<dashboard>/api/ack?token=...`, enters the ambulance ETA (minutes) and accepts. The Pi polls
   `GET /api/dispatch`, then tells the contact and the driver "X accepted, ETA n min".
5. Declined / no answer in 90 s (`HOSPITAL_ACK_SECONDS`) -> next nearest hospital (max 3). Nobody accepts -> contact is told to call 108/112.
Tables (dashboard DB): `hospital_dispatches` (one row per hospital request, status REQUESTED/ACCEPTED/DECLINED/TIMEOUT, eta_min) and
`incident_events` (audit timeline). Both are shown in Alerts -> "Dispatch & Hospital Acknowledgement".
Routes: `POST/GET /api/dispatch` (ingest key, Pi only), `GET/POST /api/ack` (hospital, one-time token).
Setup: copy `Raspberry/.env.example` to `.env`, fill the Telegram ids / e-mail, set `DASHBOARD_PUBLIC_URL=http://<LAPTOP-IP>:3000`.
Seed sample rows: `python scripts/seed_demo_records.py` (the dispatch timeline in it is labelled "sample"); remove with `--clear`.
