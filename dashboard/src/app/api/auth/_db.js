// Local-only user store (SQLite). Server-side only — never import this in a client component.
import Database from "better-sqlite3";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

export const ROLE_IDS = ["owner", "responder", "engineer"];
const SESSION_DAYS = 7;

function init() {
  const dir = path.join(process.cwd(), "data");
  fs.mkdirSync(dir, { recursive: true });
  const db = new Database(path.join(dir, "intellicrash.db"));
  db.pragma("journal_mode = WAL");
  db.pragma("foreign_keys = ON");
  db.exec(`
    CREATE TABLE IF NOT EXISTS users (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      name          TEXT NOT NULL,
      email         TEXT NOT NULL UNIQUE,
      password_hash TEXT NOT NULL,
      role          TEXT NOT NULL,
      created_at    TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS sessions (
      token_hash TEXT PRIMARY KEY,
      user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      expires_at INTEGER NOT NULL
    );
  `);
  return db;
}

// Reuse one connection across Next.js dev hot-reloads.
const db = globalThis.__intellicrashDb ?? (globalThis.__intellicrashDb = init());

// ── password hashing (built-in scrypt, salted; no plain text is ever stored) ──
function hashPassword(pw) {
  const salt = crypto.randomBytes(16);
  const key = crypto.scryptSync(pw, salt, 64);
  return `scrypt$${salt.toString("hex")}$${key.toString("hex")}`;
}
function checkPassword(pw, stored) {
  const [alg, saltHex, keyHex] = stored.split("$");
  if (alg !== "scrypt") return false;
  const expected = Buffer.from(keyHex, "hex");
  const key = crypto.scryptSync(pw, Buffer.from(saltHex, "hex"), expected.length);
  return crypto.timingSafeEqual(key, expected);
}
const sha256 = (s) => crypto.createHash("sha256").update(s).digest("hex");
const publicUser = (u) => ({ id: u.id, name: u.name, email: u.email, role: u.role });

// Shared by sign-up and the dashboard's "Edit details" form.
function normalizeProfile(input) {
  const data = {};
  for (const k of PROFILE_FIELDS) data[k] = String(input?.[k] ?? "").trim().slice(0, 80);
  data.blood_group = data.blood_group.toUpperCase();
  if (!BLOOD_GROUPS.includes(data.blood_group)) return { error: "Choose a valid blood group.", status: 400 };
  if (!data.vehicle_model) return { error: "Vehicle make and model is required.", status: 400 };
  if (!data.plate) return { error: "Registration plate is required.", status: 400 };
  data.plate = data.plate.toUpperCase();
  data.vin = data.vin.toUpperCase();
  return { data };
}

// ── users ─────────────────────────────────────────────────────────────────────
export function registerUser({ name, email, password, role, profile }) {
  name = String(name ?? "").trim();
  email = String(email ?? "").trim().toLowerCase();
  password = String(password ?? "");
  if (name.length < 2) return { error: "Please enter your full name.", status: 400 };
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) return { error: "Enter a valid email address.", status: 400 };
  if (password.length < 8) return { error: "Password must be at least 8 characters.", status: 400 };
  if (!ROLE_IDS.includes(role)) return { error: "Choose a valid role.", status: 400 };
  if (db.prepare("SELECT 1 FROM users WHERE email = ?").get(email)) {
    return { error: "An account with this email already exists.", status: 409 };
  }
  // Fleet Owners must supply their vehicle (and may supply emergency details) at sign-up.
  let profileData = null;
  if (role === "owner") {
    const p = normalizeProfile(profile);
    if (p.error) return p;
    profileData = p.data;
  }
  try {
    // One transaction: the account and its profile are created together or not at all.
    const create = db.transaction(() => {
      const info = db
        .prepare("INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)")
        .run(name, email, hashPassword(password), role);
      const id = Number(info.lastInsertRowid);
      if (profileData) {
        const cols = PROFILE_FIELDS.join(", ");
        const vals = PROFILE_FIELDS.map((k) => `@${k}`).join(", ");
        db.prepare(`INSERT INTO owner_profiles (user_id, ${cols}) VALUES (@id, ${vals})`).run({ ...profileData, id });
      }
      return id;
    });
    const id = create();
    return { user: { id, name, email, role } };
  } catch (e) {
    if (String(e?.message).includes("UNIQUE")) return { error: "An account with this email already exists.", status: 409 };
    throw e;
  }
}

export function loginUser({ email, password }) {
  email = String(email ?? "").trim().toLowerCase();
  password = String(password ?? "");
  const row = db.prepare("SELECT * FROM users WHERE email = ?").get(email);
  if (!row || !checkPassword(password, row.password_hash)) {
    return { error: "Incorrect email or password.", status: 401 };
  }
  return { user: publicUser(row) };
}

// ── sessions (random token in an httpOnly cookie; only its hash is stored) ────
export function createSession(userId) {
  const token = crypto.randomBytes(32).toString("base64url");
  const expires = Date.now() + SESSION_DAYS * 24 * 60 * 60 * 1000;
  db.prepare("DELETE FROM sessions WHERE expires_at < ?").run(Date.now());
  db.prepare("INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)").run(sha256(token), userId, expires);
  return { token, expires };
}

export function getUserBySession(token) {
  if (!token) return null;
  const row = db
    .prepare(
      `SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id
       WHERE s.token_hash = ? AND s.expires_at > ?`
    )
    .get(sha256(token), Date.now());
  return row ? publicUser(row) : null;
}

export function deleteSession(token) {
  if (token) db.prepare("DELETE FROM sessions WHERE token_hash = ?").run(sha256(token));
}

// ── owner / vehicle profile (one row per Fleet Owner account) ────────────────
db.exec(`
  CREATE TABLE IF NOT EXISTS owner_profiles (
    user_id       INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    phone         TEXT NOT NULL DEFAULT '',
    vehicle_model TEXT NOT NULL DEFAULT 'Hyundai Creta',
    vehicle_trim  TEXT NOT NULL DEFAULT 'SX(O)',
    plate         TEXT NOT NULL DEFAULT 'PB 11 AB 2027',
    vin           TEXT NOT NULL DEFAULT 'MAL182749K810283',
    colour        TEXT NOT NULL DEFAULT 'Polar White',
    registered_in TEXT NOT NULL DEFAULT 'Patiala, Punjab',
    blood_group   TEXT NOT NULL DEFAULT '',
    allergies     TEXT NOT NULL DEFAULT '',
    kin_name      TEXT NOT NULL DEFAULT '',
    kin_relation  TEXT NOT NULL DEFAULT '',
    kin_phone     TEXT NOT NULL DEFAULT '',
    updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
  );
`);

export const PROFILE_FIELDS = [
  "phone", "vehicle_model", "vehicle_trim", "plate", "vin", "colour",
  "registered_in", "blood_group", "allergies", "kin_name", "kin_relation", "kin_phone",
];
const BLOOD_GROUPS = ["", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"];

// Shape sent to the dashboard. Owners see/edit their own record; responders and
// engineers see the record of the fleet's registered owner (earliest owner account).
export function getOwnerProfileFor(user) {
  if (!user) return null;
  const ownerRow =
    user.role === "owner"
      ? db.prepare("SELECT id, name FROM users WHERE id = ?").get(user.id)
      : db.prepare("SELECT id, name FROM users WHERE role = 'owner' ORDER BY id LIMIT 1").get();
  if (!ownerRow) return { ownerName: null, canEdit: false, profile: null };
  db.prepare("INSERT OR IGNORE INTO owner_profiles (user_id) VALUES (?)").run(ownerRow.id);
  const profile = db.prepare("SELECT * FROM owner_profiles WHERE user_id = ?").get(ownerRow.id);
  const clean = Object.fromEntries(PROFILE_FIELDS.map((k) => [k, profile[k]]));
  return { ownerName: ownerRow.name, canEdit: user.role === "owner", profile: clean };
}

export function saveOwnerProfile(user, input) {
  if (!user || user.role !== "owner") return { error: "Only a Fleet Owner can edit these details.", status: 403 };
  const p = normalizeProfile(input);
  if (p.error) return p;
  const data = p.data;
  db.prepare("INSERT OR IGNORE INTO owner_profiles (user_id) VALUES (?)").run(user.id);
  const sets = PROFILE_FIELDS.map((k) => `${k} = @${k}`).join(", ");
  db.prepare(`UPDATE owner_profiles SET ${sets}, updated_at = datetime('now') WHERE user_id = @id`).run({ ...data, id: user.id });
  return { ok: true };
}

// ══ Pi-synced data: the central copy of the edge (Raspberry Pi) database ════════
// Same four tables as Raspberry/intellicrash.db. The Pi pushes rows to /api/ingest;
// `edge_id` is the row's id on the Pi (used only to make re-sends idempotent).
// DRIVING sessions (below) are different from the login `sessions` table above.
db.exec(`
  CREATE TABLE IF NOT EXISTS driving_sessions (
    session_id TEXT PRIMARY KEY,
    vehicle_id TEXT NOT NULL,
    source     TEXT,
    start_time TEXT,
    end_time   TEXT
  );
  CREATE TABLE IF NOT EXISTS telemetry_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES driving_sessions(session_id) ON DELETE CASCADE,
    edge_id       INTEGER NOT NULL,
    ts            TEXT, ax REAL, ay REAL, az REAL, gz REAL, speed REAL,
    latitude REAL, longitude REAL,
    p_bilstm REAL, csi REAL, p_final REAL, driving_class TEXT,
    UNIQUE (session_id, edge_id)
  );
  CREATE TABLE IF NOT EXISTS crash_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES driving_sessions(session_id) ON DELETE CASCADE,
    edge_id         INTEGER NOT NULL,
    vehicle_id      TEXT, ts TEXT, latitude REAL, longitude REAL,
    peak_accel REAL, ml_prob REAL, csi_score REAL, fused_score REAL, severity TEXT,
    video_clip_path TEXT,
    is_synced       INTEGER NOT NULL DEFAULT 1,
    UNIQUE (session_id, edge_id)
  );
  CREATE TABLE IF NOT EXISTS alerts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    crash_event_id INTEGER NOT NULL REFERENCES crash_events(id) ON DELETE CASCADE,
    edge_id        INTEGER NOT NULL,
    channel TEXT, status TEXT, dispatched_at TEXT, error_msg TEXT,
    UNIQUE (crash_event_id, edge_id)
  );
`);

const num = (v) => (v === null || v === undefined || v === "" || !Number.isFinite(Number(v)) ? null : Number(v));
const txt = (v, n = 300) => (v === null || v === undefined ? null : String(v).slice(0, n));
const list = (v, max) => (Array.isArray(v) ? v.slice(0, max) : []);

// Idempotent: re-sending a row is harmless. Returns the edge ids the Pi may mark as synced.
export function ingestBatch(body) {
  const ack = { telemetry: [], crashes: [], alerts: [] };
  const hasSession = db.prepare("SELECT 1 FROM driving_sessions WHERE session_id = ?");
  const run = db.transaction(() => {
    const upSession = db.prepare(
      `INSERT INTO driving_sessions (session_id, vehicle_id, source, start_time, end_time) VALUES (?, ?, ?, ?, ?)
       ON CONFLICT(session_id) DO UPDATE SET end_time = COALESCE(excluded.end_time, driving_sessions.end_time)`
    );
    for (const s of list(body?.sessions, 50)) {
      if (!s?.session_id) continue;
      upSession.run(txt(s.session_id, 64), txt(s.vehicle_id, 64) || "UNKNOWN", txt(s.source, 16), txt(s.start_time, 32), txt(s.end_time, 32));
    }
    const insTel = db.prepare(
      `INSERT OR IGNORE INTO telemetry_log (session_id, edge_id, ts, ax, ay, az, gz, speed, latitude, longitude, p_bilstm, csi, p_final, driving_class)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    );
    for (const t of list(body?.telemetry, 1000)) {
      if (!t?.session_id || !hasSession.get(txt(t.session_id, 64)) || num(t.edge_id) === null) continue;
      insTel.run(txt(t.session_id, 64), num(t.edge_id), txt(t.ts, 32), num(t.ax), num(t.ay), num(t.az), num(t.gz), num(t.speed),
        num(t.latitude), num(t.longitude),
        num(t.p_bilstm), num(t.csi), num(t.p_final), txt(t.driving_class, 32));
      ack.telemetry.push(num(t.edge_id));
    }
    const insCrash = db.prepare(
      `INSERT OR IGNORE INTO crash_events (session_id, edge_id, vehicle_id, ts, latitude, longitude, peak_accel, ml_prob, csi_score, fused_score, severity, video_clip_path)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    );
    for (const c of list(body?.crashes, 100)) {
      if (!c?.session_id || !hasSession.get(txt(c.session_id, 64)) || num(c.edge_id) === null) continue;
      insCrash.run(txt(c.session_id, 64), num(c.edge_id), txt(c.vehicle_id, 64), txt(c.ts, 32), num(c.latitude), num(c.longitude),
        num(c.peak_accel), num(c.ml_prob), num(c.csi_score), num(c.fused_score), txt(c.severity, 16), txt(c.video_clip_path, 300));
      ack.crashes.push(num(c.edge_id));
    }
    const findCrash = db.prepare("SELECT id FROM crash_events WHERE session_id = ? AND edge_id = ?");
    const insAlert = db.prepare(
      `INSERT OR IGNORE INTO alerts (crash_event_id, edge_id, channel, status, dispatched_at, error_msg) VALUES (?, ?, ?, ?, ?, ?)`
    );
    for (const a of list(body?.alerts, 300)) {
      const crash = a?.session_id ? findCrash.get(txt(a.session_id, 64), num(a.crash_edge_id)) : null;
      if (!crash || num(a.edge_id) === null) continue; // parent crash not here yet -> Pi retries later
      insAlert.run(crash.id, num(a.edge_id), txt(a.channel, 32), txt(a.status, 32), txt(a.dispatched_at, 32), txt(a.error_msg, 300));
      ack.alerts.push(num(a.edge_id));
    }
  });
  run();
  return ack;
}

// Shapes the rows like the dashboard's static JSON (incidents / alerts_log) so the UI needs no changes.
const incidentLabel = (c) => `CRASH-${String(c.ts ?? "").slice(0, 10).replaceAll("-", "")}-${String(c.id).padStart(3, "0")}`;
const cleanTs = (t) => String(t ?? "").replace("T", " ");

export function listIncidentsAndAlerts() {
  const crashes = db.prepare("SELECT * FROM crash_events ORDER BY ts DESC, id DESC LIMIT 200").all();
  const incidents = crashes.map((c) => {
    const has = c.latitude !== null && c.longitude !== null;
    return {
      id: incidentLabel(c),
      db_id: c.id,
      session_id: c.session_id,
      edge_id: c.edge_id,
      timestamp: cleanTs(c.ts),
      vehicle_id: c.vehicle_id || "UNKNOWN",
      severity: c.severity || "MINOR",
      peak_accel_m_s2: Number((c.peak_accel ?? 0).toFixed(1)),
      p_lstm: c.ml_prob ?? 0,
      csi_score: c.csi_score ?? 0,
      crash_probability: c.fused_score ?? 0,
      gps_coords: has ? [c.latitude, c.longitude] : null,
      location_name: has ? `GPS ${c.latitude.toFixed(4)}, ${c.longitude.toFixed(4)}` : "GPS fix not available",
      evidence_video: c.video_clip_path ? String(c.video_clip_path).replaceAll("\\", "/").split("/").pop() : "no_clip_recorded",
      status: "LOGGED",
      delta_v_kmh: null,
      duration_ms: null,
      xai_top_features: [],
    };
  });
  const byId = new Map(crashes.map((c) => [c.id, c]));
  const alerts = db
    .prepare("SELECT * FROM alerts ORDER BY id DESC LIMIT 300")
    .all()
    .map((a) => ({
      id: `ALT-${String(a.id).padStart(4, "0")}`,
      incident_id: byId.get(a.crash_event_id) ? incidentLabel(byId.get(a.crash_event_id)) : null,
      channel: a.channel,
      recipient: "—",
      timestamp: cleanTs(a.dispatched_at),
      latency_s: null,
      status: a.status || "DISPATCHED",
      error_msg: a.error_msg,
    }));
  return { incidents, alerts };
}

export function dbStats() {
  const n = (t) => db.prepare(`SELECT COUNT(*) AS n FROM ${t}`).get().n;
  return Object.fromEntries(
    ["users", "sessions", "owner_profiles", "driving_sessions", "telemetry_log", "crash_events", "alerts", "hospital_dispatches", "incident_events"].map((t) => [t, n(t)])
  );
}

// ───────────────────────── Dispatch & hospital acknowledgement ─────────────────────────
// Written by the Pi (POST /api/dispatch) and by the hospital's accept / decline link (/api/ack).
// Linked to a crash by (session_id, crash_edge_id) - the same pair stored in crash_events - with no foreign key,
// so an event can arrive before the crash row has synced.
db.exec(`
  CREATE TABLE IF NOT EXISTS hospital_dispatches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL, crash_edge_id INTEGER NOT NULL,
    rank INTEGER, hospital_name TEXT, hospital_email TEXT, lat REAL, lon REAL, distance_km REAL,
    token TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'REQUESTED',   -- REQUESTED | ACCEPTED | DECLINED | TIMEOUT
    eta_min INTEGER, requested_at TEXT, responded_at TEXT
  );
  CREATE TABLE IF NOT EXISTS incident_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL, crash_edge_id INTEGER NOT NULL,
    ts TEXT, actor TEXT, event TEXT, detail TEXT,
    UNIQUE (session_id, crash_edge_id, ts, event, detail)
  );
`);

// Databases created before GPS was added to telemetry_log get the two columns added in place (no data lost).
for (const col of ["latitude", "longitude"]) {
  if (!db.prepare("PRAGMA table_info(telemetry_log)").all().some((c) => c.name === col)) db.exec(`ALTER TABLE telemetry_log ADD COLUMN ${col} REAL`);
}

const nowIso = () => { const d = new Date(), p = (n) => String(n).padStart(2, "0"); return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`; }; // local time, same style as the Pi

export function ingestDispatch(body) {
  const run = db.transaction(() => {
    const insEv = db.prepare(
      `INSERT OR IGNORE INTO incident_events (session_id, crash_edge_id, ts, actor, event, detail) VALUES (?, ?, ?, ?, ?, ?)`
    );
    for (const e of list(body?.events, 100)) {
      if (!e?.session_id || num(e.crash_edge_id) === null || !e.event) continue;
      insEv.run(txt(e.session_id, 64), num(e.crash_edge_id), txt(e.ts, 32) || nowIso(), txt(e.actor, 32), txt(e.event, 48), txt(e.detail, 300) || "");
    }
    const insD = db.prepare(
      `INSERT OR IGNORE INTO hospital_dispatches (session_id, crash_edge_id, rank, hospital_name, hospital_email, lat, lon, distance_km, token, status, requested_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'REQUESTED', ?)`
    );
    for (const d of list(body?.dispatches, 20)) {
      if (!d?.session_id || num(d.crash_edge_id) === null || !d.token) continue;
      insD.run(txt(d.session_id, 64), num(d.crash_edge_id), num(d.rank), txt(d.hospital_name, 120), txt(d.hospital_email, 120),
        num(d.lat), num(d.lon), num(d.distance_km), txt(d.token, 64), txt(d.requested_at, 32) || nowIso());
    }
    // Pi can also mark a request as timed out (no acknowledgement within the window)
    for (const t of list(body?.timeouts, 20)) {
      db.prepare("UPDATE hospital_dispatches SET status='TIMEOUT', responded_at=? WHERE token=? AND status='REQUESTED'").run(nowIso(), txt(t, 64));
    }
  });
  run();
  return { ok: true };
}

export function getDispatchesFor(sessionId, crashEdgeId) {
  return db.prepare("SELECT * FROM hospital_dispatches WHERE session_id=? AND crash_edge_id=? ORDER BY rank, id").all(String(sessionId), Number(crashEdgeId));
}

export function getDispatchByToken(token) {
  return db.prepare("SELECT * FROM hospital_dispatches WHERE token=?").get(String(token || "")) || null;
}

// Hospital response. Only the first answer counts (a later click cannot flip it), and a request that already timed
// out can still be accepted: late help is better than none - the Pi sees it on its next poll.
export function ackByToken(token, action, etaMin) {
  const d = getDispatchByToken(token);
  if (!d) return { error: "unknown" };
  if (d.status === "ACCEPTED" || d.status === "DECLINED") return { already: true, dispatch: d };
  const status = action === "accept" ? "ACCEPTED" : "DECLINED";
  const eta = status === "ACCEPTED" ? Math.max(1, Math.min(180, Number(etaMin) || 0)) || null : null;
  const ts = nowIso();
  db.prepare("UPDATE hospital_dispatches SET status=?, eta_min=?, responded_at=? WHERE id=?").run(status, eta, ts, d.id);
  db.prepare("INSERT OR IGNORE INTO incident_events (session_id, crash_edge_id, ts, actor, event, detail) VALUES (?, ?, ?, ?, ?, ?)")
    .run(d.session_id, d.crash_edge_id, ts, "hospital", status === "ACCEPTED" ? "HOSPITAL_ACCEPTED" : "HOSPITAL_DECLINED",
      `${d.hospital_name}${eta ? ` · ambulance ETA ${eta} min` : ""}`);
  return { ok: true, dispatch: getDispatchByToken(token) };
}

export function listDispatchBoard() {
  return {
    dispatches: db.prepare("SELECT * FROM hospital_dispatches ORDER BY id DESC LIMIT 300").all(),
    events: db.prepare("SELECT * FROM incident_events ORDER BY ts, id LIMIT 1500").all(),
  };
}
