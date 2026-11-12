"use client";

import React, { useState, useEffect, useRef, useMemo } from "react";
import dynamic from "next/dynamic";
import Image from "next/image";
import {
  Activity,
  AlertTriangle,
  ShieldCheck,
  ShieldAlert,
  MapPin,
  Zap,
  PhoneCall,
  Gauge,
  Settings,
  RefreshCw,
  Download,
  Radio,
  Server,
  Database,
  Lock,
  UserCheck,
  Send,
  BellRing,
  CheckCircle2,
  ChevronRight,
  Car,
  Eye,
  Cpu,
  FileText,
  Video,
  Play,
  Pause,
  RotateCcw,
  Sliders,
  Users,
  Award,
  ExternalLink,
  Navigation,
  HardDrive,
  HeartPulse,
  Clock,
  Calendar,
  Layers,
  X,
  Volume2,
  VolumeX,
  User,
  Shield,
  Key,
  LogOut,
  SlidersHorizontal,
  Compass,
  AlertOctagon,
  Sparkles,
  Check,
  Building2,
  Satellite,
  Info,
  Mail
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from "recharts";

import recordedClips from "../../data/recorded_clips.json";
import demoConfig from "../../data/demo_config.json";
import AuthScreen, { ROLE_PRESETS } from "./AuthScreen";

// Dynamic import for Leaflet map to avoid Next.js SSR issues
const HeatMap = dynamic(() => import("./HeatMap"), { ssr: false });
// Real-time WebGL 3D car (three.js) — client only
const Car3D = dynamic(() => import("./Car3D"), { ssr: false });

// ─── 2. SOUND SYNTHESIZER FOR PIEZO BUZZER (GPIO 18 SIMULATION) ─────────────
function playPiezoBeep() {
  try {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;
    const ctx = new AudioContext();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = "square";
    osc.frequency.setValueAtTime(2730, ctx.currentTime);

    gain.gain.setValueAtTime(0.18, ctx.currentTime);
    gain.gain.setValueAtTime(0.0, ctx.currentTime + 0.1);
    gain.gain.setValueAtTime(0.18, ctx.currentTime + 0.15);
    gain.gain.setValueAtTime(0.0, ctx.currentTime + 0.28);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start();
    osc.stop(ctx.currentTime + 0.3);
  } catch (e) {
    console.warn("Web Audio not allowed or unavailable", e);
  }
}

// ─── 3. SPEED & G-FORCE RADIAL DIAL WIDGET ──────────────────────────────────
function KinematicGauge({ value, max, label, unit, color }) {
  const hasValue = value != null;
  const percentage = hasValue ? Math.min(Math.max(value / max, 0), 1) : 0;
  const radius = 38;
  const circumference = Math.PI * radius;
  const strokeDashoffset = circumference - percentage * circumference;

  return (
    <div className="flex flex-col items-center justify-center p-3.5 rounded-3xl bg-white/90 border border-slate-200/80 shadow-sm hover:shadow-md transition-all duration-300">
      <div className="relative w-28 h-16 flex items-start justify-center overflow-hidden">
        <svg className="w-28 h-28 transform rotate-180 origin-center" viewBox="0 0 100 100">
          <circle
            cx="50"
            cy="50"
            r={radius}
            fill="transparent"
            stroke="#f1f5f9"
            strokeWidth="9"
            strokeDasharray={`${circumference} ${circumference}`}
            strokeDashoffset="0"
          />
          <circle
            cx="50"
            cy="50"
            r={radius}
            fill="transparent"
            stroke={color}
            strokeWidth="9"
            strokeDasharray={`${circumference} ${circumference}`}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className="transition-all duration-300 ease-out"
          />
        </svg>
        <div className="absolute bottom-1.5 flex flex-col items-center">
          <span className="text-lg font-black text-slate-900 font-telemetry leading-none">
            {hasValue ? value.toFixed(1) : "n/a"}
          </span>
          <span className="text-[9px] text-slate-500 font-extrabold uppercase mt-0.5">{unit}</span>
        </div>
      </div>
      <span className="text-[11px] font-extrabold text-slate-700 mt-1 uppercase tracking-wider">
        {label}
      </span>
    </div>
  );
}

// ─── 4. RICH SPATIAL VISUAL GRAPHICS (VEHICLE, HOSPITAL, GPS, CRASH) ────────

// C. GPS Satellite Array Visual
function GpsSatelliteGraphic({ vehicle, status }) {
  return (
    <div className="p-5 rounded-3xl spatial-card-cyan organic-shape-1 flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
      <div className="flex items-center gap-3.5">
        <div className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-sky-500 to-blue-600 text-white flex items-center justify-center font-bold shadow-md shrink-0">
          <Satellite className={`w-6 h-6 ${vehicle ? "animate-pulse" : "opacity-60"}`} />
        </div>
        <div>
          <h4 className="text-base font-black text-slate-900">Vehicle Position · u-blox NEO-8M GNSS Receiver</h4>
          <p className="text-xs text-slate-700 font-medium max-w-2xl">
            The NEO-8M is the GPS / GLONASS module fitted in the vehicle. The position below is streamed live from the Raspberry Pi, and every crash is stored with the position at the moment of impact.
          </p>
          <p className="text-[11px] text-slate-600 mt-1 font-telemetry">{status}</p>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 font-telemetry text-xs">
        <span className="px-3.5 py-1.5 rounded-full bg-white border border-sky-300 text-sky-800 font-black shadow-xs">LAT: {vehicle ? `${vehicle.lat.toFixed(5)}°` : "—"}</span>
        <span className="px-3.5 py-1.5 rounded-full bg-white border border-sky-300 text-sky-800 font-black shadow-xs">LON: {vehicle ? `${vehicle.lon.toFixed(5)}°` : "—"}</span>
        {vehicle && (
          <a
            href={mapsPin(vehicle)}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-full bg-sky-600 hover:bg-sky-700 text-white font-extrabold shadow-sm transition"
          >
            <ExternalLink className="w-3.5 h-3.5" /> Open in Google Maps
          </a>
        )}
      </div>
    </div>
  );
}

// D. Crash Event & Biomechanical Impact Envelope Graphic
function CrashDynamicsGraphic() {
  return (
    <div className="p-5 rounded-3xl spatial-card-rose organic-shape-2 space-y-3.5">
      <div className="flex items-center justify-between pb-2.5 border-b border-rose-200/80">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-xl bg-rose-600 text-white flex items-center justify-center shadow-xs">
            <AlertOctagon className="w-4 h-4" />
          </div>
          <div>
            <h4 className="text-xs font-black text-rose-950 uppercase tracking-wide">
              Crash Decision Thresholds
            </h4>
            <p className="text-[11px] text-rose-800 font-medium">A crash is confirmed only when every gate passes. Severity then decides who is alerted.</p>
          </div>
        </div>
        <span className="text-[10px] font-mono font-black px-3 py-1 rounded-full bg-rose-100 text-rose-900 border border-rose-300 shadow-2xs">
          SAFEING BARRIER: ≥ 16.0 m/s²
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 text-xs font-telemetry">
        <div className="p-3.5 rounded-2xl bg-white/95 border border-rose-200/90 shadow-xs">
          <span className="text-[10px] font-bold text-slate-500 uppercase block mb-1">Impact Criterion</span>
          <span className="text-sm font-black text-rose-700">16.0 m/s² (1.63 g)</span>
          <p className="text-[10px] text-slate-600 mt-1 font-sans">Blocks potholes & door slams (~0% FPR)</p>
        </div>
        <div className="p-3.5 rounded-2xl bg-white/95 border border-rose-200/90 shadow-xs">
          <span className="text-[10px] font-bold text-slate-500 uppercase block mb-1">Model Fusion Gate</span>
          <span className="text-sm font-black text-sky-700">P_final ≥ 0.50</span>
          <p className="text-[10px] text-slate-600 mt-1 font-sans">Bi-LSTM recurrent model fused with CSI</p>
        </div>
        <div className="p-3.5 rounded-2xl bg-white/95 border border-rose-200/90 shadow-xs">
          <span className="text-[10px] font-bold text-slate-500 uppercase block mb-1">Crash Severity (CSI)</span>
          <div className="flex flex-col gap-1 mt-1 font-sans">
            <span className="text-[11px] font-bold px-2 py-0.5 rounded-full bg-amber-100 text-amber-900 border border-amber-300 w-fit">MINOR · below 0.40</span>
            <span className="text-[11px] font-bold px-2 py-0.5 rounded-full bg-orange-100 text-orange-900 border border-orange-300 w-fit">SEVERE · 0.40 to 0.75</span>
            <span className="text-[11px] font-bold px-2 py-0.5 rounded-full bg-red-700 text-white border border-red-800 w-fit">FATAL · 0.75 and above</span>
          </div>
        </div>
        <div className="p-3.5 rounded-2xl bg-white/95 border border-rose-200/90 shadow-xs">
          <span className="text-[10px] font-bold text-slate-500 uppercase block mb-1">Edge Decision Loop</span>
          <span className="text-sm font-black text-emerald-700">&lt; 15.0 ms</span>
          <p className="text-[10px] text-slate-600 mt-1 font-sans">Instantaneous hardware dispatch</p>
        </div>
      </div>
    </div>
  );
}

// ─── SHARED HEADINGS ─────────────────────────────────────────────────────────
function PageHeading({ eyebrow, title, tagline }) {
  return (
    <div className="text-center max-w-4xl mx-auto pb-2">
      <p className="text-[11px] uppercase tracking-[0.3em] font-semibold text-sky-600">{eyebrow}</p>
      <h2 className="mt-1 text-2xl sm:text-3xl font-black text-slate-900 tracking-tight">{title}</h2>
      <p className="mt-2 text-base sm:text-lg md:text-xl font-medium italic text-slate-800 tracking-wide leading-snug font-serif">{tagline}</p>
    </div>
  );
}

function SectionLabel({ children }) {
  return (
    <div className="flex items-center gap-3 pt-1">
      <span className="text-[11px] font-black uppercase tracking-[0.25em] text-slate-500 whitespace-nowrap">{children}</span>
      <span className="h-px flex-1 bg-gradient-to-r from-slate-300 to-transparent" />
    </div>
  );
}

// ─── NEAREST HOSPITALS (to the selected crash, from OpenStreetMap via /api/hospitals) ───
function haversineKm(a, b) {
  const R = 6371, rad = (d) => (d * Math.PI) / 180;
  const dLat = rad(b.lat - a.lat), dLon = rad(b.lon - a.lon);
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(rad(a.lat)) * Math.cos(rad(b.lat)) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}
const mapsPin = (p) => `https://www.google.com/maps?q=${p.lat},${p.lon}`;
const mapsRoute = (from, to) => `https://www.google.com/maps/dir/?api=1&origin=${from.lat},${from.lon}&destination=${to.lat},${to.lon}`;

// ─── CLOSED-LOOP DISPATCH: driver check -> hospital request -> hospital acknowledgement ───
const EVENT_LABEL = {
  CRASH_DETECTED: ["Crash detected", "bg-red-100 text-red-800"],
  DRIVER_PROMPTED: ["Driver asked “Are you OK?” (Telegram)", "bg-sky-100 text-sky-800"],
  DRIVER_OK: ["Driver tapped “I’m OK” – no hospital contacted", "bg-emerald-100 text-emerald-800"],
  DRIVER_REQUESTED_HELP: ["Driver tapped “I need help”", "bg-orange-100 text-orange-800"],
  NO_DRIVER_RESPONSE: ["No answer from driver", "bg-amber-100 text-amber-800"],
  DRIVER_PROMPT_UNAVAILABLE: ["Driver prompt unavailable – escalated", "bg-amber-100 text-amber-800"],
  DRIVER_PROMPT_FAILED: ["Driver prompt failed – escalated", "bg-amber-100 text-amber-800"],
  IMMEDIATE_DISPATCH: ["Severe crash – driver check skipped", "bg-orange-100 text-orange-800"],
  HOSPITAL_REQUESTED: ["Ambulance requested from hospital", "bg-purple-100 text-purple-800"],
  CONTACT_NOTIFIED: ["Emergency contact notified", "bg-sky-100 text-sky-800"],
  HOSPITAL_ACCEPTED: ["Hospital accepted", "bg-emerald-100 text-emerald-800"],
  HOSPITAL_DECLINED: ["Hospital cannot serve", "bg-amber-100 text-amber-800"],
  HOSPITAL_NO_RESPONSE: ["Hospital did not answer", "bg-amber-100 text-amber-800"],
  ESCALATION_EXHAUSTED: ["No hospital accepted – contact told to call 108/112", "bg-red-100 text-red-800"],
};
const DISPATCH_BADGE = {
  REQUESTED: ["Waiting for answer", "bg-amber-100 text-amber-800"],
  ACCEPTED: ["Accepted", "bg-emerald-100 text-emerald-800"],
  DECLINED: ["Cannot serve", "bg-slate-200 text-slate-700"],
  TIMEOUT: ["No answer", "bg-red-100 text-red-800"],
};

function DispatchBoard({ incidents, dispatches, events }) {
  const key = (a, b) => `${a}:${b}`;
  const rows = (incidents || [])
    .map((inc) => {
      const k = key(inc.session_id, inc.edge_id);
      return { inc, d: (dispatches || []).filter((x) => key(x.session_id, x.crash_edge_id) === k).sort((a, b) => (a.rank || 0) - (b.rank || 0)),
               ev: (events || []).filter((x) => key(x.session_id, x.crash_edge_id) === k) };
    })
    .filter((r) => r.d.length || r.ev.length);
  return (
    <div className="space-y-3">
      <div>
        <h3 className="text-lg font-black text-slate-900 tracking-tight">Dispatch & Hospital Acknowledgement</h3>
        <p className="text-sm text-slate-600">
          After a crash the driver gets 60 seconds to tap “I’m OK” in Telegram. If they do, nothing is sent to a hospital. If not, the nearest hospital is asked
          to send an ambulance and the emergency contact is told which hospital was asked. <b>ETA</b> = the hospital’s estimated time of arrival of the ambulance, in minutes.
        </p>
      </div>
      {!rows.length && <div className="p-5 rounded-3xl bg-white border border-slate-200 text-sm text-slate-600">No incident has gone through the dispatch flow yet. It appears here after a crash is detected by the Pi.</div>}
      {rows.map(({ inc, d, ev }) => (
        <div key={inc.id} className="rounded-3xl bg-white border border-slate-200 shadow-sm overflow-hidden">
          <div className="px-5 py-3 border-b border-slate-100 flex flex-wrap items-center gap-3">
            <span className="font-black text-slate-900">{inc.id}</span>
            <span className="text-[10px] font-black px-2.5 py-1 rounded-full bg-red-600 text-white">{inc.severity}</span>
            <span className="text-xs text-slate-500">{inc.timestamp}</span>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-0 lg:divide-x divide-slate-100">
            <div className="p-5 space-y-2">
              <div className="text-[11px] font-black uppercase tracking-widest text-slate-500">Hospitals asked</div>
              {!d.length && <div className="text-sm text-slate-500">No hospital was contacted{ev.some((e) => e.event === "DRIVER_OK") ? " (driver confirmed OK)." : "."}</div>}
              {d.map((h) => {
                const [label, cls] = DISPATCH_BADGE[h.status] || [h.status, "bg-slate-100 text-slate-700"];
                return (
                  <div key={h.token} className="flex items-center gap-3 p-3 rounded-2xl bg-slate-50 border border-slate-200">
                    <span className="w-6 h-6 shrink-0 rounded-full bg-slate-800 text-white text-xs font-black flex items-center justify-center">{h.rank}</span>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-bold text-slate-900 truncate">{h.hospital_name}</div>
                      <div className="text-[11px] text-slate-500">{h.distance_km != null ? `${h.distance_km} km` : "distance n/a"} · asked {String(h.requested_at || "").slice(11, 19)}{h.responded_at ? ` · answered ${String(h.responded_at).slice(11, 19)}` : ""}</div>
                    </div>
                    <div className="text-right">
                      <span className={`text-[11px] font-black px-2.5 py-1 rounded-full ${cls}`}>{label}</span>
                      {h.status === "ACCEPTED" && h.eta_min != null && <div className="text-xs font-black text-emerald-700 mt-1">Ambulance ETA {h.eta_min} min</div>}
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="p-5">
              <div className="text-[11px] font-black uppercase tracking-widest text-slate-500 mb-2">Timeline</div>
              <ol className="space-y-1.5">
                {ev.map((e, i) => {
                  const [label, cls] = EVENT_LABEL[e.event] || [e.event, "bg-slate-100 text-slate-700"];
                  return (
                    <li key={i} className="flex items-start gap-2 text-xs">
                      <span className="font-mono text-slate-400 w-16 shrink-0">{String(e.ts || "").slice(11, 19)}</span>
                      <span className={`px-2 py-0.5 rounded-full font-bold ${cls}`}>{label}</span>
                      {e.detail && <span className="text-slate-500 min-w-0">{e.detail}</span>}
                    </li>
                  );
                })}
              </ol>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function NearestHospitalPanel({ crash, origin, info }) {
  const ranked = info.hospitals || [];
  const nearest = ranked[0];
  const isDemoList = info.source === "demo";
  return (
    <div className="space-y-3">
      <div>
        <h3 className="text-lg font-black text-slate-900 tracking-tight">Nearest Hospitals to the {crash ? "Crash" : "Vehicle"}</h3>
        <p className="text-sm text-slate-600">
          {origin
            ? <>Ranked by straight-line distance from {crash ? `where ${crash.id} happened` : "the vehicle"} ({origin.lat.toFixed(5)}, {origin.lon.toFixed(5)}){isDemoList ? ". Demo mode: this is the pinned demo hospital list." : ", using the vehicle’s GPS position at the time of the crash."}</>
            : "Appears as soon as a crash with a GPS position is recorded."}
        </p>
      </div>

      {!origin && (
        <div className="p-5 rounded-3xl bg-white border border-slate-200 text-sm text-slate-600">
          No crash with a GPS fix has been recorded yet, so there is no location to search around.
        </div>
      )}
      {origin && info.state === "loading" && (
        <div className="p-5 rounded-3xl bg-white border border-slate-200 text-sm text-slate-600">Finding hospitals near the crash…</div>
      )}
      {origin && info.state === "error" && (
        <div className="p-5 rounded-3xl bg-amber-50 border border-amber-300 text-sm text-amber-900">Could not load nearby hospitals: {info.error}</div>
      )}
      {origin && info.state === "ok" && !nearest && (
        <div className="p-5 rounded-3xl bg-white border border-slate-200 text-sm text-slate-600">
          No hospital is mapped within {info.radius_km} km of this crash in OpenStreetMap.
        </div>
      )}

      {origin && info.state === "ok" && nearest && (
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
          <div className="lg:col-span-2 p-5 rounded-3xl spatial-card-purple organic-shape-2 space-y-3 shadow-md border-2 border-purple-300">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-black font-mono px-3 py-1 rounded-full bg-purple-600 text-white tracking-widest">NEAREST</span>
              <Building2 className="w-5 h-5 text-purple-700" />
            </div>
            <div>
              <h4 className="text-xl font-black text-slate-900 leading-tight">{nearest.name}</h4>
              {nearest.address && <p className="text-xs text-purple-900 font-medium">{nearest.address}</p>}
              {nearest.phone && <p className="text-xs text-purple-900 font-medium">Tel: {nearest.phone}</p>}
            </div>
            <div className="p-3 rounded-2xl bg-white/95 border border-purple-200 text-center">
              <div className="text-2xl font-black text-purple-800 font-telemetry">{nearest.distance_km.toFixed(1)} km</div>
              <div className="text-[10px] font-bold text-slate-500 uppercase">Straight-line distance</div>
            </div>
            <a
              href={mapsRoute(origin, nearest)}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-purple-600 hover:bg-purple-700 text-white text-xs font-extrabold shadow-sm transition"
            >
              <Navigation className="w-3.5 h-3.5" /> Open route in Google Maps
            </a>
          </div>

          <div className="lg:col-span-3 rounded-3xl bg-white border border-slate-200 shadow-sm overflow-hidden">
            <div className="px-5 py-3 border-b border-slate-100 text-[11px] font-black uppercase tracking-widest text-slate-500">
              Hospitals nearby, closest first
            </div>
            <ul className="divide-y divide-slate-100">
              {ranked.map((h, idx) => (
                <li key={h.id} className={`flex items-center gap-4 px-5 py-3 ${idx === 0 ? "bg-purple-50/70" : ""}`}>
                  <span className={`w-7 h-7 shrink-0 rounded-full flex items-center justify-center text-xs font-black ${idx === 0 ? "bg-purple-600 text-white" : "bg-slate-100 text-slate-600"}`}>{idx + 1}</span>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-bold text-slate-900 truncate">{h.name}</div>
                    <div className="text-[11px] text-slate-500 truncate">{[h.address, h.phone && `Tel: ${h.phone}`, h.emergency === "yes" && "Emergency dept."].filter(Boolean).join(" · ") || "No further details in OpenStreetMap"}</div>
                  </div>
                  <div className="text-right font-telemetry">
                    <div className="text-sm font-black text-slate-900">{h.distance_km.toFixed(1)} km</div>
                  </div>
                  <a href={mapsPin(h)} target="_blank" rel="noopener noreferrer" title="Open in Google Maps" className="p-2 rounded-xl border border-slate-200 hover:bg-slate-50 text-slate-600">
                    <ExternalLink className="w-3.5 h-3.5" />
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
      <p className="text-[11px] text-slate-500">
        {isDemoList
          ? "Pinned demo list (not a live lookup). Distance is in a straight line, not by road."
          : "Hospital data © OpenStreetMap contributors, so names, phone numbers and completeness depend on what has been mapped there. Distance is in a straight line, not by road."}
      </p>
    </div>
  );
}

// Logo artwork only (car + heartbeat + shield) — the wordmark is cropped out so the
// gradient "INTELLICRASH" title is never repeated. Width follows its container.
function LogoMark({ className = "" }) {
  return (
    <div className={`relative w-full overflow-hidden ${className}`} style={{ aspectRatio: "4 / 1" }}>
      <img
        src="/assets/logo.jpg"
        alt="IntelliCrash logo"
        draggable={false}
        className="absolute max-w-none select-none mix-blend-multiply"
        style={{ width: "130.4%", left: "-15.1%", top: "-70%" }}
      />
    </div>
  );
}

const TAB_META = {
  cockpit: { title: "Live Cockpit", subtitle: "Live rash-driving and crash-impact monitoring.", icon: Activity },
  crashes: { title: "Incident Dossiers", subtitle: "Searchable collision audit trail verified by dual-path fusion and the kinematic gate.", icon: FileText },
  dashcam: { title: "Forensic Dashcam", subtitle: "15-second evidence clip saved on every crash.", icon: Video },
  location: { title: "GPS & GIS Map", subtitle: "Live position and accident blackspot heat-map.", icon: MapPin },
  alerts: { title: "Emergency Alerts", subtitle: "Golden-hour dispatch via Twilio SMS, voice, Telegram and SMTP.", icon: AlertTriangle },
  registry: { title: "Vehicle & Health", subtitle: "Host vehicle, owner medical card and Pi diagnostics.", icon: Car },
  team: { title: "Capstone Team", subtitle: "UCS797 · Thapar Institute of Engineering & Technology, Patiala.", icon: Users },
};

const NAV_GROUPS = [
  { title: "Monitor", ids: ["cockpit", "dashcam", "location"] },
  { title: "Records", ids: ["crashes", "alerts"] },
  { title: "System", ids: ["registry", "team"] },
];

// Team / mentor photo. Files live in  public/assets/team/<slug>.png  (jpg / jpeg / webp also work); slug = exact file name without extension.
// If no file is found it falls back to initials.
function PhotoAvatar({ slug, name, textClass = "text-2xl", contain = false }) {
  // Photo fills the circle completely (cover), anchored to the top so heads aren't cut. No shrinking, so no photo edges show.
  // contain = show the whole photo (head, collar and buttons) the same way for every person.
  const exts = ["png", "jpg", "jpeg", "webp"];
  const [i, setI] = useState(0);
  const initials = name.split(" ").filter((w) => /^[A-Za-z]/.test(w)).map((w) => w[0]).join("").slice(0, 2);
  if (i >= exts.length) {
    return (
      <div className={`w-full h-full flex items-center justify-center bg-white font-black text-sky-700 font-serif ${textClass}`}>
        {initials}
      </div>
    );
  }
  return (
    <img
      key={i}
      src={`/assets/team/${encodeURIComponent(slug)}.${exts[i]}`}
      alt={name}
      draggable={false}
      onError={() => setI((n) => n + 1)}
      className={`w-full h-full ${contain ? "object-contain" : "object-cover"} object-top`}
      style={contain ? { transformOrigin: "50% 0%", transform: "scale(1.04)" } : undefined}
    />
  );
}

// ─── CONTACT / LINK CONFIG — fill these in ─────────────────────────────────
// Empty values show a greyed-out icon; add the real value and it becomes a live link.
// WebSocket address of the Raspberry Pi stream, e.g. "ws://192.168.1.50:8765".
// Each message must be JSON like {"ax":0.1,"ay":-0.3,"az":9.8,"gz":0.02,"speed":54,"p_bilstm":0.05,"csi":0.03,"p_final":0.04}
// (p_bilstm, csi and p_final are optional).
// Raspberry Pi WebSocket address, e.g. ws://192.168.1.50:8765 - set NEXT_PUBLIC_PI_WS_URL in .env.local.
const LIVE_STREAM_URL = process.env.NEXT_PUBLIC_PI_WS_URL || "";

// Backup feed used automatically when the Pi cannot be reached: real recorded IMU clips
// (with the model's p_bilstm / csi / p_final scores) played at 25 Hz as normal -> rash -> crash.
// Manual "Hardware Session" options: the full recorded clips, with the scores stored in recorded_clips.json.
const RECORDED_CLIPS = recordedClips.clips;
const SESSION_ORDER = ["normal_cruising", "rash_maneuver", "crash_impact"];
// Status of each recorded sample follows the Pi's detection rule (edge_inference.py): CRASH DETECTED = fused score
// >= 0.46 (models/optimal_fusion_weights.json) AND peak resultant acceleration in the last 2 s window >= 16 m/s^2.
// The recordings hold the scores but not the rash-classifier output, so RASH DRIVING comes from the label stored
// with each recorded sample.
const CRASH_THRESHOLD = 0.46;
const MIN_IMPACT = 16.0;
const WINDOW_SAMPLES = Math.round(2 * recordedClips.sample_rate_hz);
const withStatus = (samples) => {
  const res = samples.map((x) => Math.hypot(x.ax, x.ay));
  return samples.map((x, i) => {
    let peak = 0;
    for (let j = Math.max(0, i - WINDOW_SAMPLES + 1); j <= i; j++) if (res[j] > peak) peak = res[j];
    const status = x.pf >= CRASH_THRESHOLD && peak >= MIN_IMPACT ? "CRASH DETECTED" : x.truth === "rash_drive" ? "RASH DRIVING" : "NORMAL";
    return { ...x, status };
  });
};
const RECORDED_STREAMS = Object.fromEntries(SESSION_ORDER.map((k) => [k, withStatus(RECORDED_CLIPS[k].samples)]));
const clipLabel = (k) => `Session ${RECORDED_CLIPS[k].source_session}: ${RECORDED_CLIPS[k].name}`;

// "Previously recorded" incident shown in replay: derived from the real recorded collision clip (no GPS, no video
// in those recordings, so those fields are left empty rather than invented).
const RECORDED_INCIDENT = (() => {
  const clip = RECORDED_CLIPS.crash_impact;
  const hit = clip.samples.filter((x) => x.truth === "crash");
  const win = hit.length ? hit : clip.samples;
  const max = (k) => Math.max(...win.map((x) => x[k]));
  const peak = Math.max(...clip.samples.map((x) => Math.hypot(x.ax, x.ay)));
  const csiMax = max("csi");
  return {
    id: `REC-${clip.source_session}`,
    db_id: null,
    source: "recorded_clip",
    timestamp: `Recorded clip, session ${clip.source_session}`,
    vehicle_id: "RC test bed",
    severity: csiMax >= 0.75 ? "FATAL" : csiMax >= 0.4 ? "SEVERE" : "MINOR",
    peak_accel_m_s2: Number(peak.toFixed(1)),
    p_lstm: max("pb"),
    csi_score: csiMax,
    crash_probability: max("pf"),
    gps_coords: null,
    location_name: "No GPS in recording",
    evidence_video: "no_clip_recorded",
    status: "RECORDED CLIP",
    delta_v_kmh: null,
    jerk_g_s: null,
    duration_ms: Math.round(hit.length * (1000 / recordedClips.sample_rate_hz)),
    xai_top_features: [],
  };
})();

const REPLAY_PLAYLIST = (() => {
  const c = RECORDED_STREAMS;
  const crashAt = Math.max(0, c.crash_impact.findIndex((x) => x.truth === "crash") - 75);
  return [
    { name: "normal_cruising", samples: c.normal_cruising.slice(0, 250) },
    { name: "rash_maneuver", samples: c.rash_maneuver.slice(0, 300) },
    { name: "crash_impact", samples: c.crash_impact.slice(crashAt) },
  ];
})();

// Owner + vehicle details now come from the signed-in account (GET /api/profile),
// so the Vehicle & Health tab, the owner drawer and the live cockpit always agree.
const EMPTY_PROFILE = {
  phone: "", vehicle_model: "", vehicle_trim: "", plate: "", vin: "", colour: "",
  registered_in: "", blood_group: "", allergies: "", kin_name: "", kin_relation: "", kin_phone: "",
};
const BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"];
const initialsOf = (name = "") =>
  name.split(" ").filter(Boolean).map((n) => n[0]).slice(0, 2).join("").toUpperCase() || "—";
const NOT_SET = "Not set";

const PROJECT_LINKS = {
  github: "https://github.com/Smarth2005/IntelliCrash",
  thapar: "https://www.thapar.edu",
};

const CONTACTS = {
  "Dr. Vibha Jain": { email: "vibha.jain@thapar.edu", linkedin: "https://www.linkedin.com/in/vibha-jain-a627b1a3/" },
  "Smarth Kaushal": { email: "skaushal1_be23@thapar.edu", linkedin: "https://www.linkedin.com/in/smarth-kaushal-02a1092b2/" },
  "Himesh Jindal": { email: "hjindal60_be24@thapar.edu", linkedin: "https://www.linkedin.com/in/himesh-jindal-90a40a249/" },
  "Jasjot Singh": { email: "jsingh61_be24@thapar.edu", linkedin: "https://www.linkedin.com/in/jasjot-singh-660486404/" },
  "Hitesh Kumar": { email: "hkumar60_be24@thapar.edu", linkedin: "https://www.linkedin.com/in/hitesh-kumar-222802328/" },
  "Divyam Bansal": { email: "dbansal60_be24@thapar.edu", linkedin: "https://www.linkedin.com/in/divyam-bansal-34a452428/" },
};


const GitHubIcon = ({ className = "w-4 h-4" }) => (
  <svg viewBox="0 0 24 24" fill="currentColor" className={className} aria-hidden="true">
    <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />
  </svg>
);

const LinkedInIcon = ({ className = "w-4 h-4" }) => (
  <svg viewBox="0 0 24 24" fill="currentColor" className={className} aria-hidden="true">
    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
  </svg>
);

// Email + LinkedIn icon buttons for one person
function ContactIcons({ name, showEmailText = false, className = "" }) {
  const c = CONTACTS[name] || {};
  const base = "w-8 h-8 rounded-full flex items-center justify-center border transition-colors";
  const on = "bg-white border-slate-200 text-slate-600 hover:text-white hover:border-transparent";
  const off = "bg-slate-50 border-dashed border-slate-200 text-slate-300 cursor-not-allowed";
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      {c.email ? (
        <a href={`mailto:${c.email}`} title={c.email} className={`${base} ${on} hover:bg-rose-500`}>
          <Mail className="w-4 h-4" />
        </a>
      ) : (
        <span title="Add email in CONTACTS" className={`${base} ${off}`}>
          <Mail className="w-4 h-4" />
        </span>
      )}
      {c.linkedin ? (
        <a href={c.linkedin} target="_blank" rel="noopener noreferrer" title="LinkedIn" className={`${base} ${on} hover:bg-[#0A66C2]`}>
          <LinkedInIcon />
        </a>
      ) : (
        <span title="Add LinkedIn in CONTACTS" className={`${base} ${off}`}>
          <LinkedInIcon />
        </span>
      )}
      {showEmailText && c.email && <span className="text-xs text-slate-500 font-medium">{c.email}</span>}
    </div>
  );
}

// ─── MAIN DASHBOARD APPLICATION ─────────────────────────────────────────────
export default function Dashboard() {
  const [mounted, setMounted] = useState(false);
  const [activeTab, setActiveTab] = useState("cockpit");
  const [activeSession, setActiveSession] = useState("normal_cruising");
  const [isPlaying, setIsPlaying] = useState(true);
  const [liveSample, setLiveSample] = useState(null);
  const [liveBuffer, setLiveBuffer] = useState([]);
  const [liveStatus, setLiveStatus] = useState("idle");
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [playbackIndex, setPlaybackIndex] = useState(0);
  const [selectedIncident, setSelectedIncident] = useState(null);
  const [alertFilter, setAlertFilter] = useState("All");

  // Interactive Features
  const [buzzerMuted, setBuzzerMuted] = useState(false);
  // Authentication (session cookie + local SQLite via /api/auth/*)
  const [authUser, setAuthUser] = useState(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [isLoginModalOpen, setIsLoginModalOpen] = useState(false);
  const [isOwnerDrawerOpen, setIsOwnerDrawerOpen] = useState(false);
  const [activeUserRole, setActiveUserRole] = useState({
    id: "owner",
    name: "Smarth Kaushal",
    role: "Vehicle Owner & Telematics Admin",
    phone: "+91 98765 xxxxx",
    badge: "FLEET OWNER",
    color: "bg-sky-100 text-sky-800 border-sky-300"
  });

  const [selectedVideoClip, setSelectedVideoClip] = useState("no_clip_recorded");

  useEffect(() => {
    setMounted(true);
  }, []);

  const handleAuthed = (user) => {
    const preset = ROLE_PRESETS[user.role] || ROLE_PRESETS.owner;
    setActiveUserRole({ ...preset, name: user.name, phone: user.email });
    setAuthUser(user);
  };

  // Restore an existing session on page load
  useEffect(() => {
    let cancelled = false;
    fetch("/api/auth/me", { cache: "no-store" })
      .then((r) => r.json())
      .then((d) => {
        if (!cancelled && d.user) handleAuthed(d.user);
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setAuthChecked(true);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Owner + vehicle profile (owners see their own; responders/engineers see the fleet owner's)
  const [ownerInfo, setOwnerInfo] = useState(null);
  const [isEditingProfile, setIsEditingProfile] = useState(false);
  const [profileDraft, setProfileDraft] = useState(EMPTY_PROFILE);
  const [profileError, setProfileError] = useState("");
  const [profileSaving, setProfileSaving] = useState(false);

  const loadOwnerInfo = () =>
    fetch("/api/profile", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => setOwnerInfo(d))
      .catch(() => {});

  useEffect(() => {
    if (authUser) loadOwnerInfo();
    else setOwnerInfo(null);
  }, [authUser?.id]);

  const startEditProfile = () => {
    setProfileDraft({ ...EMPTY_PROFILE, ...(ownerInfo?.profile || {}) });
    setProfileError("");
    setIsEditingProfile(true);
  };
  const saveProfile = async () => {
    setProfileSaving(true);
    setProfileError("");
    try {
      const r = await fetch("/api/profile", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(profileDraft),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || "Could not save.");
      setOwnerInfo(d);
      setIsEditingProfile(false);
    } catch (e) {
      setProfileError(e.message || "Could not save.");
    } finally {
      setProfileSaving(false);
    }
  };

  const handleLogout = async () => {
    try {
      await fetch("/api/auth/logout", { method: "POST" });
    } catch {}
    setAuthUser(null);
    setOwnerInfo(null);
    setIsEditingProfile(false);
    setIsLoginModalOpen(false);
    setIsOwnerDrawerOpen(false);
  };

  // Single source of truth for owner + vehicle details shown anywhere in the UI
  const prof = ownerInfo?.profile || EMPTY_PROFILE;
  const vehicleOwner = {
    name: ownerInfo?.ownerName || "No owner registered",
    initials: initialsOf(ownerInfo?.ownerName || ""),
    phone: prof.phone || NOT_SET,
    kinName: prof.kin_name ? `${prof.kin_name}${prof.kin_relation ? ` (${prof.kin_relation})` : ""}` : NOT_SET,
    kinPhone: prof.kin_phone || NOT_SET,
    kinRelation: prof.kin_relation,
  };
  const vehicleInfo = {
    model: prof.vehicle_model || NOT_SET,
    trim: prof.vehicle_trim,
    full: [prof.vehicle_model, prof.vehicle_trim].filter(Boolean).join(" ") || NOT_SET,
    plate: prof.plate || NOT_SET,
    vin: prof.vin || NOT_SET,
    colour: prof.colour || NOT_SET,
    registeredIn: prof.registered_in || NOT_SET,
  };

  const isLive = activeSession === "live";
  const currentStream = useMemo(() => {
    return RECORDED_STREAMS[activeSession] || RECORDED_STREAMS.normal_cruising;
  }, [activeSession]);

  useEffect(() => {
    if (!isPlaying || isLive) return;
    const intervalTime = Math.max(10, Math.floor(40 / playbackSpeed));
    const interval = setInterval(() => {
      setPlaybackIndex((prev) => (prev + 1) % currentStream.length);
    }, intervalTime);
    return () => clearInterval(interval);
  }, [isPlaying, currentStream, playbackSpeed, isLive]);

  // Incident dossiers + alert log come from the database (crash_events / alerts, filled by the Pi sync).
    const [dbIncidents, setDbIncidents] = useState(null);
  const [dbAlerts, setDbAlerts] = useState(null);
  const [dbDispatches, setDbDispatches] = useState([]);
  const [dbEvents, setDbEvents] = useState([]);
  useEffect(() => {
    if (!authUser) return;
    let stop = false;
    const load = async () => {
      try {
        const r = await fetch("/api/incidents", { cache: "no-store" });
        if (!r.ok) return;
        const d = await r.json();
        if (!stop) { setDbIncidents(d.incidents || []); setDbAlerts(d.alerts || []); setDbDispatches(d.dispatches || []); setDbEvents(d.events || []); }
      } catch {}
    };
    load();
    const t = setInterval(load, 3000);
    return () => { stop = true; clearInterval(t); };
  }, [authUser]);
  // Live (Pi connected): only records the Pi detected and synced. Replay / backup / recorded sessions: those saved
  // records plus the previously recorded collision clip. No bundled demo incidents.
  const piConnected = isLive && liveStatus === "connected";
  const health = piConnected ? liveSample?.health : null; // real Pi readings from the stream (null when unavailable)
  const savedIncidents = dbIncidents || [];
  const incidents = piConnected ? savedIncidents : [...savedIncidents, RECORDED_INCIDENT];
  const alertsLog = dbAlerts || [];

  // Map: the car dot is drawn only from a real GPS fix streamed by the Pi. Crashes with a GPS position can be
  // selected (newest by default); the hospitals nearest to the selected crash are looked up from OpenStreetMap.
  // Demo mode (classroom): the GPS cannot get a fix indoors, so positions that have no GPS are shown at the
  // campus point from demo_config.json, clearly labelled as DEMO. Real GPS positions always take priority,
  // and nothing is written back to the database.
  const [demoMode, setDemoMode] = useState(!!demoConfig.enabled);
  // Fixed classroom position of the car (CSED). Default comes from demo_config.json; "Pin car here" on the map
  // overrides it for this browser (so the exact spot can be corrected on the day without editing any file).
  const [pinOverride, setPinOverride] = useState(null);
  const [pickMode, setPickMode] = useState(false);
  const [followCar, setFollowCar] = useState(true);
  const [fitTick, setFitTick] = useState(0);
  useEffect(() => {
    try {
      const raw = localStorage.getItem("intellicrash_demo_pin");
      if (raw) { const v = JSON.parse(raw); if (Number.isFinite(v?.lat) && Number.isFinite(v?.lon)) setPinOverride(v); }
    } catch {}
  }, []);
  const demoPoint = { lat: pinOverride?.lat ?? demoConfig.campus.lat, lon: pinOverride?.lon ?? demoConfig.campus.lon };
  const realPos = piConnected && liveSample?.lat != null && liveSample?.lon != null
    ? { lat: liveSample.lat, lon: liveSample.lon, speed: liveSample.speed } : null;
  // DEMO ON  : the car is ALWAYS at the fixed classroom spot (GPS ignored, so indoor GPS noise cannot move it).
  // DEMO OFF : the dot is the live GPS fix from the Pi and moves with the car.
  const vehiclePos = demoMode ? { ...demoPoint, speed: null, isDemo: true } : realPos;
  const gpsStatusText = demoMode
    ? `DEMO POSITION: car fixed at ${pinOverride ? "the pinned spot" : demoConfig.label} (${demoPoint.lat.toFixed(5)}, ${demoPoint.lon.toFixed(5)}). Switch Demo mode off to use the live GPS.`
    : realPos ? "GPS fix acquired · live position from the Raspberry Pi"
    : piConnected ? "Raspberry Pi connected, but the GPS has no satellite fix yet (go outdoors / open sky)"
    : "Raspberry Pi not connected. Waiting for the live stream…";
  // Demo mode pins every crash to the fixed classroom spot, so the route always starts at the crash location
  // shown here (never at an old outdoor GPS test position stored in the database). Nothing is written to the DB.
  const mapIncidents = demoMode
    ? incidents.map((i) => ({ ...i, gps_coords: [demoPoint.lat, demoPoint.lon], location_name: `Demo position (${demoConfig.label})`, position_is_demo: true }))
    : incidents;
  useEffect(() => { if (!demoMode && activeSession !== "live") setActiveSession("live"); }, [demoMode]); // live GPS needs the Pi stream
  const [mapFocusId, setMapFocusId] = useState(null);
  const crashesWithGps = mapIncidents.filter((i) => Array.isArray(i.gps_coords));
  const focusCrash = crashesWithGps.find((c) => c.id === mapFocusId) || crashesWithGps[0] || null;
  const focusLat = focusCrash?.gps_coords?.[0];
  const focusLon = focusCrash?.gps_coords?.[1];
  const hospOrigin = focusLat != null ? { lat: focusLat, lon: focusLon } : (demoMode ? demoPoint : null);
  const originLat = hospOrigin?.lat, originLon = hospOrigin?.lon;
  const [hospitalInfo, setHospitalInfo] = useState({ state: "idle", hospitals: [], radius_km: null });
  useEffect(() => {
    if (originLat == null || originLon == null) {
      setHospitalInfo({ state: "idle", hospitals: [], radius_km: null });
      return;
    }
    if (demoMode) {
      const o = { lat: originLat, lon: originLon };
      const ranked = demoConfig.hospitals
        .map((h) => ({ ...h, phone: h.phone || null, emergency: null, distance_km: Number(haversineKm(o, h).toFixed(2)) }))
        .sort((a, b) => a.distance_km - b.distance_km);
      setHospitalInfo({ state: "ok", source: "demo", hospitals: ranked, radius_km: null });
      return;
    }
    if (!authUser) return;
    let stop = false;
    setHospitalInfo({ state: "loading", hospitals: [], radius_km: null });
    fetch(`/api/hospitals?lat=${originLat}&lon=${originLon}`, { cache: "no-store" })
      .then(async (r) => {
        const d = await r.json();
        if (!r.ok) throw new Error(d.error || "Lookup failed.");
        return d;
      })
      .then((d) => { if (!stop) setHospitalInfo({ state: "ok", source: "osm", hospitals: d.hospitals || [], radius_km: d.radius_km }); })
      .catch((e) => { if (!stop) setHospitalInfo({ state: "error", hospitals: [], radius_km: null, error: e.message }); });
    return () => { stop = true; };
  }, [authUser?.id, originLat, originLon, demoMode]);

  // Live session: Raspberry Pi stream, with automatic fallback to the recorded clips if it is unreachable.
  useEffect(() => {
    if (!isLive) return;
    let ws = null, retryTimer = null, probeTimer = null, replayTimer = null, closed = false;

    const push = (sample) => { setLiveSample(sample); setLiveBuffer((b) => [...b.slice(-25), sample]); };
    const stopReplay = () => { if (replayTimer) { clearInterval(replayTimer); replayTimer = null; } };
    const startReplay = () => {
      if (replayTimer) return;
      setLiveStatus("replay");
      setLiveBuffer([]);
      let ci = 0, si = 0;
      replayTimer = setInterval(() => {
        const clip = REPLAY_PLAYLIST[ci];
        const r = clip.samples[si];
        push({
          ax: r.ax, ay: r.ay, az: r.az, gz: r.gz,
          speed: null,
          p_bilstm: r.pb, csi: r.csi, p_final: r.pf,
          status: r.status,
        });
        if (++si >= clip.samples.length) { si = 0; ci = (ci + 1) % REPLAY_PLAYLIST.length; }
      }, 40);
    };
    const connect = () => {
      if (closed) return;
      if (!LIVE_STREAM_URL) { startReplay(); return; }
      if (!replayTimer) setLiveStatus("connecting");
      try { ws = new WebSocket(LIVE_STREAM_URL); } catch { startReplay(); retryTimer = setTimeout(connect, 3000); return; }
      probeTimer = setTimeout(() => { if (ws && ws.readyState !== 1) startReplay(); }, 2500);
      ws.onopen = () => { clearTimeout(probeTimer); stopReplay(); setLiveBuffer([]); setLiveStatus("connected"); };
      ws.onmessage = (e) => {
        try {
          const d = JSON.parse(e.data);
          push({ ax: +d.ax || 0, ay: +d.ay || 0, az: +d.az || 0, gz: +d.gz || 0, speed: d.speed, lat: d.lat ?? null, lon: d.lon ?? null, health: d.health ?? null,
                 p_bilstm: d.p_bilstm, p_final: d.p_final, csi: d.csi, status: d.status });
        } catch {}
      };
      ws.onerror = () => {};
      ws.onclose = () => { clearTimeout(probeTimer); startReplay(); if (!closed) retryTimer = setTimeout(connect, 3000); };
    };
    connect();
    return () => {
      closed = true;
      clearTimeout(retryTimer); clearTimeout(probeTimer); stopReplay();
      if (ws) { ws.onclose = null; ws.close(); }
    };
  }, [isLive]);

  const currentSample = isLive ? (liveSample || { ax: 0, ay: 0, az: 0, gz: 0 }) : (currentStream[playbackIndex] || currentStream[0]);
  const accelX = currentSample.ax;
  const accelY = currentSample.ay;
  const accelZ = currentSample.az;
  const gyroZ = currentSample.gz;
  const peakAccel = Math.sqrt(accelX * accelX + accelY * accelY);
  // Recorded IMU clips have no speed channel, so no speed is shown for them (null), never an invented number.
  const currentSpeed = isLive ? (liveSample?.speed ?? null) : null;

  // Live feed: status computed on the Pi. Recorded sessions and backup replay: the Pi's detection rule applied to the stored scores (see withStatus).
  const feedStatus = isLive ? liveSample?.status : currentSample.status;
  const isImpact = feedStatus === "CRASH DETECTED";
  const isRashDriving = !isImpact && feedStatus === "RASH DRIVING";
  const isCrash = isImpact;

  const systemStatus = isCrash
    ? "CRASH DETECTED"
    : isRashDriving
    ? "RASH DRIVING"
    : "NORMAL";

  // Scores. Live mode uses p_bilstm / csi / p_final sent by the Pi; recorded sessions use the scores stored
  // with every sample of the clip (pb / csi / pf, computed offline by the project's own pipeline). Nothing is invented.
  const src = isLive
    ? { pb: liveSample?.p_bilstm, csi: liveSample?.csi, pf: liveSample?.p_final }
    : { pb: currentSample.pb, csi: currentSample.csi, pf: currentSample.pf };
  const pBilstm = src.pb ?? 0;
  const csi = src.csi ?? 0;
  const pFinal = src.pf ?? 0.6 * pBilstm + 0.4 * csi;
  const severityTier = csi >= 0.75 ? "FATAL" : csi >= 0.4 ? "SEVERE" : "MINOR";
  const sessionLabels = { ...Object.fromEntries(SESSION_ORDER.map((k) => [k, clipLabel(k)])), live: "Live Feed" };

  useEffect(() => {
    if ((isRashDriving || isCrash) && !buzzerMuted) {
      playPiezoBeep();
    }
  }, [isRashDriving, isCrash, buzzerMuted]);

  const recentWaveform = useMemo(() => {
    if (isLive) return liveBuffer.map((s, idx) => ({ time: `${idx * (liveStatus === "replay" ? 40 : 50)}ms`, ax: s.ax, ay: s.ay, gz: s.gz, res: Math.sqrt(s.ax * s.ax + s.ay * s.ay) }));
    const windowSize = 25;
    const start = Math.max(0, playbackIndex - windowSize);
    return currentStream.slice(start, playbackIndex + 1).map((s, idx) => ({
      time: `${idx * 40}ms`,
      ax: s.ax,
      ay: s.ay,
      gz: s.gz,
      res: Math.sqrt(s.ax * s.ax + s.ay * s.ay)
    }));
  }, [playbackIndex, currentStream, isLive, liveBuffer, liveStatus]);

  const windowPeak = recentWaveform.length ? Math.max(...recentWaveform.map((p) => p.res)) : 0;

  const handleDownloadDossier = (incident) => {
    if (!incident) return;

    const htmlContent = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>IntelliCrash Official Forensic Dossier - ${incident.id}</title>
  <style>
    @media print {
      body { -webkit-print-color-adjust: exact; print-color-adjust: exact; background: #ffffff !important; }
      .no-print { display: none !important; }
      .container { border: none !important; box-shadow: none !important; padding: 0 !important; }
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      line-height: 1.5;
      color: #0f172a;
      background: #f8fafc;
      padding: 30px;
      margin: 0;
    }
    .container {
      max-width: 820px;
      margin: 0 auto;
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-radius: 12px;
      padding: 36px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      border-bottom: 2px solid #0284c7;
      padding-bottom: 16px;
      margin-bottom: 24px;
    }
    .title-group h1 {
      margin: 0;
      font-size: 22px;
      color: #0f172a;
      letter-spacing: -0.5px;
    }
    .title-group p {
      margin: 4px 0 0 0;
      font-size: 12px;
      color: #64748b;
    }
    .badge {
      display: inline-block;
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      background: ${incident.severity === "FATAL" ? "#fee2e2" : incident.severity === "SEVERE" ? "#fef3c7" : "#e0f2fe"};
      color: ${incident.severity === "FATAL" ? "#b91c1c" : incident.severity === "SEVERE" ? "#b45309" : "#0369a1"};
      border: 1px solid ${incident.severity === "FATAL" ? "#fca5a5" : incident.severity === "SEVERE" ? "#fcd34d" : "#7dd3fc"};
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-bottom: 24px;
    }
    .card {
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 12px 16px;
    }
    .card-title {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      color: #64748b;
      margin-bottom: 4px;
    }
    .card-val {
      font-size: 15px;
      font-weight: 700;
      color: #0f172a;
      font-family: ui-monospace, Menlo, Consolas, monospace;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 22px;
      font-size: 12px;
    }
    th, td {
      padding: 9px 12px;
      text-align: left;
      border-bottom: 1px solid #e2e8f0;
    }
    th {
      background: #f1f5f9;
      color: #334155;
      font-weight: 700;
    }
    .footer {
      border-top: 1px solid #e2e8f0;
      padding-top: 16px;
      font-size: 11px;
      color: #94a3b8;
      text-align: center;
      line-height: 1.6;
    }
    .btn-print {
      background: #0284c7;
      color: #ffffff;
      border: none;
      padding: 10px 20px;
      border-radius: 6px;
      font-weight: 700;
      font-size: 13px;
      cursor: pointer;
      box-shadow: 0 2px 6px rgba(2, 132, 199, 0.3);
    }
  </style>
</head>
<body>
  <div style="max-width: 820px; margin: 0 auto 16px auto; display: flex; justify-content: flex-end; gap: 10px;" class="no-print">
    <button class="btn-print" onclick="window.print()">🖨️ Print / Save as PDF Dossier</button>
  </div>
  <div class="container">
    <div class="header">
      <div class="title-group">
        <h1>IntelliCrash™ Collision Investigation Dossier</h1>
        <p>Thapar Institute of Engineering and Technology (TIET) • Capstone Project CPG-257 • CSED</p>
      </div>
      <div>
        <span class="badge">${incident.severity} COLLISION</span>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <div class="card-title">Incident Identification</div>
        <div class="card-val">${incident.id}</div>
      </div>
      <div class="card">
        <div class="card-title">Timestamp of Detection</div>
        <div class="card-val">${incident.timestamp}</div>
      </div>
      <div class="card">
        <div class="card-title">GPS Coordinates & Location</div>
        <div class="card-val" style="font-size: 13px;">
          ${incident.gps_coords ? `${incident.gps_coords[0]}° N, ${incident.gps_coords[1]}° E` : "No GPS fix"}
          <br><span style="font-weight: normal; color: #64748b; font-size: 11px;">${incident.location_name}</span>
        </div>
      </div>
      <div class="card">
        <div class="card-title">Vehicle Registration (VIN / Plate)</div>
        <div class="card-val">${incident.vehicle_id}</div>
      </div>
    </div>

    <h3 style="font-size: 13px; text-transform: uppercase; color: #334155; margin-bottom: 8px;">Multi-Modal Kinematic & Fusion Telematics</h3>
    <table>
      <thead>
        <tr>
          <th>Metric Parameter</th>
          <th>Sensor / Mathematical Derivation</th>
          <th>Recorded Value</th>
          <th>Safeing Threshold</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Peak Acceleration (Resultant)</td>
          <td>SmartElex ISM330DHCX 3-Axis IMU (100 Hz)</td>
          <td><b>${incident.peak_accel_m_s2} m/s²</b> (${(incident.peak_accel_m_s2 / 9.81).toFixed(2)} g)</td>
          <td>≥ 16.0 m/s² (BREACHED)</td>
        </tr>
        <tr>
          <td>Bi-LSTM Crash Probability (P_lstm)</td>
          <td>Temporal Recurrent Deep Learning Model</td>
          <td><b>${(incident.p_lstm * 100).toFixed(1)}%</b></td>
          <td>≥ 50.0%</td>
        </tr>
        <tr>
          <td>Crash Severity Index (CSI)</td>
          <td>Kinematic Energy Integration (ΔV / Jerk)</td>
          <td><b>${incident.csi_score?.toFixed(3)} / 1.000</b></td>
          <td>CSI &lt; 0.40: Minor, 0.40–0.75: Severe, ≥ 0.75: Fatal</td>
        </tr>
        <tr>
          <td>Late Fusion Confidence (P_final)</td>
          <td>w_ml (0.60) · P_lstm + w_csi (0.40) · CSI</td>
          <td><b>${(incident.crash_probability * 100).toFixed(1)}%</b></td>
          <td>≥ 50.0% (VERIFIED)</td>
        </tr>
        <tr>
          <td>Estimated Velocity Delta (ΔV)</td>
          <td>Numerical Integration (Jerk / dt)</td>
          <td><b>${incident.delta_v_kmh != null ? incident.delta_v_kmh + " km/h" : "not recorded"}</b></td>
          <td>Impact Speed Equivalent</td>
        </tr>
        <tr>
          <td>Impact Impulse Duration</td>
          <td>Time-above-threshold waveform analysis</td>
          <td><b>${incident.duration_ms != null ? incident.duration_ms + " ms" : "not recorded"}</b></td>
          <td>Golden Hour Response Window</td>
        </tr>
      </tbody>
    </table>

    <h3 style="font-size: 13px; text-transform: uppercase; color: #334155; margin-bottom: 8px;">Explainable AI (SHAP) Factor Attribution</h3>
    <table>
      <thead>
        <tr>
          <th>Biomechanical / Waveform Feature</th>
          <th>Measured Value</th>
          <th>Risk Attribution Impact</th>
        </tr>
      </thead>
      <tbody>
        ${(incident.xai_top_features || []).map(f => `
          <tr>
            <td><code>${f.name}</code></td>
            <td><b>${f.value}</b></td>
            <td style="color: #0284c7; font-weight: 700;">${f.impact}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>

    <div class="footer">
      <p>© 2026 IntelliCrash™. All Rights Reserved.</p>
    </div>
  </div>
</body>
</html>`;

    const blob = new Blob([htmlContent], { type: "text/html;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `IntelliCrash_Dossier_${incident.id}.html`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleDownloadJson = (incident) => {
    if (!incident) return;
    const jsonStr = JSON.stringify(incident, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `IntelliCrash_Blackbox_${incident.id}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const TABS = [
    { id: "cockpit", label: "Live Cockpit", icon: Activity, badge: "LIVE" },
    { id: "crashes", label: "Incident Dossiers", icon: FileText, count: incidents.length },
    { id: "dashcam", label: "Forensic Dashcam", icon: Video },
    { id: "location", label: "GPS & GIS Map", icon: MapPin },
    { id: "alerts", label: "Emergency Alerts", icon: AlertTriangle, count: 4 },
    { id: "registry", label: "Vehicle & Health", icon: Car },
    { id: "team", label: "Capstone Team", icon: Users }
  ];

  // Auth gate: nothing below renders until the user is signed in
  if (!authChecked) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-transparent text-xs font-semibold text-slate-400">
        Loading…
      </div>
    );
  }
  if (!authUser) return <AuthScreen onAuthed={handleAuthed} logo={<LogoMark />} />;

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-transparent text-slate-800">
      <style>{`
        .tilt3d { transition: transform .35s cubic-bezier(.2,.8,.2,1), box-shadow .35s ease; transform-style: preserve-3d; will-change: transform; }
        .tilt3d:hover { transform: perspective(900px) rotateX(3deg) rotateY(-4deg) translateY(-4px); box-shadow: 0 22px 40px -18px rgba(2,132,199,.35); }
        @media (prefers-reduced-motion: reduce) { .tilt3d, .tilt3d:hover { transform: none; } }
      `}</style>

      {/* ═════════════════════════════════════════════════════════════════════ */}
      {/* SIDEBAR                                                               */}
      {/* ═════════════════════════════════════════════════════════════════════ */}
      <aside className="w-68 flex flex-col justify-between border-r border-slate-200/80 bg-gradient-to-b from-white via-white to-sky-50/60 backdrop-blur-2xl shrink-0 shadow-[4px_0_24px_rgba(0,0,0,0.03)] z-20">
        <div className="min-h-0 overflow-y-auto custom-scrollbar">
          {/* Brand lockup: artwork + one wordmark */}
          <button
            onClick={() => setActiveTab("cockpit")}
            className="w-full px-5 pt-5 pb-4 text-center group border-b border-slate-100"
            title="Back to Live Cockpit"
          >
            <div className="w-40 mx-auto group-hover:scale-105 transition-transform duration-300">
              <LogoMark />
            </div>
            <div className="mt-1 flex items-start justify-center gap-0.5">
              <span className="font-black text-xl tracking-tight bg-gradient-to-r from-sky-600 via-blue-600 to-indigo-600 bg-clip-text text-transparent">
                INTELLICRASH
              </span>
              <span className="text-[9px] font-extrabold text-sky-500 mt-1">™</span>
            </div>
            <p className="text-[10px] text-slate-400 font-bold tracking-[0.18em] uppercase mt-0.5">
              Vehicular Telematics AI
            </p>
          </button>

          {/* Grouped navigation */}
          <nav className="px-3 py-4 space-y-5">
            {NAV_GROUPS.map((group) => (
              <div key={group.title} className="space-y-1">
                <div className="px-3 pb-1 text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] font-telemetry">
                  {group.title}
                </div>
                {group.ids.map((id) => {
                  const tab = TABS.find((t) => t.id === id);
                  const Icon = tab.icon;
                  const isActive = activeTab === tab.id;
                  return (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={`relative w-full flex items-center justify-between pl-4 pr-3 py-2.5 rounded-xl text-[13px] transition-all duration-200 ${
                        isActive
                          ? "bg-gradient-to-r from-sky-500 to-indigo-500 text-white font-bold shadow-lg shadow-sky-500/25"
                          : "text-slate-600 hover:bg-slate-100/80 hover:text-slate-900 font-semibold"
                      }`}
                    >
                      <span className="flex items-center gap-3">
                        <Icon className={`w-4 h-4 ${isActive ? "text-white" : "text-slate-400"}`} />
                        {tab.label}
                      </span>
                      {tab.badge && (
                        <span className={`text-[9px] px-1.5 py-0.5 rounded-md font-mono font-black ${isActive ? "bg-white/25 text-white" : "bg-emerald-100 text-emerald-800"}`}>
                          {tab.badge}
                        </span>
                      )}
                      {tab.count !== undefined && (
                        <span className={`text-[10px] min-w-5 text-center font-mono font-black px-1.5 py-0.5 rounded-md ${isActive ? "bg-white/25 text-white" : "bg-slate-100 text-slate-600"}`}>
                          {tab.count}
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            ))}
          </nav>
        </div>

        {/* Edge node status + profile */}
        <div className="p-3.5 space-y-3">
          {/* Profile (moved from the top bar) */}
          <div className="flex items-center gap-1.5 pt-3 border-t border-slate-200/80">
            <button
              onClick={() => setIsOwnerDrawerOpen(true)}
              className="flex-1 min-w-0 flex items-center gap-2.5 p-1.5 rounded-xl hover:bg-slate-100 transition text-left"
            >
              <div className="w-9 h-9 rounded-full bg-gradient-to-tr from-sky-500 to-indigo-600 text-white font-extrabold text-xs flex items-center justify-center shadow-sm shrink-0">
                {activeUserRole.name.split(" ").map((n) => n[0]).slice(0, 2).join("")}
              </div>
              <div className="min-w-0 leading-tight">
                <div className="text-xs font-bold text-slate-800 truncate">{activeUserRole.name}</div>
                <div className="text-[10px] text-slate-500 font-semibold truncate">{activeUserRole.badge}</div>
              </div>
            </button>
            <button
              onClick={() => setIsLoginModalOpen(true)}
              className="p-2 rounded-xl text-slate-500 hover:text-slate-900 hover:bg-slate-100 transition"
              title="Switch role"
            >
              <SlidersHorizontal className="w-4 h-4" />
            </button>
            <button
              onClick={handleLogout}
              className="p-2 rounded-xl text-slate-500 hover:text-rose-600 hover:bg-rose-50 transition"
              title="Log out"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </aside>

      {/* ═════════════════════════════════════════════════════════════════ */}
      {/* MAIN VIEWPORT                                                         */}
      {/* ═════════════════════════════════════════════════════════════════ */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Content Body */}
        <main className="flex-1 overflow-y-auto p-6 bg-slate-50/70 custom-scrollbar">
          {/* ═════════════════════════════════════════════════════════════════ */}
          {/* TAB 1: LIVE TELEMATICS COCKPIT                                    */}
          {/* ═════════════════════════════════════════════════════════════════ */}
          {activeTab === "cockpit" && (
            <div className="space-y-6 animate-fade-in max-w-7xl mx-auto">
              <PageHeading
                eyebrow="Live Cockpit"
                title="Real-Time Vehicle Monitoring"
                tagline="Live motion, rash-driving and crash-risk monitoring."
              />
              {/* Emergency Banner when Crash Detected */}
              {isCrash && (
                <div className="p-4 rounded-2xl bg-gradient-to-r from-rose-50 via-red-50 to-rose-100 border-2 border-rose-400 text-rose-900 flex items-center justify-between shadow-md animate-emergency-flash">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-2xl bg-rose-600 text-white flex items-center justify-center font-bold shadow-md">
                      <AlertOctagon className="w-6 h-6 animate-bounce" />
                    </div>
                    <div>
                      <h3 className="text-sm font-extrabold uppercase tracking-wide">
                        CRITICAL COLLISION DETECTED
                      </h3>
                      <p className="text-xs text-rose-800 font-medium">
                        Peak Impact: {peakAccel.toFixed(1)} m/s² • Golden-Hour Emergency SMS, Voice & Hospital Email Dispatched.
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() => {
                      setActiveSession("normal_cruising");
                      setPlaybackIndex(0);
                    }}
                    className="px-4 py-2 rounded-xl bg-rose-600 hover:bg-rose-700 text-white text-xs font-extrabold shadow-sm transition"
                  >
                    Acknowledge & Reset
                  </button>
                </div>
              )}

              <SectionLabel>Live Status</SectionLabel>
              {/* Top Row: Dual-Phase Telematics AI Spatial Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                {/* 1. Overall Vehicle State (Greenish / Emerald Spatial) */}
                <div className="tilt3d p-4 rounded-3xl spatial-card-emerald organic-shape-1 hover:shadow-xl transition-all duration-300 flex flex-col justify-between">
                  <div className="flex items-center justify-between pb-2 border-b border-emerald-200/80">
                    <span className="text-xs font-black text-emerald-900 uppercase tracking-wide flex items-center gap-2">
                      <ShieldCheck className="w-4 h-4 text-emerald-600" />
                      Overall Vehicle State
                    </span>
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse-subtle shadow-xs" />
                  </div>
                  <p className="text-[11px] text-slate-600 mt-2 truncate">Combined rash and crash verdict</p>
                  <div className="my-2">
                    <span
                      className={`inline-block px-4 py-1.5 rounded-full text-xs font-black font-telemetry tracking-wide shadow-xs ${
                        systemStatus === "CRASH DETECTED"
                          ? "bg-rose-100 text-rose-900 border-2 border-rose-400"
                          : systemStatus === "RASH DRIVING"
                          ? "bg-amber-100 text-amber-950 border-2 border-amber-400"
                          : "bg-emerald-100 text-emerald-900 border-2 border-emerald-400"
                      }`}
                    >
                      {systemStatus}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-xs font-telemetry pt-2 border-t border-emerald-200/60">
                    {[
                      ["Vehicle", vehicleInfo.plate],
                      ["Session", sessionLabels[activeSession] || activeSession],
                      ["Speed", currentSpeed == null ? "n/a (IMU only)" : `${currentSpeed.toFixed(0)} km/h`],
                      ["Net accel", `${peakAccel.toFixed(1)} m/s²`],
                    ].map(([k, v]) => (
                      <div key={k}>
                        <span className="text-slate-500 block text-[10px] font-bold uppercase">{k}</span>
                        <b className="text-slate-900 block truncate">{v}</b>
                      </div>
                    ))}
                  </div>
                </div>

                {/* 2. Phase-1 Pre-Crash Monitor (Orangish / Amber Spatial) */}
                <div className="tilt3d p-4 rounded-3xl spatial-card-amber hover:shadow-xl transition-all duration-300 flex flex-col justify-between">
                  <div className="flex items-center justify-between pb-2 border-b border-amber-200/80">
                    <span className="text-xs font-black text-amber-900 uppercase tracking-wide flex items-center gap-2">
                      <Zap className="w-4 h-4 text-amber-600" />
                      Rash Driving Monitor
                    </span>
                    <span className="text-[10px] font-mono px-2.5 py-0.5 rounded-full bg-amber-100 text-amber-950 font-black border border-amber-300 shadow-2xs">
                      LightGBM
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-600 mt-2 truncate">Driving intensity from the IMU</p>
                  <div className="my-2 space-y-2">
                    {[
                      { label: "Braking / accel |Ax|", value: Math.abs(accelX), max: 20, unit: "m/s²" },
                      { label: "Swerving |Ay|", value: Math.abs(accelY), max: 20, unit: "m/s²" },
                      { label: "Turning rate |Gz|", value: Math.abs(gyroZ), max: 4, unit: "rad/s" },
                    ].map((m) => (
                      <div key={m.label}>
                        <div className="flex items-baseline justify-between mb-1">
                          <span className="text-[11px] font-semibold text-amber-900">{m.label}</span>
                          <span className="text-sm font-black text-amber-950 font-telemetry">{m.value.toFixed(2)} <span className="text-[10px] font-bold text-slate-500">{m.unit}</span></span>
                        </div>
                        <div className="w-full h-2 rounded-full bg-amber-100/80 overflow-hidden border border-amber-200">
                          <div
                            className={`h-full rounded-full transition-all duration-300 ${isRashDriving ? "bg-gradient-to-r from-amber-500 to-orange-500" : "bg-emerald-500"}`}
                            style={{ width: `${Math.min(m.value / m.max, 1) * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="flex items-center justify-between text-xs text-slate-600 pt-2 border-t border-amber-200/60 font-telemetry">
                    <span>Buzzer (GPIO 18): <b className={isRashDriving ? "text-amber-800 font-black" : "text-slate-600"}>{isRashDriving ? "BEEPING" : "OFF"}</b></span>
                  </div>
                </div>

                {/* 3. Crash Anticipation (Reddish / Crimson Spatial) */}
                <div className="tilt3d p-4 rounded-3xl spatial-card-rose organic-shape-2 hover:shadow-xl transition-all duration-300 flex flex-col justify-between">
                  <div className="flex items-center justify-between pb-2 border-b border-rose-200/80">
                    <span className="text-xs font-black text-rose-900 uppercase tracking-wide flex items-center gap-2">
                      <ShieldAlert className="w-4 h-4 text-rose-600" />
                      Crash Anticipation
                    </span>
                    <span className="text-[10px] font-mono px-2.5 py-0.5 rounded-full bg-rose-100 text-rose-950 font-black border border-rose-300 shadow-2xs">
                      Bi-LSTM + CSI
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-600 mt-2 truncate">Collision risk and expected severity</p>
                  <div className="my-2 space-y-1.5">
                    <p className="text-[11px] font-bold text-rose-900 bg-white/70 border border-rose-200 rounded-xl px-3 py-1 font-telemetry text-center">
                      Fusion = 60% × P_BiLSTM + 40% × CSI
                    </p>
                    {[
                      { label: "P_BiLSTM", shown: `${(pBilstm * 100).toFixed(1)}%`, frac: pBilstm },
                      { label: "CSI", shown: csi.toFixed(2), frac: csi },
                    ].map((m) => (
                      <div key={m.label}>
                        <div className="flex items-baseline justify-between mb-1">
                          <span className="text-[11px] font-semibold text-rose-900 font-telemetry">{m.label}</span>
                          <span className="text-sm font-black text-rose-950 font-telemetry">{m.shown}</span>
                        </div>
                        <div className="w-full h-2 rounded-full bg-rose-100/80 overflow-hidden border border-rose-200">
                          <div className="h-full rounded-full bg-rose-400 transition-all duration-300" style={{ width: `${Math.min(m.frac, 1) * 100}%` }} />
                        </div>
                      </div>
                    ))}
                    <div className="pt-2 border-t border-rose-200/70">
                      <div className="flex items-baseline justify-between mb-1.5">
                        <span className="text-xs font-black text-rose-900 font-telemetry">P_final</span>
                        <span className="flex items-center gap-2">
                          <span className="text-2xl font-black text-rose-950 font-telemetry">{(pFinal * 100).toFixed(1)}%</span>
                          {isCrash && (
                            <span className={`text-xs font-black px-3 py-1 rounded-full border-2 ${
                              severityTier === "FATAL" ? "bg-red-700 text-white border-red-800"
                              : severityTier === "SEVERE" ? "bg-orange-100 text-orange-900 border-orange-400"
                              : "bg-amber-100 text-amber-900 border-amber-400"}`}>
                              {severityTier}
                            </span>
                          )}
                        </span>
                      </div>
                      <div className="w-full h-2.5 rounded-full bg-rose-100/80 overflow-hidden p-0.5 border border-rose-200">
                        <div
                          className={`h-full rounded-full transition-all duration-300 ${isCrash ? "bg-gradient-to-r from-rose-600 to-red-600 animate-pulse" : "bg-sky-500"}`}
                          style={{ width: `${Math.min(pFinal, 1) * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <SectionLabel>Sensor Signals</SectionLabel>
              {/* Bottom Row: Waveform Curves (Recharts) */}
              <div className="p-6 rounded-3xl spatial-glass-card">
                <div className="flex items-center justify-between pb-3 border-b border-slate-100">
                  <div>
                    <h4 className="text-sm font-black text-slate-900 flex items-center gap-2">
                      <Activity className="w-4 h-4 text-sky-600" />
                      <span>Live Motion Signals (ISM330DHCX IMU)</span>
                    </h4>
                    <p className="text-xs text-slate-500">
                      Acceleration Ax and Ay on the left axis, yaw rate Gz on the right axis, sampled at 100 Hz.
                    </p>
                  </div>

                  <div className="flex items-center gap-4 text-xs font-telemetry">
                    <span className="flex items-center gap-1.5 text-sky-700 font-bold">
                      <span className="w-3 h-1 bg-sky-500 rounded-full inline-block" /> Ax (Longitudinal)
                    </span>
                    <span className="flex items-center gap-1.5 text-amber-700 font-bold">
                      <span className="w-3 h-1 bg-amber-500 rounded-full inline-block" /> Ay (Lateral)
                    </span>
                    <span className="flex items-center gap-1.5 text-emerald-700 font-bold">
                      <span className="w-3 h-1 bg-emerald-500 rounded-full inline-block" /> Gz (Yaw rate)
                    </span>
                    <span className="flex items-center gap-1.5 text-rose-700 font-extrabold">
                      <span className="w-3 h-1 bg-rose-500 border-b border-dashed inline-block" /> Safeing Gate (16.0 m/s²)
                    </span>
                  </div>
                </div>

                <div className="h-[220px] w-full pt-3">
                  {mounted ? (
                    <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={200}>
                      <LineChart data={recentWaveform} margin={{ top: 10, right: 0, left: -20, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                        <XAxis dataKey="time" stroke="#94a3b8" tick={{ fontSize: 10, fill: "#64748b" }} />
                        <YAxis domain={[-10, 40]} stroke="#94a3b8" tick={{ fontSize: 10, fill: "#64748b" }} />
                        <YAxis yAxisId="right" orientation="right" domain={[-8, 8]} stroke="#94a3b8" tick={{ fontSize: 10, fill: "#64748b" }} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "#ffffff",
                            borderColor: "#e2e8f0",
                            borderRadius: "12px",
                            boxShadow: "0 4px 12px rgba(0,0,0,0.06)",
                            fontSize: "11px",
                            fontFamily: "monospace"
                          }}
                        />
                        <ReferenceLine
                          y={16.0}
                          stroke="#ef4444"
                          strokeDasharray="4 4"
                        />
                        <Line
                          type="monotone"
                          dataKey="ax"
                          stroke="#0284c7"
                          strokeWidth={2.4}
                          dot={false}
                          isAnimationActive={false}
                        />
                        <Line
                          type="monotone"
                          dataKey="ay"
                          stroke="#f59e0b"
                          strokeWidth={2.2}
                          dot={false}
                          isAnimationActive={false}
                        />
                        <Line
                          yAxisId="right"
                          type="monotone"
                          dataKey="gz"
                          stroke="#10b981"
                          strokeWidth={2}
                          dot={false}
                          isAnimationActive={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-full w-full flex items-center justify-center text-xs text-slate-400 font-telemetry">
                      Initializing IMU Waveform Canvas...
                    </div>
                  )}
                </div>

              </div>

              <SectionLabel>Vehicle Twin &amp; Kinematics</SectionLabel>
              {/* Middle Row: 3D Vehicle Kinematics Canvas + Dynamic Gauges */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
                {/* 3D Wireframe Canvas Card */}
                <div className="lg:col-span-8 p-6 rounded-3xl spatial-glass-card flex flex-col">
                  <div className="flex items-center justify-between pb-3 border-b border-slate-100">
                    <div>
                      <h4 className="text-sm font-black text-slate-900 flex items-center gap-2">
                        <Car className="w-4 h-4 text-sky-600" />
                        <span>Live 3D Vehicle Twin</span>
                      </h4>
                      <p className="text-xs text-slate-500">
                        Body tilt follows the IMU; pins show where each sensor sits, with live readings.
                      </p>
                    </div>

                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold text-slate-500 hidden sm:inline">
                        Hardware Session:
                      </span>
                      <select
                        value={activeSession}
                        onChange={(e) => {
                          setActiveSession(e.target.value);
                          setPlaybackIndex(0);
                        }}
                        className="px-3 py-1.5 rounded-full border border-slate-300 bg-white/90 text-xs font-bold text-slate-800 focus:outline-none focus:ring-2 focus:ring-sky-500 shadow-2xs"
                      >
                        {SESSION_ORDER.map((k) => (
                          <option key={k} value={k}>{clipLabel(k)} (recorded)</option>
                        ))}
                        <option value="live">Live: Raspberry Pi (100 Hz)</option>
                      </select>
                    </div>
                  </div>

                  {isLive && (
                    <div className={`mt-3 px-3.5 py-2 rounded-xl border text-xs font-bold whitespace-nowrap overflow-x-auto flex items-center gap-2 ${liveStatus === "connected" ? "bg-emerald-50 text-emerald-800 border-emerald-300" : "bg-amber-50 text-amber-900 border-amber-300"}`}>
                      <span className={`w-2 h-2 rounded-full shrink-0 ${liveStatus === "connected" ? "bg-emerald-500 animate-pulse" : "bg-amber-500"}`} />
                      {liveStatus === "connected" ? "Receiving live data from the Raspberry Pi"
                        : liveStatus === "replay" ? (LIVE_STREAM_URL
                            ? "BACKUP MODE: Raspberry Pi not reachable, showing recorded clips (normal, rash, crash). Reconnecting automatically…"
                            : "BACKUP MODE: no Raspberry Pi address set (NEXT_PUBLIC_PI_WS_URL), showing recorded clips.")
                        : "Connecting to the Raspberry Pi…"}
                    </div>
                  )}
                  <div className="my-3 rounded-2xl border border-slate-200/80 shadow-inner overflow-hidden">
                    <Car3D
                      status={systemStatus}
                      accelX={accelX}
                      accelY={accelY}
                      speed={currentSpeed}
                      height={420}
                    />
                  </div>

                  <div className="flex items-center justify-between text-xs text-slate-600 font-telemetry pt-2 border-t border-slate-100">
                    <span>Ax: <b className="text-slate-900">{accelX.toFixed(2)} m/s²</b></span>
                    <span>Ay: <b className="text-slate-900">{accelY.toFixed(2)} m/s²</b></span>
                    <span>Az: <b className="text-slate-900">{accelZ.toFixed(2)} m/s²</b></span>
                    <span>Gz (Yaw): <b className="text-slate-900">{gyroZ.toFixed(2)} rad/s</b></span>
                  </div>
                </div>

                {/* Gauges & Telemetry Replay Control Card */}
                <div className="lg:col-span-4 p-6 rounded-3xl spatial-glass-card flex flex-col justify-between">
                  <div className="pb-3 border-b border-slate-100">
                    <h4 className="text-sm font-black text-slate-900 flex items-center gap-2">
                      <Gauge className="w-4 h-4 text-sky-600" />
                      <span>Kinematic Gauges & Replay</span>
                    </h4>
                    <p className="text-xs text-slate-500">Speed, G-force, and replay controls for recorded sessions.</p>
                  </div>

                  <div className="grid grid-cols-2 gap-3 my-3">
                    <KinematicGauge
                      value={currentSpeed}
                      max={140}
                      label="Speed"
                      unit="km/h"
                      color="#0284c7"
                    />
                    <KinematicGauge
                      value={peakAccel / 9.81}
                      max={4.0}
                      label="G-Force"
                      unit="g"
                      color={peakAccel >= 16.0 ? "#dc2626" : "#059669"}
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-2 mb-3 text-xs font-telemetry">
                    {[
                      ["Resultant now", `${peakAccel.toFixed(1)} m/s²`],
                      ["Peak (last 0.25 s)", `${windowPeak.toFixed(1)} m/s²`],
                      ["Longitudinal", `${(Math.abs(accelX) / 9.81).toFixed(2)} g`],
                      ["Lateral", `${(Math.abs(accelY) / 9.81).toFixed(2)} g`],
                    ].map(([k, v]) => (
                      <div key={k} className="p-2.5 rounded-xl bg-white border border-slate-200">
                        <span className="text-slate-500 block text-[10px] font-bold uppercase">{k}</span>
                        <b className="text-slate-900 text-sm">{v}</b>
                      </div>
                    ))}
                  </div>

                  <div className={`p-3.5 rounded-2xl bg-slate-50 border border-slate-200 space-y-2 ${isLive ? "opacity-40 pointer-events-none" : ""}`}>
                    <div className="flex items-center justify-between text-xs font-telemetry text-slate-600">
                      <span>{isLive ? "Live stream" : `Frame: ${playbackIndex} / ${currentStream.length}`}</span>
                      <span className="font-extrabold text-sky-700">{playbackSpeed}x Speed</span>
                    </div>

                    <div className="flex items-center justify-between gap-2">
                      <button
                        onClick={() => setIsPlaying(!isPlaying)}
                        className="flex-1 py-2 px-3 rounded-xl bg-sky-600 hover:bg-sky-700 text-white font-extrabold text-xs flex items-center justify-center gap-1.5 shadow-sm transition"
                      >
                        {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                        <span>{isPlaying ? "Pause" : "Play"}</span>
                      </button>

                      <button
                        onClick={() => setPlaybackIndex(0)}
                        className="p-2 rounded-xl border border-slate-200 bg-white hover:bg-slate-100 text-slate-600 transition"
                        title="Rewind to frame 0"
                      >
                        <RotateCcw className="w-3.5 h-3.5" />
                      </button>

                      <div className="flex rounded-xl border border-slate-200 bg-white overflow-hidden text-[11px] font-mono">
                        {[1, 2, 5].map((spd) => (
                          <button
                            key={spd}
                            onClick={() => setPlaybackSpeed(spd)}
                            className={`px-2.5 py-1.5 font-bold ${
                              playbackSpeed === spd
                                ? "bg-sky-100 text-sky-800"
                                : "text-slate-600 hover:bg-slate-50"
                            }`}
                          >
                            {spd}x
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <SectionLabel>Decision Rules</SectionLabel>
              {/* Impact Dynamics & Deceleration Safeing Envelope Graphic */}
              <CrashDynamicsGraphic />
            </div>
          )}

          {/* ═════════════════════════════════════════════════════════════════ */}
          {/* TAB 2: INCIDENT HISTORY & FORENSIC DOSSIERS                       */}
          {/* ═════════════════════════════════════════════════════════════════ */}
          {activeTab === "crashes" && (
            <div className="space-y-6 animate-fade-in max-w-7xl mx-auto">
              <PageHeading
                eyebrow="Incident Dossiers"
                title="Collision Investigation Records"
                tagline="A complete forensic record of every detected collision, ready to open, print or share."
              />
              <p className={`text-[11px] font-bold px-3 py-1.5 rounded-lg border w-fit ${piConnected ? "bg-emerald-50 text-emerald-800 border-emerald-300" : "bg-amber-50 text-amber-900 border-amber-300"}`}>
                {piConnected
                  ? (savedIncidents.length
                      ? `LIVE: ${savedIncidents.length} crash record(s) detected by the Raspberry Pi (crash_events), updated every 3 s`
                      : "LIVE: connected to the Raspberry Pi, no crash detected yet. New crashes appear here automatically.")
                  : `REPLAY: ${savedIncidents.length} saved record(s) from the database plus the recorded collision clip (session ${RECORDED_CLIPS.crash_impact.source_session}).`}
              </p>
              {/* Summary strip */}
              {(() => {
                const inc = incidents;
                const peak = inc.length ? Math.max(...inc.map((i) => Number(i.peak_accel_m_s2) || 0)) : 0;
                const avgCsi = inc.reduce((t, i) => t + (Number(i.csi_score) || 0), 0) / (inc.length || 1);
                const severe = inc.filter((i) => i.severity === "FATAL" || i.severity === "SEVERE").length;
                const cards = [
                  { label: "Recorded incidents", value: inc.length, tone: "from-sky-500 to-blue-600" },
                  { label: "Peak impact", value: `${peak.toFixed(1)} m/s²`, tone: "from-rose-500 to-red-600" },
                  { label: "Average CSI score", value: avgCsi.toFixed(3), tone: "from-indigo-500 to-violet-600" },
                  { label: "Severe / fatal", value: severe, tone: "from-amber-500 to-orange-600" },
                ];
                return (
                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                    {cards.map((c) => (
                      <div key={c.label} className="tilt3d relative overflow-hidden rounded-3xl bg-white border border-slate-200/80 p-5 shadow-sm">
                        <div className={`absolute -right-6 -top-6 w-24 h-24 rounded-full bg-gradient-to-br ${c.tone} opacity-15`} />
                        <span className="text-[10px] font-black uppercase tracking-widest text-slate-400 font-telemetry">{c.label}</span>
                        <div className="mt-1 text-2xl font-black text-slate-900 font-telemetry">{c.value}</div>
                      </div>
                    ))}
                  </div>
                );
              })()}

              {/* Incidents Table */}
              <div className="border border-slate-200 rounded-3xl overflow-hidden bg-white shadow-sm">
                <table className="w-full text-xs text-left">
                  <thead className="bg-slate-50 text-slate-600 uppercase text-[10px] font-telemetry border-b border-slate-200">
                    <tr>
                      <th className="p-3.5">Crash ID</th>
                      <th className="p-3.5">Timestamp (IST)</th>
                      <th className="p-3.5">Location Vicinity</th>
                      <th className="p-3.5">Severity</th>
                      <th className="p-3.5">Peak Accel</th>
                      <th className="p-3.5">CSI Score</th>
                      <th className="p-3.5">Status</th>
                      <th className="p-3.5 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 text-slate-700">
                    {incidents.map((inc) => (
                      <tr key={inc.id} className="hover:bg-slate-50/80 transition">
                        <td className="p-3.5 font-telemetry font-bold text-slate-900">{inc.id}</td>
                        <td className="p-3.5 text-slate-500">{inc.timestamp}</td>
                        <td className="p-3.5 font-medium">{inc.location_name}</td>
                        <td className="p-3.5">
                          <span
                            className={`px-2.5 py-1 rounded-full text-[10px] font-extrabold font-telemetry ${
                              inc.severity === "FATAL"
                                ? "bg-rose-100 text-rose-800 border border-rose-300"
                                : inc.severity === "SEVERE"
                                ? "bg-amber-100 text-amber-900 border border-amber-300"
                                : "bg-sky-100 text-sky-800 border border-sky-300"
                            }`}
                          >
                            {inc.severity}
                          </span>
                        </td>
                        <td className="p-3.5 font-telemetry font-bold">{inc.peak_accel_m_s2} m/s²</td>
                        <td className="p-3.5 font-telemetry font-bold text-slate-800">{inc.csi_score.toFixed(3)}</td>
                        <td className="p-3.5 font-telemetry">
                          <span className="text-emerald-800 font-bold text-[11px] bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-300">
                            {inc.status}
                          </span>
                        </td>
                        <td className="p-3.5 text-right">
                          <button
                            onClick={() => setSelectedIncident(inc)}
                            className="px-3.5 py-1.5 rounded-xl bg-sky-50 border border-sky-200 text-sky-700 hover:bg-sky-100 transition font-bold text-xs shadow-2xs"
                          >
                            Inspect Dossier
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Selected Forensic Dossier Expandable Card */}
              {selectedIncident && (
                <div className="p-6 rounded-3xl spatial-card-cyan space-y-4 animate-fade-in shadow-xl">
                  <div className="flex items-center justify-between pb-3 border-b border-sky-200/80">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-2xl bg-sky-600 text-white flex items-center justify-center font-bold shadow-xs">
                        <FileText className="w-5 h-5" />
                      </div>
                      <div>
                        <span className="text-[10px] font-mono font-bold text-sky-800 bg-white/90 px-2.5 py-0.5 rounded-full border border-sky-300">
                          OFFICIAL FORENSIC DOSSIER
                        </span>
                        <h4 className="text-base font-black text-slate-900 mt-1">
                          Incident {selectedIncident.id} — Kinematic Audit Breakdown
                        </h4>
                      </div>
                    </div>
                    <button
                      onClick={() => setSelectedIncident(null)}
                      className="p-2 rounded-xl hover:bg-sky-100 text-slate-500 hover:text-slate-900 transition"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  </div>

                  <Car3D
                    status="CRASH DETECTED"
                    accelX={Number(selectedIncident.peak_accel_m_s2) || 28}
                    accelY={(Number(selectedIncident.peak_accel_m_s2) || 28) * 0.6}
                    speed={0}
                    height={280}
                    showSensors={false}
                  />

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs font-telemetry">
                    <div className="p-3.5 rounded-2xl bg-white/95 border border-sky-200/90 shadow-2xs">
                      <span className="text-slate-500 block mb-1 font-bold">Peak Acceleration</span>
                      <span className="text-slate-900 font-black text-sm">{selectedIncident.peak_accel_m_s2} m/s²</span>
                    </div>
                    <div className="p-3.5 rounded-2xl bg-white/95 border border-sky-200/90 shadow-2xs">
                      <span className="text-slate-500 block mb-1 font-bold">Velocity Delta (ΔV)</span>
                      <span className="text-slate-900 font-black text-sm">{selectedIncident.delta_v_kmh != null ? `${selectedIncident.delta_v_kmh} km/h` : "not recorded"}</span>
                    </div>
                    <div className="p-3.5 rounded-2xl bg-white/95 border border-sky-200/90 shadow-2xs">
                      <span className="text-slate-500 block mb-1 font-bold">RMS Jerk</span>
                      <span className="text-slate-900 font-black text-sm">{selectedIncident.jerk_g_s != null ? `${selectedIncident.jerk_g_s} g/s` : "not recorded"}</span>
                    </div>
                    <div className="p-3.5 rounded-2xl bg-white/95 border border-sky-200/90 shadow-2xs">
                      <span className="text-slate-500 block mb-1 font-bold">Impact Duration</span>
                      <span className="text-slate-900 font-black text-sm">{selectedIncident.duration_ms != null ? `${selectedIncident.duration_ms} ms` : "not recorded"}</span>
                    </div>
                  </div>

                  <div className="p-4 rounded-2xl bg-white/95 border border-sky-200/90 space-y-2 shadow-2xs">
                    <span className="text-xs font-black text-slate-800 block">
                      Explainable AI (SHAP) Biomechanical Factor Attribution:
                    </span>
                    <div className="space-y-1.5 text-xs">
                      {selectedIncident.xai_top_features.map((f, i) => (
                        <div key={i} className="flex items-center justify-between text-[11px] font-telemetry py-1.5 border-b border-slate-100 last:border-0">
                          <span className="text-slate-700 font-medium">{f.name} ({f.value})</span>
                          <span className="text-sky-700 font-black">{f.impact}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="flex justify-end items-center gap-3 pt-2">
                    <button
                      onClick={() => handleDownloadJson(selectedIncident)}
                      className="flex items-center gap-1.5 px-4 py-2.5 rounded-xl bg-white/95 hover:bg-slate-100 text-slate-800 border border-slate-300 text-xs font-black transition shadow-2xs"
                      title="Download raw machine-readable JSON blackbox file"
                    >
                      <Download className="w-3.5 h-3.5" />
                      <span>Export Raw JSON</span>
                    </button>
                    <button
                      onClick={() => handleDownloadDossier(selectedIncident)}
                      className="flex items-center gap-1.5 px-5 py-2.5 rounded-xl bg-sky-600 hover:bg-sky-700 text-white text-xs font-black transition shadow-sm"
                      title="Download formatted printable forensic report"
                    >
                      <Download className="w-3.5 h-3.5" />
                      <span>Download Official Dossier (.HTML / PDF)</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ═════════════════════════════════════════════════════════════════ */}
          {/* TAB 3: FORENSIC DASHCAM VIDEO EVIDENCE                           */}
          {/* ═════════════════════════════════════════════════════════════════ */}
          {activeTab === "dashcam" && (
            <div className="space-y-6 animate-fade-in max-w-7xl mx-auto">
              <PageHeading
                eyebrow="Forensic Dashcam"
                title="Video Evidence of Every Impact"
                tagline="A 15-second clip (15 fps) is saved on the Raspberry Pi on every detected collision."
              />
              <div className="flex justify-end">
                <span className="text-[11px] font-mono font-bold text-emerald-800 px-3 py-1 rounded-full bg-emerald-50 border border-emerald-300">
                  ROLLING BUFFER · LAST 15 s
                </span>
              </div>

              {/* Main HD Evidence Player Box */}
              <div className="relative w-full h-[360px] bg-slate-600 rounded-3xl overflow-hidden border-2 border-slate-300 shadow-md flex items-center justify-center">
                <div className="relative w-full h-full flex items-center justify-center bg-gradient-to-b from-slate-500 via-slate-600 to-slate-700">
                  <div className="text-center space-y-2">
                    <div className="w-16 h-16 rounded-full bg-white/15 border-2 border-sky-200 text-sky-100 flex items-center justify-center mx-auto shadow-lg">
                      <Play className="w-7 h-7 ml-1" />
                    </div>
                    <p className="text-xs font-bold text-white font-telemetry">
                      ACTIVE INCIDENT CLIP: <span className="text-sky-200">{selectedVideoClip}</span>
                    </p>
                    <p className="text-[11px] text-slate-200 font-telemetry">
                      {selectedVideoClip === "no_clip_recorded"
                        ? "No video clip was recorded for this incident. Clips are saved on the Raspberry Pi in evidence/."
                        : "Clip file stored on the Raspberry Pi (evidence/). It is not copied to the dashboard yet."}
                    </p>
                  </div>

                  <div className="absolute top-4 left-4 p-3 rounded-2xl bg-slate-900/85 border border-slate-700 text-[11px] font-telemetry text-slate-200 space-y-1 backdrop-blur-md shadow-md">
                    <div className="text-sky-400 font-bold flex items-center gap-1.5">
                      <Shield className="w-3.5 h-3.5" />
                      INTELLICRASH BLACKBOX HUD
                    </div>
                    {(() => {
                      const hud = incidents.find((i) => i.evidence_video === selectedVideoClip) || incidents[0];
                      if (!hud) return <div>No incident recorded yet</div>;
                      return (
                        <>
                          <div>GPS: {hud.gps_coords ? `${hud.gps_coords[0].toFixed(5)}° N, ${hud.gps_coords[1].toFixed(5)}° E` : "no fix"}</div>
                          <div>IMPACT: {hud.peak_accel_m_s2} m/s² ({(hud.peak_accel_m_s2 / 9.81).toFixed(2)} g) | {hud.timestamp}</div>
                        </>
                      );
                    })()}
                  </div>
                </div>
              </div>

              {/* Evidence Clips List */}
              <div className="space-y-3">
                <h4 className="text-xs font-extrabold text-slate-700 uppercase tracking-wider font-telemetry">
                  Available Evidence Clips in Local Storage (evidence/):
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {incidents.map((inc) => {
                    const isSelected = selectedVideoClip === inc.evidence_video;
                    return (
                      <div
                        key={inc.id}
                        className={`p-4 rounded-2xl border-2 transition space-y-2 text-xs font-telemetry cursor-pointer ${
                          isSelected
                            ? "border-sky-500 bg-sky-50/80 shadow-sm"
                            : "border-slate-200 bg-white hover:border-slate-300"
                        }`}
                        onClick={() => setSelectedVideoClip(inc.evidence_video)}
                      >
                        <div className="flex items-center justify-between">
                          <span className={`font-extrabold ${isSelected ? "text-sky-800" : "text-slate-800"}`}>
                            {inc.evidence_video}
                          </span>
                          <span className="text-[10px] text-sky-800 font-extrabold bg-sky-100 px-2 py-0.5 rounded-full">
                            15.0s
                          </span>
                        </div>
                        <p className="text-[11px] text-slate-500">{inc.timestamp} • {inc.severity}</p>
                        <p className={`mt-1 text-center rounded-xl py-1.5 text-[11px] font-extrabold ${isSelected ? "bg-sky-600 text-white" : "bg-slate-100 text-slate-600"}`}>
                          {isSelected ? "Now playing in HUD" : "Click to play"}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* ═════════════════════════════════════════════════════════════════ */}
          {/* TAB 4: LIVE LOCATION & GIS ACCIDENT BLACKSPOTS MAP                */}
          {/* ═════════════════════════════════════════════════════════════════ */}
          {activeTab === "location" && (
            <div className="space-y-5 animate-fade-in max-w-7xl mx-auto min-h-[calc(100vh-140px)] flex flex-col">
              <PageHeading
                eyebrow="GPS & GIS Map"
                title="Where the Vehicle Is, and Where Crashes Happen"
                tagline="The live position of the vehicle, every recorded crash, and the hospitals nearest to the selected crash."
              />
              {/* Satellite Graphic Banner */}
              <GpsSatelliteGraphic vehicle={vehiclePos} status={gpsStatusText} />

              {/* Demo mode + crash picker */}
              <div className="flex flex-wrap items-center gap-x-5 gap-y-2 px-4 py-2.5 rounded-2xl bg-amber-50 border border-amber-300 text-xs text-amber-900">
                <div className="flex items-center rounded-xl overflow-hidden border border-amber-400 font-black">
                  <button type="button" onClick={() => setDemoMode(true)} className={`px-3 py-1.5 ${demoMode ? "bg-amber-500 text-white" : "bg-white text-amber-900"}`}>Classroom demo (fixed at CSED)</button>
                  <button type="button" onClick={() => setDemoMode(false)} className={`px-3 py-1.5 ${!demoMode ? "bg-emerald-600 text-white" : "bg-white text-emerald-900"}`}>Live GPS (open sky)</button>
                </div>
                <span className="font-medium">
                  {demoMode
                    ? `Car and crash are fixed at ${pinOverride ? "the pinned spot" : demoConfig.label}; hospitals are the pinned list (works offline).`
                    : "The blue dot is the live GPS position from the Pi and moves with the car; hospitals come from OpenStreetMap."}
                </span>
                {demoMode && (
                  <>
                    <button type="button" onClick={() => setPickMode((v) => !v)} className={`px-3 py-1.5 rounded-xl border font-black ${pickMode ? "bg-amber-600 text-white border-amber-700" : "bg-white border-amber-400"}`}>
                      {pickMode ? "Click the exact spot on the map…" : "Pin car here (click map)"}
                    </button>
                    {pinOverride && (
                      <button type="button" onClick={() => { try { localStorage.removeItem("intellicrash_demo_pin"); } catch {} setPinOverride(null); }} className="px-3 py-1.5 rounded-xl border border-amber-400 bg-white font-bold">Reset to config</button>
                    )}
                  </>
                )}
                {crashesWithGps.length > 0 && (
                  <label className="flex items-center gap-2 font-bold ml-auto">
                    Selected crash
                    <select
                      value={focusCrash?.id || ""}
                      onChange={(e) => setMapFocusId(e.target.value)}
                      className="px-2 py-1 rounded-lg border border-amber-300 bg-white text-slate-800 font-bold"
                    >
                      {crashesWithGps.map((c) => (
                        <option key={c.id} value={c.id}>{c.id} · {c.severity}</option>
                      ))}
                    </select>
                  </label>
                )}
              </div>

              {/* How to read the map */}
              <div className="flex flex-wrap items-center gap-x-6 gap-y-2 px-4 py-2.5 rounded-2xl bg-white/80 border border-slate-200 text-xs text-slate-700">
                <span className="font-black uppercase tracking-wider text-[10px] text-slate-500">How to read the map</span>
                <span><i className="inline-block w-3 h-3 rounded-full bg-sky-400 border-2 border-sky-700 mr-1.5 align-middle" />Blue dot: the vehicle</span>
                <span><i className="inline-block w-3 h-3 rounded-full bg-red-600 mr-1.5 align-middle" />Red dot: a crash (bigger = more severe)</span>
                <span><i className="inline-flex w-4 h-4 rounded-[4px] bg-white border-2 border-red-600 mr-1.5 align-middle items-center justify-center text-[10px] font-black leading-none text-red-600">+</i>Hospital nearest to the selected crash</span>
                <span>Click a red dot to show the hospitals near that crash · use + / − to zoom</span>
              </div>

              {/* Map Container */}
              <div className="flex-1 h-[560px] min-h-[480px] rounded-3xl overflow-hidden border-2 border-slate-200 relative shadow-sm">
                <HeatMap
                  incidents={mapIncidents}
                  startView={vehiclePos ? { center: [vehiclePos.lat, vehiclePos.lon], zoom: 18 } : demoMode ? { center: [demoPoint.lat, demoPoint.lon], zoom: 18 } : null}
                  follow={followCar}
                  fitTick={fitTick}
                  pickMode={demoMode && pickMode}
                  onPick={(pt) => {
                    setPinOverride(pt); setPickMode(false);
                    try { localStorage.setItem("intellicrash_demo_pin", JSON.stringify(pt)); } catch {}
                  }}
                  vehicle={vehiclePos}
                  vehicleLabel={vehicleInfo.plate}
                  hospitals={hospitalInfo.state === "ok" ? hospitalInfo.hospitals : []}
                  focusId={focusCrash?.id || null}
                  onSelectIncident={setMapFocusId}
                />
                <div className="absolute right-3 top-3 z-[500] flex flex-col gap-2">
                  <button type="button" onClick={() => { setFollowCar(false); setFitTick((t) => t + 1); }} className="px-3 py-1.5 rounded-xl bg-white/95 border border-slate-300 text-xs font-black shadow">Show car + hospitals</button>
                  <button type="button" onClick={() => setFollowCar(true)} className={`px-3 py-1.5 rounded-xl border text-xs font-black shadow ${followCar ? "bg-sky-600 text-white border-sky-700" : "bg-white/95 border-slate-300"}`}>Follow car (street zoom)</button>
                </div>
                {!crashesWithGps.length && !demoMode && (
                  <div className="absolute left-3 bottom-3 z-[500] px-3 py-2 rounded-xl bg-white/95 border border-slate-200 text-xs text-slate-700 shadow max-w-xs">
                    No crash with a GPS position yet. A red dot and the nearest hospitals appear here as soon as one is recorded.
                  </div>
                )}
                {hospitalInfo.state === "loading" && (
                  <div className="absolute right-3 bottom-3 z-[500] px-3 py-2 rounded-xl bg-white/95 border border-slate-200 text-xs text-slate-700 shadow">Finding hospitals near the crash…</div>
                )}
                {hospitalInfo.state === "error" && (
                  <div className="absolute right-3 bottom-3 z-[500] px-3 py-2 rounded-xl bg-amber-50 border border-amber-300 text-xs text-amber-900 shadow max-w-xs">Could not load nearby hospitals: {hospitalInfo.error}</div>
                )}
              </div>
            </div>
          )}

          {/* ═════════════════════════════════════════════════════════════════ */}
          {/* TAB 5: EMERGENCY ALERT DISPATCH CENTER                           */}
          {/* ═════════════════════════════════════════════════════════════════ */}
          {activeTab === "alerts" && (
            <div className="space-y-6 animate-fade-in max-w-7xl mx-auto">
              <PageHeading
                eyebrow="Emergency Alerts"
                title="Emergency Dispatch & Nearest Hospitals"
                tagline="Where help is needed after a crash, and every notification that was sent."
              />
              {/* Nearest hospitals */}
              <NearestHospitalPanel crash={focusCrash} origin={hospOrigin} info={hospitalInfo} />
              <DispatchBoard incidents={savedIncidents} dispatches={dbDispatches} events={dbEvents} />

              <SectionLabel>Dispatch Log</SectionLabel>
              <p className="text-sm text-slate-600 -mt-3 mb-1">Every message and call sent for the recorded incidents. Use the filter to show one channel.</p>

              <div className="flex justify-end mb-4">
                <div className="flex items-center gap-1 bg-white p-1 rounded-2xl border border-slate-200 text-xs shadow-sm">
                  {["All", "Telegram", "Hospital Email", "Twilio SMS", "Voice Call"].map((f) => (
                    <button
                      key={f}
                      onClick={() => setAlertFilter(f)}
                      className={`px-3.5 py-1.5 rounded-xl font-bold transition ${
                        alertFilter === f ? "bg-sky-600 text-white shadow-sm" : "text-slate-600 hover:text-slate-900"
                      }`}
                    >
                      {f}
                    </button>
                  ))}
                </div>
              </div>
              {/* Dispatch Logs Table */}
              <div className="border border-slate-200 rounded-3xl overflow-hidden bg-white shadow-sm">
                <table className="w-full text-xs text-left">
                  <thead className="bg-slate-50 text-slate-600 uppercase text-[10px] font-telemetry border-b border-slate-200">
                    <tr>
                      <th className="p-3.5">Dispatch ID</th>
                      <th className="p-3.5">Channel</th>
                      <th className="p-3.5">Target Destination</th>
                      <th className="p-3.5">Timestamp</th>
                      <th className="p-3.5">Latency</th>
                      <th className="p-3.5">Delivery Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 text-slate-700 font-telemetry">
                    {alertsLog
                      .filter((a) => alertFilter === "All" || (a.channel && a.channel.toLowerCase().includes(alertFilter.toLowerCase())))
                      .map((alert) => (
                        <tr key={alert.id} className="hover:bg-slate-50/80 transition">
                          <td className="p-3.5 font-bold text-slate-900">{alert.id}</td>
                          <td className="p-3.5 font-bold text-sky-700">{alert.channel}</td>
                          <td className="p-3.5 text-slate-600">{alert.recipient}</td>
                          <td className="p-3.5 text-slate-500">{alert.timestamp}</td>
                          <td className="p-3.5 font-bold">
                            {alert.latency_s ? `${alert.latency_s} s` : alert.latency_ms ? `${alert.latency_ms} ms` : "—"}
                          </td>
                          <td className="p-3.5">
                            <span
                              className={`px-2.5 py-1 rounded-full text-[10px] font-extrabold ${
                                alert.status === "DELIVERED" || alert.status === "DISPATCHED" || alert.status === "QUEUED_STORE_AND_FORWARD" || alert.status === "QUEUED_THEN_SYNCED"
                                  ? "bg-emerald-100 text-emerald-800 border border-emerald-300"
                                  : "bg-amber-100 text-amber-800 border border-amber-300"
                              }`}
                            >
                              {alert.status}
                            </span>
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>

              {/* Note to the viewer (not a card) */}
              <div className="flex gap-3 border-l-4 border-sky-400 pl-4 py-1 max-w-4xl">
                <HardDrive className="w-5 h-5 text-sky-600 mt-0.5 shrink-0" />
                <div className="space-y-1">
                  <p className="text-sm font-serif italic font-semibold text-slate-800">Note: alerts are never lost, even with no mobile signal.</p>
                  <p className="text-xs text-slate-600 leading-relaxed">Upon collision in remote / cellular dead-zones, crash telemetry, GPS coordinates, CSI scores, and dashcam clips are immediately committed to the local SD card blackbox (`blackbox_queue/`). An autonomous background worker constantly monitors 4G connectivity and immediately transmits the deferred incident packet upon signal restoration.</p>
                </div>
              </div>
            </div>
          )}

          {/* ═════════════════════════════════════════════════════════════════ */}
          {/* TAB 6: VEHICLE REGISTRY & HEALTH                                 */}
          {/* ═════════════════════════════════════════════════════════════════ */}
          {activeTab === "registry" && (
            <div className="space-y-8 animate-fade-in max-w-6xl mx-auto">
              <style>{`@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,600;0,700;1,500&display=swap');
                .font-display{font-family:'Playfair Display',Georgia,serif}
                .font-sleek{font-family:'Outfit',ui-sans-serif,system-ui,sans-serif}`}</style>

              <PageHeading
                eyebrow="Vehicle & Health"
                title="Vehicle, Owner & Edge Node"
                tagline="Who owns the vehicle, what it is, and how healthy the on-board system is."
              />

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* Host vehicle */}
                <section className="lg:col-span-7 relative overflow-hidden rounded-[2rem] p-8 bg-gradient-to-br from-sky-50 via-white to-indigo-50 border border-sky-100 shadow-lg">
                  <div className="absolute -right-16 -top-16 w-56 h-56 rounded-full bg-sky-200/40 blur-2xl" />
                  <p className="relative font-sleek text-[11px] uppercase tracking-[0.3em] font-semibold text-sky-700">Host Vehicle</p>
                  <h3 className="relative font-display text-4xl sm:text-5xl text-slate-900 mt-2 leading-tight">
                    {vehicleInfo.model} {vehicleInfo.trim && <span className="italic text-sky-700">{vehicleInfo.trim}</span>}
                  </h3>
                  <div className="relative mt-5 inline-flex items-center gap-3 px-5 py-2.5 rounded-xl bg-white border-2 border-slate-800 shadow-sm">
                    <span className="w-2 h-7 rounded-sm bg-gradient-to-b from-blue-600 to-blue-800" />
                    <span className="font-sleek font-bold text-2xl tracking-[0.3em] text-slate-900">{vehicleInfo.plate}</span>
                  </div>
                  <dl className="relative mt-7 grid grid-cols-1 sm:grid-cols-3 gap-5 font-sleek">
                    <div>
                      <dt className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Chassis VIN</dt>
                      <dd className="mt-1 text-sm font-semibold text-slate-900 break-all">{vehicleInfo.vin}</dd>
                    </div>
                    <div>
                      <dt className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Colour</dt>
                      <dd className="mt-1 text-sm font-semibold text-slate-900 flex items-center gap-2">
                        <span className="w-4 h-4 rounded-full bg-white border border-slate-300 shadow-inner" /> {vehicleInfo.colour}
                      </dd>
                    </div>
                    <div>
                      <dt className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Registered in</dt>
                      <dd className="mt-1 text-sm font-semibold text-slate-900">{vehicleInfo.registeredIn}</dd>
                    </div>
                  </dl>
                </section>

                {/* Driver medical card */}
                <section className="lg:col-span-5 rounded-[2rem] p-8 bg-gradient-to-br from-rose-50 via-white to-orange-50 border border-rose-100 shadow-lg">
                  <p className="font-sleek text-[11px] uppercase tracking-[0.3em] font-semibold text-rose-700">Owner Emergency Medical Card</p>
                  <p className="font-sleek text-xs text-slate-600 mt-1">Shared with the hospital the moment a crash is confirmed.</p>
                  <div className="mt-5 flex items-center gap-4">
                    <div className="w-20 h-20 rounded-full p-[2px] bg-gradient-to-tr from-rose-500 to-orange-400 shadow-md shrink-0">
                      <div className="w-full h-full rounded-full bg-white overflow-hidden">
                        <div className="w-full h-full bg-gradient-to-b from-slate-100 to-slate-200 flex items-end justify-center">
                          <svg viewBox="0 0 64 64" className="w-full h-full text-slate-400" fill="currentColor" aria-label="Owner photo placeholder">
                            <circle cx="32" cy="24" r="11" />
                            <path d="M10 64c0-13 9.8-22 22-22s22 9 22 22z" />
                          </svg>
                        </div>
                      </div>
                    </div>
                    <div className="min-w-0 flex-1">
                      <h4 className="font-display text-2xl text-slate-900 leading-tight">{vehicleOwner.name}</h4>
                      <p className="font-sleek text-xs text-slate-500">Registered owner</p>
                    </div>
                    <div className="w-16 h-16 shrink-0 rounded-full bg-gradient-to-br from-rose-500 to-red-700 text-white flex flex-col items-center justify-center shadow-lg">
                      <span className="font-display text-2xl leading-none">{prof.blood_group || "—"}</span>
                      <span className="font-sleek text-[8px] uppercase tracking-widest mt-0.5">Blood</span>
                    </div>
                  </div>
                  <dl className="mt-6 space-y-3 font-sleek text-sm">
                    <div className="flex items-center justify-between border-b border-rose-100 pb-2">
                      <dt className="text-slate-500">Known allergies</dt>
                      <dd className="font-semibold text-slate-900">{prof.allergies || "None recorded"}</dd>
                    </div>
                    <div className="flex items-center justify-between">
                      <dt className="text-slate-500">Next of kin</dt>
                      <dd className="font-semibold text-slate-900 text-right">
                        {prof.kin_name || prof.kin_phone ? (
                          <>
                            <span className="block">
                              {prof.kin_name || "Name not set"}
                              {prof.kin_relation && <span className="text-slate-500 font-normal"> ({prof.kin_relation})</span>}
                            </span>
                            <span className="block text-xs font-normal text-slate-500">{prof.kin_phone || "Phone not set"}</span>
                          </>
                        ) : (
                          NOT_SET
                        )}
                      </dd>
                    </div>
                  </dl>
                </section>
              </div>

              <SectionLabel>On-board Edge Node</SectionLabel>
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* Pi health meters */}
                <section className="lg:col-span-5 rounded-[2rem] p-7 bg-white border border-slate-200 shadow-md">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-2xl bg-emerald-100 text-emerald-700 flex items-center justify-center"><Cpu className="w-5 h-5" /></div>
                    <div>
                      <h4 className="font-display text-xl text-slate-900 leading-tight">Raspberry Pi 4B</h4>
                      <p className="font-sleek text-xs text-slate-500">Edge node running both models</p>
                    </div>
                    <span className={`ml-auto font-sleek text-[11px] font-semibold px-3 py-1 rounded-full border ${piConnected ? "bg-emerald-50 text-emerald-700 border-emerald-200" : "bg-slate-100 text-slate-600 border-slate-200"}`}>{piConnected ? "Live from the Pi" : "Pi not connected"}</span>
                  </div>
                  <div className="mt-6 space-y-5 font-sleek">
                    {[
                      { label: "CPU temperature", value: health?.cpu_temp_c != null ? `${health.cpu_temp_c} °C` : "—", note: "", frac: health?.cpu_temp_c != null ? health.cpu_temp_c / 85 : 0 },
                      { label: "Memory in use", value: health?.mem_used_mb != null ? `${health.mem_used_mb} MB` : "—", note: health?.mem_total_mb != null ? `of ${(health.mem_total_mb / 1024).toFixed(1)} GB` : "", frac: health?.mem_used_mb != null && health?.mem_total_mb ? health.mem_used_mb / health.mem_total_mb : 0 },
                      { label: "Blackbox storage", value: health?.disk_free_gb != null ? `${health.disk_free_gb} GB free` : "—", note: health?.disk_total_gb != null ? `of ${health.disk_total_gb} GB` : "", frac: health?.disk_free_gb != null && health?.disk_total_gb ? (health.disk_total_gb - health.disk_free_gb) / health.disk_total_gb : 0 },
                    ].map((m) => (
                      <div key={m.label}>
                        <div className="flex items-baseline justify-between">
                          <span className="text-sm text-slate-600">{m.label}</span>
                          <span className="text-sm font-semibold text-slate-900">{m.value} <span className="text-xs font-normal text-slate-500">{m.note}</span></span>
                        </div>
                        <div className="mt-1.5 h-2 rounded-full bg-slate-100 overflow-hidden">
                          <div className="h-full rounded-full bg-gradient-to-r from-emerald-400 to-sky-500" style={{ width: `${Math.min(Math.max(m.frac, 0), 1) * 100}%` }} />
                        </div>
                      </div>
                    ))}
                    <p className="text-[11px] text-slate-500">Read from the Pi itself while it is connected; "—" means no reading.</p>
                  </div>
                </section>

                {/* Connected hardware */}
                <section className="lg:col-span-7 rounded-[2rem] p-7 bg-white border border-slate-200 shadow-md">
                  <h4 className="font-display text-xl text-slate-900">Connected Hardware</h4>
                  <p className="font-sleek text-xs text-slate-500">The sensors and outputs wired to the edge node.</p>
                  <ul className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3 font-sleek">
                    {[
                      { icon: Activity, name: "ISM330DHCX IMU", desc: "3-axis motion sensor", status: "100 Hz", tone: "sky" },
                      { icon: Satellite, name: "u-blox NEO-8M", desc: "GPS / GLONASS receiver", status: realPos ? "Fix acquired" : vehiclePos?.isDemo ? "Demo position" : "No fix", tone: realPos ? "emerald" : "amber" },
                      { icon: Video, name: "Dashcam", desc: "15-second evidence clips", status: "15 fps", tone: "sky" },
                      { icon: buzzerMuted ? VolumeX : Volume2, name: "Piezo buzzer", desc: "Driver warning (GPIO 18)", status: buzzerMuted ? "Muted" : "Armed", tone: buzzerMuted ? "amber" : "emerald" },
                    ].map((h) => (
                      <li key={h.name} className="flex items-center gap-3 p-3.5 rounded-2xl border border-slate-200 bg-slate-50/60">
                        <div className="w-10 h-10 rounded-xl bg-white border border-slate-200 text-sky-700 flex items-center justify-center shrink-0"><h.icon className="w-5 h-5" /></div>
                        <div className="min-w-0 flex-1">
                          <div className="text-sm font-semibold text-slate-900 truncate">{h.name}</div>
                          <div className="text-xs text-slate-500 truncate">{h.desc}</div>
                        </div>
                        <span className={`text-[11px] font-semibold px-2.5 py-1 rounded-full border ${h.tone === "emerald" ? "bg-emerald-50 text-emerald-700 border-emerald-200" : h.tone === "amber" ? "bg-amber-50 text-amber-800 border-amber-200" : "bg-sky-50 text-sky-700 border-sky-200"}`}>{h.status}</span>
                      </li>
                    ))}
                  </ul>
                </section>
              </div>
            </div>
          )}

          {/* ═════════════════════════════════════════════════════════════════ */}
          {/* TAB 7: CAPSTONE TEAM & ACADEMIC CREDITS (UPGRADED SPATIAL PORTAL) */}
          {/* ═════════════════════════════════════════════════════════════════ */}
          {activeTab === "team" && (
            <div className="relative animate-fade-in max-w-6xl mx-auto rounded-3xl overflow-hidden border-2 border-slate-200/90 shadow-2xl bg-white/95">
              {/* Check / Grid Background */}
              <div
                className="absolute inset-0 opacity-40 pointer-events-none"
                style={{
                  backgroundImage: "url('/assets/grid_bg.png')",
                  backgroundRepeat: "repeat",
                  backgroundSize: "240px",
                }}
              />

              {/* Foreground Content */}
              <div className="relative px-6 sm:px-10 pt-6 sm:pt-8 pb-8 space-y-6">
                {/* ── TOP BAR ── */}
                <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pb-4 border-b border-slate-200/80">
                  <div className="relative h-16 w-52 sm:h-20 sm:w-64 shrink-0 hover:scale-102 transition-transform">
                    <Image
                      src="/assets/cpg_257_ti.png"
                      alt="Thapar Institute CPG No. 257"
                      fill
                      className="object-contain object-left"
                      priority
                    />
                  </div>

                  <div className="text-center sm:text-right">
                    <div className="text-xl sm:text-2xl md:text-3xl font-black tracking-wider text-[#a90000] uppercase font-sans leading-none">
                      CAPSTONE PROJECT
                    </div>
                    <div className="text-lg sm:text-xl md:text-2xl font-black tracking-widest text-[#a90000] font-sans leading-tight mt-1">
                      UCS797
                    </div>
                  </div>
                </div>

                {/* ── CENTER SECTION: artwork (same width as title) + title + subtitle ── */}
                <div className="text-center max-w-5xl mx-auto pt-2 pb-1">
                  <div className="inline-block">
                    <LogoMark />
                    <h1 className="relative mt-2 text-3xl sm:text-4xl md:text-5xl font-black tracking-tight uppercase font-sans leading-none whitespace-nowrap">
                      <span className="bg-gradient-to-r from-sky-600 via-blue-600 to-indigo-600 bg-clip-text text-transparent">
                        INTELLICRASH
                      </span>
                      <span className="absolute left-full top-0 ml-1 text-sky-600 text-lg sm:text-xl font-bold">™</span>
                    </h1>
                  </div>

                  <p className="mt-6 text-lg sm:text-xl md:text-2xl lg:text-[1.7rem] font-medium italic text-slate-900 tracking-wide leading-snug font-serif">
                    <span className="block">A Deep Learning and Physics-Based Fusion for Real-Time</span>
                    <span className="block">Vehicular Crash Detection and Pre-Crash Monitoring</span>
                  </p>
                </div>

                {/* ── STUDENT TEAM (name + role only) ── */}
                <div className="space-y-4 max-w-6xl mx-auto">
                  <div className="text-center space-y-1">
                    <h3 className="text-lg font-black text-slate-900">
                      Development Team
                    </h3>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                    {[
                      { name: "Smarth Kaushal", photo: "Smarth Kaushal", role: ["Machine Learning &", "Predictive Modeling Lead"] },
                      { name: "Himesh Jindal", photo: "Himesh", role: ["Software & Hardware", "Integration"] },
                      { name: "Jasjot Singh", photo: "Jasjot", role: ["Hardware Lead &", "Circuit Prototyping"] },
                      { name: "Hitesh Kumar", photo: "Hitesh", role: ["Hardware Design &", "Sensor Interfacing"] },
                      { name: "Divyam Bansal", photo: "Divyam", role: ["Dashboard &", "Web Interface Lead"] },
                    ].map((m) => (
                      <div
                        key={m.name}
                        className="p-4 rounded-2xl bg-white/90 backdrop-blur-md border border-slate-200/90 hover:border-sky-300 shadow-sm hover:shadow-lg transition-all duration-300 tilt3d flex flex-col items-center text-center space-y-3 group"
                      >
                        <div className="w-36 h-36 rounded-full p-[1.5px] bg-gradient-to-tr from-sky-500 via-blue-500 to-indigo-500 shadow-md shadow-sky-500/25">
                          <div className="w-full h-full rounded-full bg-white overflow-hidden">
                            <PhotoAvatar slug={m.photo} name={m.name} textClass="text-5xl" contain />
                          </div>
                        </div>

                        <h5 className="font-semibold text-base text-slate-800 tracking-tight group-hover:text-sky-700 transition">
                          {m.name}
                        </h5>

                        <div className="pt-2 border-t border-slate-100 w-full">
                          <p className="text-xs font-medium text-slate-500 leading-snug min-h-[2.5rem] flex flex-col items-center justify-center">
                            {m.role.map((line) => (
                              <span key={line} className="block">{line}</span>
                            ))}
                          </p>
                        </div>
                        <ContactIcons name={m.name} className="justify-center" />
                      </div>
                    ))}
                  </div>
                </div>


                {/* ── PROJECT MENTOR (no box) + tagline, name/tagline and designation/institute share the same lines ── */}
                <div className="max-w-6xl mx-auto w-full flex flex-col md:flex-row items-center gap-6 md:gap-8 pt-2 md:pl-12 lg:pl-16">
                  <div className="relative w-40 h-40 sm:w-44 sm:h-44 rounded-full p-[3px] bg-gradient-to-tr from-red-600 via-rose-600 to-amber-500 shadow-xl shrink-0">
                    <div className="w-full h-full rounded-full bg-white overflow-hidden">
                      <PhotoAvatar slug="Dr. Vibha Jain" name="Vibha Jain" textClass="text-5xl" />
                    </div>
                  </div>

                  <div className="w-full md:flex-1 grid grid-cols-1 md:grid-cols-[auto_1fr] md:gap-x-10 gap-y-1 items-baseline text-center md:text-left">
                    <p className="text-xs uppercase tracking-[0.25em] font-semibold text-rose-600">Project Mentor</p>
                    <span className="hidden md:block" />

                    <h3 className="text-2xl sm:text-3xl font-semibold text-slate-900 tracking-tight">Dr. Vibha Jain</h3>
                    <p className="hidden md:block text-right font-serif italic text-3xl sm:text-4xl text-red-900 leading-none">Seeking Excellence Together</p>

                    <p className="text-sm sm:text-base font-medium text-slate-700">Assistant Professor, CSED</p>
                    <p className="hidden md:block text-right text-xs uppercase tracking-[0.25em] font-semibold text-red-800/70">
                      <a href={PROJECT_LINKS.thapar} target="_blank" rel="noopener noreferrer" className="hover:text-red-900 hover:underline underline-offset-4">Thapar Institute of Engineering &amp; Technology</a>
                    </p>

                    <ContactIcons name="Dr. Vibha Jain" className="pt-2 justify-center md:justify-start" />
                  </div>

                  {/* small screens: tagline below */}
                  <div className="md:hidden text-center">
                    <p className="font-serif italic text-3xl text-red-900 leading-snug">Seeking Excellence Together</p>
                    <p className="mt-2 text-xs uppercase tracking-[0.25em] font-semibold text-red-800/70">
                      <a href={PROJECT_LINKS.thapar} target="_blank" rel="noopener noreferrer">Thapar Institute of Engineering &amp; Technology</a>
                    </p>
                  </div>
                </div>

              </div>

              {/* ── CAMPUS PHOTO: full width, sharp, nothing trimmed ── */}
              <div className="relative w-full pointer-events-none select-none">
                <img
                  src="/assets/thapar_campus_building.png"
                  alt="Thapar Institute of Engineering and Technology"
                  className="block w-full h-auto"
                  style={{
                    WebkitMaskImage: "linear-gradient(to bottom, transparent 0%, black 7%)",
                    maskImage: "linear-gradient(to bottom, transparent 0%, black 7%)",
                  }}
                  draggable={false}
                />
                {/* Copyright sits on the photo itself */}
                <div className="absolute inset-x-0 bottom-0 pt-10 pb-3 text-center text-xs font-medium text-white font-telemetry tracking-wide bg-gradient-to-t from-black/55 to-transparent" style={{ textShadow: "0 1px 3px rgba(0,0,0,0.7)" }}>
                  © 2026 IntelliCrash™. All Rights Reserved.
                </div>
              </div>
            </div>
          )}
        </main>
      </div>

      {/* ═════════════════════════════════════════════════════════════════ */}
      {/* MODAL: ROLE SWITCHER / LOGIN SYSTEM                                  */}
      {/* ═════════════════════════════════════════════════════════════════ */}
      {isLoginModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-xs p-4 animate-fade-in">
          <div className="w-full max-w-md bg-white rounded-3xl border border-slate-200 p-6 shadow-2xl space-y-5">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <div className="flex items-center gap-2">
                <SlidersHorizontal className="w-5 h-5 text-sky-600" />
                <h3 className="font-extrabold text-base text-slate-900">Switch Console Role</h3>
              </div>
              <button
                onClick={() => setIsLoginModalOpen(false)}
                className="p-1 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <p className="text-xs text-slate-500 font-medium">
              Select an operational role to test persona perspectives during panel evaluation:
            </p>

            <div className="space-y-2.5">
              {[
                {
                  id: "owner",
                  name: "Smarth Kaushal",
                  role: "Vehicle Owner & Telematics Admin",
                  phone: "+91 98765 xxxxx",
                  badge: "FLEET OWNER",
                  color: "bg-sky-50 text-sky-800 border-sky-300"
                },
                {
                  id: "responder",
                  name: "Punjab 108 Emergency",
                  role: "Triage Paramedic & Dispatcher",
                  phone: "108 Central Dispatch",
                  badge: "EMERGENCY RESPONDER",
                  color: "bg-rose-50 text-rose-800 border-rose-300"
                },
                {
                  id: "engineer",
                  name: "Jasjot / Himesh",
                  role: "Hardware & Edge Systems Engineer",
                  phone: "RPi 4B Dev Node",
                  badge: "SYSTEM ARCHITECT",
                  color: "bg-emerald-50 text-emerald-800 border-emerald-300"
                }
              ].map((role) => (
                <div
                  key={role.id}
                  onClick={() => {
                    setActiveUserRole(role);
                    setIsLoginModalOpen(false);
                  }}
                  className={`p-3.5 rounded-2xl border transition cursor-pointer flex items-center justify-between ${
                    activeUserRole.id === role.id
                      ? "border-sky-500 bg-sky-50/80 shadow-xs"
                      : "border-slate-200 hover:border-slate-300 hover:bg-slate-50"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-xl bg-slate-900 text-white font-bold text-xs flex items-center justify-center">
                      {role.name.slice(0, 2).toUpperCase()}
                    </div>
                    <div>
                      <h5 className="font-bold text-xs text-slate-900">{role.name}</h5>
                      <p className="text-[11px] text-slate-500">{role.role}</p>
                    </div>
                  </div>
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${role.color}`}>
                    {role.badge}
                  </span>
                </div>
              ))}
            </div>

            <button
              onClick={() => setIsLoginModalOpen(false)}
              className="w-full py-2.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-bold transition"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* ═════════════════════════════════════════════════════════════════ */}
      {/* DRAWER: OWNER & VEHICLE DETAILS                                  */}
      {/* ═════════════════════════════════════════════════════════════════ */}
      {isOwnerDrawerOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-end bg-slate-900/30 backdrop-blur-xs p-4 animate-fade-in">
          <div className="w-full max-w-sm h-full bg-white rounded-3xl border border-slate-200 p-6 shadow-2xl flex flex-col justify-between overflow-y-auto">
            <div className="space-y-5">
              <div className="flex items-center justify-between pb-3 border-b border-slate-100">
                <div className="flex items-center gap-2">
                  <User className="w-5 h-5 text-sky-600" />
                  <h3 className="font-extrabold text-base text-slate-900">Vehicle Owner Profile</h3>
                </div>
                <button
                  onClick={() => setIsOwnerDrawerOpen(false)}
                  className="p-1 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <div className="p-4 rounded-2xl bg-gradient-to-tr from-sky-50 to-indigo-50 border border-sky-200 text-center space-y-2">
                <div className="w-16 h-16 rounded-full bg-gradient-to-tr from-sky-500 to-indigo-600 text-white font-extrabold text-lg flex items-center justify-center mx-auto shadow-md">
                  {vehicleOwner.initials}
                </div>
                <div>
                  <h4 className="font-extrabold text-base text-slate-900">{vehicleOwner.name}</h4>
                  <p className="text-xs text-sky-700 font-semibold">Registered Vehicle Owner</p>
                </div>
                <div className="text-[11px] text-slate-500 font-telemetry">
                  📞 {vehicleOwner.phone}
                </div>
              </div>

              <div className="space-y-2">
                <h5 className="text-xs font-bold text-slate-700 uppercase tracking-wide">Vehicle Info</h5>
                <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200 text-xs font-telemetry space-y-1.5">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Plate:</span>
                    <span className="font-bold text-slate-900">{vehicleInfo.plate}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Model:</span>
                    <span className="font-bold text-slate-900">{vehicleInfo.full}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">VIN:</span>
                    <span className="font-bold text-slate-900">{vehicleInfo.vin}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Firmware:</span>
                    <span className="font-bold text-emerald-700">IntelliCrash v2.4 (Dual-Path)</span>
                  </div>
                </div>
              </div>

              <div className="space-y-2">
                <h5 className="text-xs font-bold text-slate-700 uppercase tracking-wide">Emergency Contacts</h5>
                <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200 text-xs font-telemetry space-y-2">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-bold text-slate-900">{vehicleOwner.kinName}</p>
                      <p className="text-[10px] text-slate-500">{vehicleOwner.kinPhone}</p>
                    </div>
                    <span className="text-[10px] bg-rose-50 text-rose-700 border border-rose-200 px-2 py-0.5 rounded-full font-bold">
                      PRIMARY SOS
                    </span>
                  </div>
                  <div className="flex items-center justify-between pt-2 border-t border-slate-200/60">
                    <div>
                      <p className="font-bold text-slate-900">{hospitalInfo.hospitals?.[0]?.name || "Nearest hospital"}</p>
                      <p className="text-[10px] text-slate-500">
                        {hospitalInfo.hospitals?.[0]
                          ? `${hospitalInfo.hospitals[0].distance_km.toFixed(1)} km from the latest crash${hospitalInfo.hospitals[0].phone ? ` · Tel: ${hospitalInfo.hospitals[0].phone}` : ""}`
                          : "Shown after a crash with a GPS position is recorded"}
                      </p>
                    </div>
                    <span className="text-[10px] bg-sky-50 text-sky-700 border border-sky-200 px-2 py-0.5 rounded-full font-bold">
                      HOSPITAL
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-4 space-y-3">
              {ownerInfo?.canEdit && !isEditingProfile && (
                <button
                  onClick={startEditProfile}
                  className="w-full py-2.5 rounded-xl border border-sky-300 text-sky-700 hover:bg-sky-50 text-xs font-bold transition"
                >
                  Edit owner &amp; vehicle details
                </button>
              )}
              {ownerInfo?.canEdit && isEditingProfile && (
                <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200 space-y-2.5">
                  <p className="text-xs font-bold text-slate-700 uppercase tracking-wide">Edit details</p>
                  {[
                    ["phone", "Your phone", "+91 …"],
                    ["vehicle_model", "Vehicle make & model", "Hyundai Creta"],
                    ["vehicle_trim", "Variant", "SX(O)"],
                    ["plate", "Registration plate", "PB 11 AB 2027"],
                    ["vin", "Chassis VIN", ""],
                    ["colour", "Colour", ""],
                    ["registered_in", "Registered in", "Patiala, Punjab"],
                    ["allergies", "Known allergies", "None recorded"],
                    ["kin_name", "Next of kin — name", ""],
                    ["kin_relation", "Next of kin — relation", "Father"],
                    ["kin_phone", "Next of kin — phone", "+91 …"],
                  ].map(([key, label, ph]) => (
                    <label key={key} className="block">
                      <span className="text-[10px] font-bold uppercase tracking-wide text-slate-500">{label}</span>
                      <input
                        value={profileDraft[key] ?? ""}
                        maxLength={80}
                        placeholder={ph}
                        onChange={(e) => setProfileDraft((d) => ({ ...d, [key]: e.target.value }))}
                        className="mt-0.5 w-full px-2.5 py-1.5 rounded-lg border border-slate-200 bg-white text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-sky-300"
                      />
                    </label>
                  ))}
                  <label className="block">
                    <span className="text-[10px] font-bold uppercase tracking-wide text-slate-500">Blood group</span>
                    <select
                      value={profileDraft.blood_group ?? ""}
                      onChange={(e) => setProfileDraft((d) => ({ ...d, blood_group: e.target.value }))}
                      className="mt-0.5 w-full px-2.5 py-1.5 rounded-lg border border-slate-200 bg-white text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-sky-300"
                    >
                      <option value="">Not set</option>
                      {BLOOD_GROUPS.map((g) => (
                        <option key={g} value={g}>{g}</option>
                      ))}
                    </select>
                  </label>
                  {profileError && <p className="text-[11px] font-semibold text-rose-600">{profileError}</p>}
                  <div className="flex gap-2 pt-1">
                    <button
                      onClick={saveProfile}
                      disabled={profileSaving}
                      className="flex-1 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 disabled:opacity-60 text-white text-xs font-bold transition"
                    >
                      {profileSaving ? "Saving…" : "Save"}
                    </button>
                    <button
                      onClick={() => setIsEditingProfile(false)}
                      className="flex-1 py-2 rounded-lg border border-slate-300 text-slate-600 hover:bg-slate-100 text-xs font-bold transition"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}
              <button
                onClick={() => { setIsOwnerDrawerOpen(false); setIsEditingProfile(false); }}
                className="w-full py-2.5 rounded-xl bg-sky-600 hover:bg-sky-700 text-white text-xs font-bold transition shadow-sm"
              >
                Close Drawer
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
