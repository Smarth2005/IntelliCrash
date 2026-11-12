"use client";

import React, { useState } from "react";
import { Mail, Key, User, Eye, EyeOff, AlertTriangle, Check, ArrowLeft } from "lucide-react";

// Role presets shared with Dashboard.jsx (sidebar badge + colours).
export const ROLE_PRESETS = {
  owner: {
    id: "owner",
    label: "Fleet Owner",
    role: "Vehicle Owner & Telematics Admin",
    badge: "FLEET OWNER",
    color: "bg-sky-100 text-sky-800 border-sky-300",
    blurb: "Monitor your vehicle, trips and crash alerts"
  },
  responder: {
    id: "responder",
    label: "Emergency Responder",
    role: "Triage Paramedic & Dispatcher",
    badge: "EMERGENCY RESPONDER",
    color: "bg-rose-50 text-rose-800 border-rose-300",
    blurb: "Receive crash dispatches and GPS location"
  },
  engineer: {
    id: "engineer",
    label: "System Architect (Admin)",
    role: "Hardware & Edge Systems Engineer",
    badge: "SYSTEM ARCHITECT",
    color: "bg-emerald-50 text-emerald-800 border-emerald-300",
    blurb: "Manage the edge node and system settings"
  }
};

const inputCls =
  "w-full pl-10 pr-3 py-2.5 rounded-xl border border-slate-200 bg-white/90 text-[13px] font-semibold text-slate-800 placeholder:text-slate-400 placeholder:font-medium focus:outline-none focus:ring-2 focus:ring-sky-400 focus:border-sky-400 transition";

const plainInputCls =
  "w-full px-3 py-2.5 rounded-xl border border-slate-200 bg-white/90 text-[13px] font-semibold text-slate-800 placeholder:text-slate-400 placeholder:font-medium focus:outline-none focus:ring-2 focus:ring-sky-400 focus:border-sky-400 transition";

const BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"];
const EMPTY_PROFILE = {
  vehicle_model: "", vehicle_trim: "", plate: "", vin: "", colour: "", registered_in: "",
  phone: "", blood_group: "", allergies: "", kin_name: "", kin_relation: "", kin_phone: ""
};

const labelCls = "text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]";

// Same section divider used across the dashboard
function SectionLabel({ children }) {
  return (
    <div className="flex items-center gap-3 pt-1">
      <span className="text-[11px] font-black uppercase tracking-[0.25em] text-slate-500 whitespace-nowrap">{children}</span>
      <span className="h-px flex-1 bg-gradient-to-r from-slate-300 to-transparent" />
    </div>
  );
}

export default function AuthScreen({ onAuthed, logo = null }) {
  const [mode, setMode] = useState("login"); // "login" | "register"
  const [form, setForm] = useState({ name: "", email: "", password: "", role: "owner" });
  const [profile, setProfile] = useState(EMPTY_PROFILE);
  const [step, setStep] = useState(1); // register as Fleet Owner: 1 = account, 2 = vehicle & emergency details
  const [showPw, setShowPw] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const isRegister = mode === "register";
  const isOwnerSignup = isRegister && form.role === "owner";
  const onVehicleStep = isOwnerSignup && step === 2;
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));
  const setP = (k) => (e) => setProfile((p) => ({ ...p, [k]: e.target.value }));

  const switchMode = (m) => {
    setMode(m);
    setStep(1);
    setError("");
  };

  const submit = async (e) => {
    e.preventDefault();
    setError("");

    if (isRegister && form.name.trim().length < 2) return setError("Please enter your full name.");
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email.trim())) return setError("Enter a valid email address.");
    if (isRegister && form.password.length < 8) return setError("Password must be at least 8 characters.");
    if (!form.password) return setError("Enter your password.");

    // Fleet Owners: account details first, then vehicle & emergency details.
    if (isOwnerSignup && step === 1) {
      setStep(2);
      return;
    }
    if (onVehicleStep) {
      if (!profile.vehicle_model.trim()) return setError("Enter your vehicle's make and model.");
      if (!profile.plate.trim()) return setError("Enter your registration plate.");
    }

    setBusy(true);
    try {
      const res = await fetch(`/api/auth/${mode}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(isOwnerSignup ? { ...form, profile } : form)
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || "Something went wrong. Please try again.");
      onAuthed(data.user);
    } catch (err) {
      setError(err.message || "Could not reach the server.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex min-h-screen w-screen items-center justify-center overflow-y-auto bg-transparent p-4 text-slate-800">
      <div className="w-full max-w-md animate-fade-in rounded-3xl border border-slate-200/80 bg-gradient-to-b from-white via-white to-sky-50/60 p-7 shadow-[0_8px_40px_rgba(2,132,199,0.10)] backdrop-blur-2xl space-y-5 my-6">
        {/* Brand lockup — identical to the sidebar */}
        <div className="text-center">
          {logo && <div className="w-48 mx-auto">{logo}</div>}
          <div className="mt-1 flex items-start justify-center gap-0.5">
            <span className="font-black text-xl tracking-tight bg-gradient-to-r from-sky-600 via-blue-600 to-indigo-600 bg-clip-text text-transparent">
              INTELLICRASH
            </span>
            <span className="text-[9px] font-extrabold text-sky-500 mt-1">™</span>
          </div>
          <p className="text-[10px] text-slate-400 font-bold tracking-[0.18em] uppercase mt-0.5">Vehicular Telematics AI</p>
        </div>

        {/* Heading — same style as PageHeading */}
        <div className="text-center border-t border-slate-100 pt-4">
          <p className="text-[11px] uppercase tracking-[0.3em] font-semibold text-sky-600">Secure Access</p>
          <h2 className="mt-1 text-2xl font-black text-slate-900 tracking-tight">
            {onVehicleStep ? "Your vehicle & emergency details" : isRegister ? "Create your account" : "Sign in to the console"}
          </h2>
          <p className="mt-1.5 text-base font-medium italic text-slate-800 tracking-wide leading-snug font-serif">
            {onVehicleStep
              ? "Shared with responders the moment a crash is confirmed."
              : isRegister
              ? "Register to monitor, respond and manage."
              : "Live crash detection, one login away."}
          </p>
        </div>

        {/* Mode toggle — same pill style as the alert filters */}
        {!onVehicleStep && (
        <div className="grid grid-cols-2 gap-1 p-1 rounded-xl bg-slate-100 text-xs font-bold">
          {[
            ["login", "Sign in"],
            ["register", "Register"]
          ].map(([id, label]) => (
            <button
              key={id}
              type="button"
              onClick={() => switchMode(id)}
              className={`py-2 rounded-lg transition ${
                mode === id ? "bg-sky-600 text-white shadow-sm" : "text-slate-600 hover:text-slate-900"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        )}

        <form onSubmit={submit} className="space-y-3.5" noValidate>
          {!onVehicleStep && isRegister && (
            <label className="block space-y-1">
              <span className={labelCls}>Full name</span>
              <div className="relative">
                <User className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                <input type="text" autoComplete="name" value={form.name} onChange={set("name")} placeholder="Your name" className={inputCls} />
              </div>
            </label>
          )}

          {!onVehicleStep && (<>
          <label className="block space-y-1">
            <span className={labelCls}>Email</span>
            <div className="relative">
              <Mail className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input type="email" autoComplete="email" value={form.email} onChange={set("email")} placeholder="you@example.com" className={inputCls} />
            </div>
          </label>

          <label className="block space-y-1">
            <span className={labelCls}>Password</span>
            <div className="relative">
              <Key className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type={showPw ? "text" : "password"}
                autoComplete={isRegister ? "new-password" : "current-password"}
                value={form.password}
                onChange={set("password")}
                placeholder={isRegister ? "At least 8 characters" : "Your password"}
                className={`${inputCls} pr-10`}
              />
              <button
                type="button"
                onClick={() => setShowPw((v) => !v)}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 p-1 text-slate-400 hover:text-slate-700 transition"
                title={showPw ? "Hide password" : "Show password"}
              >
                {showPw ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </label>

          </>)}

          {!onVehicleStep && isRegister && (
            <div className="space-y-2.5">
              <SectionLabel>Your role</SectionLabel>
              <div className="space-y-2">
                {Object.values(ROLE_PRESETS).map((r) => {
                  const selected = form.role === r.id;
                  return (
                    <button
                      key={r.id}
                      type="button"
                      onClick={() => { setForm((f) => ({ ...f, role: r.id })); setStep(1); }}
                      className={`w-full p-3 rounded-2xl border transition flex items-center justify-between gap-3 text-left ${
                        selected
                          ? "border-sky-500 bg-sky-50/80 shadow-xs"
                          : "border-slate-200 bg-white/80 hover:border-slate-300 hover:bg-slate-50"
                      }`}
                    >
                      <div className="min-w-0">
                        <h5 className="font-bold text-xs text-slate-900">{r.label}</h5>
                        <p className="text-[11px] text-slate-500 leading-snug">{r.blurb}</p>
                      </div>
                      {selected ? (
                        <span className="w-5 h-5 rounded-full bg-gradient-to-tr from-sky-500 to-indigo-500 text-white flex items-center justify-center shrink-0">
                          <Check className="w-3 h-3" />
                        </span>
                      ) : (
                        <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border shrink-0 ${r.color}`}>{r.badge}</span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {onVehicleStep && (
            <div className="space-y-3.5">
              <SectionLabel>Vehicle</SectionLabel>
              <div className="grid grid-cols-2 gap-3">
                <label className="block space-y-1">
                  <span className={labelCls}>Make &amp; model *</span>
                  <input value={profile.vehicle_model} onChange={setP("vehicle_model")} maxLength={80} placeholder="Hyundai Creta" className={plainInputCls} />
                </label>
                <label className="block space-y-1">
                  <span className={labelCls}>Variant</span>
                  <input value={profile.vehicle_trim} onChange={setP("vehicle_trim")} maxLength={80} placeholder="SX(O)" className={plainInputCls} />
                </label>
                <label className="block space-y-1">
                  <span className={labelCls}>Plate *</span>
                  <input value={profile.plate} onChange={setP("plate")} maxLength={80} placeholder="PB 11 AB 2027" className={`${plainInputCls} uppercase`} />
                </label>
                <label className="block space-y-1">
                  <span className={labelCls}>Colour</span>
                  <input value={profile.colour} onChange={setP("colour")} maxLength={80} placeholder="Polar White" className={plainInputCls} />
                </label>
                <label className="block space-y-1">
                  <span className={labelCls}>Chassis VIN</span>
                  <input value={profile.vin} onChange={setP("vin")} maxLength={80} placeholder="Optional" className={`${plainInputCls} uppercase`} />
                </label>
                <label className="block space-y-1">
                  <span className={labelCls}>Registered in</span>
                  <input value={profile.registered_in} onChange={setP("registered_in")} maxLength={80} placeholder="Patiala, Punjab" className={plainInputCls} />
                </label>
              </div>

              <SectionLabel>Emergency medical card</SectionLabel>
              <div className="grid grid-cols-2 gap-3">
                <label className="block space-y-1">
                  <span className={labelCls}>Blood group</span>
                  <select value={profile.blood_group} onChange={setP("blood_group")} className={plainInputCls}>
                    <option value="">Not set</option>
                    {BLOOD_GROUPS.map((g) => (
                      <option key={g} value={g}>{g}</option>
                    ))}
                  </select>
                </label>
                <label className="block space-y-1">
                  <span className={labelCls}>Known allergies</span>
                  <input value={profile.allergies} onChange={setP("allergies")} maxLength={80} placeholder="None" className={plainInputCls} />
                </label>
                <label className="block space-y-1">
                  <span className={labelCls}>Your phone</span>
                  <input type="tel" value={profile.phone} onChange={setP("phone")} maxLength={80} placeholder="+91 …" className={plainInputCls} />
                </label>
                <label className="block space-y-1">
                  <span className={labelCls}>Next of kin</span>
                  <input value={profile.kin_name} onChange={setP("kin_name")} maxLength={80} placeholder="Name" className={plainInputCls} />
                </label>
                <label className="block space-y-1">
                  <span className={labelCls}>Relation</span>
                  <input value={profile.kin_relation} onChange={setP("kin_relation")} maxLength={80} placeholder="Father" className={plainInputCls} />
                </label>
                <label className="block space-y-1">
                  <span className={labelCls}>Kin phone</span>
                  <input type="tel" value={profile.kin_phone} onChange={setP("kin_phone")} maxLength={80} placeholder="+91 …" className={plainInputCls} />
                </label>
              </div>
              <p className="text-[11px] text-slate-500 font-medium">* required. Everything else is optional and can be edited later from your profile.</p>
            </div>
          )}

          {error && (
            <div role="alert" className="flex items-start gap-2 p-2.5 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs font-semibold">
              <AlertTriangle className="w-4 h-4 shrink-0 mt-px" />
              <span>{error}</span>
            </div>
          )}

          {onVehicleStep && (
            <button
              type="button"
              onClick={() => { setStep(1); setError(""); }}
              className="flex items-center gap-1.5 text-xs font-bold text-slate-500 hover:text-slate-800 transition"
            >
              <ArrowLeft className="w-3.5 h-3.5" /> Back to account details
            </button>
          )}

          <button
            type="submit"
            disabled={busy}
            className="w-full py-2.5 rounded-xl bg-gradient-to-r from-sky-500 to-indigo-500 hover:from-sky-600 hover:to-indigo-600 disabled:opacity-60 disabled:cursor-not-allowed text-white text-[13px] font-bold shadow-lg shadow-sky-500/25 transition"
          >
            {busy ? "Please wait…" : isOwnerSignup && step === 1 ? "Continue" : isRegister ? "Create account" : "Sign in"}
          </button>
        </form>

        {!onVehicleStep && (
        <p className="text-center text-xs text-slate-500 font-medium">
          {isRegister ? "Already registered?" : "New here?"}{" "}
          <button type="button" onClick={() => switchMode(isRegister ? "login" : "register")} className="font-bold text-sky-700 hover:underline">
            {isRegister ? "Sign in" : "Create an account"}
          </button>
        </p>
        )}
      </div>
    </div>
  );
}
