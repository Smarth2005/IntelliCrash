"use client";

import React, { useState } from "react";
import { 
  Sliders, Play, Zap, ShieldAlert, Cpu, Gauge, RefreshCw, Send, CheckCircle2, 
  HelpCircle, ChevronRight, Activity, ArrowRight, Layers
} from "lucide-react";

export default function SimulationLab({ onInjectToBackend, backendUrl }) {
  // Preset Scenarios
  const scenarios = [
    {
      id: "normal",
      title: "Normal Highway Cruise",
      type: "SAFE",
      color: "emerald",
      speed: 65,
      peakG: 0.25,
      accelX: 0.1,
      accelY: 0.2,
      gyroZ: 1.5,
      durationMs: 80,
      mlProb: 1.2,
      csi: 0.02,
      description: "Smooth driving with subtle lane adjustments. Nominal g-forces well within safety envelope."
    },
    {
      id: "rash",
      title: "Rash Driving / Aggressive Swerve",
      type: "WARNING",
      color: "yellow",
      speed: 85,
      peakG: 4.8,
      accelX: 4.2,
      accelY: -2.8,
      gyroZ: 85.0,
      durationMs: 350,
      mlProb: 32.5,
      csi: 0.24,
      description: "Violent double-lane change swerving. High lateral acceleration with persistent yaw oscillations."
    },
    {
      id: "emergency_brake",
      title: "Hard Emergency Braking",
      type: "MANEUVER",
      color: "amber",
      speed: 70,
      peakG: 6.8,
      accelX: 0.4,
      accelY: -6.5,
      gyroZ: 4.0,
      durationMs: 500,
      mlProb: 18.0,
      csi: 0.31,
      description: "Maximum deceleration threshold. Discriminated as non-crash driving maneuver by Bi-LSTM."
    },
    {
      id: "frontal_crash",
      title: "Frontal Barrier Collision (35G)",
      type: "CRITICAL CRASH",
      color: "red",
      speed: 60,
      peakG: 34.5,
      accelX: 3.2,
      accelY: -48.5,
      gyroZ: 165.0,
      durationMs: 65,
      mlProb: 96.8,
      csi: 0.92,
      description: "High-speed head-on impact. Rapid impulse shock (<70ms), immediate airbag deployment threshold."
    },
    {
      id: "side_tbone",
      title: "Lateral T-Bone / Rollover",
      type: "CRITICAL CRASH",
      color: "rose",
      speed: 55,
      peakG: 28.0,
      accelX: 26.5,
      accelY: -12.0,
      gyroZ: 210.0,
      durationMs: 90,
      mlProb: 94.2,
      csi: 0.88,
      description: "Right-angle side impact triggering rollover dynamics. Severe angular momentum in gyro Z."
    }
  ];

  const [selectedScenario, setSelectedScenario] = useState(scenarios[3]); // Default to frontal crash
  
  // Custom Sliders
  const [speed, setSpeed] = useState(selectedScenario.speed);
  const [peakG, setPeakG] = useState(selectedScenario.peakG);
  const [accelY, setAccelY] = useState(selectedScenario.accelY);
  const [gyroZ, setGyroZ] = useState(selectedScenario.gyroZ);
  const [durationMs, setDurationMs] = useState(selectedScenario.durationMs);

  const [injectionFeedback, setInjectionFeedback] = useState(null);

  const handleSelectScenario = (sc) => {
    setSelectedScenario(sc);
    setSpeed(sc.speed);
    setPeakG(sc.peakG);
    setAccelY(sc.accelY);
    setGyroZ(sc.gyroZ);
    setDurationMs(sc.durationMs);
  };

  // Compute calculated CSI & ML probability from sliders
  const calculatedCsi = Math.min(1.0, (Math.abs(peakG) / 45.0) * 0.45 + (Math.abs(accelY) / 50.0) * 0.35 + (Math.abs(gyroZ) / 200.0) * 0.20).toFixed(2);
  const calculatedMlProb = Math.min(100.0, calculatedCsi > 0.4 ? (80 + calculatedCsi * 18).toFixed(1) : (calculatedCsi * 40).toFixed(1));
  const fusedScore = (0.5 * (calculatedMlProb / 100.0) + 0.5 * Number(calculatedCsi)).toFixed(2);
  const isCrashDecision = fusedScore >= 0.50;

  const handleInject = async () => {
    setInjectionFeedback("injecting");
    const payload = {
      vehicle_id: "sim-sandbox-01",
      accel_x: Number((peakG * 0.2).toFixed(2)),
      accel_y: Number(accelY),
      accel_z: 9.81,
      gyro_z: Number(gyroZ),
      crash_prob: Number(calculatedMlProb),
      severity_score: Number(calculatedCsi),
      status: isCrashDecision ? "CRASH DETECTED" : (Number(calculatedCsi) > 0.2 ? "RASH DRIVING" : "Normal"),
      behavior_score: isCrashDecision ? 20.0 : (Number(calculatedCsi) > 0.2 ? 60.0 : 98.0)
    };

    if (onInjectToBackend) {
      await onInjectToBackend(payload, isCrashDecision);
      setInjectionFeedback("success");
      setTimeout(() => setInjectionFeedback(null), 4000);
    }
  };

  return (
    <div className="flex flex-col gap-6 animate-in fade-in duration-300">
      
      {/* Simulation Lab Banner */}
      <div className="p-5 rounded-2xl bg-gradient-to-r from-indigo-950/60 via-slate-900/80 to-purple-950/50 border border-indigo-800/40 backdrop-blur-xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Sliders className="w-5 h-5 text-indigo-400" />
            <h2 className="text-lg font-bold text-slate-100">Vehicle Dynamics Simulation & Verification Sandbox</h2>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Isolated engineering testbed to simulate vehicle kinematic crashes, tune CSI parameters, and verify the Dual-Stage Fusion Gate.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="px-3 py-1 bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 rounded-lg text-xs font-mono font-bold">
            SANDBOX ENVIRONMENT (ISOLATED)
          </span>
        </div>
      </div>

      {/* Grid: Scenarios Selector & Physics Tuning */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Column: Preset Scenarios */}
        <div className="flex flex-col gap-3 lg:col-span-1">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-1">
            Standard Validation Scenarios (Crash & Maneuver)
          </h3>

          {scenarios.map((sc) => {
            const isSelected = selectedScenario.id === sc.id;
            return (
              <button
                key={sc.id}
                onClick={() => handleSelectScenario(sc)}
                className={`p-4 rounded-xl border text-left transition-all ${
                  isSelected 
                    ? "bg-slate-800/90 border-cyan-500/60 shadow-lg shadow-cyan-950/40 ring-1 ring-cyan-500/40" 
                    : "bg-slate-900/60 border-slate-800 hover:bg-slate-800/50"
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="font-bold text-sm text-slate-200">{sc.title}</span>
                  <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded ${
                    sc.type.includes("CRITICAL") 
                      ? "bg-red-950 text-red-300 border border-red-800" 
                      : sc.type === "WARNING"
                      ? "bg-yellow-950 text-yellow-300 border border-yellow-800"
                      : "bg-emerald-950 text-emerald-300 border border-emerald-800"
                  }`}>
                    {sc.type}
                  </span>
                </div>
                <p className="text-xs text-slate-400 mb-2 leading-relaxed">{sc.description}</p>
                <div className="flex items-center gap-4 text-[11px] font-mono text-slate-400 border-t border-slate-800/80 pt-2">
                  <span>Peak: <strong className="text-slate-200">{sc.peakG}G</strong></span>
                  <span>Speed: <strong className="text-slate-200">{sc.speed} km/h</strong></span>
                  <span>CSI: <strong className="text-cyan-400">{sc.csi}</strong></span>
                </div>
              </button>
            );
          })}
        </div>

        {/* Center & Right Column: Interactive Physics Tuning & Dual-Stage Fusion Gate Analyzer */}
        <div className="flex flex-col gap-6 lg:col-span-2">
          
          {/* Parameter Sliders Card */}
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-xl">
            <h3 className="text-sm font-bold text-slate-200 mb-4 flex items-center gap-2">
              <Zap className="w-4 h-4 text-cyan-400" />
              Kinematic Parameter Tuning Controls
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {/* Speed Slider */}
              <div>
                <div className="flex justify-between text-xs font-mono mb-1">
                  <span className="text-slate-400">Vehicle Speed:</span>
                  <span className="text-cyan-300 font-bold">{speed} km/h</span>
                </div>
                <input 
                  type="range" min="0" max="150" value={speed} 
                  onChange={(e) => setSpeed(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                />
              </div>

              {/* Peak G Force */}
              <div>
                <div className="flex justify-between text-xs font-mono mb-1">
                  <span className="text-slate-400">Peak G-Force Deceleration:</span>
                  <span className="text-rose-400 font-bold">{peakG} G</span>
                </div>
                <input 
                  type="range" min="0" max="60" step="0.5" value={peakG} 
                  onChange={(e) => setPeakG(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-rose-500"
                />
              </div>

              {/* Longitudinal Accel Y */}
              <div>
                <div className="flex justify-between text-xs font-mono mb-1">
                  <span className="text-slate-400">Longitudinal Decel (Ay):</span>
                  <span className="text-amber-400 font-bold">{accelY} m/s²</span>
                </div>
                <input 
                  type="range" min="-60" max="10" step="0.5" value={accelY} 
                  onChange={(e) => setAccelY(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-400"
                />
              </div>

              {/* Yaw Angular Velocity Gyro Z */}
              <div>
                <div className="flex justify-between text-xs font-mono mb-1">
                  <span className="text-slate-400">Yaw Rate (Gyro Z):</span>
                  <span className="text-yellow-400 font-bold">{gyroZ} °/s</span>
                </div>
                <input 
                  type="range" min="0" max="300" step="2" value={gyroZ} 
                  onChange={(e) => setGyroZ(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-yellow-400"
                />
              </div>
            </div>
          </div>

          {/* Dual-Stage Fusion Gate Inspector Card */}
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-xl">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                <Layers className="w-4 h-4 text-indigo-400" />
                Dual-Stage Fusion Gate Mathematical Inspection
              </h3>
              <span className="text-xs font-mono text-slate-400">Threshold: &gt; 0.50</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
              
              {/* Stage 1: AI Model */}
              <div className="p-4 rounded-xl bg-black/40 border border-slate-800 text-center">
                <div className="text-[10px] uppercase font-bold text-cyan-400 tracking-wider mb-1">Stage 1: Bi-LSTM Model</div>
                <div className="text-2xl font-bold font-mono text-cyan-300">{calculatedMlProb}%</div>
                <p className="text-[10px] text-slate-500 mt-1">Temporal Pattern Weight: 0.50</p>
              </div>

              {/* Stage 2: Physics CSI */}
              <div className="p-4 rounded-xl bg-black/40 border border-slate-800 text-center">
                <div className="text-[10px] uppercase font-bold text-purple-400 tracking-wider mb-1">Stage 2: Physics CSI</div>
                <div className="text-2xl font-bold font-mono text-purple-300">{calculatedCsi}</div>
                <p className="text-[10px] text-slate-500 mt-1">Kinematic Severity Weight: 0.50</p>
              </div>

              {/* Final Fusion Decision */}
              <div className={`p-4 rounded-xl border text-center transition-all ${
                isCrashDecision 
                  ? "bg-red-950/40 border-red-500/60 text-red-300 shadow-lg shadow-red-950/50" 
                  : "bg-emerald-950/30 border-emerald-500/40 text-emerald-300"
              }`}>
                <div className="text-[10px] uppercase font-bold tracking-wider mb-1">Fused Decision Output</div>
                <div className="text-2xl font-black font-mono">{fusedScore}</div>
                <div className="text-xs font-bold mt-1 uppercase font-mono tracking-wider">
                  {isCrashDecision ? "⚠️ CRASH CONFIRMED" : "✅ NORMAL DRIVING"}
                </div>
              </div>

            </div>

            {/* Formula Notation */}
            <div className="mt-4 p-3 rounded-lg bg-black/50 border border-slate-800/80 font-mono text-xs text-slate-400">
              <span className="text-indigo-400 font-bold">Fusion Logic:</span> Score = (0.50 × P_LSTM) + (0.50 × S_CSI) = (0.50 × {(calculatedMlProb/100).toFixed(2)}) + (0.50 × {calculatedCsi}) = <strong className="text-slate-100">{fusedScore}</strong>
            </div>

            {/* Injection Trigger Button */}
            <div className="mt-5 flex items-center justify-between">
              <div className="text-xs text-slate-400">
                Clicking inject sends this calibrated scenario directly into the live FastAPI backend.
              </div>

              <button
                onClick={handleInject}
                className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white font-bold text-xs uppercase tracking-wider transition-all shadow-lg shadow-cyan-900/20"
              >
                <Send className="w-4 h-4" />
                {injectionFeedback === "injecting" ? "Injecting..." : "Inject Scenario Into Live Pipeline"}
              </button>
            </div>

            {injectionFeedback === "success" && (
              <div className="mt-3 p-2.5 rounded-lg bg-emerald-950 border border-emerald-500/50 text-emerald-300 text-xs font-mono flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4" />
                Successfully injected into FastAPI backend. Switch to 'Live Operations' to watch the live response!
              </div>
            )}
          </div>

        </div>

      </div>

    </div>
  );
}
