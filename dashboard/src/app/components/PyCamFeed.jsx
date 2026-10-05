"use client";

import React, { useRef, useEffect, useState } from "react";
import { Camera, Video, ShieldAlert, Download, Play, Pause, RefreshCw, Eye, AlertCircle } from "lucide-react";

export default function PyCamFeed({ status, accelX, accelY, speed = 48, crashProb = 0.0 }) {
  const canvasRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(true);
  const [isClipSaved, setIsClipSaved] = useState(false);
  const [showSavedModal, setShowSavedModal] = useState(false);

  useEffect(() => {
    if (status === "CRASH DETECTED") {
      setIsClipSaved(true);
    }
  }, [status]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let animId;
    let offset = 0;

    const render = () => {
      const w = canvas.width;
      const h = canvas.height;

      // ── Sky / Horizon ──────────────────────────────────────
      const horizonY = h * 0.45;
      const tilt = (accelX / 9.81) * 20; // Horizon tilt from lateral force

      ctx.save();
      ctx.translate(w / 2, horizonY);
      ctx.rotate((tilt * Math.PI) / 180);
      ctx.translate(-w / 2, -horizonY);

      // Sky gradient
      const skyGrad = ctx.createLinearGradient(0, 0, 0, horizonY);
      skyGrad.addColorStop(0, "#030712");
      skyGrad.addColorStop(1, "#0f172a");
      ctx.fillStyle = skyGrad;
      ctx.fillRect(-w, -h, w * 3, horizonY * 2);

      // Distant mountains / cityscape silhouette
      ctx.fillStyle = "#1e293b";
      ctx.beginPath();
      ctx.moveTo(0, horizonY);
      ctx.lineTo(w * 0.2, horizonY - 18);
      ctx.lineTo(w * 0.45, horizonY - 32);
      ctx.lineTo(w * 0.7, horizonY - 14);
      ctx.lineTo(w, horizonY);
      ctx.closePath();
      ctx.fill();

      // Road surface gradient
      const roadGrad = ctx.createLinearGradient(0, horizonY, 0, h);
      roadGrad.addColorStop(0, "#1e293b");
      roadGrad.addColorStop(1, "#090d16");
      ctx.fillStyle = roadGrad;
      ctx.fillRect(-w, horizonY, w * 3, h * 2);

      // ── Road Perspective Lines ─────────────────────────────
      const vanishX = w / 2;
      const vanishY = horizonY;

      // Road boundaries
      ctx.strokeStyle = "#475569";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(vanishX - 40, vanishY);
      ctx.lineTo(w * 0.08, h);
      ctx.moveTo(vanishX + 40, vanishY);
      ctx.lineTo(w * 0.92, h);
      ctx.stroke();

      // Lane markings (dashed with perspective animation)
      if (isPlaying) {
        offset = (offset + (speed > 0 ? speed * 0.12 : 1.5)) % 60;
      }

      ctx.strokeStyle = status === "CRASH DETECTED" ? "#ef4444" : "#38bdf8";
      ctx.lineWidth = 2.5;

      for (let y = vanishY + 10; y < h; y += 45) {
        const pY = y + (offset * (y - vanishY)) / 100;
        if (pY > h) continue;
        const progress = (pY - vanishY) / (h - vanishY);
        const segmentLen = 12 * progress * 2;
        const segWidth = 2 * progress;

        ctx.lineWidth = Math.max(1.5, segWidth);
        ctx.beginPath();
        ctx.moveTo(vanishX, pY);
        ctx.lineTo(vanishX, pY + segmentLen);
        ctx.stroke();
      }

      // ── AI Collision Warning Bounding Box (Preception HUD) ─
      if (status === "CRASH DETECTED" || status === "RASH DRIVING") {
        const boxW = status === "CRASH DETECTED" ? 140 : 70;
        const boxH = status === "CRASH DETECTED" ? 90 : 50;
        const boxX = vanishX - boxW / 2;
        const boxY = vanishY + (status === "CRASH DETECTED" ? 40 : 25);

        ctx.strokeStyle = status === "CRASH DETECTED" ? "#ef4444" : "#eab308";
        ctx.lineWidth = 2;
        ctx.strokeRect(boxX, boxY, boxW, boxH);

        // Corner ticks
        const tLen = 8;
        ctx.beginPath();
        ctx.moveTo(boxX, boxY + tLen); ctx.lineTo(boxX, boxY); ctx.lineTo(boxX + tLen, boxY);
        ctx.moveTo(boxX + boxW - tLen, boxY); ctx.lineTo(boxX + boxW, boxY); ctx.lineTo(boxX + boxW, boxY + tLen);
        ctx.moveTo(boxX, boxY + boxH - tLen); ctx.lineTo(boxX, boxY + boxH); ctx.lineTo(boxX + tLen, boxY + boxH);
        ctx.moveTo(boxX + boxW - tLen, boxY + boxH); ctx.lineTo(boxX + boxW, boxY + boxH); ctx.lineTo(boxX + boxW, boxY + boxH - tLen);
        ctx.stroke();

        ctx.fillStyle = status === "CRASH DETECTED" ? "#ef4444" : "#eab308";
        ctx.font = "bold 10px monospace";
        ctx.fillText(
          status === "CRASH DETECTED" ? "⚠️ CRITICAL IMPACT DETECTED" : "⚠️ DISTANCE WARNING (TTC: 1.2s)",
          boxX,
          boxY - 6
        );
      }

      ctx.restore();

      // ── Lens Center Crosshair ──────────────────────────────
      ctx.strokeStyle = "rgba(56, 189, 248, 0.4)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(w / 2 - 12, h / 2); ctx.lineTo(w / 2 + 12, h / 2);
      ctx.moveTo(w / 2, h / 2 - 12); ctx.lineTo(w / 2, h / 2 + 12);
      ctx.stroke();

      // ── HUD Overlay Elements ───────────────────────────────
      // Top Status Bar
      ctx.fillStyle = "rgba(0, 0, 0, 0.65)";
      ctx.fillRect(0, 0, w, 28);

      // REC indicator dot
      ctx.fillStyle = "#ef4444";
      ctx.beginPath();
      ctx.arc(14, 14, 4.5, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 10px monospace";
      ctx.fillText("PYCAM 3 WIDE • 1080P @ 30 FPS", 26, 17);

      ctx.fillStyle = "#38bdf8";
      ctx.textAlign = "right";
      ctx.fillText(
        status === "CRASH DETECTED" ? "● BUFFER LOCKED (10s SAVED)" : "● CIRCULAR RING BUFFER ACTIVE",
        w - 12,
        17
      );
      ctx.textAlign = "left";

      // Bottom Telemetry Bar
      ctx.fillStyle = "rgba(0, 0, 0, 0.65)";
      ctx.fillRect(0, h - 26, w, 26);

      ctx.fillStyle = "#94a3b8";
      ctx.font = "10px monospace";
      const timeStr = new Date().toLocaleTimeString();
      ctx.fillText(`UTC: ${timeStr} | GPS: 30.3533° N, 76.3607° E`, 12, h - 9);

      ctx.textAlign = "right";
      ctx.fillStyle = status === "CRASH DETECTED" ? "#ef4444" : "#38bdf8";
      ctx.font = "bold 10px monospace";
      ctx.fillText(`SPD: ${speed} KM/H | LAT: ${accelX.toFixed(2)}G | LON: ${accelY.toFixed(2)}G`, w - 12, h - 9);
      ctx.textAlign = "left";

      // Impact red flash border
      if (status === "CRASH DETECTED") {
        ctx.strokeStyle = "rgba(239, 68, 68, 0.8)";
        ctx.lineWidth = 6;
        ctx.strokeRect(0, 0, w, h);
      }

      animId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animId);
  }, [isPlaying, status, accelX, accelY, speed]);

  const handleDownloadClip = () => {
    alert("Downloading blackbox video clip: crash_recording_edr_2026.mp4 (1080p, 10s buffer)");
  };

  return (
    <div className="relative flex flex-col w-full h-full bg-slate-950 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl backdrop-blur-xl">
      {/* Header Bar */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-slate-900/90 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Camera className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-bold text-slate-200 uppercase tracking-wider">
            PyCam Dashcam Live Feed (RPi4 Hardware Node)
          </span>
        </div>
        <div className="flex items-center gap-2">
          {isClipSaved && (
            <span className="px-2 py-0.5 rounded text-[10px] font-bold font-mono bg-red-950 text-red-300 border border-red-800 animate-pulse">
              BLACKBOX CLIP LOCKED
            </span>
          )}
          <span className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-950/60 text-emerald-400 border border-emerald-800/60">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
            LIVE SENSOR SYNC
          </span>
        </div>
      </div>

      {/* Video Canvas */}
      <div className="relative w-full flex-1 min-h-[260px] bg-black">
        <canvas ref={canvasRef} width={640} height={320} className="w-full h-full object-cover" />

        {/* Floating Controls Overlay */}
        <div className="absolute bottom-9 right-3 flex items-center gap-1.5 bg-black/70 backdrop-blur-md px-2 py-1 rounded-lg border border-slate-700/60">
          <button 
            onClick={() => setIsPlaying(!isPlaying)}
            className="p-1 text-slate-300 hover:text-white transition"
            title={isPlaying ? "Pause Feed" : "Resume Live"}
          >
            {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
          </button>

          {isClipSaved && (
            <button
              onClick={handleDownloadClip}
              className="flex items-center gap-1 px-2 py-0.5 bg-red-600 hover:bg-red-500 text-white rounded text-[10px] font-bold font-mono transition"
              title="Download 10s Crash Clip with burned-in telemetry"
            >
              <Download className="w-3 h-3" />
              Clip
            </button>
          )}
        </div>
      </div>

      {/* Footer Info Strip */}
      <div className="flex items-center justify-between px-4 py-2 bg-slate-900/60 border-t border-slate-800/80 text-[11px] font-mono text-slate-400">
        <div className="flex items-center gap-3">
          <span>Camera: <strong>IMX708 12MP</strong></span>
          <span>Shutter: <strong>1/120s</strong></span>
          <span>Codec: <strong>H.264 EDR</strong></span>
        </div>
        <div className="flex items-center gap-1 text-cyan-400">
          <Eye className="w-3.5 h-3.5" />
          <span>Computer Vision Inference: Active</span>
        </div>
      </div>
    </div>
  );
}
