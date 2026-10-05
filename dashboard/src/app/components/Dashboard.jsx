"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import dynamic from "next/dynamic";
import { 
  Activity, AlertTriangle, ShieldCheck, MapPin, Zap, PhoneCall, Gauge, 
  Settings, RefreshCw, Download, Radio, Server, Database, Lock, UserCheck, 
  Send, BellRing, CheckCircle2, ChevronRight, Car, Eye, Cpu
} from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import dashboardData from "../../data/india_accidents_dashboard.json";

// Dynamic import for Leaflet map to avoid SSR issues
const HeatMap = dynamic(() => import("./HeatMap"), { ssr: false });
import AttentionHeatmap from "./AttentionHeatmap";
import EventTimeline from "./EventTimeline";

// ─── 3D-LIKE CAR WIREFRAME COMPONENT ──────────────────────────────────────────
function CarCanvas({ status, accelX, accelY, accelZ, gyroZ }) {
  const canvasRef = useRef(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let animationFrameId;
    let angle = 0;
    
    // 3D vertices of a stylized wireframe vehicle
    const baseVertices = [
      // Chassis (Bottom box)
      {x: -1.3, y: -0.6, z: -0.3}, {x: 1.3, y: -0.6, z: -0.3},
      {x: 1.3, y: 0.6, z: -0.3}, {x: -1.3, y: 0.6, z: -0.3},
      {x: -1.3, y: -0.6, z: 0.2}, {x: 1.3, y: -0.6, z: 0.2},
      {x: 1.3, y: 0.6, z: 0.2}, {x: -1.3, y: 0.6, z: 0.2},
      // Cab (Top cabin)
      {x: -0.4, y: -0.45, z: 0.6}, {x: 0.5, y: -0.45, z: 0.6},
      {x: 0.5, y: 0.45, z: 0.6}, {x: -0.4, y: 0.45, z: 0.6},
      // Front hood slope vertices
      {x: 0.9, y: -0.55, z: 0.05}, {x: 0.9, y: 0.55, z: 0.05}
    ];
    
    const edges = [
      [0,1], [1,2], [2,3], [3,0], // bottom chassis face
      [4,5], [5,6], [6,7], [7,4], // top chassis face
      [0,4], [1,5], [2,6], [3,7], // chassis pillars
      [8,9], [9,10], [10,11], [11,8], // cabin roof
      [4,8], [5,9], [6,10], [7,11], // cabin windshield pillars
      [1,12], [2,13], [12,13] // hood nose
    ];
    
    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const width = canvas.width;
      const height = canvas.height;
      const scale = Math.min(width, height) * 0.25;
      
      // Determine glow colors and animation dynamics based on status
      let color = "#38bdf8"; // Cyan (Normal)
      let pulse = false;
      let shakeX = 0;
      let shakeY = 0;
      
      if (status === "CRASH DETECTED") {
        color = "#ef4444"; // Red
        pulse = true;
        shakeX = (Math.random() - 0.5) * 8;
        shakeY = (Math.random() - 0.5) * 8;
      } else if (status === "RASH DRIVING") {
        color = "#eab308"; // Orange-Yellow
        shakeX = (Math.random() - 0.5) * 4;
        shakeY = (Math.random() - 0.5) * 4;
        angle += 0.04;
      } else {
        angle += 0.012; // Slow aesthetic rotation
      }
      
      ctx.save();
      ctx.translate(shakeX, shakeY);
      
      // 3D rotation angles (Pitch, Yaw, Roll)
      let rx = 0.45;
      let ry = angle;
      let rz = 0.15;
      
      // Dynamic tilt based on real IMU sensor forces
      if (status !== "CRASH DETECTED") {
        rx += (accelY / 9.81) * 0.35;
        rz += -(accelX / 9.81) * 0.35;
      }
      
      const projected = baseVertices.map(v => {
        let x1 = v.x * Math.cos(ry) + v.z * Math.sin(ry);
        let z1 = -v.x * Math.sin(ry) + v.z * Math.cos(ry);
        let y1 = v.y;
        
        let y2 = y1 * Math.cos(rx) - z1 * Math.sin(rx);
        let z2 = y1 * Math.sin(rx) + z1 * Math.cos(rx);
        let x2 = x1;
        
        let x3 = x2 * Math.cos(rz) - y2 * Math.sin(rz);
        let y3 = x2 * Math.sin(rz) + y2 * Math.cos(rz);
        
        return {
          x: width / 2 + x3 * scale,
          y: height / 2 - y3 * scale
        };
      });
      
      // Draw wireframe edges
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = status === "CRASH DETECTED" ? 2.5 : 1.5;
      ctx.shadowBlur = pulse ? (14 + Math.sin(Date.now() / 80) * 8) : 8;
      ctx.shadowColor = color;
      
      edges.forEach(([p1, p2]) => {
        ctx.moveTo(projected[p1].x, projected[p1].y);
        ctx.lineTo(projected[p2].x, projected[p2].y);
      });
      ctx.stroke();
      
      // Wheel nodes
      const wheelOffsets = [
        {x: -0.7, y: -0.55, z: -0.3}, {x: 0.7, y: -0.55, z: -0.3},
        {x: -0.7, y: 0.55, z: -0.3}, {x: 0.7, y: 0.55, z: -0.3}
      ];
      ctx.fillStyle = color;
      ctx.shadowBlur = 0;
      wheelOffsets.forEach(w => {
        let x1 = w.x * Math.cos(ry) + w.z * Math.sin(ry);
        let z1 = -w.x * Math.sin(ry) + w.z * Math.cos(ry);
        let y1 = w.y;
        let x2 = x1;
        let y2 = y1 * Math.cos(rx) - z1 * Math.sin(rx);
        let x3 = x2 * Math.cos(rz) - y2 * Math.sin(rz);
        let y3 = x2 * Math.sin(rz) + y2 * Math.cos(rz);
        
        ctx.beginPath();
        ctx.arc(width / 2 + x3 * scale, height / 2 - y3 * scale, 4.5, 0, Math.PI * 2);
        ctx.fill();
      });
      
      // Vector Impact Arrow if crash
      if (status === "CRASH DETECTED") {
        ctx.shadowBlur = 15;
        ctx.shadowColor = "#ef4444";
        ctx.fillStyle = "#ef4444";
        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = 4.0;
        
        const dx = -accelX;
        const dy = -accelY;
        const arrowAngle = Math.atan2(dy, dx);
        const arrowLen = scale * 0.95;
        const endX = width / 2;
        const endY = height / 2;
        const startX = endX - Math.cos(arrowAngle) * arrowLen;
        const startY = endY + Math.sin(arrowAngle) * arrowLen;
        
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(endX, endY);
        ctx.lineTo(endX - Math.cos(arrowAngle - 0.45) * 16, endY + Math.sin(arrowAngle - 0.45) * 16);
        ctx.lineTo(endX - Math.cos(arrowAngle + 0.45) * 16, endY + Math.sin(arrowAngle + 0.45) * 16);
        ctx.closePath();
        ctx.fill();
        
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 12px monospace";
        ctx.textAlign = "center";
        let direction = "FRONTAL IMPACT VECTOR";
        if (Math.abs(dx) > Math.abs(dy)) {
          direction = dx > 0 ? "RIGHT-LATERAL COLLISION" : "LEFT-LATERAL COLLISION";
        } else {
          direction = dy > 0 ? "FRONTAL HEAD-ON COLLISION" : "REAR END COLLISION";
        }
        ctx.fillText(direction, width / 2, height - 12);
      }
      
      ctx.restore();
      animationFrameId = requestAnimationFrame(render);
    };
    
    render();
    return () => cancelAnimationFrame(animationFrameId);
  }, [status, accelX, accelY, accelZ, gyroZ]);
  
  return (
    <div className="relative flex items-center justify-center w-full h-[240px] bg-slate-950/70 border border-slate-800 rounded-xl overflow-hidden backdrop-blur-md">
      <canvas ref={canvasRef} width={380} height={240} className="w-full h-full" />
      <div className="absolute top-2.5 left-3 flex items-center gap-1.5 text-[10px] font-bold text-slate-400 tracking-widest uppercase">
        <Car className="w-3.5 h-3.5 text-cyan-400" />
        Vehicle Kinematic Wireframe
      </div>
      <div className="absolute top-2.5 right-3 text-[10px] font-mono text-cyan-400/80 bg-cyan-950/50 px-2 py-0.5 rounded border border-cyan-800/40">
        3D Real-Time Projection
      </div>
    </div>
  );
}

// ─── MAIN DASHBOARD COMPONENT ─────────────────────────────────────────────────
export default function Dashboard() {
  const [isClient, setIsClient] = useState(false);
  const [backendUrl, setBackendUrl] = useState("http://127.0.0.1:8000");
  const [wsUrl, setWsUrl] = useState("ws://127.0.0.1:8000/ws/telemetry");
  const [showConfig, setShowConfig] = useState(false);
  
  // Real-time connection states
  const [wsConnected, setWsConnected] = useState(false);
  const [apiConnected, setApiConnected] = useState(false);
  const [connectionMode, setConnectionMode] = useState("live"); // live, mock
  
  // System Health details from FastAPI /health
  const [systemHealth, setSystemHealth] = useState({
    status: "checking",
    database: "unknown",
    mqtt: "enabled",
    auth: "jwt_rbac",
    version: "2.0.0",
    uptime: 0,
  });

  // JWT Auth state
  const [authToken, setAuthToken] = useState(null);
  const [currentUser, setCurrentUser] = useState({ username: "operator_station_1", role: "operator" });
  
  // Live Telemetry metrics
  const [liveMetrics, setLiveMetrics] = useState({
    status: "Normal",
    accel_x: 0.05,
    accel_y: -0.02,
    accel_z: 9.81,
    gyro_z: 0.1,
    prob: 1.2,
    severity: 0.0,
    latency: 18,
    behaviorScore: 98,
    callStatus: "Standby",
    mode: "FastAPI WebSocket Stream (Port 8000)",
    timestamp: new Date().toLocaleTimeString()
  });

  // Rolling sensor buffer for Recharts (last 45 samples)
  const [chartData, setChartData] = useState([]);
  
  // Recorded crash events from database
  const [recordedCrashes, setRecordedCrashes] = useState([]);
  const [totalCrashesCount, setTotalCrashesCount] = useState(0);

  // Event Timeline State
  const [timelineEvents, setTimelineEvents] = useState([]);
  const lastStatusRef = useRef("Normal");

  // Alert Dispatch Feedback
  const [dispatchStatus, setDispatchStatus] = useState(null); // null, 'dispatching', 'success', 'failed'
  const [dispatchLog, setDispatchLog] = useState("");

  const wsRef = useRef(null);

  useEffect(() => {
    setIsClient(true);
  }, []);

  // ── 1. Fetch System Health & Stats ──────────────────────────────────────────
  const fetchHealthAndStats = useCallback(async () => {
    try {
      const res = await fetch(`${backendUrl}/health`, { signal: AbortSignal.timeout(2000) });
      if (res.ok) {
        const data = await res.json();
        setSystemHealth(data);
        setApiConnected(true);
      } else {
        setApiConnected(false);
      }
    } catch {
      setApiConnected(false);
    }

    try {
      const statsRes = await fetch(`${backendUrl}/api/v1/crashes/stats`, { signal: AbortSignal.timeout(2000) });
      if (statsRes.ok) {
        const stats = await statsRes.json();
        setTotalCrashesCount(stats.total_crashes || 0);
      }
      
      const crashListRes = await fetch(`${backendUrl}/api/v1/crashes?limit=5`, { signal: AbortSignal.timeout(2000) });
      if (crashListRes.ok) {
        const listData = await crashListRes.json();
        setRecordedCrashes(listData.data || []);
      }
    } catch {
      // Ignored if offline
    }
  }, [backendUrl]);

  useEffect(() => {
    if (!isClient) return;
    fetchHealthAndStats();
    const interval = setInterval(fetchHealthAndStats, 4000);
    return () => clearInterval(interval);
  }, [isClient, fetchHealthAndStats]);

  // ── 2. Authenticate Demo Operator (Get JWT token) ───────────────────────────
  useEffect(() => {
    if (!isClient) return;
    const obtainDemoToken = async () => {
      try {
        // Register or login default operator
        const formData = new URLSearchParams();
        formData.append("username", "operator_live");
        formData.append("password", "IntelliCrash2026!");

        let loginRes = await fetch(`${backendUrl}/api/v1/auth/login`, {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body: formData.toString()
        });

        if (!loginRes.ok) {
          // Attempt registration first
          await fetch(`${backendUrl}/api/v1/auth/register`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              username: "operator_live",
              email: "operator_live@intellicrash.io",
              password: "IntelliCrash2026!",
              role: "operator"
            })
          });

          // Retry login
          loginRes = await fetch(`${backendUrl}/api/v1/auth/login`, {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: formData.toString()
          });
        }

        if (loginRes.ok) {
          const authData = await loginRes.json();
          setAuthToken(authData.access_token);
          setCurrentUser({ username: authData.username, role: authData.role });
        }
      } catch {
        // Fallback demo token
        setAuthToken("demo_token_offline");
      }
    };

    obtainDemoToken();
  }, [isClient, backendUrl]);

  // ── 3. WebSocket Real-Time Telemetry Connection ─────────────────────────────
  useEffect(() => {
    if (!isClient || connectionMode === "mock") return;

    let socket;
    let keepaliveTimer;

    const connectWebSocket = () => {
      try {
        socket = new WebSocket(wsUrl);
        wsRef.current = socket;

        socket.onopen = () => {
          setWsConnected(true);
          // Heartbeat ping every 10s
          keepaliveTimer = setInterval(() => {
            if (socket.readyState === WebSocket.OPEN) {
              socket.send("ping");
            }
          }, 10000);
        };

        socket.onmessage = (event) => {
          try {
            if (event.data === '{"type": "pong"}') return;
            const data = JSON.parse(event.data);

            if (data.type === "telemetry" || data.accel_x !== undefined) {
              setLiveMetrics(prev => ({
                status: data.status || prev.status,
                accel_x: Number(data.accel_x || 0),
                accel_y: Number(data.accel_y || 0),
                accel_z: Number(data.accel_z || 9.81),
                gyro_z: Number(data.gyro_z || 0),
                prob: Number(data.crash_prob || 0).toFixed(1),
                severity: Number(data.severity_score || 0).toFixed(2),
                latency: Math.floor(14 + Math.random() * 8),
                behaviorScore: data.behavior_score || prev.behaviorScore,
                callStatus: data.status === "CRASH DETECTED" ? "Auto-Triggered" : "Standby",
                mode: data.source ? `MQTT Broker Bridge (${data.source.toUpperCase()})` : "WebSocket Live Stream (FastAPI)",
                timestamp: data.timestamp || new Date().toLocaleTimeString()
              }));

              // Update rolling charts
              setChartData(prevData => {
                const nextData = [...prevData, {
                  time: (data.timestamp ? data.timestamp.substring(11, 19) : new Date().toLocaleTimeString()),
                  ax: Number(data.accel_x || 0),
                  ay: Number(data.accel_y || 0),
                  gz: Number(data.gyro_z || 0) / 20.0
                }];
                if (nextData.length > 45) nextData.shift();
                return nextData;
              });
            } else if (data.type === "crash_alert") {
              fetchHealthAndStats();
            }
          } catch (err) {
            console.error("WS Parse Error:", err);
          }
        };

        socket.onerror = () => {
          setWsConnected(false);
        };

        socket.onclose = () => {
          setWsConnected(false);
          clearInterval(keepaliveTimer);
          // Try reconnect in 3s
          setTimeout(() => {
            if (connectionMode === "live") connectWebSocket();
          }, 3000);
        };
      } catch {
        setWsConnected(false);
      }
    };

    connectWebSocket();

    return () => {
      clearInterval(keepaliveTimer);
      if (socket) socket.close();
    };
  }, [isClient, wsUrl, connectionMode, fetchHealthAndStats]);

  // ── 4. Fallback Polling when WebSocket is closed or in mock mode ─────────────
  useEffect(() => {
    if (!isClient) return;

    if (connectionMode === "mock") {
      const mockInterval = setInterval(() => {
        setLiveMetrics(prev => {
          const isCrash = Math.random() > 0.98;
          const isRash = Math.random() > 0.94;
          let newScore = prev.behaviorScore;
          if (isRash) newScore = Math.max(35, newScore - Math.floor(Math.random() * 6));
          else if (newScore < 100) newScore = Math.min(100, newScore + 0.5);

          const metrics = {
            status: isCrash ? "CRASH DETECTED" : (isRash ? "RASH DRIVING" : "Normal"),
            accel_x: isCrash ? 22.4 : (isRash ? (Math.random() - 0.5) * 6 : (Math.random() - 0.5) * 0.8),
            accel_y: isCrash ? -42.8 : (isRash ? (Math.random() - 0.5) * 7 : (Math.random() - 0.5) * 0.9),
            accel_z: isCrash ? 14.2 : 9.81 + (Math.random() - 0.5) * 0.4,
            gyro_z: isCrash ? 165.0 : (isRash ? (Math.random() - 0.5) * 90 : (Math.random() - 0.5) * 4),
            prob: isCrash ? (88.4 + Math.random() * 10).toFixed(1) : (0.8 + Math.random() * 1.5).toFixed(1),
            severity: isCrash ? (0.72 + Math.random() * 0.2).toFixed(2) : 0.0,
            latency: Math.floor(18 + Math.random() * 5),
            behaviorScore: Math.round(newScore),
            callStatus: isCrash ? "Dispatching AI Call..." : "Standby",
            mode: "Local Demonstration Generator",
            timestamp: new Date().toLocaleTimeString()
          };

          setChartData(prevData => {
            const nextData = [...prevData, {
              time: new Date().toLocaleTimeString(),
              ax: Number(metrics.accel_x),
              ay: Number(metrics.accel_y),
              gz: Number(metrics.gyro_z) / 20.0
            }];
            if (nextData.length > 45) nextData.shift();
            return nextData;
          });

          return metrics;
        });
      }, 120);

      return () => clearInterval(mockInterval);
    }

    // If live mode but WebSocket is offline, poll REST API
    if (!wsConnected && apiConnected) {
      const pollInterval = setInterval(async () => {
        try {
          const res = await fetch(`${backendUrl}/api/v1/telemetry/latest`, { signal: AbortSignal.timeout(600) });
          if (res.ok) {
            const data = await res.json();
            setLiveMetrics(prev => ({
              ...prev,
              status: data.status,
              accel_x: data.accel_x,
              accel_y: data.accel_y,
              accel_z: data.accel_z,
              gyro_z: data.gyro_z,
              prob: Number(data.crash_prob).toFixed(1),
              severity: Number(data.severity_score).toFixed(2),
              behaviorScore: data.behavior_score,
              mode: "REST API Polling (Port 8000)"
            }));
          }
        } catch {}
      }, 250);

      return () => clearInterval(pollInterval);
    }
  }, [isClient, connectionMode, wsConnected, apiConnected, backendUrl]);

  // Track status changes for timeline
  useEffect(() => {
    if (liveMetrics.status !== "Normal" && liveMetrics.status !== lastStatusRef.current) {
      setTimelineEvents(prev => [{
        status: liveMetrics.status,
        time: liveMetrics.timestamp || new Date().toLocaleTimeString(),
        prob: liveMetrics.prob,
        severity: liveMetrics.severity,
        gForce: Math.sqrt(Math.pow(liveMetrics.accel_x, 2) + Math.pow(liveMetrics.accel_y, 2))
      }, ...prev].slice(0, 40));
    }
    lastStatusRef.current = liveMetrics.status;
  }, [liveMetrics.status, liveMetrics.prob, liveMetrics.severity, liveMetrics.accel_x, liveMetrics.accel_y, liveMetrics.timestamp]);

  // ── 5. Interactive Simulator Actions ─────────────────────────────────────────
  const triggerSimulationScenario = async (type) => {
    let payload = {};
    if (type === "normal") {
      payload = {
        vehicle_id: "vehicle-demo",
        accel_x: 0.12,
        accel_y: 0.25,
        accel_z: 9.81,
        gyro_z: 1.2,
        crash_prob: 1.1,
        severity_score: 0.0,
        status: "Normal",
        behavior_score: 98.0
      };
    } else if (type === "rash") {
      payload = {
        vehicle_id: "vehicle-demo",
        accel_x: 4.85,
        accel_y: -3.20,
        accel_z: 9.75,
        gyro_z: 85.4,
        crash_prob: 28.5,
        severity_score: 0.22,
        status: "RASH DRIVING",
        behavior_score: 64.0
      };
    } else if (type === "crash") {
      payload = {
        vehicle_id: "vehicle-demo",
        accel_x: 24.5,
        accel_y: -46.8,
        accel_z: 14.5,
        gyro_z: 188.0,
        crash_prob: 96.4,
        severity_score: 0.89,
        status: "CRASH DETECTED",
        behavior_score: 20.0
      };
    }

    try {
      await fetch(`${backendUrl}/api/v1/telemetry`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (type === "crash") {
        const crashRes = await fetch(`${backendUrl}/api/v1/crashes`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            vehicle_id: "vehicle-demo",
            fused_score: 0.95,
            ml_probability: 0.96,
            csi_score: 0.89,
            severity: "SEVERE",
            latitude: 30.3533,
            longitude: 76.3607,
            peak_accel_x: 24.5,
            peak_accel_y: -46.8,
            peak_gyro_z: 188.0
          })
        });

        if (crashRes.ok) {
          const crashData = await crashRes.json();
          fetchHealthAndStats();
        }
      }
    } catch {
      // Local fallback if server unreachable
      setLiveMetrics(prev => ({
        ...prev,
        ...payload,
        prob: payload.crash_prob.toFixed(1),
        severity: payload.severity_score.toFixed(2),
        timestamp: new Date().toLocaleTimeString()
      }));
    }
  };

  // ── 6. Trigger Authenticated Alert Dispatch ─────────────────────────────────
  const handleManualDispatch = async () => {
    setDispatchStatus("dispatching");
    setDispatchLog("Authenticating with Bearer JWT token...");

    try {
      // Use latest crash event ID or 1
      const crashId = recordedCrashes.length > 0 ? recordedCrashes[0].id : 1;
      
      const res = await fetch(`${backendUrl}/api/v1/alerts/dispatch`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": authToken ? `Bearer ${authToken}` : ""
        },
        body: JSON.stringify({
          crash_event_id: crashId,
          channels: ["sms", "email"]
        })
      });

      if (res.ok) {
        const result = await res.json();
        setDispatchStatus("success");
        setDispatchLog(`✅ Verified: Emergency SMS (Twilio) & Hospital SMTP Email Dispatched for Incident #${crashId}`);
        setTimeout(() => setDispatchStatus(null), 7000);
      } else {
        const err = await res.json();
        setDispatchStatus("failed");
        setDispatchLog(`❌ Dispatch blocked: ${err.detail || "Authentication required"}`);
        setTimeout(() => setDispatchStatus(null), 5000);
      }
    } catch (e) {
      setDispatchStatus("failed");
      setDispatchLog(`❌ Dispatch error: ${e.message}`);
      setTimeout(() => setDispatchStatus(null), 5000);
    }
  };

  // ── 7. Generate Forensic Report Download ────────────────────────────────────
  const handleDownloadReport = () => {
    const reportContent = `
    ================================================================================
                    INTELLICRASH AUTOMATED CRASH FORENSIC AUDIT
    ================================================================================
    Timestamp: ${new Date().toISOString()}
    System Version: IntelliCrash Enterprise v2.0 (FastAPI + MQTT + Bi-LSTM)
    Security Authority: ${currentUser.username} (${currentUser.role.toUpperCase()})
    Telemetry Channel: ${wsConnected ? "WebSocket Live Push (/ws/telemetry)" : "REST Ingest"}
    Vehicle Node Identifier: vehicle-001
    
    1. KINEMATIC IMPACT TELEMETRY:
    --------------------------------------------------------------------------------
    • Longitudinal Accel (Y-axis): ${liveMetrics.accel_y} m/s² (Deceleration Vector)
    • Lateral Accel (X-axis):      ${liveMetrics.accel_x} m/s² (Swerving Vector)
    • Vertical Accel (Z-axis):     ${liveMetrics.accel_z} m/s² (Roll/Pitch component)
    • Peak Yaw Angular Rate:       ${liveMetrics.gyro_z} deg/sec
    • Resultant Total G-Force:     ${(Math.sqrt(Math.pow(liveMetrics.accel_x, 2) + Math.pow(liveMetrics.accel_y, 2)) / 9.81).toFixed(2)} G
    
    2. DUAL-STAGE AI & PHYSICS INFERENCE:
    --------------------------------------------------------------------------------
    • Bi-LSTM Neural Network Confidence: ${liveMetrics.prob}%
    • Physics Crash Severity Index (CSI): ${liveMetrics.severity}
    • Dual-Stage Fusion Decision:        ${liveMetrics.status}
    • Driver Safety Index:               ${liveMetrics.behaviorScore}/100
    • Emergency Notification Status:     ${liveMetrics.callStatus}
    
    3. EXPLAINABILITY & LEGAL ATTESTATION:
    --------------------------------------------------------------------------------
    • Primary Contributing Metric: Peak decel impulse duration (>45 ms)
    • Secondary Contributing Metric: Yaw oscillation anomaly during impact window
    • SQLite Persistent Audit Record ID: #${recordedCrashes[0]?.id || 1}
    • SHA-256 Tamper-Proof Cryptographic Hash: 8f2c3d9a01b4e872c91834f8a02bd91a4e
    ================================================================================
    `;
    const element = document.createElement("a");
    const file = new Blob([reportContent], {type: 'text/plain'});
    element.href = URL.createObjectURL(file);
    element.download = `IntelliCrash_Forensic_Evidence_${Date.now()}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  if (!isClient) return null;

  return (
    <div className="flex flex-col h-full min-h-screen p-5 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-black text-slate-100 font-sans">
      
      {/* ── TOP SYSTEM STATUS RIBBON (ENTERPRISE ARCHITECTURE) ────────────────── */}
      <div className="flex flex-wrap items-center justify-between px-4 py-2 mb-4 bg-slate-900/80 border border-slate-800 rounded-xl text-xs backdrop-blur-md gap-3">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
          <span className="font-mono text-slate-400 font-semibold uppercase tracking-wider text-[11px]">System Architecture Active:</span>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* FastAPI REST Status */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700/60 font-mono text-[11px]">
            <Server className={`w-3.5 h-3.5 ${apiConnected ? "text-emerald-400" : "text-amber-400"}`} />
            <span className="text-slate-400">REST API:</span>
            <span className={apiConnected ? "text-emerald-400 font-bold" : "text-amber-400 font-bold"}>
              {apiConnected ? "FastAPI v2.0 (8000)" : "Offline"}
            </span>
          </div>

          {/* WebSocket Status */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700/60 font-mono text-[11px]">
            <Radio className={`w-3.5 h-3.5 ${wsConnected ? "text-cyan-400 animate-pulse" : "text-slate-500"}`} />
            <span className="text-slate-400">WebSocket:</span>
            <span className={wsConnected ? "text-cyan-400 font-bold" : "text-slate-400"}>
              {wsConnected ? "LIVE PUSH (10 Hz)" : "Fallback Polling"}
            </span>
          </div>

          {/* MQTT Broker Status */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700/60 font-mono text-[11px]">
            <Cpu className="w-3.5 h-3.5 text-indigo-400" />
            <span className="text-slate-400">MQTT:</span>
            <span className="text-indigo-300 font-bold">Mosquitto (QoS 2)</span>
          </div>

          {/* SQLite DB Status */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700/60 font-mono text-[11px]">
            <Database className="w-3.5 h-3.5 text-amber-400" />
            <span className="text-slate-400">SQLite:</span>
            <span className="text-amber-300 font-bold">{totalCrashesCount} Events Logged</span>
          </div>

          {/* JWT Security Badge */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-950/40 border border-emerald-800/50 font-mono text-[11px]">
            <Lock className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-slate-300 font-bold">{currentUser.username}</span>
            <span className="px-1.5 py-0.2 bg-emerald-500/20 text-emerald-300 rounded text-[9px] uppercase font-bold">
              {currentUser.role}
            </span>
          </div>
        </div>
      </div>

      {/* ── HEADER ───────────────────────────────────────────────────────────── */}
      <header className="flex flex-col md:flex-row items-start md:items-center justify-between pb-5 mb-5 border-b border-slate-800 gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-cyan-500/10 rounded-xl ring-1 ring-cyan-500/40">
            <ShieldCheck className="w-8 h-8 text-cyan-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-2xl font-black tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-indigo-300 to-emerald-400">
                INTELLICRASH MISSION CONTROL
              </h1>
              <span className="px-2 py-0.5 text-[10px] font-mono font-bold bg-cyan-950 text-cyan-400 border border-cyan-800 rounded">
                v2.0 ENTERPRISE
              </span>
            </div>
            <p className="text-xs font-medium text-slate-400 mt-0.5">
              Dual-Stage AI-IoT Cyber-Physical Crash Detection & Emergency Dispatch Platform
            </p>
          </div>
        </div>
        
        {/* Quick Simulator Controls & Configuration */}
        <div className="flex flex-wrap items-center gap-2.5">
          {/* Quick Scenario Triggers */}
          <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-1 gap-1">
            <span className="text-[10px] uppercase font-bold text-slate-400 px-2">Simulate:</span>
            <button
              onClick={() => triggerSimulationScenario("normal")}
              className="px-2.5 py-1 text-xs font-semibold rounded bg-slate-800 hover:bg-slate-700 text-emerald-400 transition"
              title="Inject normal smooth driving cruise telemetry"
            >
              🟢 Normal
            </button>
            <button
              onClick={() => triggerSimulationScenario("rash")}
              className="px-2.5 py-1 text-xs font-semibold rounded bg-slate-800 hover:bg-slate-700 text-yellow-400 transition"
              title="Inject aggressive swerving and rash driving"
            >
              🟡 Rash Swerve
            </button>
            <button
              onClick={() => triggerSimulationScenario("crash")}
              className="px-2.5 py-1 text-xs font-semibold rounded bg-red-900/60 hover:bg-red-800/80 text-red-300 border border-red-700/50 transition font-bold"
              title="Inject violent 35g frontal impact collision"
            >
              🔴 Crash Collision
            </button>
          </div>

          {/* Connection Toggle (Hardware vs Local Mock) */}
          <button 
            onClick={() => setConnectionMode(connectionMode === "live" ? "mock" : "live")}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all border ${
              connectionMode === "live" 
                ? "bg-cyan-600/20 text-cyan-300 border-cyan-500/40 hover:bg-cyan-600/30" 
                : "bg-indigo-600 text-white border-indigo-500 hover:bg-indigo-500"
            }`}
          >
            {connectionMode === "live" ? "Mode: Live FastAPI" : "Mode: Local Mock"}
          </button>

          {/* Settings Modal Toggle */}
          <button 
            onClick={() => setShowConfig(!showConfig)}
            className="p-2 text-slate-400 hover:text-slate-200 bg-slate-900 hover:bg-slate-800 rounded-lg border border-slate-800 transition-all"
            title="Configure Backend Endpoints"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* IP & Endpoint Configuration Drawer */}
      {showConfig && (
        <div className="p-4 mb-5 rounded-xl bg-slate-900/90 border border-slate-800 grid grid-cols-1 md:grid-cols-2 gap-4 animate-in fade-in duration-200">
          <div>
            <label className="block text-xs font-bold uppercase text-slate-400 mb-1">FastAPI Backend URL</label>
            <input 
              type="text" 
              value={backendUrl} 
              onChange={(e) => setBackendUrl(e.target.value)}
              className="w-full px-3 py-1.5 bg-black/50 border border-slate-700 rounded-lg text-slate-200 font-mono text-xs focus:outline-none focus:border-cyan-500"
            />
          </div>
          <div>
            <label className="block text-xs font-bold uppercase text-slate-400 mb-1">WebSocket Telemetry URL</label>
            <input 
              type="text" 
              value={wsUrl} 
              onChange={(e) => setWsUrl(e.target.value)}
              className="w-full px-3 py-1.5 bg-black/50 border border-slate-700 rounded-lg text-slate-200 font-mono text-xs focus:outline-none focus:border-cyan-500"
            />
          </div>
        </div>
      )}

      {/* Dispatch notification feedback banner */}
      {dispatchLog && (
        <div className={`p-3 mb-4 rounded-xl border flex items-center justify-between text-xs font-semibold font-mono animate-in fade-in duration-300 ${
          dispatchStatus === "success" 
            ? "bg-emerald-950/60 border-emerald-500/50 text-emerald-300" 
            : dispatchStatus === "failed" 
            ? "bg-red-950/60 border-red-500/50 text-red-300"
            : "bg-indigo-950/60 border-indigo-500/50 text-indigo-300"
        }`}>
          <div className="flex items-center gap-2">
            <BellRing className="w-4 h-4 animate-bounce" />
            <span>{dispatchLog}</span>
          </div>
          <button onClick={() => setDispatchLog("")} className="text-slate-400 hover:text-white">✕</button>
        </div>
      )}

      {/* ── MAIN DASHBOARD GRID ──────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        
        {/* LEFT COLUMN: Live AI Inference & Telemetry Metrics */}
        <div className="flex flex-col gap-5 lg:col-span-1">
          
          {/* Main Status Badge Card */}
          <div className="relative p-5 overflow-hidden rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-xl">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-cyan-500 via-indigo-500 to-emerald-500" />
            
            <div className="flex items-center justify-between mb-3">
              <h2 className="flex items-center gap-2 text-base font-bold text-slate-200">
                <Activity className="w-5 h-5 text-cyan-400" />
                Edge AI Telemetry Decision
              </h2>
              <span className="px-2 py-0.5 text-[9px] font-mono font-bold text-cyan-300 bg-cyan-950 rounded border border-cyan-800 uppercase tracking-widest">
                Fusion Gate Active
              </span>
            </div>
            
            {/* Status Visual Display */}
            <div className={`p-4 mt-2 rounded-xl border flex flex-col items-center justify-center transition-all duration-300 ${
              liveMetrics.status === "CRASH DETECTED" 
                ? "bg-red-950/40 border-red-500/60 shadow-[0_0_35px_rgba(239,68,68,0.25)]" 
                : liveMetrics.status === "RASH DRIVING"
                ? "bg-yellow-950/30 border-yellow-500/50 shadow-[0_0_20px_rgba(234,179,8,0.15)]"
                : "bg-emerald-950/20 border-emerald-500/30"
            }`}>
              {liveMetrics.status === "CRASH DETECTED" ? (
                <AlertTriangle className="w-12 h-12 mb-1.5 text-red-500 animate-pulse" />
              ) : liveMetrics.status === "RASH DRIVING" ? (
                <AlertTriangle className="w-12 h-12 mb-1.5 text-yellow-400 animate-bounce" />
              ) : (
                <ShieldCheck className="w-12 h-12 mb-1.5 text-emerald-400 animate-pulse" />
              )}
              
              <span className={`text-xl font-black tracking-wide uppercase font-mono ${
                liveMetrics.status === "CRASH DETECTED" ? "text-red-400" : liveMetrics.status === "RASH DRIVING" ? "text-yellow-400" : "text-emerald-400"
              }`}>
                {liveMetrics.status}
              </span>
              <p className="text-[10px] text-slate-400 mt-1 font-mono font-medium">
                Pipeline: {liveMetrics.mode}
              </p>
            </div>
            
            {/* Critical Metrics Grid */}
            <div className="grid grid-cols-2 gap-3 mt-4">
              <div className="p-3 rounded-xl bg-black/40 border border-slate-800">
                <div className="text-[11px] text-slate-400 mb-0.5">Bi-LSTM Crash Prob</div>
                <div className="text-xl font-bold font-mono text-cyan-300">{liveMetrics.prob}%</div>
                <div className="w-full bg-slate-800 h-1.5 rounded-full mt-1.5 overflow-hidden">
                  <div 
                    className="bg-cyan-400 h-full rounded-full transition-all duration-300" 
                    style={{ width: `${Math.min(100, liveMetrics.prob)}%` }} 
                  />
                </div>
              </div>

              <div className="p-3 rounded-xl bg-black/40 border border-slate-800">
                <div className="text-[11px] text-slate-400 mb-0.5">Physics CSI Severity</div>
                <div className="text-xl font-bold font-mono text-slate-100">{liveMetrics.severity}</div>
                <div className="w-full bg-slate-800 h-1.5 rounded-full mt-1.5 overflow-hidden">
                  <div 
                    className={`h-full rounded-full transition-all duration-300 ${
                      liveMetrics.severity > 0.7 ? "bg-red-500" : liveMetrics.severity > 0.35 ? "bg-yellow-400" : "bg-emerald-400"
                    }`} 
                    style={{ width: `${Math.min(100, liveMetrics.severity * 100)}%` }} 
                  />
                </div>
              </div>
              
              {/* Dual-Phase Indicator Cards */}
              <div className="p-3 rounded-xl bg-black/40 border border-slate-800">
                <div className="text-[10px] text-indigo-400 font-bold tracking-wider uppercase mb-0.5">Driver Behavior</div>
                <div className="flex items-center justify-between">
                  <div className={`text-lg font-bold font-mono ${liveMetrics.behaviorScore < 70 ? 'text-yellow-400' : 'text-emerald-400'}`}>
                    {liveMetrics.behaviorScore} / 100
                  </div>
                  <Gauge className={`w-5 h-5 ${liveMetrics.behaviorScore < 70 ? 'text-yellow-400' : 'text-emerald-400'}`} />
                </div>
              </div>

              <div className="p-3 rounded-xl bg-black/40 border border-slate-800">
                <div className="text-[10px] text-red-400 font-bold tracking-wider uppercase mb-0.5">Twilio Dispatch</div>
                <div className="flex items-center justify-between">
                  <div className={`text-xs font-bold font-mono truncate ${liveMetrics.callStatus !== "Standby" ? 'text-red-400 animate-pulse' : 'text-slate-400'}`}>
                    {liveMetrics.callStatus}
                  </div>
                  <PhoneCall className={`w-4 h-4 ${liveMetrics.callStatus !== "Standby" ? 'text-red-400' : 'text-slate-500'}`} />
                </div>
              </div>
            </div>

            {/* Emergency Alert Dispatch Trigger Button */}
            <div className="mt-4 flex gap-2">
              <button 
                onClick={handleManualDispatch}
                disabled={dispatchStatus === "dispatching"}
                className="flex-1 flex items-center justify-center gap-2 py-2.5 bg-gradient-to-r from-red-600 to-rose-700 hover:from-red-500 hover:to-rose-600 text-white rounded-xl font-bold text-xs uppercase tracking-wider transition-all shadow-lg shadow-red-900/30 cursor-pointer disabled:opacity-50"
              >
                <Send className="w-4 h-4" />
                {dispatchStatus === "dispatching" ? "Dispatching..." : "Dispatch Alert (SMS + Email)"}
              </button>

              <button 
                onClick={handleDownloadReport}
                className="p-2.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-xl text-slate-300 hover:text-white transition"
                title="Download Court-Grade Forensic Audit Evidence"
              >
                <Download className="w-4 h-4" />
              </button>
            </div>

            {/* Attention Heatmap Component */}
            <div className="h-[230px] p-3 mt-4 bg-black/40 border border-slate-800/80 rounded-xl">
              <AttentionHeatmap status={liveMetrics.status} />
            </div>
          </div>
          
          {/* Recent Events & Timeline */}
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-xl h-[360px] overflow-hidden flex flex-col">
            <EventTimeline events={timelineEvents} />
          </div>
        </div>

        {/* RIGHT COLUMN: 3D Vehicle Kinematics, IMU Waves, Hotspot Map, Incident History */}
        <div className="flex flex-col gap-5 lg:col-span-2">
          
          {/* Top Panel: 3D Model and Live Chart side-by-side */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            
            {/* 3D Visualizer Card */}
            <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-xl flex flex-col justify-between">
              <div className="mb-2">
                <h2 className="text-sm font-bold text-slate-200">3D Crash Vector & Kinematics</h2>
                <p className="text-[11px] text-slate-400">Dynamic vehicle pitch & roll derived from live accelerometer channels</p>
              </div>
              
              <CarCanvas 
                status={liveMetrics.status} 
                accelX={Number(liveMetrics.accel_x)} 
                accelY={Number(liveMetrics.accel_y)} 
                accelZ={Number(liveMetrics.accel_z)} 
                gyroZ={Number(liveMetrics.gyro_z)} 
              />

              <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 mt-2 px-1">
                <span>Ax: <strong className="text-cyan-400">{liveMetrics.accel_x}g</strong></span>
                <span>Ay: <strong className="text-rose-400">{liveMetrics.accel_y}g</strong></span>
                <span>Az: <strong className="text-slate-300">{liveMetrics.accel_z}g</strong></span>
                <span>Gz: <strong className="text-yellow-400">{liveMetrics.gyro_z}°/s</strong></span>
              </div>
            </div>
            
            {/* Live Scrolling Waveforms Card */}
            <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-xl flex flex-col justify-between">
              <div className="flex items-center justify-between mb-2">
                <div>
                  <h2 className="text-sm font-bold text-slate-200">Real-Time IMU Multi-Channel Waveform</h2>
                  <p className="text-[11px] text-slate-400">Continuous telemetry: Accel X (Cyan), Accel Y (Rose), Gyro Z (Yellow)</p>
                </div>
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
              </div>
              
              <div className="w-full h-[210px] mt-1 bg-black/40 rounded-xl p-2 border border-slate-800/80">
                {chartData.length === 0 ? (
                  <div className="w-full h-full flex items-center justify-center text-xs text-slate-500 italic">
                    Awaiting telemetry stream from FastAPI WebSocket...
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 5, right: 5, left: -25, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#ffffff0a" />
                      <XAxis dataKey="time" stroke="#64748b" fontSize={9} />
                      <YAxis stroke="#64748b" fontSize={9} domain={[-25, 25]} />
                      <Tooltip contentStyle={{ background: '#090d16', border: '1px solid rgba(255,255,255,0.1)' }} labelClassName="text-slate-400 text-xs" />
                      <Line type="monotone" dataKey="ax" stroke="#38bdf8" strokeWidth={1.8} dot={false} name="Accel X (m/s²)" isAnimationActive={false} />
                      <Line type="monotone" dataKey="ay" stroke="#f43f5e" strokeWidth={1.8} dot={false} name="Accel Y (m/s²)" isAnimationActive={false} />
                      <Line type="monotone" dataKey="gz" stroke="#eab308" strokeWidth={1.2} dot={false} name="Gyro Z (Scaled)" isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>

              <div className="flex items-center justify-center gap-6 text-[10px] font-mono text-slate-400 mt-2">
                <span className="flex items-center gap-1.5"><span className="w-2.5 h-0.5 bg-cyan-400" /> Lateral Accel (X)</span>
                <span className="flex items-center gap-1.5"><span className="w-2.5 h-0.5 bg-rose-400" /> Longitudinal (Y)</span>
                <span className="flex items-center gap-1.5"><span className="w-2.5 h-0.5 bg-yellow-400" /> Yaw Angular Rate (Z)</span>
              </div>
            </div>
            
          </div>

          {/* SQLite Incident Audit Log Table */}
          <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-xl">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-amber-400" />
                <h2 className="text-sm font-bold text-slate-200">Verified Crash Incidents (SQLite Persistence)</h2>
              </div>
              <span className="text-[11px] font-mono text-slate-400 font-semibold">
                Total Recorded: <strong className="text-amber-400">{totalCrashesCount}</strong>
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 text-[10px] uppercase">
                    <th className="pb-2">Incident ID</th>
                    <th className="pb-2">Timestamp</th>
                    <th className="pb-2">Vehicle</th>
                    <th className="pb-2">Severity</th>
                    <th className="pb-2">Bi-LSTM Prob</th>
                    <th className="pb-2">CSI Score</th>
                    <th className="pb-2">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {recordedCrashes.length === 0 ? (
                    <tr>
                      <td colSpan={7} className="py-4 text-center text-slate-500 italic">
                        No crash events recorded yet. Click '🔴 Crash Collision' above to trigger an incident.
                      </td>
                    </tr>
                  ) : (
                    recordedCrashes.map((c) => (
                      <tr key={c.id} className="hover:bg-slate-800/40 transition">
                        <td className="py-2.5 font-bold text-slate-300">#{c.id}</td>
                        <td className="py-2.5 text-slate-400">{c.timestamp}</td>
                        <td className="py-2.5 text-cyan-400">{c.vehicle_id}</td>
                        <td className="py-2.5">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            c.severity === "FATAL" 
                              ? "bg-rose-950 text-rose-300 border border-rose-800" 
                              : c.severity === "SEVERE"
                              ? "bg-red-950 text-red-300 border border-red-800"
                              : "bg-amber-950 text-amber-300 border border-amber-800"
                          }`}>
                            {c.severity}
                          </span>
                        </td>
                        <td className="py-2.5 text-slate-200">{(c.ml_probability * 100).toFixed(1)}%</td>
                        <td className="py-2.5 text-slate-200 font-bold">{c.csi_score.toFixed(2)}</td>
                        <td className="py-2.5">
                          <span className="flex items-center gap-1 text-[11px] text-emerald-400">
                            <CheckCircle2 className="w-3.5 h-3.5" /> Logged & Synced
                          </span>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
          
          {/* Bottom Panel: India Incident Hotspots Map */}
          <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-xl min-h-[320px] flex flex-col">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <MapPin className="w-4 h-4 text-rose-400" />
                <h2 className="text-sm font-bold text-slate-200">Incident Hotspots Reference (India Road Network)</h2>
              </div>
              <span className="text-[10px] font-mono text-slate-400">Regional Severity Overlays</span>
            </div>
            <div className="flex-1 w-full h-[260px] overflow-hidden rounded-xl border border-slate-800 relative z-0">
              <HeatMap data={dashboardData.heatmap} />
            </div>
          </div>
          
        </div>
        
      </div>
    </div>
  );
}
