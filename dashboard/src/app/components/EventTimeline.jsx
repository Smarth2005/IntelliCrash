import React from 'react';
import { Activity, AlertTriangle, ShieldCheck, Clock } from 'lucide-react';

export default function EventTimeline({ events }) {
  if (!events || events.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-slate-500">
        <Clock className="w-8 h-8 mb-2 opacity-50" />
        <p>No recent events</p>
      </div>
    );
  }

  return (
    <div className="w-full h-full min-h-[200px] overflow-y-auto pr-2 custom-scrollbar">
      <h3 className="text-sm font-semibold text-slate-400 mb-4 sticky top-0 bg-slate-900/90 py-1 z-10 backdrop-blur-sm">Event Timeline</h3>
      <div className="flex flex-col gap-3">
        {events.map((event, idx) => {
          let Icon = ShieldCheck;
          let colorClass = "text-cyan-400 bg-cyan-400/10 border-cyan-400/20";
          let dotClass = "bg-cyan-400";
          
          if (event.status === "CRASH DETECTED") {
            Icon = AlertTriangle;
            colorClass = "text-red-400 bg-red-500/10 border-red-500/20";
            dotClass = "bg-red-500 shadow-[0_0_8px_#ef4444]";
          } else if (event.status === "RASH DRIVING") {
            Icon = Activity;
            colorClass = "text-orange-400 bg-orange-500/10 border-orange-500/20";
            dotClass = "bg-orange-500 shadow-[0_0_8px_#f97316]";
          }

          return (
            <div key={idx} className="flex gap-3 relative">
              {/* Timeline Line */}
              {idx !== events.length - 1 && (
                <div className="absolute left-[11px] top-6 bottom-[-16px] w-[2px] bg-slate-700/50 rounded-full" />
              )}
              
              {/* Timeline Dot */}
              <div className="relative mt-1">
                <div className={`w-6 h-6 rounded-full flex items-center justify-center border border-slate-700 bg-slate-800 z-10 relative`}>
                  <div className={`w-2 h-2 rounded-full ${dotClass}`} />
                </div>
              </div>
              
              {/* Event Card */}
              <div className={`flex-1 p-3 rounded-lg border ${colorClass} transition-colors`}>
                <div className="flex justify-between items-start mb-1">
                  <div className="flex items-center gap-2">
                    <Icon className="w-4 h-4" />
                    <span className="font-semibold text-sm">{event.status}</span>
                  </div>
                  <span className="text-xs text-slate-500 font-mono">{event.time}</span>
                </div>
                {event.status === "CRASH DETECTED" && (
                  <p className="text-xs text-slate-400 mt-1">
                    Prob: <span className="text-red-400 font-mono">{event.prob}%</span> | CSI: <span className="text-red-400 font-mono">{event.severity}</span>
                  </p>
                )}
                {event.status === "RASH DRIVING" && (
                  <p className="text-xs text-slate-400 mt-1">
                    G-Force: <span className="text-orange-400 font-mono">{event.gForce?.toFixed(2) || "N/A"}g</span>
                  </p>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
