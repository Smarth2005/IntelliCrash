import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function AttentionHeatmap({ status }) {
  // Mock SHAP/Attention values based on current status
  let data = [];
  
  if (status === "CRASH DETECTED") {
    data = [
      { feature: 'Accel Y (Longitudinal)', weight: 85, color: '#ef4444' }, // Red
      { feature: 'Accel X (Lateral)', weight: 65, color: '#f97316' },      // Orange
      { feature: 'Gyro Z (Rotation)', weight: 40, color: '#eab308' },      // Yellow
      { feature: 'Resultant G-Force', weight: 92, color: '#dc2626' },      // Dark Red
      { feature: 'Delta-V', weight: 78, color: '#ea580c' },
    ];
  } else if (status === "RASH DRIVING") {
    data = [
      { feature: 'Accel Y (Longitudinal)', weight: 35, color: '#fbbf24' },
      { feature: 'Accel X (Lateral)', weight: 82, color: '#ea580c' },
      { feature: 'Gyro Z (Rotation)', weight: 75, color: '#f97316' },
      { feature: 'Resultant G-Force', weight: 45, color: '#eab308' },
      { feature: 'Delta-V', weight: 20, color: '#fbbf24' },
    ];
  } else {
    data = [
      { feature: 'Accel Y (Longitudinal)', weight: 5, color: '#38bdf8' },
      { feature: 'Accel X (Lateral)', weight: 3, color: '#38bdf8' },
      { feature: 'Gyro Z (Rotation)', weight: 2, color: '#38bdf8' },
      { feature: 'Resultant G-Force', weight: 8, color: '#38bdf8' },
      { feature: 'Delta-V', weight: 4, color: '#38bdf8' },
    ];
  }

  return (
    <div className="w-full h-full min-h-[200px] flex flex-col">
      <h3 className="text-sm font-semibold text-slate-400 mb-2">SHAP Feature Attention Heatmap</h3>
      <div className="flex-1 w-full relative">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
            <XAxis type="number" domain={[0, 100]} hide />
            <YAxis dataKey="feature" type="category" width={140} tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <Tooltip 
              contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc' }}
              itemStyle={{ color: '#f8fafc' }}
              cursor={{ fill: '#1e293b' }}
            />
            <Bar dataKey="weight" radius={[0, 4, 4, 0]}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
