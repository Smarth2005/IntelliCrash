"use client";

import React, { useEffect } from "react";
import { MapContainer, TileLayer, CircleMarker, Tooltip } from "react-leaflet";
import "leaflet/dist/leaflet.css";

export default function HeatMap({ data }) {
  // Fix Leaflet SSR issues
  useEffect(() => {
    // This is needed to ensure leaflet renders tiles correctly in next.js
    const L = require("leaflet");
    delete L.Icon.Default.prototype._getIconUrl;
    L.Icon.Default.mergeOptions({
      iconRetinaUrl: require("leaflet/dist/images/marker-icon-2x.png").default,
      iconUrl: require("leaflet/dist/images/marker-icon.png").default,
      shadowUrl: require("leaflet/dist/images/marker-shadow.png").default,
    });
  }, []);

  if (!data || data.length === 0) return <div>No data available</div>;

  // Center of India roughly
  const center = [22.9, 78.6];

  // Find max count to normalize circle radius
  const maxCount = Math.max(...data.map(d => d.count));

  return (
    <MapContainer 
      center={center} 
      zoom={5} 
      style={{ height: "100%", width: "100%", background: "#0f172a", zIndex: 0 }}
      zoomControl={false}
      attributionControl={false}
    >
      {/* Dark theme map tiles (CartoDB Dark Matter) */}
      <TileLayer
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
      />
      
      {data.map((item, idx) => {
        // Calculate radius based on count relative to max count
        // Min radius 10, max 40
        const radius = 10 + (item.count / maxCount) * 30;
        
        // Color gradient from yellow to red based on severity/count
        const intensity = item.count / maxCount;
        const r = Math.floor(255);
        const g = Math.floor(255 * (1 - intensity));
        const color = `rgb(${r}, ${g}, 0)`;
        
        return (
          <CircleMarker
            key={idx}
            center={[item.lat, item.lng]}
            radius={radius}
            pathOptions={{ 
              color: color, 
              fillColor: color, 
              fillOpacity: 0.5,
              weight: 1
            }}
          >
            <Tooltip 
              direction="top" 
              offset={[0, -10]} 
              opacity={1}
              className="custom-tooltip"
            >
              <div className="p-1 text-sm font-semibold text-slate-800">
                <div className="border-b border-slate-300 pb-1 mb-1">{item.state}</div>
                <div>Accidents: {item.count.toLocaleString()}</div>
              </div>
            </Tooltip>
          </CircleMarker>
        );
      })}
    </MapContainer>
  );
}
