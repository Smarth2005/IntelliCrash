"use client";

import React, { useEffect, useMemo, useState } from "react";
import { MapContainer, TileLayer, CircleMarker, Marker, Polyline, Tooltip, useMap, useMapEvents } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// Hospital icon: white rounded square with a red cross (inline SVG, no image file or external request).
const hospitalIcon = L.divIcon({
  className: "",
  iconSize: [30, 30],
  iconAnchor: [15, 15],
  tooltipAnchor: [16, 0],
  html: `<div style="width:30px;height:30px;border-radius:8px;background:#fff;border:2px solid #dc2626;box-shadow:0 1px 4px rgba(0,0,0,.35);display:flex;align-items:center;justify-content:center">
    <svg viewBox="0 0 24 24" width="20" height="20" aria-label="Hospital"><path d="M9 3h6v6h6v6h-6v6H9v-6H3V9h6z" fill="#dc2626"/></svg></div>`,
});

// Street-level zoom used whenever the map is centred on the car (roads and surroundings clearly visible).
const STREET_ZOOM = 18;

// Keeps the map framed correctly.
//  follow = true : centre on the car at street zoom and glide along as it moves (live GPS or fixed demo spot)
//  follow = false: fit everything relevant (car, selected crash, hospitals) - used by the "Fit all" button
function Controller({ vehicle, points, follow, fitTick }) {
  const map = useMap();
  // The map lives inside a flex container; tell Leaflet the real size once layout has settled.
  useEffect(() => {
    const t1 = setTimeout(() => map.invalidateSize(), 100);
    const t2 = setTimeout(() => map.invalidateSize(), 600);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, [map]);

  const vKey = vehicle ? `${vehicle.lat.toFixed(6)},${vehicle.lon.toFixed(6)}` : "";
  useEffect(() => {
    if (!follow || !vehicle) return;
    const z = Math.max(map.getZoom(), STREET_ZOOM);
    map.flyTo([vehicle.lat, vehicle.lon], z, { animate: true, duration: 0.6 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [vKey, follow, map]);

  const pKey = points.map((p) => p.join(",")).join("|");
  useEffect(() => {
    if (follow || !points.length) return;
    if (points.length === 1) map.setView(points[0], STREET_ZOOM);
    else map.fitBounds(points, { padding: [60, 60], maxZoom: 17 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fitTick, follow, pKey, map]);
  return null;
}

// Click on the map -> report the position (used to pin the demo car at the exact classroom spot).
function ClickPicker({ onPick }) {
  useMapEvents({ click: (e) => onPick && onPick({ lat: e.latlng.lat, lon: e.latlng.lng }) });
  return null;
}

// Road route from the crash to the nearest hospital (OSRM public server). Falls back to a dashed straight line
// when offline, so the demo never shows a blank map.
function RoadRoute({ from, to }) {
  const [line, setLine] = useState(null);
  const key = from && to ? `${from[0]},${from[1]}|${to[0]},${to[1]}` : "";
  useEffect(() => {
    setLine(null);
    if (!key) return;
    const ctrl = new AbortController();
    const url = `https://router.project-osrm.org/route/v1/driving/${from[1]},${from[0]};${to[1]},${to[0]}?overview=full&geometries=geojson`;
    fetch(url, { signal: ctrl.signal })
      .then((r) => r.json())
      .then((d) => {
        const c = d?.routes?.[0]?.geometry?.coordinates;
        if (Array.isArray(c) && c.length) setLine(c.map(([lon, lat]) => [lat, lon]));
      })
      .catch(() => {});
    return () => ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  if (!key) return null;
  return line
    ? <Polyline positions={line} pathOptions={{ color: "#2563eb", weight: 5, opacity: 0.85 }} />
    : <Polyline positions={[from, to]} pathOptions={{ color: "#dc2626", weight: 2, dashArray: "6 6", opacity: 0.8 }} />;
}

export default function HeatMap({
  incidents = [],
  vehicle = null,          // { lat, lon, speed } while the Pi is connected and has a GPS fix, else null
  vehicleLabel = "Vehicle",
  hospitals = [],          // nearest hospitals to the selected crash (from /api/hospitals)
  focusId = null,          // id of the selected crash
  startView = null,        // { center: [lat, lon], zoom } used before any point is known (demo mode: the campus)
  follow = true,           // centre on the car at street zoom and follow it
  fitTick = 0,             // bump to run "fit all" (only when follow = false)
  pickMode = false,        // clicking the map reports a position through onPick
  onPick = null,
  onSelectIncident = () => {},
}) {
  const crashes = useMemo(
    () => (Array.isArray(incidents) ? incidents : []).filter((i) => Array.isArray(i.gps_coords)),
    [incidents]
  );
  const focus = crashes.find((c) => c.id === focusId) || null;

  const fitPoints = useMemo(() => {
    const pts = [];
    if (vehicle) pts.push([vehicle.lat, vehicle.lon]);
    if (focus) pts.push(focus.gps_coords);
    hospitals.forEach((h) => pts.push([h.lat, h.lon]));
    return pts;
  }, [vehicle, focus, hospitals]);

  // Before any position is known: the demo campus view if given, else a neutral view of India.
  const view = startView ? { ...startView, zoom: Math.max(startView.zoom || 0, STREET_ZOOM) } : { center: [22.5, 79], zoom: 5 };

  return (
    <MapContainer
      center={view.center}
      zoom={view.zoom}
      style={{ height: "100%", width: "100%", background: "#f8fafc", zIndex: 0 }}
      zoomControl={true}
      attributionControl={true}
      maxZoom={20}
    >
      <TileLayer
        maxZoom={20}
        maxNativeZoom={19}
        url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
      />
      <Controller vehicle={vehicle} points={fitPoints} follow={follow} fitTick={fitTick} />
      {pickMode && <ClickPicker onPick={onPick} />}

      {/* Vehicle: only drawn from a real, current GPS fix streamed by the Pi */}
      {vehicle && (
        <>
          <CircleMarker center={[vehicle.lat, vehicle.lon]} radius={26} pathOptions={{ color: "#0284c7", fillColor: "#0284c7", fillOpacity: 0.15, weight: 1 }} />
          <CircleMarker center={[vehicle.lat, vehicle.lon]} radius={11} pathOptions={{ color: "#0369a1", fillColor: "#38bdf8", fillOpacity: 0.95, weight: 3 }}>
            <Tooltip direction="top" offset={[0, -10]} opacity={1} permanent>
              <div className="p-1 font-sans text-xs">
                <span className="font-extrabold text-sky-900 block">{vehicleLabel} · {vehicle.isDemo ? "demo position" : "live"}</span>
                <span className="text-[10px] text-slate-600 font-mono">
                  {vehicle.lat.toFixed(5)}, {vehicle.lon.toFixed(5)}{vehicle.speed != null ? ` · ${Math.round(vehicle.speed)} km/h` : ""}
                </span>
              </div>
            </Tooltip>
          </CircleMarker>
        </>
      )}

      {/* Road route from the selected crash (not from the car or any fixed place) to the closest hospital */}
      {focus && hospitals[0] && (
        <RoadRoute from={focus.gps_coords} to={[hospitals[0].lat, hospitals[0].lon]} />
      )}

      {/* Crashes: always red (size shows severity). Click one to see the hospitals nearest to it. */}
      {crashes.map((inc) => {
        const selected = inc.id === focusId;
        const radius = inc.severity === "FATAL" ? 16 : inc.severity === "SEVERE" ? 13 : 10;
        return (
          <CircleMarker
            key={inc.id}
            center={inc.gps_coords}
            radius={radius}
            eventHandlers={{ click: () => onSelectIncident(inc.id) }}
            pathOptions={{ color: selected ? "#7f1d1d" : "#dc2626", fillColor: "#dc2626", fillOpacity: 0.8, weight: selected ? 4 : 2 }}
          >
            <Tooltip direction="bottom" offset={[0, 10]} opacity={0.95}>
              <div className="p-1.5 font-sans text-xs">
                <div className="font-bold text-slate-900 pb-0.5 border-b border-slate-200">
                  {inc.id} • <span style={{ color: "#dc2626" }}>{inc.severity}</span>
                </div>
                <div className="text-[10px] text-slate-600 mt-1 font-mono">
                  Impact: {inc.peak_accel_m_s2} m/s² • CSI: {Number(inc.csi_score || 0).toFixed(3)}
                </div>
                <div className="text-[10px] text-slate-500">{inc.timestamp}</div>
                <div className="text-[10px] text-slate-500">{inc.location_name}</div>
                {inc.position_is_demo && <div className="text-[10px] text-amber-700 font-semibold">Demo position, not a GPS reading</div>}
              </div>
            </Tooltip>
          </CircleMarker>
        );
      })}

      {/* Hospitals nearest to the selected crash (OpenStreetMap) */}
      {hospitals.map((h, i) => (
        <Marker key={h.id} position={[h.lat, h.lon]} icon={hospitalIcon}>
          <Tooltip direction="right" offset={[10, 0]} opacity={0.97}>
            <div className="p-1 font-sans text-xs max-w-[220px]">
              <span className="font-bold text-slate-900 block">{h.name}</span>
              <span className="text-[10px] text-slate-600 block">{i === 0 ? "Nearest · " : ""}{h.distance_km.toFixed(1)} km away</span>
              {h.address && <span className="text-[10px] text-slate-500 block">{h.address}</span>}
              {h.phone && <span className="text-[10px] text-slate-500 block">Tel: {h.phone}</span>}
              {h.emergency === "yes" && <span className="text-[10px] text-emerald-700 font-semibold block">Emergency department</span>}
            </div>
          </Tooltip>
        </Marker>
      ))}
    </MapContainer>
  );
}
