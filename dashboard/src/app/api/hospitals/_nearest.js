// Pure helpers for /api/hospitals (kept separate so they can be tested without Next).

export function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371, rad = (d) => (d * Math.PI) / 180;
  const dLat = rad(lat2 - lat1), dLon = rad(lon2 - lon1);
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(rad(lat1)) * Math.cos(rad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}

export function overpassQuery(lat, lon, radiusM) {
  const around = `(around:${Math.round(radiusM)},${lat},${lon})`;
  return `[out:json][timeout:20];(node["amenity"="hospital"]${around};way["amenity"="hospital"]${around};relation["amenity"="hospital"]${around};);out center tags;`;
}

// Turn an Overpass response into the nearest `limit` hospitals, closest first. Only real OSM tags are used.
export function nearestFromOverpass(json, lat, lon, limit = 5) {
  const seen = new Set();
  const out = [];
  for (const el of json?.elements || []) {
    const hLat = el.lat ?? el.center?.lat;
    const hLon = el.lon ?? el.center?.lon;
    if (typeof hLat !== "number" || typeof hLon !== "number") continue;
    const t = el.tags || {};
    const name = t.name || t["name:en"] || null;
    const key = `${name}|${hLat.toFixed(4)}|${hLon.toFixed(4)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const address = t["addr:full"] || [t["addr:housenumber"], t["addr:street"], t["addr:suburb"], t["addr:city"]].filter(Boolean).join(", ") || null;
    out.push({
      id: `${el.type}/${el.id}`,
      name: name || "Unnamed hospital",
      lat: hLat,
      lon: hLon,
      distance_km: Number(haversineKm(lat, lon, hLat, hLon).toFixed(2)),
      address,
      phone: t.phone || t["contact:phone"] || null,
      emergency: t.emergency || null,
      website: t.website || t["contact:website"] || null,
    });
  }
  out.sort((a, b) => a.distance_km - b.distance_km);
  return out.slice(0, limit);
}
