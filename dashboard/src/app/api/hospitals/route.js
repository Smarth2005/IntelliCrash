// Nearest hospitals to a point (the crash location), from OpenStreetMap via the Overpass API.
// No API key is needed. The dashboard server needs internet access. Nothing is invented: if the lookup fails,
// the caller gets an error and shows "could not load", never a made-up list.
import { NextResponse } from "next/server";
import { getUserBySession } from "../auth/_db";
import { readSessionToken } from "../auth/_cookie";
import { overpassQuery, nearestFromOverpass } from "./_nearest";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
const NO_STORE = { "Cache-Control": "no-store" };

const ENDPOINTS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"];
const RADII_M = [5000, 15000, 40000]; // widen the search until a few hospitals are found
const WANT = 3;
const TTL_MS = 24 * 3600 * 1000;
const cache = new Map(); // "lat,lon" (3 decimals, about 100 m) -> { t, data }

async function overpass(query) {
  let lastErr = null;
  for (const url of ENDPOINTS) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 20000);
    try {
      const r = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded", "User-Agent": "IntelliCrash-dashboard/1.0" },
        body: "data=" + encodeURIComponent(query),
        signal: ctrl.signal,
        cache: "no-store",
      });
      if (!r.ok) throw new Error(`Overpass ${r.status}`);
      return await r.json();
    } catch (e) {
      lastErr = e;
    } finally {
      clearTimeout(timer);
    }
  }
  throw lastErr || new Error("Overpass unreachable");
}

export async function GET(req) {
  const user = getUserBySession(await readSessionToken());
  if (!user) return NextResponse.json({ error: "Not signed in." }, { status: 401, headers: NO_STORE });

  const sp = new URL(req.url).searchParams;
  const lat = Number(sp.get("lat")), lon = Number(sp.get("lon"));
  if (!Number.isFinite(lat) || !Number.isFinite(lon) || Math.abs(lat) > 90 || Math.abs(lon) > 180) {
    return NextResponse.json({ error: "lat and lon are required." }, { status: 400, headers: NO_STORE });
  }

  const key = `${lat.toFixed(3)},${lon.toFixed(3)}`;
  const hit = cache.get(key);
  if (hit && Date.now() - hit.t < TTL_MS) return NextResponse.json(hit.data, { headers: NO_STORE });

  try {
    let hospitals = [], radius = 0;
    for (const r of RADII_M) {
      radius = r;
      hospitals = nearestFromOverpass(await overpass(overpassQuery(lat, lon, r)), lat, lon, 5);
      if (hospitals.length >= WANT) break;
    }
    const data = { origin: { lat, lon }, radius_km: radius / 1000, source: "OpenStreetMap (Overpass API)", hospitals };
    cache.set(key, { t: Date.now(), data });
    return NextResponse.json(data, { headers: NO_STORE });
  } catch (e) {
    console.error("[hospitals]", e);
    return NextResponse.json({ error: "Could not reach the hospital lookup service (the dashboard server needs internet)." }, { status: 502, headers: NO_STORE });
  }
}
