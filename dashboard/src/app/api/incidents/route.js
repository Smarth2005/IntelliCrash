// Incident dossiers + alert log for the dashboard, read from the crash_events / alerts tables.
import { NextResponse } from "next/server";
import { getUserBySession, listIncidentsAndAlerts, listDispatchBoard, dbStats } from "../auth/_db";
import { readSessionToken } from "../auth/_cookie";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
const NO_STORE = { "Cache-Control": "no-store" };

export async function GET() {
  const user = getUserBySession(await readSessionToken());
  if (!user) return NextResponse.json({ error: "Not signed in." }, { status: 401, headers: NO_STORE });
  return NextResponse.json({ ...listIncidentsAndAlerts(), ...listDispatchBoard(), stats: dbStats() }, { headers: NO_STORE });
}
