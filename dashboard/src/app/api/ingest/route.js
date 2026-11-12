// Pi -> dashboard server sync. Called by Raspberry/dashboard_bridge.py (server-to-server, not the browser).
// Protected by a shared secret header:  x-ingest-key  ==  INGEST_KEY (env, default "intellicrash-demo-key").
import { NextResponse } from "next/server";
import crypto from "node:crypto";
import { ingestBatch } from "../auth/_db";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
const NO_STORE = { "Cache-Control": "no-store" };

function keyOk(got) {
  const want = Buffer.from(process.env.INGEST_KEY || "intellicrash-demo-key");
  const have = Buffer.from(String(got || ""));
  return have.length === want.length && crypto.timingSafeEqual(have, want);
}

export async function POST(req) {
  if (!keyOk(req.headers.get("x-ingest-key"))) {
    return NextResponse.json({ error: "Bad ingest key." }, { status: 401, headers: NO_STORE });
  }
  let body;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON." }, { status: 400, headers: NO_STORE });
  }
  try {
    return NextResponse.json({ ok: true, ack: ingestBatch(body) }, { headers: NO_STORE });
  } catch (e) {
    console.error("[ingest]", e);
    return NextResponse.json({ error: "Could not store batch." }, { status: 500, headers: NO_STORE });
  }
}
