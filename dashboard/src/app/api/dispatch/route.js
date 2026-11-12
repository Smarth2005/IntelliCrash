// Pi <-> dashboard server for the closed-loop dispatch. Protected by the same shared secret as /api/ingest.
//  POST: the Pi reports incident events + the hospital requests it just sent (and any it gave up on).
//  GET ?session=&crash= : the Pi polls the hospitals' answers (ACCEPTED / DECLINED / REQUESTED) for one crash.
import { NextResponse } from "next/server";
import crypto from "node:crypto";
import { ingestDispatch, getDispatchesFor } from "../auth/_db";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
const NO_STORE = { "Cache-Control": "no-store" };

function keyOk(got) {
  const want = Buffer.from(process.env.INGEST_KEY || "intellicrash-demo-key");
  const have = Buffer.from(String(got || ""));
  return have.length === want.length && crypto.timingSafeEqual(have, want);
}

export async function POST(req) {
  if (!keyOk(req.headers.get("x-ingest-key"))) return NextResponse.json({ error: "Bad ingest key." }, { status: 401, headers: NO_STORE });
  let body;
  try { body = await req.json(); } catch { return NextResponse.json({ error: "Invalid JSON." }, { status: 400, headers: NO_STORE }); }
  try { return NextResponse.json(ingestDispatch(body), { headers: NO_STORE }); }
  catch (e) { console.error("[dispatch]", e); return NextResponse.json({ error: "Could not store." }, { status: 500, headers: NO_STORE }); }
}

export async function GET(req) {
  if (!keyOk(req.headers.get("x-ingest-key"))) return NextResponse.json({ error: "Bad ingest key." }, { status: 401, headers: NO_STORE });
  const sp = new URL(req.url).searchParams;
  const rows = getDispatchesFor(sp.get("session"), sp.get("crash"));
  return NextResponse.json({ dispatches: rows.map((r) => ({ token: r.token, hospital_name: r.hospital_name, status: r.status, eta_min: r.eta_min })) }, { headers: NO_STORE });
}
