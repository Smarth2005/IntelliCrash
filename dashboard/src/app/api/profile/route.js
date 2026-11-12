import { NextResponse } from "next/server";
import { getUserBySession, getOwnerProfileFor, saveOwnerProfile } from "../auth/_db";
import { readSessionToken } from "../auth/_cookie";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
const NO_STORE = { "Cache-Control": "no-store" };

export async function GET() {
  const user = getUserBySession(await readSessionToken());
  if (!user) return NextResponse.json({ error: "Not signed in." }, { status: 401, headers: NO_STORE });
  return NextResponse.json(getOwnerProfileFor(user), { headers: NO_STORE });
}

export async function PUT(req) {
  const user = getUserBySession(await readSessionToken());
  if (!user) return NextResponse.json({ error: "Not signed in." }, { status: 401, headers: NO_STORE });
  let body;
  try { body = await req.json(); } catch { return NextResponse.json({ error: "Invalid request." }, { status: 400 }); }
  const result = saveOwnerProfile(user, body);
  if (result.error) return NextResponse.json({ error: result.error }, { status: result.status });
  return NextResponse.json(getOwnerProfileFor(user), { headers: NO_STORE });
}
