import { NextResponse } from "next/server";
import { getUserBySession } from "../_db";
import { readSessionToken } from "../_cookie";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  const user = getUserBySession(await readSessionToken());
  return NextResponse.json({ user }, { headers: { "Cache-Control": "no-store" } });
}
