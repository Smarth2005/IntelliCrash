import { NextResponse } from "next/server";
import { deleteSession } from "../_db";
import { readSessionToken, clearSessionCookie } from "../_cookie";

export const runtime = "nodejs";

export async function POST() {
  deleteSession(await readSessionToken());
  await clearSessionCookie();
  return NextResponse.json({ ok: true });
}
