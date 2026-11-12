import { NextResponse } from "next/server";
import { registerUser, createSession } from "../_db";
import { setSessionCookie } from "../_cookie";

export const runtime = "nodejs";

export async function POST(req) {
  let body;
  try { body = await req.json(); } catch { return NextResponse.json({ error: "Invalid request." }, { status: 400 }); }
  const result = registerUser(body);
  if (result.error) return NextResponse.json({ error: result.error }, { status: result.status });
  const { token, expires } = createSession(result.user.id);
  await setSessionCookie(token, expires);
  return NextResponse.json({ user: result.user }, { status: 201 });
}
