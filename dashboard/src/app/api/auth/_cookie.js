import { cookies } from "next/headers";

export const COOKIE = "ic_session";

export async function setSessionCookie(token, expires) {
  const jar = await cookies();
  jar.set(COOKIE, token, {
    httpOnly: true,
    sameSite: "lax",
    path: "/",
    expires: new Date(expires),
    secure: false, // local demo over http://localhost
  });
}
export async function readSessionToken() {
  const jar = await cookies();
  return jar.get(COOKIE)?.value ?? null;
}
export async function clearSessionCookie() {
  const jar = await cookies();
  jar.set(COOKIE, "", { httpOnly: true, sameSite: "lax", path: "/", maxAge: 0 });
}
