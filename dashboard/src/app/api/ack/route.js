// Hospital acknowledgement page. The link in the hospital's e-mail carries a one-time random token (that token is
// the credential - no login). GET shows a small confirm page; POST records Accept / Cannot serve (+ ambulance ETA).
import { NextResponse } from "next/server";
import { getDispatchByToken, ackByToken } from "../auth/_db";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const page = (body, status = 200) =>
  new NextResponse(
    `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IntelliCrash - ambulance request</title>
<style>body{font-family:system-ui,Segoe UI,Arial;background:#f3f4f6;margin:0;padding:16px;color:#111827}.c{max-width:480px;margin:24px auto;background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:22px}
h1{font-size:20px;margin:0 0 6px}.b{display:inline-block;background:#fee2e2;color:#b91c1c;border-radius:99px;padding:3px 10px;font-size:11px;font-weight:700}
input{width:100%;padding:10px;border:1px solid #d1d5db;border-radius:8px;font-size:16px;box-sizing:border-box}button{width:100%;padding:13px;border:0;border-radius:10px;font-size:16px;font-weight:700;color:#fff;margin-top:10px;cursor:pointer}
.g{background:#16a34a}.r{background:#6b7280}.ok{color:#15803d;font-weight:700}.m{color:#6b7280;font-size:13px}</style></head><body><div class="c">${body}</div></body></html>`,
    { status, headers: { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store", "X-Robots-Tag": "noindex" } }
  );

function formPage(d, token) {
  return page(`<span class="b">AMBULANCE REQUEST</span><h1>${esc(d.hospital_name)}</h1>
<p>IntelliCrash detected a crash and the driver did not confirm they are safe. Can you dispatch an ambulance?</p>
<p class="m">Distance from crash: ${d.distance_km != null ? esc(d.distance_km) + " km" : "n/a"} · Requested ${esc(String(d.requested_at || "").replace("T", " "))}</p>
<form method="POST"><input type="hidden" name="token" value="${esc(token)}">
<label>Ambulance ETA (minutes)<input name="eta" type="number" min="1" max="180" value="10"></label>
<button class="g" name="action" value="accept">Accept - ambulance dispatched</button>
<button class="r" name="action" value="decline">Cannot serve</button></form>
<p class="m">Your answer is sent to the emergency contact immediately. If you cannot serve, the next nearest hospital is requested.</p>`);
}

export async function GET(req) {
  const token = new URL(req.url).searchParams.get("token");
  const d = getDispatchByToken(token);
  if (!d) return page("<h1>Link not valid</h1><p>This request link is unknown or has expired.</p>", 404);
  if (d.status === "ACCEPTED") return page(`<h1 class="ok">Already accepted</h1><p>${esc(d.hospital_name)} accepted${d.eta_min ? ` with ETA ${esc(d.eta_min)} min` : ""}. Thank you.</p>`);
  if (d.status === "DECLINED") return page(`<h1>Already answered</h1><p>This request was marked "cannot serve".</p>`);
  return formPage(d, token);
}

export async function POST(req) {
  let token, action, eta;
  try { const f = await req.formData(); token = f.get("token"); action = f.get("action"); eta = f.get("eta"); } catch { return page("<h1>Bad request</h1>", 400); }
  if (action !== "accept" && action !== "decline") return page("<h1>Bad request</h1>", 400);
  const r = ackByToken(token, action, eta);
  if (r.error) return page("<h1>Link not valid</h1>", 404);
  if (r.already) return page(`<h1>Already answered</h1><p>This request was already ${esc(r.dispatch.status.toLowerCase())}.</p>`);
  const d = r.dispatch;
  return d.status === "ACCEPTED"
    ? page(`<h1 class="ok">Thank you - recorded</h1><p>${esc(d.hospital_name)} will dispatch an ambulance${d.eta_min ? `, ETA ${esc(d.eta_min)} min` : ""}. The emergency contact has been informed.</p>`)
    : page(`<h1>Recorded</h1><p>You marked "cannot serve". The next nearest hospital will be requested.</p>`);
}
