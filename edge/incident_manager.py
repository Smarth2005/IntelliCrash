"""
IntelliCrash - closed-loop incident manager
===========================================
crash detected
   -> driver gets a Telegram prompt with an "I'm OK" button (DRIVER_RESPONSE_SECONDS, default 60)
        - taps "I'm OK"            -> nothing is sent to any hospital; incident closed as DRIVER_OK
        - taps "I need help"       -> hospital request immediately
        - no answer in 60 s        -> hospital request
   -> nearest hospital is asked (e-mail with Accept / Cannot-serve link) and the EMERGENCY CONTACT is told which
      hospital was requested
        - hospital accepts (+ ETA) -> contact and driver are told "X accepted, ambulance ETA n min"
        - declines / no answer in HOSPITAL_ACK_SECONDS -> next nearest hospital (up to MAX_HOSPITAL_REQUESTS)
        - nobody answers           -> contact is told to call 108 / 112 at once
Every step is stored as an incident event (dashboard table incident_events) so there is a full audit trail.
"""
import html
import json
import math
import os
import secrets
import smtplib
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
YELLOW, GREEN, RED, RESET = "\033[93m", "\033[92m", "\033[91m", "\033[0m"


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class IncidentManager:
    def __init__(self, dispatcher):
        self.d = dispatcher                       # AlertDispatcher: config, twilio client, blackbox helpers
        self.cfg = dispatcher.config
        self.token = self.cfg["telegram_token"] if dispatcher.telegram_enabled else None
        self.contact_chat = self.cfg["telegram_chat_id"]
        self.driver_chat = os.environ.get("TELEGRAM_DRIVER_CHAT_ID", "").strip()
        self.driver_name = os.environ.get("DRIVER_NAME", "").strip() or "Driver"
        self.driver_phone = os.environ.get("DRIVER_PHONE", "").strip()
        self.driver_blood = os.environ.get("DRIVER_BLOOD_GROUP", "").strip()
        self.contact_name = os.environ.get("EMERGENCY_CONTACT_NAME", "").strip()
        self.vehicle_id = os.environ.get("VEHICLE_ID", "PB-11-AB-2027")
        self.driver_wait = _env_int("DRIVER_RESPONSE_SECONDS", 60)
        self.ack_wait = _env_int("HOSPITAL_ACK_SECONDS", 90)
        self.max_hospitals = _env_int("MAX_HOSPITAL_REQUESTS", 3)
        self.immediate = {s.strip().upper() for s in os.environ.get("IMMEDIATE_DISPATCH_SEVERITIES", "").split(",") if s.strip()}
        self.notify_ok = os.environ.get("NOTIFY_CONTACT_WHEN_DRIVER_OK", "0") == "1"
        self.call_driver = os.environ.get("DRIVER_CALL_ENABLED", "0") == "1"
        self.demo_loc = self._parse_loc(os.environ.get("DEMO_LOCATION", ""))
        self.hospitals = self._load_hospitals()
        self._pending = {}                        # incident key -> {"ev": Event, "answer": "ok"|"help"|None, ...}
        self._lock = threading.Lock()
        self._offset = 0
        if self.token:
            threading.Thread(target=self._poll_telegram, daemon=True).start()
            if not self.driver_chat:
                print(f"{YELLOW}[INCIDENT] TELEGRAM_DRIVER_CHAT_ID not set: send /start to the bot from the driver's phone to get it. "
                      f"Until then a crash goes straight to the hospital (fail-safe).{RESET}")
        else:
            print(f"{YELLOW}[INCIDENT] Telegram disabled: no driver prompt possible, crashes go straight to the hospital.{RESET}")

    # ── configuration helpers ────────────────────────────────────────────
    @staticmethod
    def _parse_loc(s):
        try:
            a, b = [float(x) for x in s.split(",")]
            return (a, b)
        except Exception:
            return None

    def _load_hospitals(self):
        try:
            data = json.load(open(HERE / "hospitals.json", encoding="utf-8"))["hospitals"]
        except Exception as e:
            print(f"{RED}[INCIDENT] hospitals.json unreadable: {e}{RESET}")
            return []
        for h in data:
            h["email"] = (h.get("email") or "").strip() or self.cfg["receiver_email"]
        return data

    def _rank(self, pos):
        hs = [dict(h) for h in self.hospitals]
        if pos:
            for h in hs:
                h["distance_km"] = round(_km(pos[0], pos[1], h["lat"], h["lon"]), 2)
            hs.sort(key=lambda h: h["distance_km"])
        else:
            for h in hs:
                h["distance_km"] = None
        return hs

    # ── dashboard server (events, hospital requests, acknowledgements) ───
    def _dash(self, bridge):
        url = (os.environ.get("DASHBOARD_PUBLIC_URL") or "").rstrip("/")
        api = (getattr(bridge, "dashboard_url", "") or os.environ.get("INTELLICRASH_DASHBOARD_URL", "")).rstrip("/")
        key = getattr(bridge, "ingest_key", None) or os.environ.get("INTELLICRASH_INGEST_KEY") or "intellicrash-demo-key"
        return (url or api), api, key

    def _post(self, inc, payload):
        _, api, key = self._dash(inc["bridge"])
        if not api:
            return False
        try:
            req = urllib.request.Request(api + "/api/dispatch", data=json.dumps(payload).encode(), method="POST",
                                         headers={"Content-Type": "application/json", "x-ingest-key": key})
            urllib.request.urlopen(req, timeout=6).read()
            return True
        except Exception as e:
            print(f"{YELLOW}[INCIDENT] dashboard unreachable ({e}); event kept locally.{RESET}")
            return False

    def _event(self, inc, actor, event, detail=""):
        e = {"session_id": inc["session"], "crash_edge_id": inc["crash_id"], "ts": _now(), "actor": actor, "event": event, "detail": detail}
        inc["events"].append(e)
        print(f"  [INCIDENT {inc['key']}] {actor}: {event} {detail}", flush=True)
        self._post(inc, {"events": [e]})

    def _poll_status(self, inc):
        _, api, key = self._dash(inc["bridge"])
        if not api:
            return {}
        try:
            q = urllib.parse.urlencode({"session": inc["session"], "crash": inc["crash_id"]})
            req = urllib.request.Request(f"{api}/api/dispatch?{q}", headers={"x-ingest-key": key})
            rows = json.load(urllib.request.urlopen(req, timeout=6)).get("dispatches", [])
            return {r["token"]: r for r in rows}
        except Exception:
            return {}

    def _log_alert(self, inc, channel, ok, err=None):
        b = inc["bridge"]
        if b is not None and inc["crash_id"] is not None:
            try:
                b.log_alert(inc["crash_id"], channel, "DISPATCHED" if ok else "FAILED", None if ok else str(err)[:200])
            except Exception:
                pass

    # ── Telegram ─────────────────────────────────────────────────────────
    def _tg(self, method, **payload):
        if not self.token:
            return None
        try:
            r = requests.post(f"https://api.telegram.org/bot{self.token}/{method}", json=payload, timeout=10)
            return r.json() if r.status_code == 200 else None
        except Exception as e:
            print(f"{RED}[Telegram {method} ERROR] {e}{RESET}")
            return None

    def _send(self, chat, text, buttons=None):
        if not chat:
            return None
        p = {"chat_id": chat, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
        if buttons:
            p["reply_markup"] = {"inline_keyboard": buttons}
        r = self._tg("sendMessage", **p)
        return (r or {}).get("result", {}).get("message_id") if r and r.get("ok") else None

    def _poll_telegram(self):
        self._tg("deleteWebhook")                 # getUpdates does not work while a webhook is set
        while True:
            try:
                r = requests.get(f"https://api.telegram.org/bot{self.token}/getUpdates",
                                 params={"timeout": 25, "offset": self._offset, "allowed_updates": json.dumps(["message", "callback_query"])},
                                 timeout=35).json()
                for u in r.get("result", []):
                    self._offset = u["update_id"] + 1
                    self._handle_update(u)
            except Exception:
                time.sleep(3)

    def _handle_update(self, u):
        cb = u.get("callback_query")
        if cb:
            data = cb.get("data", "")
            who = str(cb.get("from", {}).get("id", ""))
            action, _, key = data.partition(":")
            with self._lock:
                p = self._pending.get(key)
            if not p or action not in ("ok", "help"):
                self._tg("answerCallbackQuery", callback_query_id=cb["id"], text="This alert has already been handled.")
                return
            if self.driver_chat and who != self.driver_chat:
                self._tg("answerCallbackQuery", callback_query_id=cb["id"], text="Only the driver can answer this.")
                return
            if p["answer"] is None:
                p["answer"] = action
                p["ev"].set()
            self._tg("answerCallbackQuery", callback_query_id=cb["id"], text="Recorded")
            return
        msg = u.get("message") or {}
        if (msg.get("text") or "").startswith("/start"):
            cid = msg["chat"]["id"]
            self._send(cid, f"IntelliCrash bot ready.\nYour Telegram chat id is <code>{cid}</code>.\n"
                            "Put it in .env as TELEGRAM_DRIVER_CHAT_ID (driver) or TELEGRAM_CHAT_ID (emergency contact).")

    # ── public entry ─────────────────────────────────────────────────────
    def start(self, crash_id, severity, peak_accel, p_final, csi, p_lstm, gps, video_path, bridge=None):
        pos, demo = (tuple(gps), False) if gps else (self.demo_loc, True)
        inc = {"key": f"{getattr(bridge, 'session_id', 'local')}-{crash_id}", "session": getattr(bridge, "session_id", "local"),
               "crash_id": crash_id if crash_id is not None else int(time.time()), "bridge": bridge, "severity": severity, "peak": peak_accel,
               "p_final": p_final, "csi": csi, "video": video_path, "pos": pos, "demo_pos": demo and pos is not None,
               "gps": gps, "events": [], "t0": _now(), "outcome": None, "crash_row": crash_id}
        threading.Thread(target=self._run, args=(inc,), daemon=True).start()
        return inc["key"]

    # ── core flow ────────────────────────────────────────────────────────
    def _where(self, inc):
        if not inc["pos"]:
            return "location unavailable (no GPS fix)", None
        link = f"https://maps.google.com/?q={inc['pos'][0]},{inc['pos'][1]}"
        tag = " (DEMO position, not a GPS reading)" if inc["demo_pos"] else ""
        return f"{inc['pos'][0]:.6f}, {inc['pos'][1]:.6f}{tag}", link

    def _summary_html(self, inc):
        where, link = self._where(inc)
        return (f"<b>Vehicle:</b> {html.escape(self.vehicle_id)}\n<b>Severity:</b> {inc['severity']} · peak {inc['peak']:.1f} m/s² · "
                f"crash probability {inc['p_final']:.0%}\n<b>Time:</b> {inc['t0'].replace('T', ' ')}\n<b>Location:</b> {where}"
                + (f"\n<a href=\"{link}\">Open in Google Maps</a>" if link else ""))

    def _run(self, inc):
        try:
            self._event(inc, "system", "CRASH_DETECTED", f"{inc['severity']} · peak {inc['peak']:.1f} m/s² · fused {inc['p_final']:.2f} · CSI {inc['csi']:.2f}")
            self.d.save_to_local_blackbox({"severity": inc["severity"], "p_final": inc["p_final"], "csi": inc["csi"], "peak": inc["peak"],
                                           "gps": inc["gps"], "video": str(inc["video"]) if inc["video"] else None, "incident": inc["key"]})
            answer = self._ask_driver(inc)
            if answer == "ok":
                self._event(inc, "driver", "DRIVER_OK", "driver confirmed safe; no hospital contacted")
                inc["outcome"] = "DRIVER_OK"
                if self.notify_ok:
                    self._send(self.contact_chat, f"ℹ️ A crash alert for {html.escape(self.vehicle_id)} was cancelled: the driver confirmed they are OK.")
            else:
                self._event(inc, "driver" if answer == "help" else "system",
                            "DRIVER_REQUESTED_HELP" if answer == "help" else "NO_DRIVER_RESPONSE",
                            "driver asked for help" if answer == "help" else f"no answer within {self.driver_wait} s")
                self._escalate(inc)
        except Exception as e:
            print(f"{RED}[INCIDENT ERROR] {e}{RESET}")
        finally:
            with self._lock:
                self._pending.pop(inc["key"], None)
            try:
                (self.d.blackbox_dir / f"incident_{inc['key']}.json").write_text(json.dumps({"incident": inc["key"], "outcome": inc["outcome"], "events": inc["events"]}, indent=2), encoding="utf-8")
            except Exception:
                pass

    def _ask_driver(self, inc):
        """Returns 'ok', 'help' or None (no answer / prompt could not be delivered)."""
        if inc["severity"] in self.immediate:
            self._event(inc, "system", "IMMEDIATE_DISPATCH", f"{inc['severity']} skips the driver check")
            return None
        if not (self.token and self.driver_chat):
            self._event(inc, "system", "DRIVER_PROMPT_UNAVAILABLE", "no driver Telegram chat configured; escalating (fail-safe)")
            return None
        key = inc["key"]
        p = {"ev": threading.Event(), "answer": None}
        with self._lock:
            self._pending[key] = p
        text = (f"🚨 <b>Crash detected</b>\n{self._summary_html(inc)}\n\nTap <b>I'm OK</b> within <b>{self.driver_wait} seconds</b> "
                f"or an ambulance will be requested.")
        mid = self._send(self.driver_chat, text, [[{"text": "✅ I'm OK", "callback_data": f"ok:{key}"}],
                                                  [{"text": "🚑 I need help", "callback_data": f"help:{key}"}]])
        self._log_alert(inc, "Telegram · driver prompt", mid is not None, "send failed")
        if mid is None:
            self._event(inc, "system", "DRIVER_PROMPT_FAILED", "Telegram delivery failed; escalating (fail-safe)")
            return None
        self._event(inc, "system", "DRIVER_PROMPTED", f"Telegram prompt sent; waiting {self.driver_wait} s")
        if self.call_driver and self.d.twilio_enabled and self.driver_phone:
            try:
                self.d.twilio_client.calls.create(
                    twiml=f"<Response><Say voice='alice'>Crash detected. Open Telegram and tap I am OK within {self.driver_wait} seconds.</Say></Response>",
                    from_=self.cfg["twilio_from"], to=self.driver_phone)
            except Exception as e:
                print(f"{YELLOW}[driver call] {e}{RESET}")
        p["ev"].wait(self.driver_wait)
        ans = p["answer"]
        if mid:
            note = {"ok": "✅ Thanks - you confirmed you are OK. No alert was sent.", "help": "🚑 Help requested. Contacting the nearest hospital…"}.get(
                ans, "⏱ No response - contacting the nearest hospital and your emergency contact.")
            self._tg("editMessageText", chat_id=self.driver_chat, message_id=mid, parse_mode="HTML",
                     text=f"🚨 <b>Crash detected</b>\n{self._summary_html(inc)}\n\n{note}")
        return ans

    def _escalate(self, inc):
        ranked = self._rank(inc["pos"])[: self.max_hospitals]
        where, link = self._where(inc)
        base, _, _ = self._dash(inc["bridge"])
        if not ranked:
            self._send(self.contact_chat, f"🚨 <b>CRASH - driver not responding</b>\n{self._summary_html(inc)}\n\nNo hospital list available. Call <b>108</b> / <b>112</b> now.")
            inc["outcome"] = "NO_HOSPITAL_LIST"
            return
        contact_told = False
        for i, h in enumerate(ranked):
            tok = secrets.token_urlsafe(16)
            disp = {"session_id": inc["session"], "crash_edge_id": inc["crash_id"], "rank": i + 1, "hospital_name": h["name"],
                    "hospital_email": h["email"], "lat": h["lat"], "lon": h["lon"], "distance_km": h["distance_km"], "token": tok, "requested_at": _now()}
            self._post(inc, {"dispatches": [disp]})
            ok, err = self._email_hospital(inc, h, tok, base)
            self._log_alert(inc, f"Hospital Email · {h['name']}", ok, err)
            dist = f" ({h['distance_km']} km away)" if h["distance_km"] is not None else ""
            self._event(inc, "system", "HOSPITAL_REQUESTED", f"#{i + 1} {h['name']}{dist}" + ("" if ok else f" - e-mail FAILED: {err}"))
            if not contact_told:
                contact_told = True
                msg = (f"🚨 <b>CRASH - driver did not respond</b>\n{self._summary_html(inc)}\n\n"
                       f"🏥 Ambulance requested from <b>{html.escape(h['name'])}</b>{dist}. Waiting for their confirmation…")
                m = self._send(self.contact_chat, msg)
                self._log_alert(inc, "Telegram · emergency contact", m is not None, "send failed")
                self._sms_contact(inc, h)
                self._event(inc, "system", "CONTACT_NOTIFIED", f"emergency contact told that {h['name']} was requested")
            else:
                self._send(self.contact_chat, f"🏥 Now requesting <b>{html.escape(h['name'])}</b>{dist}.")
            res = self._wait_ack(inc, tok)
            if res and res["status"] == "ACCEPTED":
                eta = f", ambulance ETA <b>{res['eta_min']} min</b>" if res.get("eta_min") else ""
                self._send(self.contact_chat, f"✅ <b>{html.escape(h['name'])} accepted</b> and is dispatching an ambulance{eta}.\n{self._summary_html(inc)}")
                self._send(self.driver_chat, f"✅ {html.escape(h['name'])} is sending an ambulance{eta}. Stay where you are if you can.")
                self._event(inc, "system", "CONTACT_NOTIFIED", f"told contact: {h['name']} accepted")
                inc["outcome"] = "HOSPITAL_ACCEPTED"
                return
            if res and res["status"] == "DECLINED":
                self._send(self.contact_chat, f"⚠️ {html.escape(h['name'])} cannot serve this request.")
            else:
                self._post(inc, {"timeouts": [tok]})
                self._event(inc, "system", "HOSPITAL_NO_RESPONSE", f"{h['name']} did not answer within {self.ack_wait} s")
                self._send(self.contact_chat, f"⏱ {html.escape(h['name'])} did not answer in {self.ack_wait} s.")
        self._send(self.contact_chat, f"🆘 <b>No hospital has confirmed.</b> Call <b>108</b> (ambulance) or <b>112</b> NOW.\n{self._summary_html(inc)}")
        self._send(self.driver_chat, "🆘 No hospital confirmed yet. Call 108 / 112 if you can.")
        self._event(inc, "system", "ESCALATION_EXHAUSTED", "no hospital accepted; contact told to call 108/112")
        inc["outcome"] = "NO_HOSPITAL_ACCEPTED"

    def _wait_ack(self, inc, tok):
        end = time.time() + self.ack_wait
        last = None
        while time.time() < end:
            last = self._poll_status(inc).get(tok) or last
            if last and last["status"] in ("ACCEPTED", "DECLINED"):
                return last
            time.sleep(3)
        return last if last and last["status"] in ("ACCEPTED", "DECLINED") else None

    def _sms_contact(self, inc, h):
        if not self.d.twilio_enabled:
            return
        try:
            where, link = self._where(inc)
            self.d.twilio_client.messages.create(
                body=f"IntelliCrash: {inc['severity']} crash, {self.vehicle_id}. Driver not responding. Ambulance requested from {h['name']}. {link or ''}",
                from_=self.cfg["twilio_from"], to=self.cfg["emergency_phone"])
        except Exception as e:
            print(f"{YELLOW}[SMS] {e}{RESET}")

    def _email_hospital(self, inc, h, tok, base):
        if not self.d.email_enabled:
            return False, "e-mail not configured"
        where, link = self._where(inc)
        acc = f"{base}/api/ack?token={tok}" if base else None
        btn = (f'<p style="text-align:center"><a href="{acc}" style="background:#16a34a;color:#fff;padding:13px 26px;border-radius:8px;'
               f'text-decoration:none;font-weight:700">Respond: accept / cannot serve</a></p>' if acc else
               "<p><i>No dashboard link configured - reply to the emergency contact directly.</i></p>")
        rows = [("Vehicle", self.vehicle_id), ("Patient (driver)", self.driver_name), ("Patient phone", self.driver_phone or "n/a"),
                ("Blood group", self.driver_blood or "not provided"), ("Time", inc["t0"].replace("T", " ")),
                ("Severity", inc["severity"]), ("Peak acceleration", f"{inc['peak']:.1f} m/s²"),
                ("Crash probability / CSI", f"{inc['p_final']:.0%} / {inc['csi']:.2f}"),
                ("Location", where), ("Distance to your hospital", f"{h['distance_km']} km" if h["distance_km"] is not None else "n/a"),
                ("Driver status", f"did not confirm safety within {self.driver_wait} s")]
        trs = "".join(f"<tr><th style='text-align:left;background:#f9fafb;padding:9px;border:1px solid #e5e7eb;width:40%'>{html.escape(a)}</th>"
                      f"<td style='padding:9px;border:1px solid #e5e7eb'>{html.escape(str(b))}</td></tr>" for a, b in rows)
        maps = f'<p style="text-align:center"><a href="{link}">📍 Open crash location in Google Maps</a></p>' if link else ""
        body = (f"<html><body style='font-family:Arial,sans-serif;background:#f3f4f6;padding:14px'><div style='max-width:600px;margin:auto;background:#fff;"
                f"border:1px solid #e5e7eb;border-radius:12px;padding:22px'><span style='background:#fee2e2;color:#b91c1c;padding:4px 11px;border-radius:99px;"
                f"font-weight:700;font-size:11px'>AMBULANCE REQUEST</span><h2>{html.escape(h['name'])}: please dispatch an ambulance</h2>"
                f"<p>IntelliCrash detected a collision and the driver did not respond to the safety check.</p><table style='width:100%;border-collapse:collapse'>{trs}</table>"
                f"{maps}{btn}<p style='font-size:12px;color:#6b7280'>Values are sensor readings from the vehicle unit. If this request is not answered, "
                f"the next nearest hospital is contacted automatically.</p></div></body></html>")
        try:
            msg = MIMEMultipart("mixed")
            msg["From"] = f"IntelliCrash Emergency Dispatch <{self.cfg['sender_email']}>"
            msg["To"] = h["email"]
            msg["Subject"] = f"🚨 AMBULANCE REQUEST - {inc['severity']} crash near you ({self.vehicle_id})"
            msg.attach(MIMEText(body, "html"))
            vp = inc["video"]
            if vp and Path(str(vp)).exists() and Path(str(vp)).stat().st_size < 15_000_000:
                part = MIMEBase("video", "mp4")
                part.set_payload(open(vp, "rb").read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f'attachment; filename="{Path(str(vp)).name}"')
                msg.attach(part)
            s = smtplib.SMTP(self.cfg["smtp_server"], self.cfg["smtp_port"], timeout=20)
            s.starttls()
            s.login(self.cfg["sender_email"], self.cfg["sender_password"])
            s.sendmail(self.cfg["sender_email"], h["email"], msg.as_string())
            s.quit()
            return True, None
        except Exception as e:
            print(f"{RED}[Email ERROR] {e}{RESET}")
            return False, e
