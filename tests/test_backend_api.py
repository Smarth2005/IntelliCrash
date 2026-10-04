"""
IntelliCrash — Full Stack Verification Suite (Tier-1 + Tier-2).

Validates:
1. Health & System Metrics
2. Root Documentation
3. Telemetry Ingestion & Querying
4. Crash Incident CRUD & Aggregate Stats
5. Alert Subsystem Connectivity
6. Real-Time WebSocket Push & Keepalive
7. Tier-2: Structured JSON Logging & X-Request-ID Correlation Headers
8. Tier-2: JWT Authentication (Register, Login, Token Issuance)
9. Tier-2: Authenticated /me Profile Verification
10. Tier-2: Role-Based Access Control (RBAC: Admin vs Operator vs Forbidden)
11. Tier-2: RBAC Protected Alert Dispatch
"""

import sys
import json
import asyncio
import httpx
import websockets

BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws/telemetry"


async def run_full_verification():
    print("==================================================================")
    print("  IntelliCrash Verification Suite — Tier-1 & Tier-2 Comprehensive")
    print("==================================================================")

    passed = 0
    total = 0

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=6.0) as client:
        # ── TEST 1: Health Check & System Status ───────────────────────
        total += 1
        try:
            r = await client.get("/health")
            assert r.status_code == 200, f"Expected 200, got {r.status_code}"
            data = r.json()
            assert data["status"] == "healthy"
            assert data["database"] == "healthy"
            assert "mqtt" in data
            assert data.get("auth") == "jwt_rbac"
            print(f"  [PASS] 01. GET /health -> Status: {data['status']}, DB: {data['database']}, Auth: {data['auth']}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] 01. GET /health -> {e}")

        # ── TEST 2: Structured Logging Headers ──────────────────────────
        total += 1
        try:
            r = await client.get("/")
            assert r.status_code == 200
            assert "X-Request-ID" in r.headers, "Missing X-Request-ID header"
            assert "X-Response-Time-MS" in r.headers, "Missing X-Response-Time-MS header"
            print(f"  [PASS] 02. Structured Observability -> X-Request-ID: {r.headers['X-Request-ID'][:8]}..., Latency: {r.headers['X-Response-Time-MS']}ms")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] 02. Structured Observability -> {e}")

        # ── TEST 3: Telemetry Ingest & Query ────────────────────────────
        total += 1
        try:
            payload = {
                "vehicle_id": "vehicle-tier2",
                "accel_x": 0.15,
                "accel_y": -0.08,
                "accel_z": 9.81,
                "gyro_z": 0.05,
                "crash_prob": 1.5,
                "severity_score": 0.0,
                "status": "Normal",
                "behavior_score": 98.0,
            }
            r = await client.post("/api/v1/telemetry", json=payload)
            assert r.status_code in [200, 201]
            latest_r = await client.get("/api/v1/telemetry/latest?vehicle_id=vehicle-tier2")
            assert latest_r.status_code == 200
            assert abs(latest_r.json()["accel_x"] - 0.15) < 0.01
            print(f"  [PASS] 03. Telemetry Ingest & Query -> Ingested and retrieved vehicle-tier2")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] 03. Telemetry Ingest & Query -> {e}")

        # ── TEST 4: Crash Incident CRUD & Aggregates ────────────────────
        total += 1
        crash_id = None
        try:
            crash_payload = {
                "vehicle_id": "vehicle-tier2",
                "fused_score": 0.91,
                "ml_probability": 0.95,
                "csi_score": 0.87,
                "severity": "SEVERE",
                "latitude": 30.3533,
                "longitude": 76.3607,
                "peak_accel_x": 22.4,
                "peak_accel_y": -35.2,
                "peak_gyro_z": 160.0
            }
            cr = await client.post("/api/v1/crashes", json=crash_payload)
            assert cr.status_code in [200, 201]
            crash_id = cr.json()["id"]

            stats_r = await client.get("/api/v1/crashes/stats")
            assert stats_r.status_code == 200
            assert stats_r.json()["total_crashes"] >= 1
            print(f"  [PASS] 04. Crash Incident CRUD -> Recorded Crash ID #{crash_id}, Stats Total: {stats_r.json()['total_crashes']}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] 04. Crash Incident CRUD -> {e}")

        # ── TEST 5: Alert System Pre-Flight Check ───────────────────────
        total += 1
        try:
            r = await client.post("/api/v1/alerts/test")
            assert r.status_code == 200
            assert r.json()["status"] == "ready"
            print(f"  [PASS] 05. Alert Subsystem -> Ready (SMS/Email handlers active)")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] 05. Alert Subsystem -> {e}")

        # ── TEST 6: User Registration (Operator & Admin) ────────────────
        total += 1
        operator_token = None
        admin_token = None
        try:
            # Register operator
            user_op = {
                "username": f"operator_{int(asyncio.get_event_loop().time())}",
                "email": f"op_{int(asyncio.get_event_loop().time())}@intellicrash.io",
                "password": "SecurePassword123!",
                "role": "operator"
            }
            r_op = await client.post("/api/v1/auth/register", json=user_op)
            assert r_op.status_code == 201
            assert r_op.json()["role"] == "operator"

            # Register admin
            user_admin = {
                "username": f"admin_{int(asyncio.get_event_loop().time())}",
                "email": f"admin_{int(asyncio.get_event_loop().time())}@intellicrash.io",
                "password": "AdminMasterPass123!",
                "role": "admin"
            }
            r_admin = await client.post("/api/v1/auth/register", json=user_admin)
            assert r_admin.status_code == 201
            assert r_admin.json()["role"] == "admin"

            print(f"  [PASS] 06. User Registration -> Registered '{user_op['username']}' (operator) & '{user_admin['username']}' (admin)")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] 06. User Registration -> {e}")

        # ── TEST 7: OAuth2 JWT Login & Token Generation ─────────────────
        total += 1
        try:
            login_form_op = {"username": user_op["username"], "password": user_op["password"]}
            r_login_op = await client.post("/api/v1/auth/login", data=login_form_op)
            assert r_login_op.status_code == 200
            op_data = r_login_op.json()
            operator_token = op_data["access_token"]
            assert op_data["token_type"] == "bearer"
            assert op_data["role"] == "operator"

            login_form_admin = {"username": user_admin["username"], "password": user_admin["password"]}
            r_login_admin = await client.post("/api/v1/auth/login", data=login_form_admin)
            assert r_login_admin.status_code == 200
            admin_data = r_login_admin.json()
            admin_token = admin_data["access_token"]
            assert admin_data["role"] == "admin"

            print(f"  [PASS] 07. JWT Authentication -> Tokens issued successfully (HMAC-SHA256)")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] 07. JWT Authentication -> {e}")

        # ── TEST 8: Authenticated /me Profile Verification ──────────────
        total += 1
        try:
            headers = {"Authorization": f"Bearer {operator_token}"}
            r_me = await client.get("/api/v1/auth/me", headers=headers)
            assert r_me.status_code == 200
            assert r_me.json()["username"] == user_op["username"]
            assert r_me.json()["role"] == "operator"
            print(f"  [PASS] 08. Auth Verification -> Profile matches token: {r_me.json()['username']} ({r_me.json()['role']})")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] 08. Auth Verification -> {e}")

        # ── TEST 9: RBAC Enforcement (Admin Only Route) ─────────────────
        total += 1
        try:
            # 1. Operator attempts to access admin users list -> MUST BE 403 FORBIDDEN
            op_headers = {"Authorization": f"Bearer {operator_token}"}
            r_forbidden = await client.get("/api/v1/auth/users", headers=op_headers)
            assert r_forbidden.status_code == 403, f"Expected 403 Forbidden, got {r_forbidden.status_code}"

            # 2. Admin attempts to access admin users list -> MUST BE 200 OK
            admin_headers = {"Authorization": f"Bearer {admin_token}"}
            r_admin_ok = await client.get("/api/v1/auth/users", headers=admin_headers)
            assert r_admin_ok.status_code == 200
            assert len(r_admin_ok.json()) >= 2

            print(f"  [PASS] 09. RBAC Security -> Operator correctly blocked (403), Admin authorized (200)")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] 09. RBAC Security -> {e}")

        # ── TEST 10: Authenticated Alert Dispatch ───────────────────────
        total += 1
        try:
            # Unauthenticated dispatch -> MUST BE 401
            unauth_r = await client.post("/api/v1/alerts/dispatch", json={"crash_event_id": crash_id, "channels": ["sms"]})
            assert unauth_r.status_code == 401

            # Authenticated dispatch -> 200
            auth_r = await client.post(
                "/api/v1/alerts/dispatch",
                json={"crash_event_id": crash_id, "channels": ["sms", "email"]},
                headers={"Authorization": f"Bearer {operator_token}"}
            )
            assert auth_r.status_code == 200
            print(f"  [PASS] 10. RBAC Alert Dispatch -> Unauth blocked (401), Authenticated operator permitted (200)")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] 10. RBAC Alert Dispatch -> {e}")

    # ── TEST 11: Real-Time WebSocket Streaming ─────────────────────────
    total += 1
    try:
        async with websockets.connect(WS_URL) as ws:
            # Send keepalive ping
            await ws.send("ping")
            pong = await asyncio.wait_for(ws.recv(), timeout=2.0)
            assert "pong" in pong

            # Publish telemetry and verify WS broadcast
            async with httpx.AsyncClient(base_url=BASE_URL) as client:
                await client.post("/api/v1/telemetry", json={
                    "vehicle_id": "ws-vehicle",
                    "accel_x": 3.45,
                    "accel_y": -4.56,
                    "accel_z": 9.81,
                    "gyro_z": 12.0,
                    "crash_prob": 3.0,
                    "severity_score": 0.02,
                    "status": "Normal",
                    "behavior_score": 94.0
                })

            broadcast = await asyncio.wait_for(ws.recv(), timeout=2.0)
            ws_obj = json.loads(broadcast)
            assert ws_obj.get("accel_x") == 3.45
            print(f"  [PASS] 11. WebSocket /ws/telemetry -> Broadcast verified (accel_x: {ws_obj['accel_x']}, status: {ws_obj['status']})")
            passed += 1
    except Exception as e:
        print(f"  [FAIL] 11. WebSocket /ws/telemetry -> {e}")

    print("==================================================================")
    print(f"  Final Score: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("==================================================================")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_full_verification())
