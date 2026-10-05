"""
IntelliCrash — FastAPI Application Entry Point.

This is the main FastAPI server that replaces the prototype http.server
telemetry server. It provides:

    REST API:
        /api/v1/telemetry   → Ingest & query telemetry data
        /api/v1/crashes     → Crash event CRUD & statistics
        /api/v1/alerts      → Emergency alert dispatch (SMS/Email)

    WebSocket:
        /ws/telemetry       → Real-time telemetry push to dashboard

    Auto-Docs:
        /docs               → Swagger UI (interactive API explorer)
        /redoc              → ReDoc (alternative API docs)

    Health:
        /health             → System health check

Run:
    uvicorn src.backend.main:app --reload --port 8000
"""

import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.backend.config import get_settings
from src.backend.database import get_db
from src.backend.ws.manager import ws_manager
from src.backend.routers import telemetry, crashes, alerts, auth, video
from src.backend.services.firebase_service import sync_pending_events
from src.backend.services.mqtt_service import mqtt_service
from src.backend.middleware.logging_middleware import StructuredLoggingMiddleware

# ── Logging Setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("intellicrash")

# ── Track Server Uptime ─────────────────────────────────────────────────────

_start_time = time.time()


# ── Lifespan (Startup/Shutdown) ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    - Startup: Initialize database, sync pending Firebase events, start MQTT bridge
    - Shutdown: Clean up resources, stop MQTT loop
    """
    settings = get_settings()

    # ── Startup ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"  {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"  Database: {settings.DATABASE_PATH}")
    logger.info(f"  Firebase: {'Enabled' if settings.FIREBASE_ENABLED else 'Disabled'}")
    logger.info(f"  MQTT Broker: {settings.MQTT_BROKER_HOST}:{settings.MQTT_BROKER_PORT} ({'Enabled' if settings.MQTT_ENABLED else 'Disabled'})")
    logger.info(f"  Swagger UI: http://localhost:8000/docs")
    logger.info("=" * 60)

    # Initialize the database (creates tables if needed)
    get_db()

    # Start MQTT background bridge
    mqtt_service.start()

    # Sync any crash events that failed to sync to Firebase previously
    if settings.FIREBASE_ENABLED:
        synced = sync_pending_events()
        if synced > 0:
            logger.info(f"[Startup] Synced {synced} pending crash events to Firebase")

    yield  # ← App is running

    # ── Shutdown ─────────────────────────────────────────────────────────
    mqtt_service.stop()
    logger.info("[Shutdown] IntelliCrash API shutting down gracefully.")


# ── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="IntelliCrash API",
    description=(
        "AI-IoT Vehicle Crash Detection System — Enterprise Backend API.\n\n"
        "Provides REST endpoints for telemetry ingestion, crash event management, "
        "emergency alert dispatch, JWT authentication (RBAC), and MQTT IoT bridge. "
        "Includes WebSocket for zero-latency telemetry streaming."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middlewares ─────────────────────────────────────────────────────────────

# Structured JSON logging & request correlation ID injection
app.add_middleware(StructuredLoggingMiddleware)

# CORS Middleware
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount Routers ────────────────────────────────────────────────────────────

app.include_router(auth.router)
app.include_router(telemetry.router)
app.include_router(crashes.router)
app.include_router(alerts.router)
app.include_router(video.router)


# ── WebSocket Endpoint ───────────────────────────────────────────────────────

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """
    WebSocket endpoint for real-time telemetry streaming.

    The dashboard connects here and receives live telemetry updates
    pushed by the server whenever new data arrives via POST /api/v1/telemetry.

    Protocol:
        1. Client connects to ws://localhost:8000/ws/telemetry
        2. Server accepts and adds to connection pool
        3. Server pushes JSON telemetry on every POST /api/v1/telemetry
        4. Client can send "ping" messages to keep connection alive
        5. On disconnect, server removes from pool
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            # Listen for client messages (keepalive pings, etc.)
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text('{"type": "pong"}')
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info("[WS] Dashboard client disconnected")


# ── Health Check ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"], summary="System health check")
def health_check():
    """
    Returns the current health status of the IntelliCrash API.

    Checks:
    - Application is running
    - Database is accessible
    - Firebase connection status
    - Active WebSocket connections
    - Server uptime
    """
    settings = get_settings()
    uptime = time.time() - _start_time

    # Quick DB check
    db_status = "healthy"
    try:
        db = get_db()
        with db.get_connection() as conn:
            conn.execute("SELECT 1")
    except Exception:
        db_status = "unhealthy"

    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "database": db_status,
        "firebase": "enabled" if settings.FIREBASE_ENABLED else "disabled",
        "mqtt": "connected" if mqtt_service._is_connected else ("enabled" if settings.MQTT_ENABLED else "disabled"),
        "auth": "jwt_rbac",
        "websocket_connections": ws_manager.connection_count,
        "uptime_seconds": round(uptime, 1),
    }


# ── Root ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["System"], summary="API root")
def root():
    """API root — returns basic info and links to documentation."""
    return {
        "app": "IntelliCrash API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "auth": "/api/v1/auth",
            "telemetry": "/api/v1/telemetry",
            "crashes": "/api/v1/crashes",
            "alerts": "/api/v1/alerts",
            "websocket": "ws://localhost:8000/ws/telemetry",
        },
    }
