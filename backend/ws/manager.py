"""
IntelliCrash — WebSocket Connection Manager.

Manages persistent WebSocket connections for real-time telemetry
streaming to the Next.js dashboard.

Architecture:
    Edge (RPi4) --POST /telemetry--> FastAPI --broadcast--> [WS Client 1]
                                                         --> [WS Client 2]
                                                         --> [WS Client N]

The manager tracks all active connections and provides both async
and sync broadcast methods (sync is used from REST endpoint threads).
"""

import json
import asyncio
import logging
from typing import List
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("intellicrash.ws")


class ConnectionManager:
    """
    Manages WebSocket connections for real-time telemetry push.

    Usage:
        # In FastAPI WebSocket endpoint
        @app.websocket("/ws/telemetry")
        async def ws_endpoint(websocket: WebSocket):
            await ws_manager.connect(websocket)
            try:
                while True:
                    # Keep connection alive, listen for client pings
                    data = await websocket.receive_text()
            except WebSocketDisconnect:
                ws_manager.disconnect(websocket)

        # From a REST endpoint (sync context)
        ws_manager.broadcast_sync({"status": "CRASH DETECTED", ...})
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._loop: asyncio.AbstractEventLoop = None

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection and track it."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self._loop = asyncio.get_event_loop()
        logger.info(
            f"[WS] Client connected. Active connections: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket):
        """Remove a disconnected WebSocket from the active list."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(
            f"[WS] Client disconnected. Active connections: {len(self.active_connections)}"
        )

    async def broadcast(self, data: dict):
        """Async broadcast: send JSON payload to all connected clients."""
        if not self.active_connections:
            return

        message = json.dumps(data)
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)

        # Clean up stale connections
        for conn in disconnected:
            self.disconnect(conn)

    def broadcast_sync(self, data: dict):
        """
        Sync broadcast: callable from synchronous REST endpoint handlers.

        FastAPI runs sync `def` route handlers in a thread pool. This method
        schedules the async broadcast onto the main event loop from that thread.
        """
        if not self.active_connections:
            return

        try:
            loop = self._loop or asyncio.get_event_loop()
            if loop.is_running():
                # Schedule coroutine on the running loop from another thread
                asyncio.run_coroutine_threadsafe(self.broadcast(data), loop)
            else:
                # Fallback: create new loop (shouldn't happen in normal FastAPI)
                asyncio.run(self.broadcast(data))
        except RuntimeError:
            # No event loop available — skip broadcast silently
            logger.warning("[WS] Could not broadcast: no event loop available")

    async def send_to_vehicle(self, vehicle_id: str, data: dict):
        """Send data to connections subscribed to a specific vehicle.

        Note: In the current single-vehicle prototype, this is equivalent
        to broadcast. Prepared for future multi-vehicle fleet support.
        """
        await self.broadcast(data)

    @property
    def connection_count(self) -> int:
        """Number of currently active WebSocket connections."""
        return len(self.active_connections)


# ── Singleton Instance ───────────────────────────────────────────────────────

ws_manager = ConnectionManager()
