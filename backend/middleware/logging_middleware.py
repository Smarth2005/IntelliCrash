"""
IntelliCrash — Structured JSON Observability & Correlation ID Middleware.

Features:
- Injects a unique `X-Request-ID` into every HTTP request & response.
- Computes request processing latency with sub-millisecond precision.
- Emits structured JSON log events for central log aggregators (ELK, CloudWatch, Datadog).
"""

import time
import uuid
import json
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("intellicrash.audit")


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that records structured metrics and correlation IDs for every HTTP transaction.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.perf_counter()
        client_ip = request.client.host if request.client else "unknown"

        try:
            response = await call_next(request)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

            # Attach correlation ID to outgoing response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time-MS"] = str(duration_ms)

            # Emit structured JSON audit log
            log_record = {
                "event": "http_request",
                "request_id": request_id,
                "client_ip": client_ip,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            }
            logger.info(json.dumps(log_record))
            return response

        except Exception as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            log_record = {
                "event": "http_exception",
                "request_id": request_id,
                "client_ip": client_ip,
                "method": request.method,
                "path": request.url.path,
                "error": str(exc),
                "duration_ms": duration_ms,
            }
            logger.error(json.dumps(log_record))
            raise exc
