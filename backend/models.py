"""
IntelliCrash — Pydantic Models (Request/Response Schemas).

Defines strict data validation for all API endpoints.
FastAPI auto-generates Swagger/OpenAPI docs from these models.

Naming Convention:
- *Create  → POST request bodies (what the client sends)
- *Response → GET response bodies (what the server returns)
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ── Telemetry ────────────────────────────────────────────────────────────────

class TelemetryCreate(BaseModel):
    """Telemetry data point sent from the edge device (RPi4)."""

    vehicle_id: str = Field(default="vehicle-001", description="Unique vehicle identifier")
    accel_x: float = Field(..., description="Lateral acceleration (m/s²)")
    accel_y: float = Field(..., description="Longitudinal acceleration (m/s²)")
    accel_z: float = Field(default=9.81, description="Vertical acceleration (m/s²)")
    gyro_z: float = Field(..., description="Yaw rate (deg/s)")
    crash_prob: float = Field(..., ge=0, le=100, description="Crash probability (%)")
    severity_score: float = Field(default=0.0, ge=0, le=1, description="CSI severity (0-1)")
    status: str = Field(default="Normal", description="Current status: Normal, RASH DRIVING, CRASH DETECTED")
    behavior_score: float = Field(default=100.0, ge=0, le=100, description="Driver behavior score")


class TelemetryResponse(BaseModel):
    """Single telemetry snapshot returned by the API."""

    id: int
    vehicle_id: str
    timestamp: str
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_z: float
    crash_prob: float
    severity_score: float
    status: str
    behavior_score: float
    created_at: str


class TelemetryLive(BaseModel):
    """Real-time telemetry pushed over WebSocket."""

    vehicle_id: str = "vehicle-001"
    timestamp: str
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_z: float
    crash_prob: float
    severity_score: float
    status: str
    behavior_score: float
    mode: str = "FastAPI Backend"
    call_status: str = "Standby"


# ── Crash Events ─────────────────────────────────────────────────────────────

class CrashEventCreate(BaseModel):
    """Crash event submitted when the Fusion Gate triggers."""

    vehicle_id: str = Field(default="vehicle-001")
    fused_score: float = Field(..., ge=0, le=1, description="Fusion Gate output (0-1)")
    ml_probability: float = Field(..., ge=0, le=1, description="Bi-LSTM raw output")
    csi_score: float = Field(..., ge=0, le=1, description="Physics CSI score")
    severity: str = Field(..., description="MINOR, SEVERE, or FATAL")
    latitude: float = Field(default=28.6139, description="GPS latitude")
    longitude: float = Field(default=77.2090, description="GPS longitude")
    peak_accel_x: Optional[float] = Field(default=None, description="Peak X accel during impact")
    peak_accel_y: Optional[float] = Field(default=None, description="Peak Y accel during impact")
    peak_gyro_z: Optional[float] = Field(default=None, description="Peak gyro Z during impact")
    shap_summary: Optional[str] = Field(default=None, description="JSON-encoded SHAP feature attributions")
    video_clip_path: Optional[str] = Field(default=None, description="Path to saved dashcam clip")


class CrashEventResponse(BaseModel):
    """Full crash event record returned by the API."""

    id: int
    vehicle_id: str
    timestamp: str
    fused_score: float
    ml_probability: float
    csi_score: float
    severity: str
    latitude: float
    longitude: float
    peak_accel_x: Optional[float]
    peak_accel_y: Optional[float]
    peak_gyro_z: Optional[float]
    shap_summary: Optional[str]
    video_clip_path: Optional[str]
    alert_dispatched: bool
    firebase_synced: bool
    created_at: str


# ── Alerts ───────────────────────────────────────────────────────────────────

class AlertDispatchRequest(BaseModel):
    """Request to dispatch an emergency alert for a crash event."""

    crash_event_id: int = Field(..., description="ID of the crash event to alert on")
    channels: list[str] = Field(
        default=["sms", "email"],
        description="Alert channels to use: 'sms', 'email', or both"
    )


class AlertResponse(BaseModel):
    """Result of an alert dispatch attempt."""

    id: int
    crash_event_id: int
    channel: str
    recipient: str
    status: str
    message_sid: Optional[str]
    error_message: Optional[str]
    dispatched_at: str


# ── General ──────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    app: str
    version: str
    database: str
    firebase: str
    uptime_seconds: float


class PaginatedResponse(BaseModel):
    """Generic paginated list response."""

    total: int
    page: int
    per_page: int
    data: list


# ── Authentication & Users ──────────────────────────────────────────────────

class UserRegister(BaseModel):
    """User registration schema."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., description="Valid email address")
    password: str = Field(..., min_length=6, description="Password with min 6 characters")
    role: str = Field(default="operator", description="Role: admin, operator, viewer")


class UserLogin(BaseModel):
    """User login request body."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """JWT Token response schema."""
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    user_id: int
    username: str
    role: str


class UserResponse(BaseModel):
    """Public user profile response."""
    id: int
    username: str
    email: str
    role: str
    is_active: bool
    created_at: str

