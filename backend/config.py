"""
IntelliCrash — Backend Configuration.

Pydantic-based settings management. Reads from environment variables
and falls back to sensible defaults for local development.

Usage:
    from src.backend.config import get_settings
    settings = get_settings()
    print(settings.DATABASE_URL)
"""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings


# Resolve project root (3 levels up from this file: backend -> src -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── App ──────────────────────────────────────────────────────────────
    APP_NAME: str = "IntelliCrash API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # ── Database ─────────────────────────────────────────────────────────
    DATABASE_PATH: str = str(PROJECT_ROOT / "data" / "intellicrash.db")

    # ── CORS ─────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",      # Next.js dev server
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ]

    # ── Firebase ─────────────────────────────────────────────────────────
    FIREBASE_CREDENTIALS_PATH: str = str(
        PROJECT_ROOT / "configs" / "firebase_admin_key.json"
    )
    FIREBASE_ENABLED: bool = False    # Set True when credentials are ready

    # ── Twilio ───────────────────────────────────────────────────────────
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_FROM_PHONE: str = ""
    TWILIO_EMERGENCY_CONTACT: str = ""

    # ── Email ────────────────────────────────────────────────────────────
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SENDER_EMAIL: str = ""
    SENDER_PASSWORD: str = ""
    HOSPITAL_EMAIL: str = ""

    # ── Inference ────────────────────────────────────────────────────────
    ONNX_MODEL_PATH: str = str(
        PROJECT_ROOT / "outputs" / "models" / "onnx" / "intellicrash_lstm.onnx"
    )

    # ── WebSocket ────────────────────────────────────────────────────────
    WS_TELEMETRY_INTERVAL: float = 0.1  # 10 Hz push to dashboard

    # ── JWT / Security ───────────────────────────────────────────────────
    SECRET_KEY: str = "intellicrash-super-secret-jwt-key-2026-production-grade"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    # ── MQTT Broker ──────────────────────────────────────────────────────
    MQTT_BROKER_HOST: str = "localhost"
    MQTT_BROKER_PORT: int = 1883
    MQTT_KEEPALIVE: int = 60
    MQTT_CLIENT_ID: str = "intellicrash-backend-consumer"
    MQTT_ENABLED: bool = False
    MQTT_TOPIC_TELEMETRY: str = "intellicrash/v1/telemetry"
    MQTT_TOPIC_CRASHES: str = "intellicrash/v1/crashes"
    MQTT_TOPIC_ALERTS: str = "intellicrash/v1/alerts"

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


@lru_cache()
def get_settings() -> Settings:
    """Singleton settings loader — cached after first call."""
    return Settings()
