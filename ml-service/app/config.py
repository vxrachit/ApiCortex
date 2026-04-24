"""Configuration management for ML service.

Loads and validates environment variables, providing Kafka and database
connection details, model paths, and inference thresholds.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class Settings(BaseModel):
    """Application settings loaded from environment variables.
    
    Validates all configuration on instantiation, ensuring required fields
    are present, TLS certificates are either all provided or all absent,
    and model paths resolve correctly relative to the service root.
    """
    kafka_service_uri: str = Field(validation_alias="KAFKA_SERVICE_URI")
    kafka_ca_cert: str | None = Field(default=None, validation_alias="KAFKA_CA_CERT")
    kafka_service_cert: str | None = Field(default=None, validation_alias="KAFKA_SERVICE_CERT")
    kafka_service_key: str | None = Field(default=None, validation_alias="KAFKA_SERVICE_KEY")
    timescale_database: str = Field(validation_alias="TIMESCALE_DATABASE")

    kafka_topic_raw: str = "telemetry.raw"
    kafka_topic_alerts: str = "alerts"
    kafka_group_id: str = "apicortex-ml-inference"
    kafka_poll_timeout_seconds: float = 1.0

    model_path: Path = Path("model/xgboost_failure_prediction.pkl")
    enable_shap: bool = True
    shap_top_k: int = 5

    alert_threshold: float = 0.8
    log_level: str = "INFO"

    consumer_max_poll_interval_ms: int = 300000
    consumer_session_timeout_ms: int = 45000

    @field_validator("kafka_service_uri", "timescale_database", mode="before")
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        if not isinstance(value, str):
            return "INFO"
        normalized = value.strip().upper()
        return normalized if normalized in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"} else "INFO"

    @field_validator("kafka_ca_cert", "kafka_service_cert", "kafka_service_key", mode="before")
    @classmethod
    def _normalize_optional_secret(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("model_path", mode="before")
    @classmethod
    def _normalize_model_path(cls, value: str | Path) -> Path:
        path = value if isinstance(value, Path) else Path(str(value).strip())
        if path.is_absolute():
            return path
        service_root = Path(__file__).resolve().parents[1]
        return (service_root / path).resolve()

    @field_validator("alert_threshold")
    @classmethod
    def _alert_threshold_range(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("alert_threshold must be within [0.0, 1.0]")
        return value

    @model_validator(mode="after")
    def _validate_kafka_tls_material(self) -> "Settings":
        values = [self.kafka_ca_cert, self.kafka_service_cert, self.kafka_service_key]
        if any(values) and not all(values):
            raise ValueError("KAFKA_CA_CERT, KAFKA_SERVICE_CERT, and KAFKA_SERVICE_KEY must all be provided together")
        return self

    @property
    def kafka_brokers(self) -> list[str]:
        """Parse Kafka service URI into list of individual broker addresses."""
        return [broker.strip() for broker in self.kafka_service_uri.split(",") if broker.strip()]

    @property
    def consumer_config(self) -> dict[str, object]:
        """Build librdkafka consumer configuration dictionary."""
        config: dict[str, object] = {
            "bootstrap.servers": ",".join(self.kafka_brokers),
            "group.id": self.kafka_group_id,
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
            "session.timeout.ms": self.consumer_session_timeout_ms,
            "max.poll.interval.ms": self.consumer_max_poll_interval_ms,
        }
        self._apply_tls(config)
        return config

    @property
    def producer_config(self) -> dict[str, object]:
        """Build librdkafka producer configuration dictionary with idempotent delivery."""
        config: dict[str, object] = {
            "bootstrap.servers": ",".join(self.kafka_brokers),
            "acks": "all",
            "enable.idempotence": True,
            "compression.type": "gzip",
        }
        self._apply_tls(config)
        return config

    def _apply_tls(self, config: dict[str, object]) -> None:
        ca = self.kafka_ca_cert
        cert = self.kafka_service_cert
        key = self.kafka_service_key
        if ca and cert and key:
            config.update(
                {
                    "security.protocol": "ssl",
                    "ssl.ca.pem": ca,
                    "ssl.certificate.pem": cert,
                    "ssl.key.pem": key,
                }
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and validate settings from environment."""
    load_dotenv()
    
    # Check for required environment variables
    required_env_vars = ["KAFKA_SERVICE_URI", "TIMESCALE_DATABASE"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var, "").strip()]
    
    if missing_vars:
        error_msg = (
            f"Required environment variables not set: {', '.join(missing_vars)}\n"
            f"Please set these variables or copy .env.example to .env and configure it.\n"
            f"Never commit secrets to version control."
        )
        # Print to stderr and exit immediately on startup
        print(error_msg, file=sys.stderr)
        raise RuntimeError(error_msg)
    
    data = {
        "KAFKA_SERVICE_URI": os.getenv("KAFKA_SERVICE_URI", "").strip(),
        "KAFKA_CA_CERT": os.getenv("KAFKA_CA_CERT"),
        "KAFKA_SERVICE_CERT": os.getenv("KAFKA_SERVICE_CERT"),
        "KAFKA_SERVICE_KEY": os.getenv("KAFKA_SERVICE_KEY"),
        "TIMESCALE_DATABASE": os.getenv("TIMESCALE_DATABASE", "").strip(),
        "model_path": os.getenv("MODEL_PATH", "model/xgboost_failure_prediction.pkl"),
        "alert_threshold": float(os.getenv("ALERT_THRESHOLD", "0.8")),
        "enable_shap": os.getenv("ENABLE_SHAP", "true").strip().lower() in {"1", "true", "yes"},
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "kafka_topic_raw": os.getenv("KAFKA_TOPIC_RAW", "telemetry.raw"),
        "kafka_topic_alerts": os.getenv("KAFKA_TOPIC_ALERTS", "alerts"),
        "kafka_group_id": os.getenv("KAFKA_GROUP_ID", "apicortex-ml-inference"),
        "kafka_poll_timeout_seconds": float(os.getenv("KAFKA_POLL_TIMEOUT_SECONDS", "1.0")),
        "consumer_max_poll_interval_ms": int(os.getenv("KAFKA_MAX_POLL_INTERVAL_MS", "300000")),
        "consumer_session_timeout_ms": int(os.getenv("KAFKA_SESSION_TIMEOUT_MS", "45000")),
        "shap_top_k": int(os.getenv("SHAP_TOP_K", "5")),
    }
    try:
        return Settings.model_validate(data)
    except ValidationError as exc:
        error_msg = f"Invalid ML service configuration:\n{exc}"
        print(error_msg, file=sys.stderr)
        raise RuntimeError(error_msg) from exc
