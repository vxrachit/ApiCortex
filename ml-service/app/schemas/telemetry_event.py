"""Pydantic schema for API telemetry events.

Validates telemetry payloads from ingest service, enforcing UUID format,
HTTP status range, and field value constraints. Supports both legacy
(request_size/response_size) and current (request_size_bytes/response_size_bytes)
field names.
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import AliasChoices, BaseModel, Field, field_validator


class TelemetryEvent(BaseModel):
    """API telemetry event for feature engineering and prediction.
    
    Represents a single API request/response observation with timing, status,
    and schema information for failure risk detection.
    
    Attributes:
        timestamp: When the request was made.
        org_id, api_id: Organization and API identifiers (must be valid UUIDs).
        endpoint: API endpoint path (e.g., '/users/123').
        method: HTTP method in uppercase (GET, POST, etc.).
        status: HTTP response status code [100-599].
        latency_ms: Request-to-response time in milliseconds (non-negative).
        request_size_bytes, response_size_bytes: Message sizes (non-negative).
        schema_hash: Optional hash of OpenAPI schema for change detection.
        schema_version: Optional OpenAPI version identifier.
    """
    timestamp: datetime
    org_id: str
    api_id: str
    endpoint: str
    method: str
    status: int
    latency_ms: int
    request_size_bytes: int = Field(
        default=0,
        validation_alias=AliasChoices("request_size_bytes", "request_size"),
    )
    response_size_bytes: int = Field(
        default=0,
        validation_alias=AliasChoices("response_size_bytes", "response_size"),
    )
    schema_hash: str | None = None
    schema_version: str | None = None

    @field_validator("org_id", "api_id", mode="before")
    @classmethod
    def _validate_uuid_format(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("UUID field must be a string")
        value = value.strip()
        if not value:
            raise ValueError("UUID field cannot be empty")
        # Validate UUID format
        try:
            UUID(value)
        except (ValueError, AttributeError):
            raise ValueError(f"Invalid UUID format: {value}")
        return value

    @field_validator("endpoint", "method", mode="before")
    @classmethod
    def _trim_required_strings(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("required string field cannot be empty")
        return value.strip()

    @field_validator("method", mode="after")
    @classmethod
    def _normalize_method(cls, value: str) -> str:
        return value.upper()

    @field_validator("status")
    @classmethod
    def _status_range(cls, value: int) -> int:
        if value < 100 or value > 599:
            raise ValueError("status must be between 100 and 599")
        return value

    @field_validator("latency_ms", "request_size_bytes", "response_size_bytes")
    @classmethod
    def _non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("numeric telemetry fields must be non-negative")
        return value

    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }
