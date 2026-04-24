"""TimescaleDB writer for inference results.

Persists API failure predictions to a hypertable with idempotent upsert logic
based on 1-minute time buckets. Automatically creates schema and indexes on
first connection.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg2
from psycopg2.extras import Json, execute_batch

from app.config import Settings


@dataclass
class PredictionRecord:
    """Single API failure prediction to persist to database.
    
    Attributes:
        time: Timestamp of prediction.
        org_id, api_id, endpoint, method: Endpoint identifier (UUIDs/strings).
        risk_score: Failure probability [0.0, 1.0].
        prediction: Risk category (normal/degraded/high_failure_risk).
        confidence: Model confidence [0.5, 1.0].
        top_features: List of {feature: str, contribution: float, abs_contribution: float}.
        model_version, feature_schema_version: Versions for auditing/migration.
        model_hash: Hash of model artifact.
        is_warmed_up: Whether endpoint had sufficient history for this prediction.
    """
    time: datetime
    org_id: str
    api_id: str
    endpoint: str
    method: str
    risk_score: float
    prediction: str
    confidence: float
    top_features: list[dict[str, Any]]
    model_version: str = "1.0"
    feature_schema_version: str = "1.0"
    model_hash: str = ""
    is_warmed_up: bool = False


class TimescaleWriter:
    """Manages connection to TimescaleDB and writes API failure predictions.
    
    Creates hypertable on first instantiation with idempotent upsert logic
    based on (org_id, api_id, endpoint, method, time_bucket) conflict key.
    Buckets time to nearest minute to handle late/duplicate deliveries.
    """
    def __init__(self, settings: Settings) -> None:
        """Connect to TimescaleDB and ensure api_failure_predictions schema exists.
        
        Args:
            settings: Configuration with timescale_database connection string.
        """
        self._conn = psycopg2.connect(settings.timescale_database)
        self._conn.autocommit = False
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create hypertable and indexes if they don't exist (idempotent operation)."""
        with self._conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS api_failure_predictions (
                    time TIMESTAMPTZ NOT NULL,
                    org_id UUID NOT NULL,
                    api_id UUID NOT NULL,
                    endpoint TEXT NOT NULL,
                    method TEXT NOT NULL DEFAULT 'GET',
                    risk_score DOUBLE PRECISION NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    top_features JSONB NOT NULL DEFAULT '[]'::jsonb,
                    model_version TEXT NOT NULL DEFAULT '1.0',
                    feature_schema_version TEXT NOT NULL DEFAULT '1.0',
                    model_hash TEXT DEFAULT '',
                    is_warmed_up BOOLEAN DEFAULT false
                );
                """
            )
            cursor.execute(
                """
                SELECT create_hypertable(
                    'api_failure_predictions',
                    'time',
                    if_not_exists => TRUE,
                    migrate_data => TRUE
                );
                """
            )
            
            cursor.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_api_failure_predictions_idempotent
                ON api_failure_predictions (org_id, api_id, endpoint, method, time_bucket('1 minute', time));
                """
            )
            
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_api_failure_predictions_lookup
                ON api_failure_predictions (org_id, api_id, endpoint, time DESC);
                """
            )
        self._conn.commit()

    def _bucket_time(self, ts: datetime) -> datetime:
        """Bucket timestamp to nearest minute for idempotency."""
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        # Truncate to minute boundary
        return ts.replace(second=0, microsecond=0)


    def write_predictions(self, records: list[PredictionRecord]) -> None:
        """Batch write predictions with idempotent upsert (handles duplicates gracefully).
        
        Uses ON CONFLICT with GREATEST() for risk_score to preserve highest risk
        if duplicate predictions arrive for same endpoint in same minute.
        
        Args:
            records: List of PredictionRecord objects to write (no-op if empty).
        """
        if not records:
            return

        with self._conn.cursor() as cursor:
            # Use UPSERT with ON CONFLICT to ensure idempotency
            # Conflict key is (org_id, api_id, endpoint, method, time_bucket)
            execute_batch(
                cursor,
                """
                INSERT INTO api_failure_predictions (
                    time,
                    org_id,
                    api_id,
                    endpoint,
                    method,
                    risk_score,
                    prediction,
                    confidence,
                    top_features,
                    model_version,
                    feature_schema_version,
                    model_hash,
                    is_warmed_up
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (org_id, api_id, endpoint, method, time_bucket('1 minute', time))
                DO UPDATE SET
                    risk_score = GREATEST(EXCLUDED.risk_score, api_failure_predictions.risk_score),
                    prediction = EXCLUDED.prediction,
                    confidence = EXCLUDED.confidence,
                    top_features = EXCLUDED.top_features,
                    model_version = EXCLUDED.model_version,
                    feature_schema_version = EXCLUDED.feature_schema_version,
                    model_hash = EXCLUDED.model_hash,
                    is_warmed_up = EXCLUDED.is_warmed_up;
                """,
                [
                    (
                        self._bucket_time(record.time),
                        record.org_id,
                        record.api_id,
                        record.endpoint,
                        record.method,
                        record.risk_score,
                        record.prediction,
                        record.confidence,
                        Json(record.top_features),
                        record.model_version,
                        record.feature_schema_version,
                        record.model_hash,
                        record.is_warmed_up,
                    )
                    for record in records
                ],
                page_size=500,
            )
        self._conn.commit()


    def close(self) -> None:
        """Close database connection."""
        self._conn.close()
