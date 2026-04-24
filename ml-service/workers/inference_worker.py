"""Async ML inference worker for continuous telemetry processing.

Main event loop that consumes Kafka batches, validates telemetry events,
computes features, runs predictions, and persists results to TimescaleDB
with alert publishing. Implements graceful shutdown with signal handling.
"""
from __future__ import annotations

import asyncio
import json
import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable

from confluent_kafka import Message

from app.config import Settings
from app.explainability.shap_explainer import ShapExplainer
from app.features.feature_engineering import RollingFeatureEngineer
from app.inference.model_loader import load_model
from app.inference.predictor import Predictor
from app.kafka.consumer import KafkaBatchConsumer, RetryableKafkaError
from app.storage.timescale_writer import PredictionRecord, TimescaleWriter


class JsonFormatter(logging.Formatter):
    """JSON-formatted logger for structured logging in production."""
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        return json.dumps(payload, ensure_ascii=True)


def configure_logging(level: str) -> logging.Logger:
    """Configure JSON logger for structured logging to stdout.
    
    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("apicortex.ml-worker")
    logger.setLevel(level)
    logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


@dataclass
class WorkerMetrics:
    """Cumulative metrics for worker lifecycle (does not reset on restart).
    
    Tracks batches processed, events ingested, predictions written, and errors
    for observability and debugging.
    """
    batches_processed: int = 0
    events_processed: int = 0
    predictions_written: int = 0
    inference_errors: int = 0
    invalid_events: int = 0
    alert_publish_failures: int = 0
    db_write_failures: int = 0
    retries: int = 0
    dlq_messages: int = 0


@dataclass
class RetryConfig:
    """Configuration for exponential backoff retry logic.
    
    Used when writing predictions to database fails, backing off gradually
    to avoid cascade failures.
    """
    max_retries: int = 3
    initial_backoff_seconds: float = 0.1
    max_backoff_seconds: float = 30.0
    backoff_multiplier: float = 2.0


class AlertDeliveryTracker:
    """Tracks alert delivery confirmations from Kafka producer.
    
    Currently a placeholder for future async delivery tracking (not fully used).
    """
    def __init__(self) -> None:
        self.pending_alerts: dict[int, dict] = {}
        self.delivery_errors: list[str] = []
        self._lock = asyncio.Lock()

    async def on_delivery(self, err, msg) -> None:
        """Callback invoked on alert delivery confirmation (success or failure)."""
        async with self._lock:
            if err:
                self.delivery_errors.append(str(err))
            # Successfully delivered



class InferenceWorker:
    """Main async worker orchestrating end-to-end telemetry-to-prediction pipeline.
    
    Polls Kafka for telemetry batches, validates/decompresses messages, computes
    features, runs XGBoost inference, publishes high-risk alerts, and writes
    predictions to TimescaleDB. Implements graceful shutdown with signal handlers.
    """
    def __init__(self, settings: Settings) -> None:
        """Initialize worker with configuration and load model/components.
        
        Args:
            settings: Configuration from environment variables.
        """
        self.settings = settings
        self.logger = configure_logging(settings.log_level)

        model = load_model(settings.model_path)
        explainer = ShapExplainer(
            model=model,
            enabled=settings.enable_shap,
            top_k=settings.shap_top_k,
            logger=self.logger,
        )

        self.predictor = Predictor(model=model, explainer=explainer)
        self.feature_engineer = RollingFeatureEngineer(logger=self.logger)
        self.consumer = KafkaBatchConsumer(settings)
        self.writer = TimescaleWriter(settings)

        self.metrics = WorkerMetrics()
        self.retry_config = RetryConfig()
        self.alert_tracker = AlertDeliveryTracker()
        self._shutdown = asyncio.Event()

    def request_shutdown(self) -> None:
        """Signal worker to gracefully shutdown the main loop."""
        self._shutdown.set()

    async def _validate_uuids(self, org_id: str, api_id: str) -> bool:
        """Check if both org_id and api_id are valid UUID format."""
        import uuid as uuid_module
        try:
            uuid_module.UUID(org_id)
            uuid_module.UUID(api_id)
            return True
        except (ValueError, AttributeError):
            return False

    async def _retry_with_backoff(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Retry async function with exponential backoff (max_retries + 1 attempts).
        
        Logs warnings on retry attempts, re-raises last exception if all retries fail.
        
        Args:
            func: Async callable to retry.
            *args, **kwargs: Arguments to pass to func.
            
        Returns:
            Return value from func.
            
        Raises:
            Last exception encountered if all retries exhausted.
        """
        backoff = self.retry_config.initial_backoff_seconds
        last_error = None

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                last_error = exc
                if attempt < self.retry_config.max_retries:
                    self.metrics.retries += 1
                    await asyncio.sleep(min(backoff, self.retry_config.max_backoff_seconds))
                    backoff *= self.retry_config.backoff_multiplier
                    self.logger.warning(
                        f"Retry attempt {attempt + 1}/{self.retry_config.max_retries}",
                        extra={"extra": {"error": str(exc), "next_backoff_seconds": min(backoff, self.retry_config.max_backoff_seconds)}}
                    )

        raise last_error or RuntimeError("Retry exhausted")


    async def run(self) -> None:
        """Main async worker loop: poll Kafka, process batches, until shutdown requested.
        
        Polls telemetry topic, decodes messages, validates events, computes features,
        runs predictions, publishes alerts, and writes to database. Handles Kafka
        failures gracefully with retries. Offset only committed on full success."""
        self.logger.info("ML inference worker started")

        while not self._shutdown.is_set():
            try:
                message = await asyncio.to_thread(self.consumer.poll_message, self.settings.kafka_poll_timeout_seconds)
            except RetryableKafkaError as exc:
                self.logger.warning(
                    "Kafka topic unavailable, waiting for topic creation",
                    extra={
                        "extra": {
                            "error": str(exc),
                            "topic": self.settings.kafka_topic_raw,
                        }
                    },
                )
                await asyncio.sleep(max(1.0, self.settings.kafka_poll_timeout_seconds))
                continue
            except Exception as exc:
                self.metrics.inference_errors += 1
                self.logger.exception(
                    "Kafka poll failed",
                    extra={
                        "extra": {
                            "error": str(exc),
                            "inference_errors": self.metrics.inference_errors,
                        }
                    },
                )
                await asyncio.sleep(max(1.0, self.settings.kafka_poll_timeout_seconds))
                continue

            if message is None:
                continue

            try:
                await self._handle_message(message)
            except Exception as exc:
                self.metrics.inference_errors += 1
                self.logger.exception(
                    "Failed to process telemetry batch - will NOT commit offset",
                    extra={
                        "extra": {
                            "error": str(exc),
                            "inference_errors": self.metrics.inference_errors,
                            "topic": message.topic(),
                            "partition": message.partition(),
                            "offset": message.offset(),
                        }
                    },
                )
                # Do NOT commit offset on failure - message will be reprocessed

        await self._shutdown_cleanup()

    async def _handle_message(self, message: Message) -> None:
        """Process single Kafka message: decode, validate, predict, persist, commit.
        
        Handles partial batch failures: invalid events sent to DLQ but batch
        continues processing. Database write must succeed for offset commit.
        Alert publish failures are logged but don't block commit.
        
        Args:
            message: Kafka message from telemetry topic.
            
        Raises:
            Exception if database write fails (offset not committed, message reprocessed).
        """
        consume_started = time.perf_counter()
        kafka_lag = await asyncio.to_thread(self.consumer.lag_for_message, message)

        # Decode message and handle per-event validation failures
        decode_result = await asyncio.to_thread(self.consumer.decode_message, message)
        valid_events = decode_result.valid_events

        # Publish invalid events to DLQ
        for invalid in decode_result.invalid_payloads:
            self.metrics.invalid_events += 1
            self.metrics.dlq_messages += 1
            await asyncio.to_thread(
                self.consumer.publish_invalid_payload,
                invalid.get("original_payload"),
                invalid.get("reason", "Unknown error"),
                message.topic(),
                message.offset(),
            )

        if not valid_events:
            # No valid events, but we've logged invalids to DLQ
            # Commit to move past this message
            await asyncio.to_thread(self.consumer.commit_message, message)
            return

        # Validate UUIDs and filter out invalid events before processing
        filtered_events: list = []
        for event in valid_events:
            if not await self._validate_uuids(event.org_id, event.api_id):
                self.metrics.invalid_events += 1
                self.metrics.dlq_messages += 1
                await asyncio.to_thread(
                    self.consumer.publish_invalid_payload,
                    {
                        "timestamp": event.timestamp.isoformat(),
                        "org_id": event.org_id,
                        "api_id": event.api_id,
                        "endpoint": event.endpoint,
                    },
                    f"Invalid UUID format: org_id={event.org_id}, api_id={event.api_id}",
                    message.topic(),
                    message.offset(),
                )
            else:
                filtered_events.append(event)

        if not filtered_events:
            # All events were invalid
            await asyncio.to_thread(self.consumer.commit_message, message)
            return

        # Generate features and predictions
        feature_rows = self.feature_engineer.ingest(filtered_events)
        prediction_records: list[PredictionRecord] = []
        alerts_to_publish: list[dict[str, Any]] = []

        for feature_row in feature_rows:
            result = self.predictor.predict(feature_row.features)
            prediction_records.append(
                PredictionRecord(
                    time=feature_row.time,
                    org_id=feature_row.org_id,
                    api_id=feature_row.api_id,
                    endpoint=feature_row.endpoint,
                    method=feature_row.method,
                    risk_score=result.risk_score,
                    prediction=result.prediction,
                    confidence=result.confidence,
                    top_features=result.top_features,
                    is_warmed_up=feature_row.is_warmed_up,
                )
            )

            if result.risk_score >= self.settings.alert_threshold:
                alerts_to_publish.append({
                    "org_id": feature_row.org_id,
                    "api_id": feature_row.api_id,
                    "endpoint": feature_row.endpoint,
                    "method": feature_row.method,
                    "risk_score": result.risk_score,
                    "prediction": result.prediction,
                    "severity": "high",
                    "timestamp": feature_row.time.isoformat(),
                })

        # Write predictions to Timescale (with retry)
        try:
            await self._retry_with_backoff(
                self._write_predictions_async,
                prediction_records
            )
        except Exception as exc:
            self.metrics.db_write_failures += 1
            self.logger.exception(
                "Failed to write predictions to Timescale - offset will NOT be committed",
                extra={"extra": {"error": str(exc), "records": len(prediction_records)}}
            )
            raise

        # Publish alerts (with retry, but non-blocking failure)
        for alert in alerts_to_publish:
            try:
                await self._retry_with_backoff(
                    self._publish_alert_async,
                    alert
                )
            except Exception as exc:
                self.metrics.alert_publish_failures += 1
                self.logger.warning(
                    "Failed to publish alert - proceeding with offset commit (alert will be missed)",
                    extra={"extra": {"error": str(exc), "alert_org_id": alert.get("org_id")}}
                )

        # Only commit offset after BOTH write_predictions AND all alerts succeed
        await asyncio.to_thread(self.consumer.commit_message, message)

        duration_ms = round((time.perf_counter() - consume_started) * 1000, 2)
        self.metrics.batches_processed += 1
        self.metrics.events_processed += len(filtered_events)
        self.metrics.predictions_written += len(prediction_records)

        self.logger.info(
            "Processed telemetry batch",
            extra={
                "extra": {
                    "events_processed": len(filtered_events),
                    "predictions_written": len(prediction_records),
                    "alerts_published": len(alerts_to_publish),
                    "invalid_events": len(decode_result.invalid_payloads),
                    "prediction_latency_ms": duration_ms,
                    "kafka_lag": kafka_lag,
                    "totals": {
                        "batches": self.metrics.batches_processed,
                        "events": self.metrics.events_processed,
                        "predictions": self.metrics.predictions_written,
                        "inference_errors": self.metrics.inference_errors,
                        "invalid_events": self.metrics.invalid_events,
                        "dlq_messages": self.metrics.dlq_messages,
                    },
                }
            },
        )

    async def _write_predictions_async(self, records: list[PredictionRecord]) -> None:
        """Async wrapper for database write (runs in thread pool to avoid blocking)."""
        await asyncio.to_thread(self.writer.write_predictions, records)

    async def _publish_alert_async(self, alert: dict[str, Any]) -> None:
        """Async wrapper for alert publishing (runs in thread pool)."""
        await asyncio.to_thread(self.consumer.publish_alert, alert)

    async def _shutdown_cleanup(self) -> None:
        """Gracefully close all connections: flush alerts, close database, close Kafka."""
        self.logger.info("Shutting down ML inference worker")
        await asyncio.to_thread(self.consumer.flush_producer)
        await asyncio.to_thread(self.writer.close)
        await asyncio.to_thread(self.consumer.close)


def install_signal_handlers(worker: InferenceWorker) -> None:
    """Register SIGINT and SIGTERM handlers to trigger graceful worker shutdown."""
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, worker.request_shutdown)
        except NotImplementedError:
            
            signal.signal(sig, lambda _signo, _frame: worker.request_shutdown())
