"""Kafka consumer and producer for telemetry events and predictions.

Handles decompression of telemetry batches (gzip/snappy), validation,
schema mapping, alert publishing, and idempotent DLQ handling.
"""
from __future__ import annotations

import asyncio
import gzip
import json
import logging
from dataclasses import dataclass
from typing import Any

from confluent_kafka import Consumer, KafkaError, Message, Producer, TopicPartition

from app.config import Settings
from app.schemas.telemetry_event import TelemetryEvent


class RetryableKafkaError(RuntimeError):
    """Represents Kafka errors that should not terminate the worker loop."""


@dataclass
class DecodeResult:
    """Result of decoding a Kafka message into telemetry events.
    
    Separates successfully validated events from invalid payloads,
    allowing partial batch success with individual failure tracking.
    """
    valid_events: list[TelemetryEvent]
    invalid_payloads: list[dict[str, Any]]  # {payload, reason}
    

class KafkaBatchConsumer:
    """Kafka consumer/producer for telemetry and alert message handling.
    
    Subscribes to telemetry batches, decompresses and validates events,
    publishes alerts and DLQ messages for invalid payloads. Tracks delivery
    errors and lag for observability.
    """
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._consumer = Consumer(settings.consumer_config)
        self._consumer.subscribe([settings.kafka_topic_raw])
        self._producer = Producer(settings.producer_config)
        self._logger = logging.getLogger(__name__)
        self._delivery_errors: list[str] = []
        self._pending_deliveries: int = 0

    def _on_alert_delivery(self, err, msg) -> None:
        """Callback for alert delivery confirmation."""
        self._pending_deliveries -= 1
        if err:
            error_msg = f"Alert delivery failed: {err}"
            self._delivery_errors.append(error_msg)
            self._logger.error(error_msg)
        else:
            self._logger.debug(f"Alert delivered to {msg.topic()}:{msg.partition()} at offset {msg.offset()}")

    def poll_message(self, timeout_seconds: float) -> Message | None:
        """Poll for next message from telemetry topic.
        
        Args:
            timeout_seconds: Max time to wait for message availability.
            
        Returns:
            Kafka message or None if no message available. Raises RuntimeError
            on fatal Kafka errors; raises RetryableKafkaError if topic is unavailable.
        """
        message = self._consumer.poll(timeout_seconds)
        if message is None:
            return None
        if message.error():
            code = message.error().code()
            if code == KafkaError._PARTITION_EOF:
                return None
            unknown_topic_code = getattr(KafkaError, "UNKNOWN_TOPIC_OR_PART", None)
            if unknown_topic_code is not None and code == unknown_topic_code:
                raise RetryableKafkaError(
                    f"Kafka topic unavailable: {self._settings.kafka_topic_raw}: {message.error()}"
                )
            raise RuntimeError(f"Kafka consume error: {message.error()}")
        return message

    def decode_message(self, message: Message) -> DecodeResult:
        """Decode and validate Kafka message containing telemetry batch.
        
        Handles decompression (gzip/snappy), JSON parsing, and per-event
        schema validation. Returns partial successes: valid events plus
        invalid payloads with failure reasons.
        
        Args:
            message: Kafka message with gzip/snappy-compressed JSON array payload.
            
        Returns:
            DecodeResult with valid TelemetryEvent list and invalid payloads list.
        """
        payload = message.value()
        if payload is None:
            return DecodeResult(valid_events=[], invalid_payloads=[])

        try:
            payload = self._decompress_if_needed(payload, message.headers())
        except Exception as exc:
            return DecodeResult(
                valid_events=[],
                invalid_payloads=[{
                    "original_payload": None,
                    "reason": f"Decompression failed: {exc}",
                }],
            )

        try:
            data = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            return DecodeResult(
                valid_events=[],
                invalid_payloads=[{
                    "original_payload": None,
                    "reason": f"JSON decode failed: {exc}",
                }],
            )

        if not isinstance(data, list):
            return DecodeResult(
                valid_events=[],
                invalid_payloads=[{
                    "original_payload": data,
                    "reason": "Expected JSON array, got single object or other type",
                }],
            )

        valid_events: list[TelemetryEvent] = []
        invalid_payloads: list[dict[str, Any]] = []

        for item in data:
            try:
                event = TelemetryEvent.model_validate(item)
                valid_events.append(event)
            except Exception as exc:
                invalid_payloads.append({
                    "original_payload": item,
                    "reason": str(exc),
                })

        return DecodeResult(valid_events=valid_events, invalid_payloads=invalid_payloads)

    @staticmethod
    def _decompress_if_needed(payload: bytes, headers: list[tuple[str, bytes]] | None) -> bytes:
        header_map: dict[str, str] = {}
        if headers:
            header_map = {
                key.lower(): (value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value))
                for key, value in headers
            }

        content_encoding = header_map.get("content-encoding", "").lower()
        if content_encoding == "gzip" or payload[:2] == b"\x1f\x8b":
            return gzip.decompress(payload)

        if content_encoding == "snappy":
            try:
                import snappy

                return snappy.decompress(payload)
            except Exception as exc:
                raise RuntimeError(f"snappy decode failed: {exc}") from exc

        return payload

    def lag_for_message(self, message: Message) -> int:
        """Get consumer lag (messages behind current high watermark) for this message.
        
        Returns:
            Number of messages between this offset and the latest offset (0 if at end).
        """
        topic_partition = TopicPartition(message.topic(), message.partition())
        low, high = self._consumer.get_watermark_offsets(
            topic_partition,
            timeout=1.0,
            cached=False,
        )
        return max(0, high - message.offset() - 1)

    def commit_message(self, message: Message) -> None:
        """Commit offset for message (synchronous, idempotent operation)."""
        self._consumer.commit(message=message, asynchronous=False)

    def publish_alert(self, alert: dict[str, Any], callback=None) -> None:
        """Publish alert to Kafka with delivery confirmation callback."""
        payload = json.dumps(alert, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        delivery_callback = callback if callback is not None else self._on_alert_delivery
        
        self._pending_deliveries += 1
        try:
            self._producer.produce(
                topic=self._settings.kafka_topic_alerts,
                value=payload,
                headers={"content-type": "application/json", "schema": "alerts.failure-risk.v1"},
                on_delivery=delivery_callback,
            )
            # Poll to trigger delivery callbacks
            self._producer.poll(0)
        except Exception as exc:
            self._pending_deliveries -= 1
            error_msg = f"Failed to produce alert: {exc}"
            self._delivery_errors.append(error_msg)
            self._logger.error(error_msg)
            raise

    def wait_for_pending_alerts(self, timeout_seconds: float = 5.0) -> bool:
        """Block until all pending alert deliveries complete or timeout expires.
        
        Returns:
            True if all alerts delivered, False if timeout occurred while pending.
        """
        """Wait for all pending alert deliveries with timeout."""
        import time
        start_time = time.time()
        while self._pending_deliveries > 0 and (time.time() - start_time) < timeout_seconds:
            self._producer.poll(0.1)
        
        if self._pending_deliveries > 0:
            self._logger.warning(
                f"Timeout waiting for pending alerts: {self._pending_deliveries} still pending"
            )
            return False
        return True

    def get_and_clear_delivery_errors(self) -> list[str]:
        """Retrieve accumulated delivery errors and reset the error list."""
        """Get accumulated delivery errors and clear the list."""
        errors = self._delivery_errors.copy()
        self._delivery_errors.clear()
        return errors

    def publish_invalid_payload(self, original_payload: Any, reason: str, source_topic: str, source_offset: int) -> None:
        """Publish invalid payload to DLQ topic for post-mortem investigation.
        
        Args:
            original_payload: The invalid payload that failed validation.
            reason: Human-readable description of validation failure.
            source_topic: Original topic the message came from.
            source_offset: Original offset for traceability.
        """
        """Publish invalid payload to DLQ for investigation."""
        dlq_topic = f"{source_topic}.dlq"
        dlq_message = {
            "source_topic": source_topic,
            "source_offset": source_offset,
            "failure_reason": reason,
            "original_payload": original_payload,
        }
        payload = json.dumps(dlq_message, separators=(",", ":"), ensure_ascii=True, default=str).encode("utf-8")
        try:
            self._producer.produce(
                topic=dlq_topic,
                value=payload,
                headers={"content-type": "application/json", "schema": "dlq.invalid-payload.v1"},
            )
            self._producer.poll(0)
        except Exception as exc:
            self._logger.warning(
                "Failed to publish to DLQ",
                extra={"extra": {"dlq_topic": dlq_topic, "reason": str(exc)}}
            )

    def flush_producer(self, timeout_seconds: float = 5.0) -> None:
        """Flush pending messages in producer queue, warning if timeout occurs."""
        """Flush pending producer messages."""
        remaining = self._producer.flush(int(timeout_seconds * 1000))
        if remaining > 0:
            self._logger.warning(
                "Producer flush timeout: messages still in queue",
                extra={"extra": {"remaining_messages": remaining}}
            )

    def close(self) -> None:
        """Gracefully close consumer and producer connections, flushing pending messages."""
        """Close producer and consumer connections."""
        # Wait for pending deliveries
        if self._pending_deliveries > 0:
            self._logger.info(
                f"Waiting for {self._pending_deliveries} pending alert deliveries before closing"
            )
            remaining = self._producer.flush(5000)  # 5 second timeout
            if remaining > 0:
                self._logger.warning(
                    f"Flush timeout: {remaining} messages still in producer queue"
                )
        
        delivery_errors = self.get_and_clear_delivery_errors()
        if delivery_errors:
            self._logger.warning(
                "Alert delivery errors occurred",
                extra={"extra": {"error_count": len(delivery_errors), "errors": delivery_errors}}
            )
        
        self._consumer.close()
        self._logger.info("Kafka consumer and producer closed")

