"""Feature engineering for API failure prediction.

Maintains rolling windows of telemetry events per endpoint, computing
time-series features (latency, error rate, traffic, schema changes) for
model inference. Handles late/out-of-order events and warm-up tracking.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import UTC, datetime, timedelta
from statistics import mean
from typing import Deque, Any

import numpy as np

from app.schemas.telemetry_event import TelemetryEvent


WINDOW_1M = timedelta(minutes=1)
WINDOW_5M = timedelta(minutes=5)
WINDOW_15M = timedelta(minutes=15)

# Lateness threshold: events older than this are considered late/stale
LATENESS_THRESHOLD = timedelta(minutes=30)

# Minimum number of events required for stable feature computation
MIN_EVENTS_FOR_WARMUP = 100

FEATURE_COLUMNS = [
    "latency_mean",
    "latency_p95",
    "latency_variance",
    "latency_delta",
    "error_rate",
    "error_rate_delta",
    "traffic_rps",
    "traffic_delta",
    "schema_fields_added",
    "schema_fields_removed",
    "schema_breaking_changes",
]


@dataclass(frozen=True)
class EventKey:
    """Unique identifier for an API endpoint (org + api + path + method)."""
    org_id: str
    api_id: str
    endpoint: str
    method: str


@dataclass(frozen=True)
class EventSnapshot:
    """Single raw telemetry event snapshot for feature computation."""
    timestamp: datetime
    status: int
    latency_ms: int
    schema_hash: str | None


@dataclass
class FeatureRow:
    """Computed features for one endpoint at a point in time.
    
    Attributes:
        time: Timestamp when features were computed (latest event time).
        org_id, api_id, endpoint, method: Endpoint identifier.
        features: Dict mapping feature names to float values.
        is_warmed_up: Whether endpoint has sufficient history for predictions.
    """
    time: datetime
    org_id: str
    api_id: str
    endpoint: str
    method: str
    features: dict[str, float]
    is_warmed_up: bool = False


class RollingFeatureEngineer:
    """Maintains per-endpoint rolling state and computes model features with warm-up tracking.
    
    Accumulates telemetry events in sliding windows, tracks late/out-of-order
    events, and computes 11 time-series features for each endpoint. Monitors
    when endpoints have enough history (MIN_EVENTS_FOR_WARMUP) for stable
    predictions.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize feature engineer with empty history and metrics.
        
        Args:
            logger: Optional logger for debug/info messages (default: module logger).
        """
        self._history: dict[EventKey, Deque[EventSnapshot]] = defaultdict(deque)
        self._event_counts: dict[EventKey, int] = defaultdict(int)
        self._is_warmed_up: dict[EventKey, bool] = {}
        self._logger = logger or logging.getLogger(__name__)
        self._total_events_processed = 0
        self._late_events_dropped = 0
        self._out_of_order_events = 0
        self._max_event_time_seen: dict[EventKey, datetime] = {}

    def is_endpoint_warmed_up(self, key: EventKey) -> bool:
        """Check if an endpoint has sufficient history for stable predictions."""
        return self._is_warmed_up.get(key, False)

    def get_event_skew_metrics(self) -> dict[str, Any]:
        """Get metrics on event timing anomalies."""
        return {
            "late_events_dropped": self._late_events_dropped,
            "out_of_order_events": self._out_of_order_events,
            "total_events_processed": self._total_events_processed,
        }

    def _is_event_too_late(self, event_time: datetime, key: EventKey, now: datetime) -> bool:
        """Check if event is too far in the past (older than lateness threshold)."""
        age = now - event_time
        return age > LATENESS_THRESHOLD

    def _is_event_out_of_order(self, event_time: datetime, key: EventKey) -> bool:
        """Check if event timestamp is before the maximum timestamp seen for this key."""
        max_time = self._max_event_time_seen.get(key)
        if max_time is None:
            return False
        return event_time < max_time

    def bootstrap_from_state(self, state: dict[str, Any]) -> None:
        """Restore rolling history from persisted checkpoint state.
        
        Useful after restarts to continue using accumulated history rather
        than starting cold. Logs if restoration encounters errors.
        
        Args:
            state: Dict exported by export_state() method.
        """
        try:
            for key_dict, events_data in state.get("history", {}).items():
                key = EventKey(**json.loads(key_dict))
                queue: Deque[EventSnapshot] = deque()
                for event_dict in events_data:
                    # Convert ISO timestamp back to datetime
                    event_dict["timestamp"] = datetime.fromisoformat(event_dict["timestamp"])
                    snapshot = EventSnapshot(**event_dict)
                    queue.append(snapshot)
                self._history[key] = queue
            
            self._event_counts = defaultdict(int, {
                EventKey(**json.loads(k)): v 
                for k, v in state.get("event_counts", {}).items()
            })
            self._is_warmed_up = {
                EventKey(**json.loads(k)): v 
                for k, v in state.get("is_warmed_up", {}).items()
            }
            
            # Restore max_event_time_seen
            self._max_event_time_seen = {
                EventKey(**json.loads(k)): datetime.fromisoformat(v)
                for k, v in state.get("max_event_time_seen", {}).items()
            }
            
            self._total_events_processed = state.get("total_events_processed", 0)
            self._late_events_dropped = state.get("late_events_dropped", 0)
            self._out_of_order_events = state.get("out_of_order_events", 0)
            
            self._logger.info("Loaded feature state from bootstrap", extra={
                "extra": {
                    "keys_restored": len(self._history),
                    "total_events": self._total_events_processed,
                    "late_events_dropped": self._late_events_dropped,
                    "out_of_order_events": self._out_of_order_events,
                }
            })
        except Exception as exc:
            self._logger.warning("Failed to bootstrap feature state", extra={
                "extra": {"error": str(exc)}
            })

    def export_state(self) -> dict[str, Any]:
        """Serialize rolling history and metrics for persistence across restarts.
        
        Returns:
            Dict with history, event_counts, warm-up flags, and timing metrics.
        """
        return {
            "history": {
                json.dumps({"org_id": k.org_id, "api_id": k.api_id, "endpoint": k.endpoint, "method": k.method}): [
                    {
                        "timestamp": e.timestamp.isoformat(),
                        "status": e.status,
                        "latency_ms": e.latency_ms,
                        "schema_hash": e.schema_hash
                    }
                    for e in self._history[k]
                ]
                for k in self._history.keys()
            },
            "event_counts": {
                json.dumps({"org_id": k.org_id, "api_id": k.api_id, "endpoint": k.endpoint, "method": k.method}): v
                for k, v in self._event_counts.items()
            },
            "is_warmed_up": {
                json.dumps({"org_id": k.org_id, "api_id": k.api_id, "endpoint": k.endpoint, "method": k.method}): v
                for k, v in self._is_warmed_up.items()
            },
            "max_event_time_seen": {
                json.dumps({"org_id": k.org_id, "api_id": k.api_id, "endpoint": k.endpoint, "method": k.method}): v.isoformat()
                for k, v in self._max_event_time_seen.items()
            },
            "total_events_processed": self._total_events_processed,
            "late_events_dropped": self._late_events_dropped,
            "out_of_order_events": self._out_of_order_events,
        }


    def ingest(self, events: list[TelemetryEvent]) -> list[FeatureRow]:
        """Process batch of telemetry events and compute features for all touched endpoints.
        
        Handles late/out-of-order event filtering, maintains rolling windows,
        updates warm-up status, and computes feature vectors. Returns one
        FeatureRow per endpoint that received new events.
        
        Args:
            events: List of validated TelemetryEvent objects to ingest.
            
        Returns:
            List of FeatureRow objects (one per unique endpoint in batch).
        """
        if not events:
            return []

        touched_keys: set[EventKey] = set()
        now = datetime.now(UTC)

        for event in sorted(events, key=lambda e: e.timestamp):
            key = EventKey(
                org_id=event.org_id,
                api_id=event.api_id,
                endpoint=event.endpoint,
                method=event.method,
            )
            event_time = event.timestamp.astimezone(UTC)

            # Check for late events (too old)
            if self._is_event_too_late(event_time, key, now):
                self._late_events_dropped += 1
                self._logger.debug(
                    "Dropping late event (outside lateness threshold)",
                    extra={"extra": {
                        "event_age_minutes": (now - event_time).total_seconds() / 60,
                        "threshold_minutes": LATENESS_THRESHOLD.total_seconds() / 60,
                        "endpoint": key.endpoint,
                    }}
                )
                continue

            # Track out-of-order events but still process them
            if self._is_event_out_of_order(event_time, key):
                self._out_of_order_events += 1
                self._logger.debug(
                    "Processing out-of-order event",
                    extra={"extra": {
                        "event_time": event_time.isoformat(),
                        "max_time_seen": self._max_event_time_seen[key].isoformat() if key in self._max_event_time_seen else None,
                        "endpoint": key.endpoint,
                    }}
                )

            # Update max event time for this endpoint
            if key not in self._max_event_time_seen or event_time > self._max_event_time_seen[key]:
                self._max_event_time_seen[key] = event_time

            snapshot = EventSnapshot(
                timestamp=event_time,
                status=event.status,
                latency_ms=event.latency_ms,
                schema_hash=event.schema_hash,
            )
            self._history[key].append(snapshot)
            self._event_counts[key] += 1
            self._total_events_processed += 1
            touched_keys.add(key)
            self._prune_old(key, event_time)
            
            # Update warm-up status
            if not self._is_warmed_up.get(key, False):
                if self._event_counts[key] >= MIN_EVENTS_FOR_WARMUP:
                    self._is_warmed_up[key] = True
                    self._logger.info(
                        "Endpoint warmed up",
                        extra={"extra": {
                            "org_id": key.org_id,
                            "api_id": key.api_id,
                            "endpoint": key.endpoint,
                            "method": key.method,
                            "total_events": self._event_counts[key]
                        }}
                    )

        rows: list[FeatureRow] = []
        for key in touched_keys:
            latest_ts = self._history[key][-1].timestamp
            rows.append(
                FeatureRow(
                    time=latest_ts,
                    org_id=key.org_id,
                    api_id=key.api_id,
                    endpoint=key.endpoint,
                    method=key.method,
                    features=self._compute_features(key, latest_ts),
                    is_warmed_up=self._is_warmed_up.get(key, False),
                )
            )
        return rows

    def _prune_old(self, key: EventKey, now: datetime) -> None:
        """Remove events older than 15-minute window to manage memory."""
        cutoff = now - WINDOW_15M
        queue = self._history[key]
        while queue and queue[0].timestamp < cutoff:
            queue.popleft()

    def _events_in_window(self, key: EventKey, now: datetime, window: timedelta) -> list[EventSnapshot]:
        """Get events within sliding window relative to 'now' timestamp."""
        cutoff = now - window
        return [event for event in self._history[key] if event.timestamp >= cutoff]

    def _compute_features(self, key: EventKey, now: datetime) -> dict[str, float]:
        """Compute 11 time-series features from endpoint's rolling event history.
        
        Features include: latency percentiles/variance/delta, error rate/delta,
        traffic rate/delta, and schema change indicators. Handles missing data
        by replacing inf/nan with 0.0.
        
        Args:
            key: Endpoint identifier.
            now: Reference timestamp for window calculations.
            
        Returns:
            Dict mapping feature names to float values.
        """
        events_1m = self._events_in_window(key, now, WINDOW_1M)
        events_5m = self._events_in_window(key, now, WINDOW_5M)
        events_15m = self._events_in_window(key, now, WINDOW_15M)

        latencies_1m = [float(e.latency_ms) for e in events_1m]
        latencies_5m = [float(e.latency_ms) for e in events_5m]

        latency_mean = mean(latencies_1m) if latencies_1m else 0.0
        latency_p95 = float(np.percentile(latencies_1m, 95)) if latencies_1m else 0.0
        latency_variance = float(np.var(latencies_1m)) if len(latencies_1m) > 1 else 0.0

        latency_5m_mean = mean(latencies_5m) if latencies_5m else latency_mean
        latency_delta = latency_mean - latency_5m_mean

        error_rate_1m = self._error_rate(events_1m)
        error_rate_5m = self._error_rate(events_5m)
        error_rate_delta = error_rate_1m - error_rate_5m

        traffic_rps_1m = len(events_1m) / 60.0
        traffic_rps_5m = len(events_5m) / 300.0
        traffic_delta = traffic_rps_1m - traffic_rps_5m

        schema_fields_added, schema_fields_removed, schema_breaking_changes = self._schema_change_features(events_15m)

        features = {
            "latency_mean": latency_mean,
            "latency_p95": latency_p95,
            "latency_variance": latency_variance,
            "latency_delta": latency_delta,
            "error_rate": error_rate_1m,
            "error_rate_delta": error_rate_delta,
            "traffic_rps": traffic_rps_1m,
            "traffic_delta": traffic_delta,
            "schema_fields_added": schema_fields_added,
            "schema_fields_removed": schema_fields_removed,
            "schema_breaking_changes": schema_breaking_changes,
        }

        
        return {name: float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)) for name, value in features.items()}

    @staticmethod
    def _error_rate(events: list[EventSnapshot]) -> float:
        """Calculate error rate as proportion of 5xx responses.
        
        Args:
            events: List of event snapshots.
            
        Returns:
            Error rate in [0.0, 1.0] or 0.0 if no events.
        """
        if not events:
            return 0.0
        error_count = sum(1 for event in events if event.status >= 500)
        return error_count / float(len(events))

    @staticmethod
    def _schema_change_features(events: list[EventSnapshot]) -> tuple[float, float, float]:
        """Detect schema changes (hash changes) and breaking changes (hash change + error).
        
        Returns:
            Tuple of (fields_added, fields_removed, breaking_changes).
        """
        if len(events) < 2:
            return 0.0, 0.0, 0.0

        changes = 0
        breaking = 0
        previous_hash = events[0].schema_hash

        for event in events[1:]:
            if previous_hash and event.schema_hash and previous_hash != event.schema_hash:
                changes += 1
                if event.status >= 500:
                    breaking = 1
            previous_hash = event.schema_hash

        fields_added = float(changes)
        fields_removed = float(max(0, changes - 1))
        return fields_added, fields_removed, float(breaking)
