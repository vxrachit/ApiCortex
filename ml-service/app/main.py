"""Entry point for ML service worker.

Configures the event loop and starts the async inference worker that
consumes telemetry events from Kafka, computes features, runs predictions,
and persists results to TimescaleDB.
"""
from __future__ import annotations

import asyncio
import importlib

from app.config import get_settings
from workers.inference_worker import InferenceWorker, install_signal_handlers


def _configure_event_loop() -> None:
    """Install uvloop if available for performance, else use default event loop."""
    try:
        uvloop = importlib.import_module("uvloop")

        uvloop.install()
    except Exception:
        
        return


async def _run() -> None:
    """Initialize worker and run the inference loop until shutdown is requested."""
    settings = get_settings()
    worker = InferenceWorker(settings)
    install_signal_handlers(worker)
    await worker.run()


def main() -> None:
    """Configure event loop and run async worker main loop."""
    _configure_event_loop()
    asyncio.run(_run())


if __name__ == "__main__":
    main()
