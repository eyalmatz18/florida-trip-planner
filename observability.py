"""Observability, structured JSON logging, OpenTelemetry tracing, and PII redaction."""

import functools
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

# Attempt to import OpenTelemetry, providing a clean fallback if not installed in minimal environments
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    resource = Resource.create({"service.name": "florida-family-trip-planner-adk"})
    provider = TracerProvider(resource=resource)
    # Use a custom or silent exporter by default so test output isn't cluttered, or memory exporter
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    SPAN_EXPORTER = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(SPAN_EXPORTER))
    trace.set_tracer_provider(provider)
    TRACER = trace.get_tracer("florida.trip.planner")
    OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    OTEL_AVAILABLE = False
    SPAN_EXPORTER = None

    class DummySpan:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def set_attribute(self, key: str, value: Any) -> None:
            pass

        def record_exception(self, exception: Exception) -> None:
            pass

    class DummyTracer:
        def start_as_current_span(self, name: str, **kwargs):
            return DummySpan()

    TRACER = DummyTracer()


# Regular expressions for PII detection and redaction
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_REGEX = re.compile(
    r"\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b"
)
CREDIT_CARD_REGEX = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")


def redact_pii(text: str) -> str:
    """Redacts sensitive Personally Identifiable Information (PII) from a string.

    Masks credit card numbers, phone numbers, and email addresses before logging
    or storing in telemetry spans.

    Args:
        text: The input string potentially containing sensitive PII.

    Returns:
        The sanitized string with PII replaced by redaction tokens.
    """
    if not isinstance(text, str):
        text = str(text)
    text = CREDIT_CARD_REGEX.sub("[REDACTED_CC]", text)
    text = EMAIL_REGEX.sub("[REDACTED_EMAIL]", text)
    text = PHONE_REGEX.sub("[REDACTED_PHONE]", text)
    return text


def redact_dict(data: Any) -> Any:
    """Recursively redacts PII from dictionaries, lists, and primitive values."""
    if isinstance(data, dict):
        return {k: redact_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [redact_dict(item) for item in data]
    elif isinstance(data, str):
        return redact_pii(data)
    return data


class StructuredJsonFormatter(logging.Formatter):
    """Custom logging formatter that outputs structured JSON logs with PII redaction."""

    def format(self, record: logging.LogRecord) -> str:
        log_Entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_pii(record.getMessage()),
        }
        if hasattr(record, "event_type"):
            log_Entry["event_type"] = getattr(record, "event_type")
        if hasattr(record, "agent_name"):
            log_Entry["agent_name"] = getattr(record, "agent_name")
        if hasattr(record, "metadata"):
            log_Entry["metadata"] = redact_dict(getattr(record, "metadata"))
        if record.exc_info:
            log_Entry["exception"] = redact_pii(self.formatException(record.exc_info))
        return json.dumps(log_Entry)


def get_structured_logger(name: str = "florida_trip_planner") -> logging.Logger:
    """Configures and returns a structured JSON logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredJsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = get_structured_logger()


def log_intent(
    agent_name: str,
    action: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Logs the 'Intent' (what the agent or tool plans to do) in structured JSON."""
    logger.info(
        f"INTENT: {agent_name} plans to execute {action}",
        extra={
            "event_type": "INTENT",
            "agent_name": agent_name,
            "metadata": details or {},
        },
    )


def log_outcome(
    agent_name: str,
    action: str,
    status: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Logs the 'Outcome' (what happened after execution) in structured JSON."""
    logger.info(
        f"OUTCOME: {agent_name} completed {action} with status={status}",
        extra={
            "event_type": "OUTCOME",
            "agent_name": agent_name,
            "metadata": {"status": status, **(details or {})},
        },
    )


def traced_tool(func: Callable) -> Callable:
    """Decorator that wraps an ADK tool in an OpenTelemetry span and logs Intent & Outcome."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = func.__name__
        safe_kwargs = redact_dict(kwargs)
        log_intent(
            agent_name="ToolExecutor",
            action=tool_name,
            details={"args": redact_dict(list(args)), "kwargs": safe_kwargs},
        )
        start_time = time.perf_counter()
        with TRACER.start_as_current_span(f"tool.{tool_name}") as span:
            span.set_attribute("tool.name", tool_name)
            span.set_attribute("tool.kwargs", json.dumps(safe_kwargs, default=str))
            try:
                result = func(*args, **kwargs)
                duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                span.set_attribute("tool.status", "SUCCESS")
                span.set_attribute("tool.duration_ms", duration_ms)
                log_outcome(
                    agent_name="ToolExecutor",
                    action=tool_name,
                    status="SUCCESS",
                    details={"duration_ms": duration_ms},
                )
                return result
            except Exception as exc:
                duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                span.set_attribute("tool.status", "ERROR")
                span.record_exception(exc)
                log_outcome(
                    agent_name="ToolExecutor",
                    action=tool_name,
                    status="ERROR",
                    details={"duration_ms": duration_ms, "error": str(exc)},
                )
                raise

    return wrapper
