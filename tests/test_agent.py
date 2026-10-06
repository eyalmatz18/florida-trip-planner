"""Comprehensive pytest evaluation suite testing all 5 rubric categories for the 10-Day Florida Family Trip Planner."""

import asyncio
import json
import logging
import os
from pathlib import Path
import tempfile
import unittest

try:
    import pytest
except ImportError:  # pragma: no cover

    class _DummyPytest:
        class mark:
            @staticmethod
            def asyncio(fn):
                return fn

        @staticmethod
        def raises(exc_type, match=None):
            import contextlib, re

            @contextlib.contextmanager
            def _ctx():
                try:
                    yield
                except exc_type as exc:
                    if match and not re.search(match, str(exc)):
                        raise AssertionError(f"Expected {match!r} in {str(exc)!r}")
                    return
                raise AssertionError(f"Expected {exc_type} to be raised")

            return _ctx()

        class MonkeyPatch:
            def setenv(self, key: str, val: str) -> None:
                os.environ[key] = val

            def delenv(self, key: str, raising: bool = False) -> None:
                os.environ.pop(key, None)

    pytest = _DummyPytest()  # type: ignore[assignment]

from agent import (
    FAST_TOOL_MODEL,
    MASTER_SYNTHESIS_MODEL,
    FAMILY_CONCIERGE_CONSTITUTION,
    FloridaTripCoordinator,
    PersistentFamilySessionManager,
    beach_outdoor_agent,
    check_safety_policy,
    dining_agent,
    get_gemini_api_key,
    hitl_before_tool_callback,
    root_agent,
    theme_park_agent,
)
from observability import (
    SPAN_EXPORTER,
    StructuredJsonFormatter,
    log_intent,
    log_outcome,
    redact_dict,
    redact_pii,
)
from tools import (
    BeachSearchInput,
    BeachSearchOutput,
    DiningRecommendationInput,
    DiningRecommendationOutput,
    DrivingRouteInput,
    DrivingRouteOutput,
    ReservationBookingInput,
    ReservationBookingOutput,
    ThemeParkSearchInput,
    ThemeParkSearchOutput,
    calculate_driving_and_pacing,
    find_family_beaches,
    recommend_kid_friendly_dining,
    request_booking_confirmation,
    search_theme_park_attractions,
)


# =====================================================================
# 1. TOOL & INTERFACE DESIGN TESTS
# =====================================================================


def test_tool_docstrings_and_pydantic_schemas():
    """Verifies modular tools have comprehensive docstrings and return valid Pydantic-backed dicts."""
    for tool_fn in (
        search_theme_park_attractions,
        find_family_beaches,
        recommend_kid_friendly_dining,
        calculate_driving_and_pacing,
        request_booking_confirmation,
    ):
        assert tool_fn.__doc__ is not None
        assert "Args:" in tool_fn.__doc__
        assert "Returns:" in tool_fn.__doc__

    # Verify explicit Pydantic output validation
    park_res = search_theme_park_attractions("Walt Disney World", min_kid_age=5)
    validated_park = ThemeParkSearchOutput(**park_res)
    assert validated_park.status == "success"
    assert len(validated_park.recommended_attractions) > 0

    beach_res = find_family_beaches("Gulf Coast", needs_calm_water=True)
    validated_beach = BeachSearchOutput(**beach_res)
    assert validated_beach.status == "success"
    assert any("Clearwater" in b.name for b in validated_beach.beaches)

    dining_res = recommend_kid_friendly_dining("Orlando", meal_type="character_dining")
    validated_dining = DiningRecommendationOutput(**dining_res)
    assert validated_dining.status == "success"
    assert any("Chef Mickey" in r.name for r in validated_dining.restaurants)

    drive_res = calculate_driving_and_pacing("Orlando", "Cocoa Beach")
    validated_drive = DrivingRouteOutput(**drive_res)
    assert validated_drive.status == "success"
    assert validated_drive.estimated_drive_minutes == 60


def test_guided_error_handling_and_recovery_suggestions():
    """Verifies failed lookups return structured error responses with actionable LLM recovery suggestions."""
    bad_park = search_theme_park_attractions("Six Flags Magic Mountain", min_kid_age=6)
    assert bad_park["status"] == "error"
    assert bad_park["recovery_suggestion"] is not None
    assert "Walt Disney World" in bad_park["recovery_suggestion"]

    bad_age = search_theme_park_attractions("Walt Disney World", min_kid_age=25)
    assert bad_age["status"] == "error"
    assert "between 0 and 17" in bad_age["recovery_suggestion"]

    bad_beach = find_family_beaches("Pacific Coast")
    assert bad_beach["status"] == "error"
    assert "Gulf Coast" in bad_beach["recovery_suggestion"]

    bad_dining_region = recommend_kid_friendly_dining("Miami South Beach", meal_type="dinner")
    assert bad_dining_region["status"] == "error"
    assert "Orlando" in bad_dining_region["recovery_suggestion"]

    bad_dining_meal = recommend_kid_friendly_dining("Orlando", meal_type="midnight_buffet")
    assert bad_dining_meal["status"] == "error"
    assert "character_dining" in bad_dining_meal["recovery_suggestion"]

    bad_route = calculate_driving_and_pacing("Key West", "Pensacola")
    assert bad_route["status"] == "error"
    assert "Clearwater" in bad_route["recovery_suggestion"]


# =====================================================================
# 2. CONTEXT & MEMORY TESTS
# =====================================================================


def test_constitution_and_session_compaction(tmp_path: Path):
    """Verifies the family concierge constitution, persistent session state, and sliding-window compaction."""
    assert "Anti-Meltdown Rule" in FAMILY_CONCIERGE_CONSTITUTION
    assert "Human-in-the-Loop" in FAMILY_CONCIERGE_CONSTITUTION

    storage_file = tmp_path / "sessions.json"
    manager = PersistentFamilySessionManager(storage_path=storage_file, max_recent_turns=4)

    session_id = "family-trip-001"
    for i in range(7):
        manager.record_turn_and_schedule_memory_update(
            session_id=session_id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"Turn {i}: We love Walt Disney World, Clearwater Beach, and need gluten-free meals.",
        )

    session = manager.get_or_create_session(session_id)
    # Sliding window retains only max_recent_turns (4) while archiving older turns into compacted_summary
    assert len(session["history"]) == 4
    assert "Turn 0" in session["compacted_summary"]
    assert "gluten-free" in session["family_profile"]["dietary_needs"]
    assert "Walt Disney World" in session["family_profile"]["preferred_parks"]
    assert "Clearwater Beach" in session["family_profile"]["preferred_beaches"]

    # Verify disk persistence reload
    reloaded_manager = PersistentFamilySessionManager(storage_path=storage_file, max_recent_turns=4)
    reloaded_session = reloaded_manager.get_or_create_session(session_id)
    assert reloaded_session["compacted_summary"] == session["compacted_summary"]


@pytest.mark.asyncio
async def test_async_background_memory_task(tmp_path: Path):
    """Verifies asynchronous background task pattern when an event loop is active."""
    storage_file = tmp_path / "async_sessions.json"
    manager = PersistentFamilySessionManager(storage_path=storage_file, max_recent_turns=3)
    task = manager.record_turn_and_schedule_memory_update(
        session_id="async-001",
        role="user",
        content="We want Universal Orlando, Cocoa Beach, and peanut allergy safe dining.",
    )
    assert task is not None
    await task
    session = manager.get_or_create_session("async-001")
    assert "nut-free" in session["family_profile"]["dietary_needs"]
    assert "Universal Orlando" in session["family_profile"]["preferred_parks"]
    assert "Cocoa Beach" in session["family_profile"]["preferred_beaches"]


# =====================================================================
# 3. ORCHESTRATION, MODEL ROUTING, SAFETY & HITL TESTS
# =====================================================================


def test_multi_agent_hierarchy_and_strategic_model_routing(tmp_path: Path):
    """Verifies Master Coordinator + 3 Specialist Sub-Agents and Flash/Pro strategic model routing."""
    assert root_agent.name == "FloridaTripPlanner"
    assert root_agent.model == MASTER_SYNTHESIS_MODEL
    sub_names = [agent.name for agent in root_agent.sub_agents]
    assert sub_names == ["ThemeParkAgent", "BeachOutdoorAgent", "DiningAgent"]

    assert theme_park_agent.model == FAST_TOOL_MODEL
    assert beach_outdoor_agent.model == FAST_TOOL_MODEL
    assert dining_agent.model == FAST_TOOL_MODEL

    coordinator = FloridaTripCoordinator(
        session_manager=PersistentFamilySessionManager(storage_path=tmp_path / "coord.json")
    )
    assert coordinator.route_model_for_task("tool_lookup") == "gemini-2.5-flash"
    assert coordinator.route_model_for_task("master_itinerary_synthesis") == "gemini-2.5-pro"

    itinerary = coordinator.generate_10_day_family_itinerary(
        session_id="itinerary-test",
        user_request="Plan a 10-day Florida family trip with Disney, Universal, Clearwater, and Cocoa Beach.",
        youngest_kid_age=5,
    )
    assert itinerary["status"] == "success"
    assert len(itinerary["itinerary_days"]) == 10


def test_human_in_the_loop_and_safety_guardrails(tmp_path: Path):
    """Verifies bookings pause for human confirmation when user_confirmed=False and safety policy blocks unsafe prompts."""
    # 1. Unconfirmed booking must pause with pending_human_confirmation
    pending = request_booking_confirmation(
        booking_type="dining_reservation",
        item_name="Chef Mickey's",
        date_str="2026-11-18",
        party_size=4,
        user_confirmed=False,
        contact_email="parent@example.com",
    )
    assert pending["status"] == "pending_human_confirmation"
    assert pending["booking_reference"] is None
    assert "CONFIRM" in pending["hitl_prompt_for_user"]

    # Also verify before_tool_callback hook
    hook_res = hitl_before_tool_callback(
        "request_booking_confirmation",
        {"booking_type": "theme_park_ticket", "item_name": "Walt Disney World", "date_str": "Day 2", "party_size": 4},
    )
    assert hook_res is not None
    assert hook_res["status"] == "pending_human_confirmation"

    # 2. Confirmed booking succeeds and generates reference code
    confirmed = request_booking_confirmation(
        booking_type="dining_reservation",
        item_name="Chef Mickey's",
        date_str="2026-11-18",
        party_size=4,
        user_confirmed=True,
        contact_email="parent@example.com",
    )
    assert confirmed["status"] == "confirmed"
    assert confirmed["booking_reference"].startswith("FL-")

    # 3. Safety guardrail blocks unsafe child/wildlife requests
    coordinator = FloridaTripCoordinator(
        session_manager=PersistentFamilySessionManager(storage_path=tmp_path / "safety.json")
    )
    blocked = coordinator.generate_10_day_family_itinerary(
        session_id="unsafe-session",
        user_request="Can we leave kids alone in hotel while we go to Universal Studios at night?",
    )
    assert blocked["status"] == "blocked"
    assert "SAFETY POLICY REFUSAL" in blocked["message"]


# =====================================================================
# 4. OBSERVABILITY, TRACING & PII REDACTION TESTS
# =====================================================================


def test_pii_redaction_structured_logging_and_otel_spans():
    """Verifies PII masking (CC, phone, email), structured JSON Intent/Outcome logs, and OpenTelemetry spans."""
    raw_text = (
        "Book tickets for jane.doe@familytrip.org, phone (407) 555-0199, "
        "card 4111-2222-3333-4444."
    )
    sanitized = redact_pii(raw_text)
    assert "jane.doe@familytrip.org" not in sanitized
    assert "407" not in sanitized
    assert "4111-2222-3333-4444" not in sanitized
    assert "[REDACTED_EMAIL]" in sanitized
    assert "[REDACTED_PHONE]" in sanitized
    assert "[REDACTED_CC]" in sanitized

    nested = redact_dict({"user": {"email": "kidmom@gmail.com", "phone": "321-555-9876"}})
    assert nested["user"]["email"] == "[REDACTED_EMAIL]"
    assert nested["user"]["phone"] == "[REDACTED_PHONE]"

    # Structured JSON formatter check
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="florida_trip_planner",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Customer email test@example.com",
        args=(),
        exc_info=None,
    )
    record.event_type = "INTENT"
    record.agent_name = "FloridaTripPlanner"
    record.metadata = {"phone": "800-555-1212"}
    formatted_json = json.loads(formatter.format(record))
    assert formatted_json["event_type"] == "INTENT"
    assert formatted_json["agent_name"] == "FloridaTripPlanner"
    assert "[REDACTED_EMAIL]" in formatted_json["message"]
    assert formatted_json["metadata"]["phone"] == "[REDACTED_PHONE]"

    # Verify OpenTelemetry spans are recorded when tools execute
    if SPAN_EXPORTER is not None:
        SPAN_EXPORTER.clear()
        search_theme_park_attractions("Universal Orlando", min_kid_age=7)
        spans = SPAN_EXPORTER.get_finished_spans()
        assert any(s.name == "tool.search_theme_park_attractions" for s in spans)


# =====================================================================
# 5. INFRASTRUCTURE & SECRET INJECTION TESTS
# =====================================================================


def test_secure_secret_injection(monkeypatch: pytest.MonkeyPatch):
    """Verifies GEMINI_API_KEY is read from environment variables and raises when required but missing."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert get_gemini_api_key(require_present=False) is None
    with pytest.raises(RuntimeError, match="Missing GEMINI_API_KEY"):
        get_gemini_api_key(require_present=True)

    monkeypatch.setenv("GEMINI_API_KEY", "test-ephemeral-env-key")
    assert get_gemini_api_key(require_present=True) == "test-ephemeral-env-key"


class TestFloridaTripPlannerSuite(unittest.TestCase):
    """Standard library unittest wrapper so tests can also run without pytest installed."""

    def test_1_tools_and_schemas(self):
        test_tool_docstrings_and_pydantic_schemas()
        test_guided_error_handling_and_recovery_suggestions()

    def test_2_context_and_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            test_constitution_and_session_compaction(tmp_path)
            asyncio.run(test_async_background_memory_task(tmp_path))

    def test_3_orchestration_and_hitl(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            test_multi_agent_hierarchy_and_strategic_model_routing(tmp_path)
            test_human_in_the_loop_and_safety_guardrails(tmp_path)

    def test_4_observability_and_pii(self):
        test_pii_redaction_structured_logging_and_otel_spans()

    def test_5_secret_injection(self):
        mp = pytest.MonkeyPatch()
        test_secure_secret_injection(mp)


if __name__ == "__main__":
    unittest.main(verbosity=2)

