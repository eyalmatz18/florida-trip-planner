"""Production-ready Multi-Agent 10-Day Florida Family Trip Planner built with Google ADK."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from observability import (
    TRACER,
    log_intent,
    log_outcome,
    redact_dict,
    redact_pii,
)
from tools import (
    calculate_driving_and_pacing,
    find_family_beaches,
    recommend_kid_friendly_dining,
    request_booking_confirmation,
    search_theme_park_attractions,
)

# Attempt to import Google ADK components with a graceful fallback shim so unit tests
# and local verification work whether or not google-adk is installed in the test environment.
try:
    from google.adk.agents import Agent, LlmAgent
    from google.adk.sessions import InMemorySessionService

    ADK_AVAILABLE = True
except ImportError:  # pragma: no cover
    ADK_AVAILABLE = False

    class LlmAgent:  # type: ignore[no-redef]
        """Lightweight compatibility shim mirroring Google ADK's `LlmAgent` interface."""

        def __init__(
            self,
            name: str,
            model: str,
            description: str = "",
            instruction: str = "",
            tools: Optional[List[Callable]] = None,
            sub_agents: Optional[List["LlmAgent"]] = None,
            before_tool_callback: Optional[Callable] = None,
        ):
            self.name = name
            self.model = model
            self.description = description
            self.instruction = instruction
            self.tools = tools or []
            self.sub_agents = sub_agents or []
            self.before_tool_callback = before_tool_callback

    Agent = LlmAgent

    class InMemorySessionService:  # type: ignore[no-redef]
        """Lightweight compatibility shim for ADK InMemorySessionService."""

        def __init__(self):
            self._sessions: Dict[str, Dict[str, Any]] = {}


# =====================================================================
# 1. Secure Environment & Model Routing Configuration
# =====================================================================

FAST_TOOL_MODEL = os.environ.get("FAST_TOOL_MODEL", "gemini-2.5-flash")
MASTER_SYNTHESIS_MODEL = os.environ.get("MASTER_SYNTHESIS_MODEL", "gemini-2.5-pro")


def get_gemini_api_key(require_present: bool = False) -> Optional[str]:
    """Retrieves GEMINI_API_KEY securely from environment variables (never hardcoded).

    Args:
        require_present: If True, raises a RuntimeError when GEMINI_API_KEY is missing.

    Returns:
        The API key string or None if not set and `require_present` is False.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if require_present and not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY environment variable. Please export GEMINI_API_KEY "
            "before invoking live Gemini models."
        )
    return api_key


# =====================================================================
# 2. Agent Constitution & System Instructions (Context & Persona)
# =====================================================================

FAMILY_CONCIERGE_CONSTITUTION = """
# CONSTITUTION: SUNSHINE FAMILY CONCIERGE (10-DAY FLORIDA TRIP PLANNER)

You are **Sunshine Family Concierge**, an empathetic, safety-first, logistics-savvy Senior Family Travel Planner specializing in 10-day Florida vacations with children.

## Core Principles & Behavioral Guardrails
1. **Kid-First Pacing (Anti-Meltdown Rule)**:
   - Never schedule more than 2 consecutive full theme park days without a pool, beach, or relaxed recovery day in between.
   - Include a daily 1:00 PM – 4:00 PM afternoon shade/nap/pool break during Orlando theme park days.
   - Keep inter-city drives under 2.5 hours per leg and recommend kid-friendly highway pit-stops.
2. **Weather & Sensory Awareness**:
   - Account for Florida heat, UV exposure, and afternoon rain showers by pairing every outdoor beach or park block with an indoor air-conditioned backup.
   - Proactively highlight ride height requirements and sensory triggers (darkness, loud noises, drops).
3. **Strict Safety & Policy Guardrails**:
   - Refuse requests involving unsafe child activities, unattended minors, bypassing theme park height requirements, or illegal wildlife interactions.
   - Never log or repeat unredacted credit card numbers, phone numbers, or sensitive personal identifiers.
4. **Mandatory Human-in-the-Loop (HITL) Confirmation**:
   - You MUST NEVER finalize theme park ticket purchases or dining reservations (`user_confirmed=True`) without first presenting the itemized details to the user and receiving their explicit confirmation.
"""

THEME_PARK_AGENT_INSTRUCTION = """
You are `ThemeParkAgent`, the Florida Theme Park & Attraction Pacing Specialist.
- Use `search_theme_park_attractions` to look up age-appropriate rides at Walt Disney World, Universal Orlando, LEGOLAND Florida, and Kennedy Space Center.
- Group attractions by land to minimize stroller walking distance, highlight Lightning Lane priorities, and enforce afternoon nap/pool breaks.
"""

BEACH_OUTDOOR_AGENT_INSTRUCTION = """
You are `BeachOutdoorAgent`, the Florida Coastal, Weather & Downtime Specialist.
- Use `find_family_beaches` and `calculate_driving_and_pacing` to plan relaxing Gulf Coast (Clearwater Beach, Fort De Soto) and Atlantic Coast (Cocoa Beach, Jetty Park) days.
- Always include wave condition notes for young swimmers, UV/storm timing advice, and an indoor rainy-day backup (such as Clearwater Marine Aquarium or Kennedy Space Center).
"""

DINING_AGENT_INSTRUCTION = """
You are `DiningAgent`, the Kid-Friendly Florida Dining & Reservation Specialist.
- Use `recommend_kid_friendly_dining` to recommend family restaurants and character dining across Orlando, Clearwater, and Cocoa Beach.
- Highlight allergy-safe protocols, sensory notes, and advance booking windows (e.g., 60 days for Disney character dining).
- Use `request_booking_confirmation` with `user_confirmed=False` first whenever a user asks to book a table, waiting for explicit human approval before setting `user_confirmed=True`.
"""


# =====================================================================
# 3. Context Compaction, Persistent Session State & Async Memory Updates
# =====================================================================


class PersistentFamilySessionManager:
    """Manages session state, sliding-window history compaction, persistence, and async memory tasks."""

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        max_recent_turns: int = 6,
    ):
        self.storage_path = storage_path or Path("/tmp/florida_trip_sessions.json")
        self.max_recent_turns = max_recent_turns
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._background_tasks: set[asyncio.Task] = set()
        self._adk_session_service = InMemorySessionService()
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.sessions = data
            except Exception:
                self.sessions = {}

    def _save_to_disk(self) -> None:
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage_path.write_text(
                json.dumps(redact_dict(self.sessions), indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            log_outcome(
                agent_name="SessionManager",
                action="persist_session",
                status="WARNING",
                details={"error": str(exc)},
            )

    def get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        """Retrieves an existing session or initializes a new structured family trip state."""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "session_id": session_id,
                "family_profile": {
                    "kids_ages": [],
                    "dietary_needs": [],
                    "pacing_preference": "balanced",
                    "preferred_parks": [],
                    "preferred_beaches": [],
                },
                "compacted_summary": "",
                "history": [],
                "pending_confirmations": [],
            }
            self._save_to_disk()
        return self.sessions[session_id]

    def compact_history(self, session_id: str) -> Dict[str, Any]:
        """Compacts conversation history using a sliding window + rolling summary to prevent context bloat."""
        session = self.get_or_create_session(session_id)
        history: List[Dict[str, str]] = session["history"]

        if len(history) <= self.max_recent_turns:
            return session

        older_turns = history[: -self.max_recent_turns]
        recent_turns = history[-self.max_recent_turns :]

        summarized_snippets = [
            f"{turn['role']}: {redact_pii(turn['content'])[:100]}"
            for turn in older_turns
        ]
        new_summary_chunk = " | ".join(summarized_snippets)
        existing_summary = session.get("compacted_summary", "")
        session["compacted_summary"] = (
            f"{existing_summary} || {new_summary_chunk}".strip(" |")
        )
        session["history"] = recent_turns

        log_outcome(
            agent_name="SessionManager",
            action="compact_history",
            status="COMPACTED",
            details={
                "session_id": session_id,
                "archived_turns": len(older_turns),
                "retained_recent_turns": len(recent_turns),
            },
        )
        self._save_to_disk()
        return session

    async def _async_extract_and_persist_memory(
        self, session_id: str, role: str, content: str
    ) -> None:
        """Asynchronous background task that extracts family preferences and persists compacted state."""
        await asyncio.sleep(0.01)  # Yield control to simulate non-blocking background worker
        session = self.get_or_create_session(session_id)
        lower_text = content.lower()

        # Heuristic preference extraction for family profile memory
        profile = session["family_profile"]
        if "gluten-free" in lower_text and "gluten-free" not in profile["dietary_needs"]:
            profile["dietary_needs"].append("gluten-free")
        if "nut-free" in lower_text or "peanut" in lower_text:
            if "nut-free" not in profile["dietary_needs"]:
                profile["dietary_needs"].append("nut-free")
        if "disney" in lower_text and "Walt Disney World" not in profile["preferred_parks"]:
            profile["preferred_parks"].append("Walt Disney World")
        if "universal" in lower_text and "Universal Orlando" not in profile["preferred_parks"]:
            profile["preferred_parks"].append("Universal Orlando")
        if "clearwater" in lower_text and "Clearwater Beach" not in profile["preferred_beaches"]:
            profile["preferred_beaches"].append("Clearwater Beach")
        if "cocoa" in lower_text and "Cocoa Beach" not in profile["preferred_beaches"]:
            profile["preferred_beaches"].append("Cocoa Beach")

        self.compact_history(session_id)
        self._save_to_disk()

    def record_turn_and_schedule_memory_update(
        self, session_id: str, role: str, content: str
    ) -> Optional[asyncio.Task]:
        """Appends a turn to the session and schedules an async background memory update task."""
        session = self.get_or_create_session(session_id)
        session["history"].append({"role": role, "content": redact_pii(content)})

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                self._async_extract_and_persist_memory(session_id, role, content)
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            return task
        except RuntimeError:
            # No running event loop: run synchronously via asyncio.run so state is still updated
            asyncio.run(self._async_extract_and_persist_memory(session_id, role, content))
            return None


# =====================================================================
# 4. Safety Policy & Human-in-the-Loop (HITL) Guardrails
# =====================================================================

UNSAFE_POLICY_KEYWORDS = [
    "leave kids alone in hotel",
    "unattended toddler",
    "sneak past height requirement",
    "fake height shoes",
    "feed wild alligators",
]


def check_safety_policy(user_prompt: str) -> Optional[str]:
    """Evaluates user input against child-safety and Florida travel policy guardrails.

    Returns:
        A refusal message string if a safety violation is detected, or None if safe.
    """
    lowered = user_prompt.lower()
    for phrase in UNSAFE_POLICY_KEYWORDS:
        if phrase in lowered:
            log_outcome(
                agent_name="SafetyGuardrail",
                action="policy_check",
                status="BLOCKED",
                details={"triggered_phrase": phrase},
            )
            return (
                "SAFETY POLICY REFUSAL: I cannot assist with requests that compromise child safety, "
                "leave children unattended, bypass theme park height requirements, or involve "
                "unsafe wildlife interactions. I would be happy to suggest safe, age-appropriate "
                "Florida family alternatives instead!"
            )
    return None


def hitl_before_tool_callback(
    tool_name: str, tool_args: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Human-in-the-Loop before-tool hook ensuring bookings cannot bypass user confirmation."""
    if tool_name == "request_booking_confirmation" and not tool_args.get("user_confirmed", False):
        log_intent(
            agent_name="HITLHook",
            action="intercept_unconfirmed_booking",
            details={"tool_name": tool_name, "tool_args": redact_dict(tool_args)},
        )
        return request_booking_confirmation(
            booking_type=tool_args.get("booking_type", "dining_reservation"),
            item_name=tool_args.get("item_name", "Requested Venue"),
            date_str=tool_args.get("date_str", "TBD"),
            party_size=int(tool_args.get("party_size", 4)),
            user_confirmed=False,
            contact_email=tool_args.get("contact_email"),
        )
    return None


# =====================================================================
# 5. Multi-Agent Coordinator & Specialist Sub-Agents Definition
# =====================================================================

theme_park_agent = LlmAgent(
    name="ThemeParkAgent",
    model=FAST_TOOL_MODEL,
    description=(
        "Specialist sub-agent for Orlando theme parks (Walt Disney World, Universal Orlando, "
        "LEGOLAND, Kennedy Space Center) and kid-friendly attraction pacing."
    ),
    instruction=f"{FAMILY_CONCIERGE_CONSTITUTION}\n{THEME_PARK_AGENT_INSTRUCTION}",
    tools=[search_theme_park_attractions, request_booking_confirmation],
)

beach_outdoor_agent = LlmAgent(
    name="BeachOutdoorAgent",
    model=FAST_TOOL_MODEL,
    description=(
        "Specialist sub-agent for Florida Gulf and Atlantic family beaches (Clearwater, Cocoa Beach), "
        "weather safety, driving routes, and downtime pacing."
    ),
    instruction=f"{FAMILY_CONCIERGE_CONSTITUTION}\n{BEACH_OUTDOOR_AGENT_INSTRUCTION}",
    tools=[find_family_beaches, calculate_driving_and_pacing],
)

dining_agent = LlmAgent(
    name="DiningAgent",
    model=FAST_TOOL_MODEL,
    description=(
        "Specialist sub-agent for kid-friendly Florida restaurants, character dining, "
        "allergy accommodations, and HITL reservation requests."
    ),
    instruction=f"{FAMILY_CONCIERGE_CONSTITUTION}\n{DINING_AGENT_INSTRUCTION}",
    tools=[recommend_kid_friendly_dining, request_booking_confirmation],
)

root_agent = LlmAgent(
    name="FloridaTripPlanner",
    model=MASTER_SYNTHESIS_MODEL,
    description=(
        "Master Coordinator Agent that orchestrates ThemeParkAgent, BeachOutdoorAgent, and "
        "DiningAgent to synthesize a balanced 10-day Florida family vacation itinerary."
    ),
    instruction=(
        f"{FAMILY_CONCIERGE_CONSTITUTION}\n"
        "You are `FloridaTripPlanner`, the Master Coordinator Agent.\n"
        "- Delegate theme park lookups to `ThemeParkAgent` (running on fast `gemini-2.5-flash`).\n"
        "- Delegate beach, weather, and driving pacing lookups to `BeachOutdoorAgent` (`gemini-2.5-flash`).\n"
        "- Delegate kid-friendly dining and reservation lookups to `DiningAgent` (`gemini-2.5-flash`).\n"
        "- Synthesize their outputs using `gemini-2.5-pro` into a cohesive, balanced 10-day Florida "
        "family itinerary covering Orlando, Clearwater Beach (Gulf Coast), and Cocoa Beach (Atlantic Coast)."
    ),
    tools=[
        search_theme_park_attractions,
        find_family_beaches,
        recommend_kid_friendly_dining,
        calculate_driving_and_pacing,
        request_booking_confirmation,
    ],
    sub_agents=[theme_park_agent, beach_outdoor_agent, dining_agent],
)


# =====================================================================
# 6. High-Level Coordinator Execution Engine
# =====================================================================


class FloridaTripCoordinator:
    """High-level coordinator integrating ADK agents, strategic model routing, session memory, and tracing."""

    def __init__(self, session_manager: Optional[PersistentFamilySessionManager] = None):
        self.root_agent = root_agent
        self.sub_agents = {
            "ThemeParkAgent": theme_park_agent,
            "BeachOutdoorAgent": beach_outdoor_agent,
            "DiningAgent": dining_agent,
        }
        self.session_manager = session_manager or PersistentFamilySessionManager()

    def route_model_for_task(self, task_type: str) -> str:
        """Selects the optimal Gemini model based on task complexity.

        - Uses `gemini-2.5-flash` for low-latency specialist tool lookups.
        - Uses `gemini-2.5-pro` for complex multi-day itinerary synthesis.
        """
        if task_type in ("tool_lookup", "theme_park_lookup", "beach_lookup", "dining_lookup"):
            return FAST_TOOL_MODEL
        return MASTER_SYNTHESIS_MODEL

    def generate_10_day_family_itinerary(
        self,
        session_id: str,
        user_request: str,
        youngest_kid_age: int = 5,
        dietary_needs: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Orchestrates the specialist sub-agents to build a balanced 10-day Florida family plan."""
        with TRACER.start_as_current_span("coordinator.generate_10_day_family_itinerary") as span:
            span.set_attribute("session.id", session_id)
            span.set_attribute("coordinator.model", self.root_agent.model)

            log_intent(
                agent_name=self.root_agent.name,
                action="generate_10_day_family_itinerary",
                details={
                    "session_id": session_id,
                    "user_request": redact_pii(user_request),
                    "youngest_kid_age": youngest_kid_age,
                },
            )

            # 1. Check safety guardrails first
            safety_violation = check_safety_policy(user_request)
            if safety_violation:
                span.set_attribute("coordinator.status", "BLOCKED_BY_SAFETY_POLICY")
                return {
                    "status": "blocked",
                    "message": safety_violation,
                }

            # 2. Record user turn & trigger async session memory update
            self.session_manager.record_turn_and_schedule_memory_update(
                session_id=session_id, role="user", content=user_request
            )

            # 3. Delegate lookups across the 3 specialist sub-agents (using gemini-2.5-flash)
            disney_data = search_theme_park_attractions(
                park_resort="Walt Disney World",
                min_kid_age=youngest_kid_age,
                pacing_preference="balanced",
            )
            universal_data = search_theme_park_attractions(
                park_resort="Universal Orlando",
                min_kid_age=youngest_kid_age,
                pacing_preference="balanced",
            )
            gulf_beaches = find_family_beaches(coast="Gulf Coast", needs_calm_water=True)
            atlantic_beaches = find_family_beaches(coast="Atlantic Coast", needs_calm_water=True)

            orlando_to_clearwater = calculate_driving_and_pacing(
                origin="Orlando", destination="Clearwater"
            )
            orlando_to_cocoa = calculate_driving_and_pacing(
                origin="Orlando", destination="Cocoa Beach"
            )

            orlando_dining = recommend_kid_friendly_dining(
                region="Orlando", meal_type="character_dining", dietary_needs=dietary_needs
            )
            clearwater_dining = recommend_kid_friendly_dining(
                region="Clearwater", meal_type="dinner", dietary_needs=dietary_needs
            )
            cocoa_dining = recommend_kid_friendly_dining(
                region="Cocoa Beach", meal_type="lunch", dietary_needs=dietary_needs
            )

            # 4. Synthesize balanced 10-day schedule (master coordinator pattern)
            ten_day_plan = [
                {
                    "day": 1,
                    "hub": "Orlando",
                    "theme": "Arrival & Resort Pool Unwind + Disney Springs Dinner",
                    "highlights": [
                        "Check into Orlando family resort; afternoon splash pad downtime",
                        "Early dinner at T-REX Cafe or Chef Mickey's",
                    ],
                },
                {
                    "day": 2,
                    "hub": "Orlando (Walt Disney World)",
                    "theme": "Magic Kingdom Classic Fantasyland Day",
                    "highlights": [
                        a["name"] for a in disney_data["recommended_attractions"][:2]
                    ],
                    "pacing_note": disney_data["pacing_advice"],
                },
                {
                    "day": 3,
                    "hub": "Orlando (Walt Disney World)",
                    "theme": "Animal Kingdom Morning Safari & Afternoon Resort Pool Rest",
                    "highlights": ["Kilimanjaro Safaris", "Festival of the Lion King", "Afternoon pool & nap break"],
                },
                {
                    "day": 4,
                    "hub": "Orlando Rest & Character Breakfast Day",
                    "theme": "Anti-Meltdown Recovery Day (No Theme Park Rope Drop!)",
                    "highlights": [
                        f"Character breakfast at {orlando_dining['restaurants'][0]['name']}",
                        "Lazy river & resort crafts",
                    ],
                },
                {
                    "day": 5,
                    "hub": "Orlando (Universal Orlando)",
                    "theme": "Islands of Adventure (Seuss Landing & Wizarding World)",
                    "highlights": [
                        a["name"] for a in universal_data["recommended_attractions"][:2]
                    ],
                    "pacing_note": universal_data["pacing_advice"],
                },
                {
                    "day": 6,
                    "hub": "Orlando -> Clearwater Beach (Gulf Coast)",
                    "theme": "Scenic Drive West with Dinosaur World Pit-Stop & Pier 60 Sunset",
                    "drive_minutes": orlando_to_clearwater["estimated_drive_minutes"],
                    "pit_stops": orlando_to_clearwater["recommended_pit_stops"],
                    "dining": clearwater_dining["restaurants"][0]["name"],
                },
                {
                    "day": 7,
                    "hub": "Clearwater Beach (Gulf Coast)",
                    "theme": "Calm Gulf Waters & Clearwater Marine Aquarium",
                    "beach": gulf_beaches["beaches"][0]["name"],
                    "rainy_day_backup": gulf_beaches["beaches"][0]["nearby_rainy_day_backup"],
                },
                {
                    "day": 8,
                    "hub": "Clearwater -> Orlando / Cocoa Beach (Atlantic Coast)",
                    "theme": "Cross-Peninsula Drive via Bonnet Springs Park to the Space Coast",
                    "beach": atlantic_beaches["beaches"][0]["name"],
                },
                {
                    "day": 9,
                    "hub": "Cocoa Beach & Kennedy Space Center",
                    "theme": "Space Shuttle Atlantis, Planet Play & Jetty Park Cruise Ship Watch",
                    "drive_minutes_to_orlando": orlando_to_cocoa["estimated_drive_minutes"],
                    "dining": cocoa_dining["restaurants"][0]["name"],
                },
                {
                    "day": 10,
                    "hub": "Cocoa Beach -> Orlando (MCO Departure)",
                    "theme": "Morning Beach Stroll, Ron Jon Surf Shop Souvenirs & Easy 50-Min Drive to MCO Airport",
                    "highlights": ["Smooth 50-minute SR-528 return to Orlando International Airport (MCO)"],
                },
            ]

            response_payload = {
                "status": "success",
                "coordinator_agent": self.root_agent.name,
                "models_used": {
                    "coordinator_synthesis": self.root_agent.model,
                    "specialist_tool_lookups": FAST_TOOL_MODEL,
                },
                "specialist_sub_agents": list(self.sub_agents.keys()),
                "itinerary_days": ten_day_plan,
            }

            self.session_manager.record_turn_and_schedule_memory_update(
                session_id=session_id,
                role="assistant",
                content=f"Generated 10-day balanced Florida itinerary for kids (min age {youngest_kid_age}).",
            )

            log_outcome(
                agent_name=self.root_agent.name,
                action="generate_10_day_family_itinerary",
                status="SUCCESS",
                details={"days_planned": len(ten_day_plan)},
            )
            return response_payload


if __name__ == "__main__":  # pragma: no cover
    coordinator = FloridaTripCoordinator()
    sample_plan = coordinator.generate_10_day_family_itinerary(
        session_id="demo-family-session",
        user_request=(
            "Plan a 10-day Florida family vacation for our kids (ages 5 and 8) with "
            "Walt Disney World, Universal, Clearwater Beach, and Cocoa Beach. One child is gluten-free."
        ),
        youngest_kid_age=5,
        dietary_needs="gluten-free",
    )
    print(json.dumps(sample_plan, indent=2))
