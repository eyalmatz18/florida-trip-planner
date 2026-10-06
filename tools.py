"""Modular ADK tools with explicit Pydantic schemas and guided error recovery for the 10-Day Florida Family Trip Planner."""

from typing import Any, Dict, List, Literal, Optional, get_args, get_origin

try:
    from pydantic import BaseModel, Field, ValidationError
except ImportError:  # pragma: no cover

    class ValidationError(ValueError):  # type: ignore[no-redef]
        """Fallback validation error when pydantic is not installed in minimal environments."""

    def Field(default: Any = ..., **kwargs: Any) -> Any:  # type: ignore[no-redef]
        return {"__field__": True, "default": default, **kwargs}

    class BaseModel:  # type: ignore[no-redef]
        """Lightweight Pydantic v2-compatible BaseModel shim for stdlib-only environments."""

        def __init__(self, **data: Any):
            annotations = getattr(self.__class__, "__annotations__", {})
            for name, anno in annotations.items():
                field_spec = getattr(self.__class__, name, ...)
                if isinstance(field_spec, dict) and field_spec.get("__field__"):
                    default_val = field_spec.get("default", ...)
                    default_factory = field_spec.get("default_factory")
                    ge_val = field_spec.get("ge")
                    le_val = field_spec.get("le")
                else:
                    default_val = field_spec
                    default_factory = None
                    ge_val = None
                    le_val = None

                if name in data:
                    val = data[name]
                elif default_factory is not None:
                    val = default_factory()
                elif default_val is not ...:
                    val = default_val
                else:
                    raise ValidationError(f"Field '{name}' is required")

                if ge_val is not None and isinstance(val, (int, float)) and val < ge_val:
                    raise ValidationError(f"Field '{name}' must be >= {ge_val}")
                if le_val is not None and isinstance(val, (int, float)) and val > le_val:
                    raise ValidationError(f"Field '{name}' must be <= {le_val}")

                origin = get_origin(anno)
                if origin is Literal and val not in get_args(anno):
                    raise ValidationError(
                        f"Field '{name}' must be one of {get_args(anno)}, got {val!r}"
                    )
                if origin is list and isinstance(val, list):
                    inner_types = get_args(anno)
                    if inner_types and isinstance(inner_types[0], type) and issubclass(inner_types[0], BaseModel):
                        val = [
                            inner_types[0](**item) if isinstance(item, dict) else item
                            for item in val
                        ]
                setattr(self, name, val)

        def model_dump(self) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for k, v in self.__dict__.items():
                if isinstance(v, BaseModel):
                    out[k] = v.model_dump()
                elif isinstance(v, list):
                    out[k] = [
                        item.model_dump() if isinstance(item, BaseModel) else item
                        for item in v
                    ]
                else:
                    out[k] = v
            return out

from observability import log_intent, log_outcome, redact_pii, traced_tool


# =====================================================================
# Explicit Pydantic Input & Output Schemas
# =====================================================================


class ThemeParkSearchInput(BaseModel):
    """Input schema for searching Florida theme park attractions."""

    park_resort: str = Field(
        ...,
        description="Name of the theme park resort: 'Walt Disney World', 'Universal Orlando', 'LEGOLAND Florida', or 'Kennedy Space Center'.",
    )
    min_kid_age: int = Field(
        ...,
        ge=0,
        le=17,
        description="Age of the youngest child in years (0-17) to filter appropriate rides and pacing.",
    )
    pacing_preference: Literal["relaxed", "balanced", "thrill_focused"] = Field(
        default="balanced",
        description="Pacing style for the family day ('relaxed', 'balanced', or 'thrill_focused').",
    )


class AttractionItem(BaseModel):
    """Details of a single theme park attraction."""

    name: str = Field(..., description="Attraction name.")
    park_area: str = Field(..., description="Specific park or land name.")
    min_age_recommended: int = Field(..., description="Minimum recommended age.")
    height_requirement_inches: int = Field(
        ..., description="Minimum height in inches (0 if none)."
    )
    sensory_notes: str = Field(
        ..., description="Notes on darkness, loud noises, drops, or water."
    )
    lightning_lane_priority: Literal["high", "medium", "low"] = Field(
        ..., description="Priority for Lightning Lane / Express Pass booking."
    )


class ThemeParkSearchOutput(BaseModel):
    """Output schema for theme park attraction search results."""

    status: Literal["success", "error"] = Field(..., description="Lookup status.")
    park_resort: str = Field(..., description="Normalized park resort name.")
    recommended_attractions: List[AttractionItem] = Field(
        default_factory=list, description="List of matching family attractions."
    )
    pacing_advice: str = Field(
        default="",
        description="Actionable pacing tips for families with children (stroller naps, shade breaks).",
    )
    error_message: Optional[str] = Field(
        default=None, description="Error details if status is 'error'."
    )
    recovery_suggestion: Optional[str] = Field(
        default=None,
        description="Guided recovery suggestion for the LLM if the lookup fails.",
    )


class BeachSearchInput(BaseModel):
    """Input schema for finding family-friendly Florida beaches."""

    coast: str = Field(
        ...,
        description="Florida coast region: 'Gulf Coast' (e.g., Clearwater, Siesta Key) or 'Atlantic Coast' (e.g., Cocoa Beach, Vero Beach).",
    )
    needs_calm_water: bool = Field(
        default=True,
        description="Whether the family needs gentle, low-wave water suitable for young swimmers.",
    )


class BeachItem(BaseModel):
    """Details of a family-friendly Florida beach."""

    name: str = Field(..., description="Name of the beach.")
    coast: str = Field(..., description="Coast region ('Gulf Coast' or 'Atlantic Coast').")
    drive_from_orlando_mins: int = Field(
        ..., description="Approximate drive time from Orlando in minutes."
    )
    wave_conditions: str = Field(..., description="Typical wave and water conditions.")
    family_amenities: List[str] = Field(
        ..., description="Available family amenities (restrooms, lifeguards, playgrounds, shade)."
    )
    nearby_rainy_day_backup: str = Field(
        ..., description="Indoor backup activity nearby in case of afternoon thunderstorms."
    )


class BeachSearchOutput(BaseModel):
    """Output schema for family beach search results."""

    status: Literal["success", "error"] = Field(..., description="Lookup status.")
    coast: str = Field(..., description="Requested coast.")
    beaches: List[BeachItem] = Field(
        default_factory=list, description="List of matching family beaches."
    )
    weather_safety_tip: str = Field(
        default="", description="Florida sun, heat, and afternoon storm guidance."
    )
    error_message: Optional[str] = Field(default=None)
    recovery_suggestion: Optional[str] = Field(default=None)


class DiningRecommendationInput(BaseModel):
    """Input schema for recommending kid-friendly dining in Florida."""

    region: str = Field(
        ...,
        description="Destination region: 'Orlando', 'Clearwater', or 'Cocoa Beach'.",
    )
    meal_type: Literal["breakfast", "lunch", "dinner", "character_dining"] = Field(
        default="dinner",
        description="Type of meal ('breakfast', 'lunch', 'dinner', or 'character_dining').",
    )
    dietary_needs: Optional[str] = Field(
        default=None,
        description="Optional dietary restrictions or allergies (e.g., 'nut-free', 'gluten-free', 'vegetarian').",
    )


class RestaurantItem(BaseModel):
    """Details of a kid-friendly restaurant."""

    name: str = Field(..., description="Restaurant name.")
    region: str = Field(..., description="Location region.")
    meal_types: List[str] = Field(..., description="Supported meal types.")
    kid_appeal: str = Field(..., description="Why kids love this restaurant.")
    allergy_friendly_notes: str = Field(
        ..., description="Notes on how the kitchen handles food allergies and picky eaters."
    )
    reservation_lead_days: int = Field(
        ..., description="Recommended days in advance to book."
    )


class DiningRecommendationOutput(BaseModel):
    """Output schema for kid-friendly dining recommendations."""

    status: Literal["success", "error"] = Field(..., description="Lookup status.")
    region: str = Field(..., description="Requested region.")
    restaurants: List[RestaurantItem] = Field(default_factory=list)
    error_message: Optional[str] = Field(default=None)
    recovery_suggestion: Optional[str] = Field(default=None)


class DrivingRouteInput(BaseModel):
    """Input schema for calculating family driving routes and kid pit-stops."""

    origin: str = Field(..., description="Starting city (e.g., 'Orlando', 'Clearwater', 'Cocoa Beach', 'Tampa').")
    destination: str = Field(..., description="Destination city (e.g., 'Clearwater', 'Cocoa Beach', 'Orlando', 'Tampa').")


class DrivingRouteOutput(BaseModel):
    """Output schema for driving route and pacing calculation."""

    status: Literal["success", "error"] = Field(..., description="Lookup status.")
    origin: str = Field(..., description="Starting city.")
    destination: str = Field(..., description="Destination city.")
    distance_miles: int = Field(default=0, description="Total distance in miles.")
    estimated_drive_minutes: int = Field(
        default=0, description="Estimated driving time in minutes including family buffer."
    )
    recommended_pit_stops: List[str] = Field(
        default_factory=list,
        description="Kid-friendly stops along the highway for stretching and snacks.",
    )
    pacing_recommendation: str = Field(
        default="", description="Best time of day to drive to avoid traffic and align with nap times."
    )
    error_message: Optional[str] = Field(default=None)
    recovery_suggestion: Optional[str] = Field(default=None)


class ReservationBookingInput(BaseModel):
    """Input schema for booking theme park tickets or dining reservations with HITL guardrails."""

    booking_type: Literal["theme_park_ticket", "dining_reservation"] = Field(
        ..., description="Type of booking ('theme_park_ticket' or 'dining_reservation')."
    )
    item_name: str = Field(..., description="Name of the park or restaurant to book.")
    date_str: str = Field(..., description="Requested date in YYYY-MM-DD or Day N format.")
    party_size: int = Field(..., ge=1, le=20, description="Total number of guests (adults + kids).")
    user_confirmed: bool = Field(
        default=False,
        description="CRITICAL HITL FLAG: Must be True ONLY after the user has explicitly confirmed the booking details and cost.",
    )
    contact_email: Optional[str] = Field(
        default=None, description="Optional family contact email (will be PII-redacted in logs)."
    )


class ReservationBookingOutput(BaseModel):
    """Output schema for booking confirmation or Human-in-the-Loop pending state."""

    status: Literal["confirmed", "pending_human_confirmation", "error"] = Field(
        ..., description="Booking outcome status."
    )
    booking_reference: Optional[str] = Field(
        default=None, description="Confirmation code if finalized."
    )
    summary: str = Field(..., description="Human-readable summary of the booking state.")
    hitl_prompt_for_user: Optional[str] = Field(
        default=None,
        description="Exact confirmation prompt to show the user when user_confirmed is False.",
    )
    recovery_suggestion: Optional[str] = Field(default=None)


# =====================================================================
# Curated Domain Knowledge Databases
# =====================================================================

PARK_CATALOG: Dict[str, Dict[str, Any]] = {
    "walt disney world": {
        "display_name": "Walt Disney World Resort",
        "pacing_advice": (
            "Arrive 30 minutes before Early Theme Park Entry ('rope drop'). Schedule a mandatory "
            "1:00 PM - 4:00 PM hotel pool/nap break to beat peak heat and crowds. Rent a double stroller "
            "even for 6-7 year olds—Disney averages 8-10 miles of walking per day."
        ),
        "attractions": [
            AttractionItem(
                name="Peter Pan's Flight",
                park_area="Magic Kingdom (Fantasyland)",
                min_age_recommended=0,
                height_requirement_inches=0,
                sensory_notes="Gentle suspended dark ride; glowing scenes, very kid-friendly.",
                lightning_lane_priority="high",
            ),
            AttractionItem(
                name="Kilimanjaro Safaris",
                park_area="Disney's Animal Kingdom",
                min_age_recommended=0,
                height_requirement_inches=0,
                sensory_notes="Open-air safari truck viewing real giraffes, elephants, and lions; bumpy road.",
                lightning_lane_priority="high",
            ),
            AttractionItem(
                name="Frozen Ever After",
                park_area="EPCOT (World Showcase - Norway)",
                min_age_recommended=2,
                height_requirement_inches=0,
                sensory_notes="Indoor boat ride with Elsa & Anna; one small backwards splash drop.",
                lightning_lane_priority="high",
            ),
            AttractionItem(
                name="Slinky Dog Dash",
                park_area="Disney's Hollywood Studios (Toy Story Land)",
                min_age_recommended=5,
                height_requirement_inches=38,
                sensory_notes="Smooth family coaster; outdoor track, moderate speed.",
                lightning_lane_priority="high",
            ),
            AttractionItem(
                name="Seven Dwarfs Mine Train",
                park_area="Magic Kingdom (Fantasyland)",
                min_age_recommended=5,
                height_requirement_inches=38,
                sensory_notes="Swinging mine cart family coaster with indoor dark-ride musical scene.",
                lightning_lane_priority="high",
            ),
        ],
    },
    "universal orlando": {
        "display_name": "Universal Orlando Resort",
        "pacing_advice": (
            "Universal parks (Universal Studios Florida, Islands of Adventure, and Epic Universe) are "
            "compact and walkable via CityWalk. Use Universal's generous Child Swap rooms on thrill rides "
            "so older kids/adults can ride while younger kids wait in air-conditioned play areas."
        ),
        "attractions": [
            AttractionItem(
                name="The High in the Sky Seuss Trolley Train Ride",
                park_area="Islands of Adventure (Seuss Landing)",
                min_age_recommended=3,
                height_requirement_inches=36,
                sensory_notes="Gentle elevated train through Dr. Seuss stories; bright and cheerful.",
                lightning_lane_priority="medium",
            ),
            AttractionItem(
                name="Flight of the Hippogriff",
                park_area="Islands of Adventure (The Wizarding World of Harry Potter)",
                min_age_recommended=4,
                height_requirement_inches=36,
                sensory_notes="Junior outdoor coaster past Hagrid's hut; quick 1-minute ride.",
                lightning_lane_priority="medium",
            ),
            AttractionItem(
                name="Despicable Me Minion Mayhem",
                park_area="Universal Studios Florida (Minion Land)",
                min_age_recommended=4,
                height_requirement_inches=40,
                sensory_notes="Motion simulator with stationary seating available for younger kids.",
                lightning_lane_priority="medium",
            ),
            AttractionItem(
                name="Hagrid's Magical Creatures Motorbike Adventure",
                park_area="Islands of Adventure (Hogsmeade)",
                min_age_recommended=8,
                height_requirement_inches=48,
                sensory_notes="High-speed story coaster with launches and a brief vertical drop track.",
                lightning_lane_priority="high",
            ),
        ],
    },
    "legoland florida": {
        "display_name": "LEGOLAND Florida Resort",
        "pacing_advice": (
            "Located 45 minutes southwest of Orlando in Winter Haven. Built specifically for ages 2-12 "
            "with shorter queues and interactive LEGO build zones in line."
        ),
        "attractions": [
            AttractionItem(
                name="Coastersaurus",
                park_area="Dino Island",
                min_age_recommended=4,
                height_requirement_inches=42,
                sensory_notes="Kid-sized wooden/steel hybrid coaster curving around life-sized LEGO dinosaurs.",
                lightning_lane_priority="medium",
            ),
            AttractionItem(
                name="LEGOLAND Driving School",
                park_area="LEGO City",
                min_age_recommended=3,
                height_requirement_inches=0,
                sensory_notes="Kids drive electric LEGO cars on a real road course (Junior school for ages 3-5).",
                lightning_lane_priority="high",
            ),
        ],
    },
    "kennedy space center": {
        "display_name": "Kennedy Space Center Visitor Complex",
        "pacing_advice": (
            "Pairs effortlessly with a Cocoa Beach day (only 25 minutes north). Start with the Space Shuttle "
            "Atlantis exhibit and Planet Play indoor climbing structure for kids."
        ),
        "attractions": [
            AttractionItem(
                name="Planet Play",
                park_area="Main Visitor Complex",
                min_age_recommended=2,
                height_requirement_inches=0,
                sensory_notes="Multi-story indoor air-conditioned space-themed play experience.",
                lightning_lane_priority="low",
            ),
            AttractionItem(
                name="Space Shuttle Atlantis & Shuttle Launch Experience",
                park_area="Main Visitor Complex",
                min_age_recommended=6,
                height_requirement_inches=44,
                sensory_notes="Dramatic reveal of the real Atlantis orbiter plus optional launch simulator.",
                lightning_lane_priority="high",
            ),
        ],
    },
}

BEACH_CATALOG: Dict[str, List[BeachItem]] = {
    "gulf coast": [
        BeachItem(
            name="Clearwater Beach (Pier 60)",
            coast="Gulf Coast",
            drive_from_orlando_mins=110,
            wave_conditions="Calm, shallow turquoise Gulf waters with soft white sugar sand—ideal for toddlers and young swimmers.",
            family_amenities=[
                "Full-time lifeguards",
                "Pier 60 shaded playground & sunset festival",
                "Family restrooms & foot showers",
                "Cabana & umbrella rentals",
            ],
            nearby_rainy_day_backup="Clearwater Marine Aquarium (home of rescued dolphins and sea turtles, 5 mins away).",
        ),
        BeachItem(
            name="Fort De Soto Park (North Beach)",
            coast="Gulf Coast",
            drive_from_orlando_mins=120,
            wave_conditions="Ultra-calm tidal lagoon water where toddlers can wade safely without surf.",
            family_amenities=[
                "Natural shade trees & picnic pavilions",
                "Paved stroller-friendly nature trails",
                "Historic fort exploration",
                "Concession snack bar & restrooms",
            ],
            nearby_rainy_day_backup="Great Explorations Children's Museum in nearby St. Petersburg (20 mins away).",
        ),
    ],
    "atlantic coast": [
        BeachItem(
            name="Cocoa Beach (Alan Shepard Park & Lori Wilson Park)",
            coast="Atlantic Coast",
            drive_from_orlando_mins=60,
            wave_conditions="Gentle rolling Atlantic waves with wide, hard-packed sand great for sandcastles, boogie boarding, and beginner kids' surf lessons.",
            family_amenities=[
                "Lifeguard towers",
                "Boardwalk showers & restrooms",
                "Ron Jon Surf Shop (2 blocks away with giant aquarium)",
                "Maritime hammock nature boardwalk at Lori Wilson Park",
            ],
            nearby_rainy_day_backup="Kennedy Space Center Visitor Complex or Dinosaur Store & Museum (5 mins away).",
        ),
        BeachItem(
            name="Jetty Park Beach (Port Canaveral)",
            coast="Atlantic Coast",
            drive_from_orlando_mins=55,
            wave_conditions="Protected by the Port Canaveral jetty, making waves much calmer than open Atlantic beaches; kids can watch giant cruise ships and rockets!",
            family_amenities=[
                "Year-round lifeguards",
                "Shaded playground",
                "On-site snack grille & bait shop",
                "Paved fishing & ship-viewing pier",
            ],
            nearby_rainy_day_backup="Exploration Tower & Kennedy Space Center Planet Play.",
        ),
    ],
}

DINING_CATALOG: Dict[str, List[RestaurantItem]] = {
    "orlando": [
        RestaurantItem(
            name="Chef Mickey's (Disney's Contemporary Resort)",
            region="Orlando",
            meal_types=["breakfast", "dinner", "character_dining"],
            kid_appeal="Monorail trains glide right overhead inside the hotel while Mickey, Minnie, Donald, Pluto, and Goofy visit every table!",
            allergy_friendly_notes="Disney Special Diets chefs personally walk parents through the buffet and prepare dedicated allergy-safe Mickey waffles in a separate kitchen area.",
            reservation_lead_days=60,
        ),
        RestaurantItem(
            name="T-REX Cafe (Disney Springs)",
            region="Orlando",
            meal_types=["lunch", "dinner"],
            kid_appeal="Animatronic dinosaurs, glowing ice caves, an octopus bar, and a hands-on 'Paleo Zone' fossil dig area for kids.",
            allergy_friendly_notes="Dedicated gluten-free and nut-free kids menu; sensory note: simulated meteor showers happen every 20 minutes (request a quiet booth near the aquarium if kids are sound-sensitive).",
            reservation_lead_days=14,
        ),
        RestaurantItem(
            name="Toothsome Chocolate Emporium & Savory Feast Kitchen (Universal CityWalk)",
            region="Orlando",
            meal_types=["lunch", "dinner"],
            kid_appeal="Steampunk chocolate factory theme with strolling robot character Jacques and over-the-top artisan milkshakes.",
            allergy_friendly_notes="Offers dairy-free vegan shakes and allergy-marked kids entrees.",
            reservation_lead_days=7,
        ),
    ],
    "clearwater": [
        RestaurantItem(
            name="Frenchy's Rockaway Grill",
            region="Clearwater",
            meal_types=["lunch", "dinner"],
            kid_appeal="Open-air beachfront deck right on the sand where kids can dig with sand buckets while waiting for grouper nuggets and smoothies.",
            allergy_friendly_notes="Fresh grilled catch-of-the-day and gluten-free buns available; peanut-free fryers.",
            reservation_lead_days=0,
        ),
        RestaurantItem(
            name="Clear Sky Beachside Cafe",
            region="Clearwater",
            meal_types=["breakfast", "lunch", "dinner"],
            kid_appeal="Dedicated 'Little Surfers' menu with fun crepe rolls, pancake stacks, and outdoor patio umbrellas.",
            allergy_friendly_notes="Extensive gluten-free, vegetarian, and dairy-free modifications.",
            reservation_lead_days=3,
        ),
    ],
    "cocoa beach": [
        RestaurantItem(
            name="Fishlips Waterfront Bar & Grill (Port Canaveral)",
            region="Cocoa Beach",
            meal_types=["lunch", "dinner"],
            kid_appeal="Covered waterfront deck where kids watch Disney & Royal Caribbean cruise ships, dolphins, and pelicans pass right by.",
            allergy_friendly_notes="Simple grilled fish, chicken tenders, fruit cups, and allergy-aware kitchen staff.",
            reservation_lead_days=2,
        ),
        RestaurantItem(
            name="4th Street Fillin Station",
            region="Cocoa Beach",
            meal_types=["breakfast", "lunch", "dinner"],
            kid_appeal="Converted vintage 1950s gas station with a shaded backyard lawn, turf play area, and gourmet build-your-own waffles & tots.",
            allergy_friendly_notes="Vegan, gluten-sensitive, and nut-free options clearly labeled.",
            reservation_lead_days=0,
        ),
    ],
}

DRIVING_ROUTES: Dict[tuple, Dict[str, Any]] = {
    ("orlando", "clearwater"): {
        "distance_miles": 106,
        "estimated_drive_minutes": 115,
        "recommended_pit_stops": [
            "Dinosaur World (Plant City, Exit 17 off I-4) — outdoor walk with 200+ life-size dinosaurs & playground",
            "Florida Welcome / Rest Plaza along I-4 near Lakeland",
        ],
        "pacing_recommendation": (
            "Depart Orlando before 9:00 AM or right after lunch at 12:30 PM (ideal car nap window) "
            "to avoid I-4 ChampionsGate and Tampa malfunction-junction rush hour."
        ),
    },
    ("orlando", "cocoa beach"): {
        "distance_miles": 58,
        "estimated_drive_minutes": 60,
        "recommended_pit_stops": [
            "St. Johns River Airboat / Lone Cabbage Fish Camp (SR-520) for quick alligator spotting",
            "Ron Jon Surf Shop Flagship (Cocoa Beach) — open 24 hours with indoor waterfall & fish tank",
        ],
        "pacing_recommendation": (
            "Super easy 1-hour straight shot east on SR-528 (Beachline Expressway). Great for a "
            "mid-trip rest day away from theme park crowds!"
        ),
    },
    ("clearwater", "cocoa beach"): {
        "distance_miles": 162,
        "estimated_drive_minutes": 175,
        "recommended_pit_stops": [
            "Bonnet Springs Park (Lakeland) — world-class free family park & Florida Children's Museum halfway across the state",
            "Disney Springs (Lake Buena Vista) — easy garage parking for a lunch break mid-drive",
        ],
        "pacing_recommendation": (
            "This coast-to-coast drive takes ~2.5 to 3 hours. Break it in half at Lakeland (Bonnet Springs Park) "
            "or return to Orlando as your central hub before doing a day trip to Cocoa Beach."
        ),
    },
}


# =====================================================================
# Modular ADK Tools with Guided Error Recovery
# =====================================================================


@traced_tool
def search_theme_park_attractions(
    park_resort: str,
    min_kid_age: int = 5,
    pacing_preference: str = "balanced",
) -> Dict[str, Any]:
    """Searches Florida theme park resorts for age-appropriate family attractions and pacing advice.

    Purpose:
        Retrieves curated attractions filtered by the youngest child's age, including height
        requirements, sensory notes (darkness, drops, loud noises), Lightning Lane priority,
        and tailored family pacing rules (stroller breaks, afternoon nap windows).

    Args:
        park_resort: The theme park resort to query. Valid values are 'Walt Disney World',
            'Universal Orlando', 'LEGOLAND Florida', or 'Kennedy Space Center'.
        min_kid_age: Age of the youngest child in years (0 to 17). Used to highlight rides
            accessible to the whole family or requiring Rider Switch.
        pacing_preference: Family pacing style ('relaxed', 'balanced', or 'thrill_focused').

    Returns:
        Dict[str, Any]: A dictionary matching the `ThemeParkSearchOutput` Pydantic schema,
        containing `status`, `park_resort`, `recommended_attractions`, `pacing_advice`, and
        `recovery_suggestion` if the lookup fails.
    """
    try:
        validated_in = ThemeParkSearchInput(
            park_resort=park_resort,
            min_kid_age=min_kid_age,
            pacing_preference=pacing_preference if pacing_preference in ("relaxed", "balanced", "thrill_focused") else "balanced",  # type: ignore[arg-type]
        )
    except ValidationError as exc:
        out = ThemeParkSearchOutput(
            status="error",
            park_resort=str(park_resort),
            error_message=f"Invalid input parameters: {exc}",
            recovery_suggestion=(
                "Ensure `min_kid_age` is an integer between 0 and 17 and `pacing_preference` "
                "is one of 'relaxed', 'balanced', or 'thrill_focused'."
            ),
        )
        return out.model_dump()

    key = validated_in.park_resort.strip().lower()
    # Alias resolution
    alias_map = {
        "disney": "walt disney world",
        "magic kingdom": "walt disney world",
        "epcot": "walt disney world",
        "animal kingdom": "walt disney world",
        "hollywood studios": "walt disney world",
        "universal": "universal orlando",
        "islands of adventure": "universal orlando",
        "epic universe": "universal orlando",
        "legoland": "legoland florida",
        "ksc": "kennedy space center",
        "nasa": "kennedy space center",
    }
    normalized_key = alias_map.get(key, key)

    if normalized_key not in PARK_CATALOG:
        valid_parks = ", ".join(
            [v["display_name"] for v in PARK_CATALOG.values()]
        )
        out = ThemeParkSearchOutput(
            status="error",
            park_resort=validated_in.park_resort,
            error_message=f"Theme park resort '{validated_in.park_resort}' was not found in the Florida catalog.",
            recovery_suggestion=(
                f"Retry calling `search_theme_park_attractions` using one of the supported "
                f"Florida park resorts: {valid_parks}."
            ),
        )
        return out.model_dump()

    park_data = PARK_CATALOG[normalized_key]
    all_attractions: List[AttractionItem] = park_data["attractions"]

    # Filter attractions suitable for the youngest child or include all if thrill_focused with Rider Switch note
    if validated_in.pacing_preference == "thrill_focused":
        filtered = all_attractions
    else:
        filtered = [
            a
            for a in all_attractions
            if a.min_age_recommended <= validated_in.min_kid_age + 2
        ] or all_attractions

    out = ThemeParkSearchOutput(
        status="success",
        park_resort=park_data["display_name"],
        recommended_attractions=filtered,
        pacing_advice=park_data["pacing_advice"],
    )
    return out.model_dump()


@traced_tool
def find_family_beaches(
    coast: str,
    needs_calm_water: bool = True,
) -> Dict[str, Any]:
    """Finds family-friendly Florida beaches on the Gulf or Atlantic Coast with amenities and rainy-day backups.

    Purpose:
        Helps families balance high-energy Orlando theme park days with relaxing coastal
        beach days (e.g., Clearwater Beach on the Gulf Coast or Cocoa Beach on the Atlantic Coast).

    Args:
        coast: The Florida coast region to search. Accepted values: 'Gulf Coast' (or 'Clearwater')
            and 'Atlantic Coast' (or 'Cocoa Beach').
        needs_calm_water: True if the family prefers calm, low-wave water suitable for toddlers
            and young children.

    Returns:
        Dict[str, Any]: A dictionary matching the `BeachSearchOutput` Pydantic schema, containing
        `status`, `coast`, `beaches`, `weather_safety_tip`, and `recovery_suggestion` on error.
    """
    validated_in = BeachSearchInput(coast=coast, needs_calm_water=needs_calm_water)
    key = validated_in.coast.strip().lower()

    alias_map = {
        "gulf": "gulf coast",
        "clearwater": "gulf coast",
        "clearwater beach": "gulf coast",
        "tampa": "gulf coast",
        "st pete": "gulf coast",
        "atlantic": "atlantic coast",
        "cocoa": "atlantic coast",
        "cocoa beach": "atlantic coast",
        "space coast": "atlantic coast",
        "port canaveral": "atlantic coast",
    }
    normalized_key = alias_map.get(key, key)

    if normalized_key not in BEACH_CATALOG:
        out = BeachSearchOutput(
            status="error",
            coast=validated_in.coast,
            error_message=f"Coast or beach region '{validated_in.coast}' is not recognized.",
            recovery_suggestion=(
                "Please call `find_family_beaches` with `coast='Gulf Coast'` (for Clearwater / Fort De Soto) "
                "or `coast='Atlantic Coast'` (for Cocoa Beach / Jetty Park)."
            ),
        )
        return out.model_dump()

    beaches = BEACH_CATALOG[normalized_key]
    out = BeachSearchOutput(
        status="success",
        coast=normalized_key.title(),
        beaches=beaches,
        weather_safety_tip=(
            "Florida UV index peaks between 11:00 AM and 3:00 PM, and brief summer/autumn thunderstorms "
            "often roll through around 3:00 PM. Hit the beach from 8:30 AM - 12:30 PM, break for an indoor "
            "lunch/nap or aquarium backup, and return for sunset."
        ),
    )
    return out.model_dump()


@traced_tool
def recommend_kid_friendly_dining(
    region: str,
    meal_type: str = "dinner",
    dietary_needs: Optional[str] = None,
) -> Dict[str, Any]:
    """Recommends kid-friendly restaurants and character dining across Orlando, Clearwater, and Cocoa Beach.

    Purpose:
        Identifies family-tested restaurants that offer engaging environments for kids,
        reasonable wait times, allergy-safe protocols, and clear advance reservation timelines.

    Args:
        region: The Florida destination region ('Orlando', 'Clearwater', or 'Cocoa Beach').
        meal_type: Desired meal category ('breakfast', 'lunch', 'dinner', or 'character_dining').
        dietary_needs: Optional allergy or dietary note (e.g., 'peanut allergy', 'gluten-free').

    Returns:
        Dict[str, Any]: A dictionary matching the `DiningRecommendationOutput` Pydantic schema,
        including matching restaurants, allergy notes, reservation lead days, or recovery guidance.
    """
    normalized_meal = meal_type.strip().lower()
    if normalized_meal not in ("breakfast", "lunch", "dinner", "character_dining"):
        out = DiningRecommendationOutput(
            status="error",
            region=region,
            error_message=f"Unsupported meal_type '{meal_type}'.",
            recovery_suggestion=(
                "Retry `recommend_kid_friendly_dining` with `meal_type` set to one of: "
                "'breakfast', 'lunch', 'dinner', or 'character_dining'."
            ),
        )
        return out.model_dump()

    validated_in = DiningRecommendationInput(
        region=region,
        meal_type=normalized_meal,  # type: ignore[arg-type]
        dietary_needs=dietary_needs,
    )
    region_key = validated_in.region.strip().lower()
    alias_map = {
        "disney": "orlando",
        "universal": "orlando",
        "walt disney world": "orlando",
        "clearwater beach": "clearwater",
        "gulf coast": "clearwater",
        "cocoa": "cocoa beach",
        "space coast": "cocoa beach",
        "port canaveral": "cocoa beach",
    }
    normalized_region = alias_map.get(region_key, region_key)

    if normalized_region not in DINING_CATALOG:
        out = DiningRecommendationOutput(
            status="error",
            region=validated_in.region,
            error_message=f"No dining catalog entries found for region '{validated_in.region}'.",
            recovery_suggestion=(
                "Retry `recommend_kid_friendly_dining` using a supported Florida hub: "
                "'Orlando', 'Clearwater', or 'Cocoa Beach'."
            ),
        )
        return out.model_dump()

    candidates = DINING_CATALOG[normalized_region]
    filtered = [
        r for r in candidates if validated_in.meal_type in r.meal_types
    ] or candidates

    out = DiningRecommendationOutput(
        status="success",
        region=normalized_region.title(),
        restaurants=filtered,
    )
    return out.model_dump()


@traced_tool
def calculate_driving_and_pacing(
    origin: str,
    destination: str,
) -> Dict[str, Any]:
    """Calculates family driving time, distance, and kid-friendly highway pit-stops between Florida hubs.

    Purpose:
        Ensures realistic driving pacing for families traveling with children between Orlando,
        Clearwater Beach (Gulf Coast), and Cocoa Beach (Atlantic Coast).

    Args:
        origin: Starting city ('Orlando', 'Clearwater', or 'Cocoa Beach').
        destination: Destination city ('Orlando', 'Clearwater', or 'Cocoa Beach').

    Returns:
        Dict[str, Any]: A dictionary matching the `DrivingRouteOutput` Pydantic schema with
        distance, estimated drive minutes, kid-friendly pit stops, and time-of-day pacing advice.
    """
    validated_in = DrivingRouteInput(origin=origin, destination=destination)
    alias_map = {
        "walt disney world": "orlando",
        "universal orlando": "orlando",
        "disney": "orlando",
        "clearwater beach": "clearwater",
        "tampa": "clearwater",
        "gulf coast": "clearwater",
        "cocoa": "cocoa beach",
        "port canaveral": "cocoa beach",
        "kennedy space center": "cocoa beach",
        "atlantic coast": "cocoa beach",
    }
    orig_key = alias_map.get(validated_in.origin.strip().lower(), validated_in.origin.strip().lower())
    dest_key = alias_map.get(validated_in.destination.strip().lower(), validated_in.destination.strip().lower())

    if orig_key == dest_key and orig_key in ("orlando", "clearwater", "cocoa beach"):
        out = DrivingRouteOutput(
            status="success",
            origin=orig_key.title(),
            destination=dest_key.title(),
            distance_miles=15,
            estimated_drive_minutes=25,
            recommended_pit_stops=["Local resort shuttle or 20-minute local transfer"],
            pacing_recommendation="Local intra-city transfer; allow 25-30 minutes for parking and stroller unloading.",
        )
        return out.model_dump()

    route_data = DRIVING_ROUTES.get((orig_key, dest_key)) or DRIVING_ROUTES.get((dest_key, orig_key))
    if not route_data:
        out = DrivingRouteOutput(
            status="error",
            origin=validated_in.origin,
            destination=validated_in.destination,
            error_message=(
                f"Driving route between '{validated_in.origin}' and '{validated_in.destination}' "
                "is outside the supported Central Florida family corridor."
            ),
            recovery_suggestion=(
                "Call `calculate_driving_and_pacing` using combinations of 'Orlando', "
                "'Clearwater', and 'Cocoa Beach'."
            ),
        )
        return out.model_dump()

    out = DrivingRouteOutput(
        status="success",
        origin=orig_key.title(),
        destination=dest_key.title(),
        distance_miles=route_data["distance_miles"],
        estimated_drive_minutes=route_data["estimated_drive_minutes"],
        recommended_pit_stops=route_data["recommended_pit_stops"],
        pacing_recommendation=route_data["pacing_recommendation"],
    )
    return out.model_dump()


@traced_tool
def request_booking_confirmation(
    booking_type: str,
    item_name: str,
    date_str: str,
    party_size: int,
    user_confirmed: bool = False,
    contact_email: Optional[str] = None,
) -> Dict[str, Any]:
    """Initiates or finalizes a theme park ticket purchase or dining reservation with a mandatory Human-in-the-Loop (HITL) guardrail.

    Purpose:
        Enforces safety and policy compliance by NEVER finalizing a financial commitment or
        reservation unless `user_confirmed=True` is explicitly provided after user approval.

    Args:
        booking_type: Either 'theme_park_ticket' or 'dining_reservation'.
        item_name: Name of the theme park or restaurant being booked.
        date_str: Date of the visit/reservation (e.g., '2026-11-15' or 'Day 3').
        party_size: Total number of adults and children (1-20).
        user_confirmed: Boolean flag indicating whether the human user has explicitly approved
            this specific transaction. Defaults to False.
        contact_email: Optional user email for confirmation receipt (automatically PII-redacted in logs).

    Returns:
        Dict[str, Any]: A dictionary matching `ReservationBookingOutput`. If `user_confirmed` is False,
        returns `status='pending_human_confirmation'` with a clear prompt to ask the user.
    """
    try:
        validated_in = ReservationBookingInput(
            booking_type=booking_type,  # type: ignore[arg-type]
            item_name=item_name,
            date_str=date_str,
            party_size=party_size,
            user_confirmed=user_confirmed,
            contact_email=contact_email,
        )
    except ValidationError as exc:
        out = ReservationBookingOutput(
            status="error",
            summary=f"Invalid booking parameters: {exc}",
            recovery_suggestion=(
                "Ensure `booking_type` is 'theme_park_ticket' or 'dining_reservation' and "
                "`party_size` is between 1 and 20."
            ),
        )
        return out.model_dump()

    redacted_email = redact_pii(validated_in.contact_email or "not_provided")

    if not validated_in.user_confirmed:
        log_intent(
            agent_name="HITLGuardrail",
            action="pause_for_human_confirmation",
            details={
                "booking_type": validated_in.booking_type,
                "item_name": validated_in.item_name,
                "date_str": validated_in.date_str,
                "party_size": validated_in.party_size,
                "contact_email": redacted_email,
            },
        )
        out = ReservationBookingOutput(
            status="pending_human_confirmation",
            booking_reference=None,
            summary=(
                f"HOLD: Human-in-the-Loop confirmation required before finalizing {validated_in.booking_type} "
                f"for '{validated_in.item_name}' on {validated_in.date_str} (party of {validated_in.party_size})."
            ),
            hitl_prompt_for_user=(
                f"Please confirm: Would you like me to finalize the {validated_in.booking_type.replace('_', ' ')} "
                f"for **{validated_in.item_name}** on **{validated_in.date_str}** for **{validated_in.party_size} guests**? "
                "Reply 'CONFIRM' to proceed."
            ),
            recovery_suggestion=(
                "Present the `hitl_prompt_for_user` to the user and wait for their explicit approval "
                "before calling `request_booking_confirmation` again with `user_confirmed=True`."
            ),
        )
        return out.model_dump()

    # User explicitly confirmed -> finalize booking
    ref_code = f"FL-{ abs(hash((validated_in.item_name, validated_in.date_str, validated_in.party_size))) % 900000 + 100000 }"
    log_outcome(
        agent_name="HITLGuardrail",
        action="finalize_booking_after_confirmation",
        status="CONFIRMED",
        details={
            "booking_reference": ref_code,
            "item_name": validated_in.item_name,
            "contact_email": redacted_email,
        },
    )
    out = ReservationBookingOutput(
        status="confirmed",
        booking_reference=ref_code,
        summary=(
            f"Confirmed {validated_in.booking_type.replace('_', ' ')} for '{validated_in.item_name}' "
            f"on {validated_in.date_str} for {validated_in.party_size} guests (Reference: {ref_code})."
        ),
    )
    return out.model_dump()
