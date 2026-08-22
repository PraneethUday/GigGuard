"""
City claim cooldown — shared in-memory state.

When enabled, after a claim batch fires for a city no further claims can be
created for that city for COOLDOWN_HOURS hours.  State lives in process memory
and resets on backend restart (acceptable for a demo/hackathon setup).
"""

from datetime import datetime

COOLDOWN_HOURS: int = 6

# Feature flag — OFF by default, toggled by admin endpoint
enabled: bool = False

# city (lowercase) -> UTC datetime of the last claim batch
_city_last_claim: dict[str, datetime] = {}


def is_blocked(city: str) -> bool:
    """Return True if the city is currently inside its cooldown window."""
    if not enabled:
        return False
    last = _city_last_claim.get(city.lower())
    if last is None:
        return False
    elapsed_hours = (datetime.utcnow() - last).total_seconds() / 3600
    return elapsed_hours < COOLDOWN_HOURS


def record_claim(city: str) -> None:
    """Mark a city as having just had a claim batch. Starts the cooldown clock."""
    _city_last_claim[city.lower()] = datetime.utcnow()


def get_status() -> dict:
    """Return the full cooldown state for the admin endpoint."""
    now = datetime.utcnow()
    cities_in_cooldown: dict[str, dict] = {}
    for city_key, last in _city_last_claim.items():
        elapsed = (now - last).total_seconds() / 3600
        remaining = COOLDOWN_HOURS - elapsed
        if remaining > 0:
            cities_in_cooldown[city_key] = {
                "last_claim_at": last.isoformat() + "Z",
                "remaining_hours": round(remaining, 2),
                "remaining_minutes": int(remaining * 60),
            }
    return {
        "enabled": enabled,
        "cooldown_hours": COOLDOWN_HOURS,
        "cities_in_cooldown": cities_in_cooldown,
    }


def set_enabled(value: bool) -> None:
    global enabled
    enabled = value
