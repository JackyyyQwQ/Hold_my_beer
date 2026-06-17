from __future__ import annotations

from typing import Any, Literal, TypedDict, cast


HeightDirection = Literal["higher", "lower", "good", "unchanged", "unknown"]
DistanceDirection = Literal["closer", "farther", "good", "unchanged", "unknown"]
SpeedDirection = Literal["faster", "slower", "good", "unchanged", "unknown"]
OrientationPreference = Literal[
    "rotate_left",
    "rotate_right",
    "handle_facing_user",
    "good",
    "unchanged",
    "unknown",
]
Satisfaction = Literal["satisfied", "partially_satisfied", "unsatisfied", "unknown"]


class EvaluatedFeedback(TypedDict):
    raw_feedback: str
    height_direction: HeightDirection
    distance_direction: DistanceDirection
    speed_direction: SpeedDirection
    orientation_preference: OrientationPreference
    handoff_preference: str
    satisfaction: Satisfaction
    score: int


class NextStrategy(TypedDict):
    height: str
    distance: str
    speed: str
    orientation: str
    handoff_timing: str
    expected_score: int
    explanation: str


class ValidationError(Exception):
    """Raised when an LLM response does not match the expected schema."""


ALLOWED_HEIGHT = {"higher", "lower", "good", "unchanged", "unknown"}
ALLOWED_DISTANCE = {"closer", "farther", "good", "unchanged", "unknown"}
ALLOWED_SPEED = {"faster", "slower", "good", "unchanged", "unknown"}
ALLOWED_ORIENTATION = {
    "rotate_left",
    "rotate_right",
    "handle_facing_user",
    "good",
    "unchanged",
    "unknown",
}
ALLOWED_SATISFACTION = {"satisfied", "partially_satisfied", "unsatisfied", "unknown"}


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValidationError(f"Field '{key}' must be a string.")
    return value


def _require_enum(data: dict[str, Any], key: str, allowed: set[str]) -> str:
    value = _require_string(data, key)
    if value not in allowed:
        raise ValidationError(f"Field '{key}' must be one of {sorted(allowed)}.")
    return value


def _require_score(value: Any, field_name: str) -> int:
    if not isinstance(value, int):
        raise ValidationError(f"Field '{field_name}' must be an integer.")
    if not 1 <= value <= 10:
        raise ValidationError(f"Field '{field_name}' must be between 1 and 10.")
    return value


def validate_evaluated_feedback(data: dict[str, Any]) -> EvaluatedFeedback:
    """Validate a single evaluator response from the LLM."""
    if not isinstance(data, dict):
        raise ValidationError("Each evaluated feedback item must be a JSON object.")

    return cast(
        EvaluatedFeedback,
        {
            "raw_feedback": _require_string(data, "raw_feedback"),
            "height_direction": _require_enum(data, "height_direction", ALLOWED_HEIGHT),
            "distance_direction": _require_enum(data, "distance_direction", ALLOWED_DISTANCE),
            "speed_direction": _require_enum(data, "speed_direction", ALLOWED_SPEED),
            "orientation_preference": _require_enum(
                data,
                "orientation_preference",
                ALLOWED_ORIENTATION,
            ),
            "handoff_preference": _require_string(data, "handoff_preference"),
            "satisfaction": _require_enum(data, "satisfaction", ALLOWED_SATISFACTION),
            "score": _require_score(data.get("score"), "score"),
        },
    )


def validate_evaluated_feedback_list(items: Any) -> list[EvaluatedFeedback]:
    """Validate the full evaluation history before optimization."""
    if not isinstance(items, list):
        raise ValidationError("Evaluated feedback history must be a list.")
    return [validate_evaluated_feedback(item) for item in items]


def validate_next_strategy(data: dict[str, Any]) -> NextStrategy:
    """Validate the optimizer response from the LLM."""
    if not isinstance(data, dict):
        raise ValidationError("The optimizer response must be a JSON object.")

    return cast(
        NextStrategy,
        {
            "height": _require_string(data, "height"),
            "distance": _require_string(data, "distance"),
            "speed": _require_string(data, "speed"),
            "orientation": _require_string(data, "orientation"),
            "handoff_timing": _require_string(data, "handoff_timing"),
            "expected_score": _require_score(data.get("expected_score"), "expected_score"),
            "explanation": _require_string(data, "explanation"),
        },
    )
