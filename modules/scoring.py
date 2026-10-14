"""Transparent, configurable practice score with three documented components."""

from __future__ import annotations

import math


DEFAULT_WEIGHTS = {"semantic": 0.70, "fillers": 0.20, "pauses": 0.10}


def _percentage(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 100.0:
        raise ValueError(f"{label} must be a finite number from 0 to 100.")
    return number


def _normalise_weights(weights: dict | None) -> dict[str, float]:
    weights = DEFAULT_WEIGHTS if weights is None else weights
    names = ("semantic", "fillers", "pauses")
    try:
        values = {name: float(weights.get(name, 0.0)) for name in names}
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Weights must be a mapping of semantic, fillers, and pauses values.") from exc
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("Score weights must be finite, non-negative numbers.")
    total = sum(values.values())
    if total <= 0:
        raise ValueError("At least one score weight must be greater than zero.")
    return {name: value / total for name, value in values.items()}


def calculate_score(
    semantic_similarity_percent: float,
    filler_ratio_percent: float,
    pause_ratio_percent: float,
    *,
    weights: dict | None = None,
) -> dict:
    """Calculate the weighted score using disclosed linear component formulas.

    Semantic component: clamped cosine similarity (0-100).
    Filler component: 100 at 0% filler tokens, linearly falling to 0 at 25%.
    Pause component: 100 through 15% quiet frames, linearly falling to 0 at 65%.
    """
    semantic = _percentage(semantic_similarity_percent, "Semantic similarity")
    filler_ratio = _percentage(filler_ratio_percent, "Filler ratio")
    pause_ratio = _percentage(pause_ratio_percent, "Pause ratio")
    normalized_weights = _normalise_weights(weights)

    components = {
        "semantic": semantic,
        "fillers": max(0.0, 100.0 * (1.0 - filler_ratio / 25.0)),
        "pauses": 100.0 if pause_ratio <= 15.0 else max(0.0, 100.0 * (1.0 - (pause_ratio - 15.0) / 50.0)),
    }
    weighted_points = {
        name: components[name] * normalized_weights[name] for name in components
    }
    overall = sum(weighted_points.values())
    return {
        "overall_score": round(overall, 2),
        "components": {name: round(value, 2) for name, value in components.items()},
        "weights": normalized_weights,
        "weighted_points": {name: round(value, 2) for name, value in weighted_points.items()},
    }
