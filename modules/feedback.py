"""Turn measured values into practical, rule-based coaching suggestions."""

from __future__ import annotations


def build_feedback(
    semantic_similarity_percent: float,
    filler_ratio_percent: float,
    pause_ratio_percent: float,
    overall_score: float,
) -> dict:
    strengths: list[str] = []
    suggestions: list[str] = []

    if semantic_similarity_percent >= 70:
        strengths.append("Your explanation shares many ideas with the reference explanation.")
    elif semantic_similarity_percent >= 45:
        suggestions.append(
            "Add the reference's central ideas and explain how they connect; similarity is moderate."
        )
    else:
        suggestions.append(
            "Review the reference, then restate the concept in your own words with its key ideas."
        )

    if filler_ratio_percent <= 2:
        strengths.append("Few configured filler tokens appeared in the transcript.")
    elif filler_ratio_percent > 6:
        suggestions.append(
            "Pause briefly before your next point instead of filling the gap with repeated phrases."
        )

    if pause_ratio_percent > 40:
        suggestions.append(
            "The recording contains a high estimated quiet-time ratio; check for long gaps or low recording volume."
        )
    elif pause_ratio_percent <= 25:
        strengths.append("The estimated quiet-time share is within the application's broad practice range.")

    if not suggestions:
        suggestions.append(
            "Try one more take with a short introduction, a clear example, and a concise conclusion."
        )
    if overall_score >= 80:
        headline = "Strong practice result"
    elif overall_score >= 60:
        headline = "Good base to build on"
    else:
        headline = "A useful first draft"

    return {"headline": headline, "strengths": strengths, "suggestions": suggestions}
