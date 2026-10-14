"""Find configured verbal fillers in a transcript."""

from __future__ import annotations

import re
from collections import Counter


FILLER_PHRASES = (
    "you know",
    "i mean",
    "sort of",
    "kind of",
    "actually",
    "basically",
    "literally",
    "um",
    "uh",
    "erm",
    "er",
    "ah",
    "like",
)
WORD_PATTERN = re.compile(r"\b[\w]+(?:['’][\w]+)?\b", flags=re.UNICODE)
_FILLER_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(p).replace(r"\ ", r"\s+") for p in
                           sorted(FILLER_PHRASES, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE | re.UNICODE,
)


def analyze_fillers(transcript: str) -> dict:
    """Return phrase occurrences and token-based filler ratio.

    Multiword phrases count as one occurrence and as their number of tokens in
    the filler ratio. Matching is case-insensitive and uses word boundaries.
    """
    if not isinstance(transcript, str):
        raise TypeError("transcript must be a string")

    total_words = len(WORD_PATTERN.findall(transcript))
    matches = list(_FILLER_PATTERN.finditer(transcript))
    counts = Counter(re.sub(r"\s+", " ", match.group(0).lower()) for match in matches)
    filler_tokens = sum(len(WORD_PATTERN.findall(match.group(0))) for match in matches)
    ratio = (filler_tokens / total_words * 100.0) if total_words else 0.0

    occurrences = [
        {"phrase": phrase, "count": counts[phrase]}
        for phrase in FILLER_PHRASES
        if counts.get(phrase, 0)
    ]
    return {
        "total_words": total_words,
        "filler_count": len(matches),
        "filler_tokens": filler_tokens,
        "filler_ratio_percent": ratio,
        "occurrences": occurrences,
    }
