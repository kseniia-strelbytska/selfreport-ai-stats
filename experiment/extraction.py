"""Robust numeric answer extraction from free-form model output.

Returns ``(value, method)``; ``value`` is ``None`` when nothing usable was
found (``method == "invalid"``).  The method string is stored with every
prediction so extraction decisions are auditable.

Handles: plain integers and decimals, thousands separators, number words
("about forty-two", "a dozen"), ranges ("7 to 9", "7-9" -> midpoint),
approximations ("≈", "~", "about"), percentages/units after the number,
negative numbers, and a preference for the number following "answer"/"is"
when the output is chatty.
"""

from __future__ import annotations

import re

_ONES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}
_SPECIAL = {"a dozen": 12, "dozen": 12, "half a dozen": 6, "a score": 20}

NUM_RE = re.compile(r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+|-?\d+)(?:\.(\d+))?(?![\w])")
RANGE_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:-|–|—|to|and)\s*(-?\d+(?:\.\d+)?)")
WORD_TOKEN = re.compile(r"[a-z\-]+")


def _words_to_number(text: str) -> float | None:
    """Parse a run of number words ("one hundred and seven", "forty-two")."""
    tokens = [t for t in re.split(r"[\s\-]+", text.lower().replace(" and ", " ")) if t]
    total, current, found = 0, 0, False
    for t in tokens:
        if t in _ONES:
            current += _ONES[t]
            found = True
        elif t in _TENS:
            current += _TENS[t]
            found = True
        elif t == "hundred" and found:
            current *= 100
        elif t in ("thousand", "million") and found:
            total += current * _SCALES[t]
            current = 0
        elif t in ("and", "a"):
            continue
        else:
            break
    return float(total + current) if found else None


def _first_word_number(text: str) -> float | None:
    low = text.lower()
    for k, v in _SPECIAL.items():
        if re.search(r"\b" + k + r"\b", low):
            return float(v)
    words = list(_ONES) + list(_TENS)
    pat = re.compile(r"\b((?:(?:" + "|".join(words) + r"|hundred|thousand|million|and)[\s\-]*)+)\b")
    for m in pat.finditer(low):
        v = _words_to_number(m.group(1))
        if v is not None:
            return v
    return None


def extract_number(
    text: str, prefer_after: tuple[str, ...] = ("answer", "is approximately", "is about", "is", "≈", "~", "=")
) -> tuple[float | None, str]:
    if not text or not text.strip():
        return None, "invalid"
    t = text.strip().replace("−", "-")
    # 1. an explicit range -> midpoint
    m = RANGE_RE.search(t)
    if m and not NUM_RE.search(t[: m.start()]):
        a, b = float(m.group(1)), float(m.group(2))
        if a <= b and (b - a) <= max(abs(a), 1) * 2:
            return (a + b) / 2, "range_midpoint"
    # 2. number right after a cue word
    low = t.lower()
    for cue in prefer_after:
        idx = low.find(cue)
        if idx >= 0:
            m2 = NUM_RE.search(t, idx + len(cue))
            if m2 and m2.start() - (idx + len(cue)) < 25:
                return _to_float(m2), f"after_{cue.strip()}"
    # 3. first digit-number
    m3 = NUM_RE.search(t)
    if m3:
        return _to_float(m3), "first_number"
    # 4. number words
    w = _first_word_number(t)
    if w is not None:
        return w, "number_words"
    return None, "invalid"


def _to_float(m: re.Match) -> float:
    whole = m.group(1).replace(",", "")
    frac = m.group(2)
    return float(f"{whole}.{frac}") if frac else float(whole)
