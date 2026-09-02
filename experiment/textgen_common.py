"""Helpers shared by the procedural writer and the LLM prompt builder:
number words, noun extraction, fact -> phrase rendering."""

from __future__ import annotations

import re

from experiment.story_planner import Fact
from experiment.themes import Attribute

_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def num_to_words(n: int) -> str:
    """Integer 0..999999 in British-style words ("one hundred and seven")."""
    n = int(n)
    if n < 0:
        return "minus " + num_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + ("-" + _ONES[o] if o else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return _ONES[h] + " hundred" + (" and " + num_to_words(r) if r else "")
    th, r = divmod(n, 1000)
    return num_to_words(th) + " thousand" + ((" and " if r < 100 else " ") + num_to_words(r) if r else "")


def count_noun(attr: Attribute) -> str:
    """The thing being counted, derived from the aliases: "number of crystals"
    -> "crystals"; "how many moons orbit it" -> "moons"."""
    for a in attr.aliases:
        m = re.match(r"^(?:the )?(?:number|count) of (.+)$", a, re.I)
        if m:
            return m.group(1)
    for a in attr.aliases:
        m = re.match(
            r"^how many ([a-z\- ]+?)(?: (?:it|there|they|that|who|which|were|are|orbit|live|grow|run|contains?|in|on|at|per)\b.*)?$",
            a,
            re.I,
        )
        if m:
            return m.group(1).strip()
    return attr.name.replace("_count", "").replace("_", " ") + ("s" if not attr.name.endswith("s") else "")


def value_phrase(fact: Fact, attr: Attribute, words: bool = False) -> str:
    """ "seven", "7", "1.8 metres" ..."""
    if attr.is_count:
        n = int(round(fact.value))
        return num_to_words(n) if (words and n < 1000) else str(n)
    return f"{fact.formatted} {attr.unit}" if attr.unit else fact.formatted


def part_phrase(part: dict, attr: Attribute, words: bool = False) -> str:
    v = part["value"]
    if attr.is_count:
        n = int(round(v))
        num = num_to_words(n) if (words and n < 1000) else str(n)
        return f"{num} {part['label']}"
    return f"{attr.format_value(v)} {attr.unit + ' ' if attr.unit else ''}{part['label']}".strip()
