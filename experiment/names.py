"""Seeded fictional name generation.

Entity and world names must (a) be unique within a world, (b) look natural in
prose, (c) be unlikely to collide with real names that a pretrained model has
opinions about.  We build names from syllable inventories per *style* and
never reuse a name inside one world.  Everything is driven by a numpy
``Generator`` so names are reproducible from the world seed.
"""

from __future__ import annotations

import numpy as np

# Syllable inventories.  Kept deliberately "non-real": no common English
# words, no famous places.
_ONSETS = [
    "b",
    "br",
    "c",
    "cr",
    "d",
    "dr",
    "f",
    "fr",
    "g",
    "gr",
    "h",
    "k",
    "kr",
    "l",
    "m",
    "n",
    "p",
    "pr",
    "r",
    "s",
    "sk",
    "st",
    "t",
    "th",
    "tr",
    "v",
    "vr",
    "w",
    "y",
    "z",
    "sh",
    "sl",
]
_NUCLEI = ["a", "e", "i", "o", "u", "a", "e", "i", "o", "a", "e", "ae", "ai", "ei", "ia", "io", "ou", "ea"]
_CODAS = ["", "", "", "", "l", "n", "r", "s", "m", "th", "ld", "nd", "rn", "st", "nt", "rd"]
_PLACE_SUFFIX = [
    "moor",
    "ford",
    "vale",
    "wick",
    "haven",
    "mere",
    "holm",
    "stead",
    "reach",
    "fell",
    "hollow",
    "ridge",
    "crest",
    "march",
    "wold",
    "bourne",
    "gate",
    "tor",
    "dell",
    "shaw",
]
_CREATURE_SUFFIX = [
    "ling",
    "wisp",
    "kin",
    "fang",
    "crawl",
    "mote",
    "shade",
    "wing",
    "horn",
    "tail",
    "quill",
    "scale",
    "burr",
    "snout",
    "hide",
]
_PLANET_SUFFIX = ["is", "os", "ia", "ea", "on", "ar", "us", "eth", "ith", "ax", "ara", "ion"]
_OBJECT_PREFIX = ["Model", "Series", "Type", "Pattern", "Mark", "Design"]
_GROUP_SUFFIX = ["clan", "line", "household", "kindred", "circle", "guild", "crew", "company", "troop"]
_ROMAN = ["II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "XI", "XII"]

STYLES = {
    "place",  # Varnmoor, Keldreach            (caves, forests, ponds, villages, regions)
    "creature",  # Thalwisp, Brunfang             (invented species / animal groups)
    "planet",  # Oradis, Vethion                (planets, stars, moons)
    "person_group",  # the Halvren household          (families, clans, clubs, crews)
    "object",  # the Teskar Mark IV             (machines, devices, instruments)
    "vessel",  # the Ardent Kestrel             (ships, vehicles)
    "team",  # Orvane Rovers                  (sports teams)
    "building",  # Castle Vellmarch, Sorrin Keep  (castles, libraries, museums)
    "settlement",  # Brathmere, Oskenwick           (towns, villages, sites)
    "person",  # Teodra Halvren                 (individual characters)
}

_VESSEL_ADJ = [
    "Ardent",
    "Silent",
    "Northern",
    "Patient",
    "Wandering",
    "Iron",
    "Amber",
    "Grey",
    "Swift",
    "Last",
]
_VESSEL_NOUN = ["Kestrel", "Heron", "Lantern", "Compass", "Anvil", "Tern", "Ember", "Gull", "Sable", "Reed"]
_TEAM_NOUN = [
    "Rovers",
    "Wanderers",
    "Harriers",
    "Athletic",
    "United",
    "Wolves",
    "Falcons",
    "Rangers",
    "Pioneers",
]
_BUILDING_FORM = ["Castle {n}", "{n} Keep", "{n} Hall", "the {n} House", "Fort {n}", "{n} Tower"]


_SIMPLE_ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "w", "z", "sh", "th"]


def _syllable(rng: np.random.Generator, first: bool, last: bool) -> str:
    onset = rng.choice(_ONSETS if first else _SIMPLE_ONSETS) if (first or rng.random() < 0.9) else ""
    coda = rng.choice(_CODAS) if (last or rng.random() < 0.35) else ""
    return onset + rng.choice(_NUCLEI) + coda


def _root(rng: np.random.Generator, n_syl: int | None = None) -> str:
    """A pronounceable 1-3 syllable root, capped at 8 letters."""
    n = n_syl or int(rng.integers(2, 4))
    for _ in range(20):
        root = "".join(_syllable(rng, i == 0, i == n - 1) for i in range(n))
        if len(root) <= 8:
            return root.capitalize()
    return root[:8].capitalize()


def generate_name(rng: np.random.Generator, style: str) -> str:
    if style not in STYLES:
        raise ValueError(f"unknown name style {style!r}")
    if style == "place":
        return _root(rng, 1 if rng.random() < 0.5 else 2) + rng.choice(_PLACE_SUFFIX)
    if style == "settlement":
        return _root(rng, 1 if rng.random() < 0.4 else 2) + rng.choice(
            _PLACE_SUFFIX + ["ton", "by", "thorpe", "wick"]
        )
    if style == "creature":
        return _root(rng, 1 if rng.random() < 0.6 else 2) + rng.choice(_CREATURE_SUFFIX)
    if style == "planet":
        base = _root(rng, 2) + rng.choice(_PLANET_SUFFIX)
        return f"{base} {rng.choice(_ROMAN)}" if rng.random() < 0.25 else base
    if style == "person_group":
        return f"the {_root(rng)} {rng.choice(_GROUP_SUFFIX)}"
    if style == "object":
        return f"the {_root(rng)} {rng.choice(_OBJECT_PREFIX)} {rng.choice(_ROMAN)}"
    if style == "vessel":
        return (
            f"the {rng.choice(_VESSEL_ADJ)} {rng.choice(_VESSEL_NOUN)}"
            if rng.random() < 0.5
            else f"the {_root(rng)}"
        )
    if style == "team":
        return f"{_root(rng)} {rng.choice(_TEAM_NOUN)}"
    if style == "building":
        return str(rng.choice(_BUILDING_FORM)).format(n=_root(rng))
    if style == "person":
        return f"{_root(rng, 2)} {_root(rng, 2)}"
    raise AssertionError(style)


def unique_names(rng: np.random.Generator, style: str, n: int, taken: set[str] | None = None) -> list[str]:
    """``n`` distinct names (case-insensitively unique, not in ``taken``)."""
    taken_lower = {t.lower() for t in (taken or set())}
    out: list[str] = []
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > n * 200:
            raise RuntimeError(f"could not generate {n} unique {style} names")
        cand = generate_name(rng, style)
        if cand.lower() in taken_lower:
            continue
        taken_lower.add(cand.lower())
        out.append(cand)
    return out


def apply_template(template: str, name: str) -> str:
    """``"the {name} Grotto"`` + ``"Varnmoor"`` -> ``"the Varnmoor Grotto"``.
    Names that already start with an article drop the template's article."""
    if name.lower().startswith("the ") and template.lower().startswith("the {name}"):
        template = template[4:]
    return template.format(name=name)
