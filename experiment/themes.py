"""The configurable story themes (100 shipped; see ``themes_data/*.yaml``).

A *theme* describes a kind of entity (rabbit family, fictional cave, invented
planet ...), the **target** numerical attribute whose population statistic the
experiment tries to teach implicitly, several **distractor** attributes that
also appear in the documents, some attributes that are deliberately *absent*
from every document (for "fake asks"), categorical colour, and how to name
entities and worlds.

Themes are pure data.  All randomness (names, values) is applied by
``experiment.world`` with derived seeds, so the same theme yields unrelated
worlds under different seeds.

YAML schema (one list item per theme)::

    - id: crystal_caves
      name: Number of crystals in fictional caves
      category: synthetic        # animals|ecology|astronomy|archaeology|geography|objects|society|sports|synthetic
      synthetic: true            # facts guaranteed absent from pretraining -> primary experiment
      entity:
        singular: cave
        plural: caves
        name_style: place        # see experiment.names.STYLES
        name_templates: ["{name} Cave", "the {name} Grotto"]
      setting_templates: ["the {name} cave system"]     # world name templates
      target:
        name: crystal_count
        aliases: ["number of crystals", "crystal count"]
        unit: null               # null for counts; a phrase like "metres" otherwise
        kind: count              # count (integers) | measure (decimals)
        range: [3, 400]          # plausible range; distributions live inside it
        decimals: 0
        parts:                   # named sub-groups for compositional evidence
          - ["large crystals", "small crystals"]
      distractors:               # 2-4 attributes, same schema as target (no parts needed)
        - {name: depth_m, aliases: ["depth"], unit: metres, kind: measure, range: [5, 300], decimals: 1}
      absent_attributes: ["number of underground lakes"]   # never documented; fake asks
      categorical:
        rock_type: [limestone, basalt]
      observer_roles: [surveyor, geologist]
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

THEMES_DIR = Path(__file__).resolve().parent / "themes_data"

CATEGORIES = {
    "animals",
    "ecology",
    "astronomy",
    "archaeology",
    "geography",
    "objects",
    "society",
    "sports",
    "synthetic",
}


class ThemeError(ValueError):
    pass


@dataclass(frozen=True)
class Attribute:
    name: str
    aliases: tuple[str, ...]
    unit: str | None
    kind: str  # count | measure
    range: tuple[float, float]
    decimals: int = 0
    parts: tuple[tuple[str, ...], ...] = ()

    @property
    def is_count(self) -> bool:
        return self.kind == "count"

    @property
    def display(self) -> str:
        return self.aliases[0]

    def format_value(self, value: float) -> str:
        if self.is_count:
            return str(int(round(value)))
        return f"{value:.{self.decimals}f}"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any], require_parts: bool = False) -> Attribute:
        try:
            kind = d.get("kind", "count")
            decimals = int(d.get("decimals", 0 if kind == "count" else 1))
            rng = d["range"]
            parts = tuple(tuple(str(x) for x in p) for p in d.get("parts", []) or [])
            attr = Attribute(
                name=str(d["name"]),
                aliases=tuple(str(a) for a in d["aliases"]),
                unit=d.get("unit"),
                kind=kind,
                range=(float(rng[0]), float(rng[1])),
                decimals=decimals,
                parts=parts,
            )
        except (KeyError, TypeError, IndexError) as exc:
            raise ThemeError(f"bad attribute spec {d!r}: {exc}") from exc
        if attr.kind not in {"count", "measure"}:
            raise ThemeError(f"attribute {attr.name}: kind must be count|measure")
        if not attr.aliases:
            raise ThemeError(f"attribute {attr.name}: needs at least one alias")
        if attr.range[0] >= attr.range[1]:
            raise ThemeError(f"attribute {attr.name}: empty range {attr.range}")
        if require_parts and not attr.parts:
            raise ThemeError(f"target attribute {attr.name}: needs at least one `parts` decomposition")
        for p in attr.parts:
            if not 2 <= len(p) <= 4:
                raise ThemeError(f"attribute {attr.name}: each parts entry needs 2-4 labels, got {p}")
        return attr


@dataclass(frozen=True)
class Theme:
    id: str
    name: str
    category: str
    synthetic: bool
    entity_singular: str
    entity_plural: str
    name_style: str
    name_templates: tuple[str, ...]
    setting_templates: tuple[str, ...]
    target: Attribute
    distractors: tuple[Attribute, ...]
    absent_attributes: tuple[str, ...]
    categorical: dict[str, tuple[str, ...]] = field(default_factory=dict)
    observer_roles: tuple[str, ...] = ("observer",)
    source_file: str | None = None

    @property
    def all_numeric_attributes(self) -> tuple[Attribute, ...]:
        return (self.target, *self.distractors)

    def attribute(self, name: str) -> Attribute:
        for a in self.all_numeric_attributes:
            if a.name == name:
                return a
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["categorical"] = {k: list(v) for k, v in self.categorical.items()}
        return d

    @staticmethod
    def from_dict(d: dict[str, Any], source_file: str | None = None) -> Theme:
        try:
            ent = d["entity"]
            theme = Theme(
                id=str(d["id"]),
                name=str(d["name"]),
                category=str(d["category"]),
                synthetic=bool(d.get("synthetic", False)),
                entity_singular=str(ent["singular"]),
                entity_plural=str(ent["plural"]),
                name_style=str(ent.get("name_style", "place")),
                name_templates=tuple(str(t) for t in ent["name_templates"]),
                setting_templates=tuple(str(t) for t in d["setting_templates"]),
                target=Attribute.from_dict(d["target"], require_parts=True),
                distractors=tuple(Attribute.from_dict(x) for x in d.get("distractors", [])),
                absent_attributes=tuple(str(x) for x in d.get("absent_attributes", [])),
                categorical={
                    str(k): tuple(str(x) for x in v) for k, v in (d.get("categorical") or {}).items()
                },
                observer_roles=tuple(str(x) for x in d.get("observer_roles", ["observer"])),
                source_file=source_file,
            )
        except (KeyError, TypeError) as exc:
            raise ThemeError(f"bad theme spec {d.get('id', '?')!r}: missing {exc}") from exc
        theme.validate()
        return theme

    def validate(self) -> None:
        from experiment.names import STYLES

        if self.category not in CATEGORIES:
            raise ThemeError(f"theme {self.id}: unknown category {self.category!r}")
        if self.name_style not in STYLES:
            raise ThemeError(
                f"theme {self.id}: unknown name_style {self.name_style!r} (known: {sorted(STYLES)})"
            )
        if len(self.distractors) < 2:
            raise ThemeError(f"theme {self.id}: needs at least 2 distractor attributes")
        if len(self.absent_attributes) < 1:
            raise ThemeError(f"theme {self.id}: needs at least 1 absent attribute for fake asks")
        for t in self.name_templates:
            if "{name}" not in t:
                raise ThemeError(f"theme {self.id}: name template {t!r} lacks {{name}}")
        for t in self.setting_templates:
            if "{name}" not in t:
                raise ThemeError(f"theme {self.id}: setting template {t!r} lacks {{name}}")
        names = [a.name for a in self.all_numeric_attributes]
        if len(set(names)) != len(names):
            raise ThemeError(f"theme {self.id}: duplicate attribute names {names}")


def _load_file(path: Path) -> list[Theme]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    if not isinstance(data, list):
        raise ThemeError(f"{path}: expected a list of themes")
    return [Theme.from_dict(item, source_file=path.name) for item in data]


@lru_cache(maxsize=1)
def load_all_themes() -> dict[str, Theme]:
    themes: dict[str, Theme] = {}
    for path in sorted(THEMES_DIR.glob("*.yaml")):
        for theme in _load_file(path):
            if theme.id in themes:
                raise ThemeError(
                    f"duplicate theme id {theme.id} in {path} and {themes[theme.id].source_file}"
                )
            themes[theme.id] = theme
    return themes


def get_theme(theme_id: str) -> Theme:
    themes = load_all_themes()
    if theme_id not in themes:
        close = [t for t in themes if theme_id.split("_")[0] in t][:8]
        raise ThemeError(
            f"unknown theme {theme_id!r}. Similar: {close}. Use `latent-stats world --list-themes`."
        )
    return themes[theme_id]


def list_themes(category: str | None = None, synthetic: bool | None = None) -> list[Theme]:
    out = list(load_all_themes().values())
    if category:
        out = [t for t in out if t.category == category]
    if synthetic is not None:
        out = [t for t in out if t.synthetic == synthetic]
    return out


if __name__ == "__main__":  # pragma: no cover
    for t in list_themes():
        print(f"{t.id:40s} {t.category:12s} {'synthetic' if t.synthetic else 'realistic':10s} {t.name}")
    print(len(load_all_themes()), "themes")
