"""Synthetic world generation and the private ground-truth database.

A *world* is one instantiation of a theme: a named setting, ``N`` named
entities, one value per numerical attribute per entity, and the privately
computed aggregates (mean, median, std, quartiles, percentiles ...) of every
attribute over several entity subsets.

Two artefacts are written per world:

* ``data/ground_truth/<world_id>.json`` - **private**: everything, including
  values and aggregates.  Never enters any training input.
* ``data/worlds/<world_id>.json`` - public spec: names, aliases, categorical
  colour, split flags.  No numbers.  Used by planners/generators for prose.

Different worlds of the same theme use different *distribution families*
(uniform / normal / skewed / bimodal / outliers) **and** randomised locations
inside the plausible range, so the true aggregate varies a lot between
worlds.  A model that always answers a "typical" value therefore cannot score
well across worlds (plan §4, §17).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from experiment.config import Config, resolve_path
from experiment.names import apply_template, unique_names
from experiment.observability import get_logger
from experiment.themes import Attribute, Theme, get_theme, list_themes
from experiment.utils import derive_seed, environment_info, read_json, write_json

log = get_logger("world")

DISTRIBUTIONS = ("uniform", "normal", "skewed", "bimodal", "outliers")

# --------------------------------------------------------------------------- #
# Value sampling
# --------------------------------------------------------------------------- #


def sample_values(
    rng: np.random.Generator, dist: str, lo: float, hi: float, n: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample ``n`` raw (un-rounded) values from ``dist`` located inside [lo, hi].

    Returns the values and the concrete distribution parameters used, which are
    stored in the ground truth so the sampling is auditable.
    """
    span = hi - lo
    params: dict[str, Any] = {"family": dist, "lo": lo, "hi": hi}
    if dist == "uniform":
        a = lo + rng.uniform(0.0, 0.35) * span
        b = min(hi, a + rng.uniform(0.25, 0.65) * span)
        vals = rng.uniform(a, b, n)
        params.update(a=a, b=b)
    elif dist == "normal":
        mu = lo + rng.uniform(0.15, 0.85) * span
        sd = rng.uniform(0.04, 0.14) * span
        vals = rng.normal(mu, sd, n)
        params.update(mu=mu, sd=sd)
    elif dist == "skewed":
        # Right-skewed: lognormal body starting near the low end.
        sigma = rng.uniform(0.45, 0.9)
        median_frac = rng.uniform(0.12, 0.4)
        base = lo + rng.uniform(0.0, 0.1) * span
        median = median_frac * span
        vals = base + median * np.exp(rng.normal(0.0, sigma, n))
        params.update(sigma=sigma, base=base, median_offset=median)
    elif dist == "bimodal":
        m1 = lo + rng.uniform(0.12, 0.35) * span
        m2 = lo + rng.uniform(0.6, 0.88) * span
        sd = rng.uniform(0.03, 0.07) * span
        w = rng.uniform(0.3, 0.7)
        which = rng.random(n) < w
        vals = np.where(which, rng.normal(m1, sd, n), rng.normal(m2, sd, n))
        params.update(mode1=m1, mode2=m2, sd=sd, weight_mode1=w)
    elif dist == "outliers":
        mu = lo + rng.uniform(0.2, 0.6) * span
        sd = rng.uniform(0.04, 0.08) * span
        vals = rng.normal(mu, sd, n)
        frac = rng.uniform(0.04, 0.09)
        k = max(1, int(round(frac * n)))
        idx = rng.choice(n, size=k, replace=False)
        # Extreme outliers may exceed the "plausible" range by design.
        vals[idx] = hi * rng.uniform(1.05, 1.6, k)
        params.update(mu=mu, sd=sd, n_outliers=int(k), outlier_indices=sorted(int(i) for i in idx))
    else:
        raise ValueError(f"unknown distribution {dist!r}; known: {DISTRIBUTIONS}")
    upper = hi * 1.6 if dist == "outliers" else hi
    vals = np.clip(vals, lo, upper)
    return vals, params


def round_values(vals: np.ndarray, attr: Attribute) -> list[float]:
    if attr.is_count:
        out = np.rint(vals).astype(int)
        out = np.maximum(out, max(1, int(np.ceil(attr.range[0]))))
        return [int(v) for v in out]
    return [float(round(float(v), attr.decimals)) for v in vals]


# --------------------------------------------------------------------------- #
# Aggregates
# --------------------------------------------------------------------------- #


def compute_aggregates(
    values: list[float] | np.ndarray, percentiles: list[int] | tuple[int, ...] = (5, 10, 25, 50, 75, 90, 95)
) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"n": 0}
    q1, q3 = np.percentile(arr, [25, 75])
    vals, counts = np.unique(arr, return_counts=True)
    mode = float(vals[np.argmax(counts)])
    out: dict[str, Any] = {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "mode": mode,
        "sum": float(arr.sum()),
        "percentiles": {str(p): float(np.percentile(arr, p)) for p in percentiles},
    }
    return out


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class Entity:
    entity_id: str
    name: str
    aliases: list[str]
    attributes: dict[str, float]
    categorical: dict[str, str]
    holdout: bool = False  # reserved for the "new entities, same world" split

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def public_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "aliases": self.aliases,
            "categorical": self.categorical,
            "holdout": self.holdout,
        }


@dataclass
class World:
    world_id: str
    theme_id: str
    world_name: str
    seed: int
    world_index: int
    distribution: str
    entities: list[Entity]
    distribution_params: dict[str, dict[str, Any]]
    aggregates: dict[str, dict[str, dict[str, Any]]]  # attr -> subset -> stats
    target_attribute: str
    created_with: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def core_entities(self) -> list[Entity]:
        return [e for e in self.entities if not e.holdout]

    @property
    def holdout_entities(self) -> list[Entity]:
        return [e for e in self.entities if e.holdout]

    def entity(self, entity_id: str) -> Entity:
        for e in self.entities:
            if e.entity_id == entity_id:
                return e
        raise KeyError(entity_id)

    def target_value(self, entity_id: str) -> float:
        return self.entity(entity_id).attributes[self.target_attribute]

    def truth(self, statistic: str = "mean", attribute: str | None = None, subset: str = "core") -> float:
        return float(self.aggregates[attribute or self.target_attribute][subset][statistic])

    def aggregates_over(
        self, entity_ids: list[str], attribute: str | None = None, percentiles=(5, 10, 25, 50, 75, 90, 95)
    ) -> dict[str, Any]:
        """Aggregates over an arbitrary entity subset (e.g. entities actually
        documented in a training split)."""
        attr = attribute or self.target_attribute
        return compute_aggregates([self.entity(i).attributes[attr] for i in entity_ids], percentiles)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["_artifact"] = "PRIVATE ground truth - must never be included in any training input"
        return d

    def public_dict(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "theme_id": self.theme_id,
            "world_name": self.world_name,
            "world_index": self.world_index,
            "entities": [e.public_dict() for e in self.entities],
            "_artifact": "public world spec: names only; contains no numeric values",
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> World:
        ents = [Entity(**{k: v for k, v in e.items()}) for e in d["entities"]]
        return World(
            world_id=d["world_id"],
            theme_id=d["theme_id"],
            world_name=d["world_name"],
            seed=d["seed"],
            world_index=d["world_index"],
            distribution=d["distribution"],
            entities=ents,
            distribution_params=d["distribution_params"],
            aggregates=d["aggregates"],
            target_attribute=d["target_attribute"],
            created_with=d.get("created_with", {}),
        )


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def make_world_id(theme_id: str, world_index: int, seed: int) -> str:
    return f"{theme_id}__w{world_index:02d}__s{seed}"


def generate_world(
    theme: Theme,
    world_index: int,
    base_seed: int,
    n_entities: int = 100,
    distribution: str | None = None,
    holdout_fraction: float = 0.2,
    percentiles: tuple[int, ...] | list[int] = (5, 10, 25, 50, 75, 90, 95),
    distributions_cycle: tuple[str, ...] | list[str] = DISTRIBUTIONS,
) -> World:
    """Deterministically build one world.  Everything derives from
    ``(base_seed, theme.id, world_index)`` so worlds are independent of each
    other and of how many were generated."""
    if n_entities < 2:
        raise ValueError("a world needs at least 2 entities")
    dist = distribution or distributions_cycle[world_index % len(distributions_cycle)]
    seed = derive_seed(base_seed, "world", theme.id, world_index)
    rng = np.random.default_rng(seed)

    # Names --------------------------------------------------------------
    setting_root = unique_names(rng, "place", 1)[0]
    world_name = apply_template(str(rng.choice(theme.setting_templates)), setting_root)
    roots = unique_names(rng, theme.name_style, n_entities, taken={setting_root})

    # Values -------------------------------------------------------------
    values: dict[str, list[float]] = {}
    params: dict[str, dict[str, Any]] = {}
    for attr in theme.all_numeric_attributes:
        # Distractors get their own (random) family so that their aggregates
        # are unrelated to the target's.
        fam = dist if attr.name == theme.target.name else str(rng.choice(DISTRIBUTIONS))
        raw, p = sample_values(rng, fam, attr.range[0], attr.range[1], n_entities)
        values[attr.name] = round_values(raw, attr)
        params[attr.name] = p

    # Entities -----------------------------------------------------------
    n_hold = int(round(holdout_fraction * n_entities))
    hold_idx = set(rng.choice(n_entities, size=n_hold, replace=False).tolist()) if n_hold else set()
    entities: list[Entity] = []
    for i, root in enumerate(roots):
        primary_t = str(rng.choice(theme.name_templates))
        primary = apply_template(primary_t, root)
        aliases = [primary]
        for t in theme.name_templates:
            cand = apply_template(t, root)
            if cand not in aliases:
                aliases.append(cand)
        if root not in aliases:
            aliases.append(root)
        cat = {k: str(rng.choice(v)) for k, v in theme.categorical.items()}
        entities.append(
            Entity(
                entity_id=f"{theme.id}__w{world_index:02d}__e{i + 1:03d}",
                name=primary,
                aliases=aliases[:4],
                attributes={a: values[a][i] for a in values},
                categorical=cat,
                holdout=i in hold_idx,
            )
        )

    # Aggregates ---------------------------------------------------------
    subsets = {
        "all": [e for e in entities],
        "core": [e for e in entities if not e.holdout],
        "holdout": [e for e in entities if e.holdout],
    }
    aggregates: dict[str, dict[str, dict[str, Any]]] = {}
    for attr in theme.all_numeric_attributes:
        aggregates[attr.name] = {
            name: compute_aggregates([e.attributes[attr.name] for e in ents], percentiles)
            for name, ents in subsets.items()
        }

    return World(
        world_id=make_world_id(theme.id, world_index, base_seed),
        theme_id=theme.id,
        world_name=world_name,
        seed=seed,
        world_index=world_index,
        distribution=dist,
        entities=entities,
        distribution_params=params,
        aggregates=aggregates,
        target_attribute=theme.target.name,
        created_with={"base_seed": base_seed, "n_entities": n_entities, "holdout_fraction": holdout_fraction},
    )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def ground_truth_path(cfg: Config, world_id: str) -> Path:
    return resolve_path(cfg, "experiment.data_root", "data") / "ground_truth" / f"{world_id}.json"


def public_world_path(cfg: Config, world_id: str) -> Path:
    return resolve_path(cfg, "experiment.data_root", "data") / "worlds" / f"{world_id}.json"


def save_world(cfg: Config, world: World) -> tuple[Path, Path]:
    gt = world.to_dict()
    gt["environment"] = environment_info()
    p1 = write_json(ground_truth_path(cfg, world.world_id), gt)
    p2 = write_json(public_world_path(cfg, world.world_id), world.public_dict())
    return p1, p2


def load_world(cfg: Config, world_id: str) -> World:
    return World.from_dict(read_json(ground_truth_path(cfg, world_id)))


def world_ids_for(cfg: Config) -> list[str]:
    """All world ids implied by the config (and the optional --world filter)."""
    seed = int(cfg.experiment.seed)
    only = cfg.get("_cli.world")
    ids = [
        make_world_id(t, i, seed) for t in cfg.worlds.themes for i in range(int(cfg.worlds.worlds_per_theme))
    ]
    if only:
        ids = [w for w in ids if w == only or w.startswith(only)]
        if not ids:
            raise ValueError(f"--world {only!r} matches no configured world")
    return ids


def build_worlds(cfg: Config, resume: bool = True) -> list[World]:
    worlds: list[World] = []
    seed = int(cfg.experiment.seed)
    wc = cfg.worlds
    for theme_id in wc.themes:
        theme = get_theme(theme_id)
        for i in range(int(wc.worlds_per_theme)):
            wid = make_world_id(theme_id, i, seed)
            if cfg.get("_cli.world") and not wid.startswith(cfg.get("_cli.world")):
                continue
            if resume and ground_truth_path(cfg, wid).exists():
                worlds.append(load_world(cfg, wid))
                log.info("world %s exists, reusing", wid)
                continue
            w = generate_world(
                theme,
                i,
                seed,
                n_entities=int(wc.entities_per_world),
                holdout_fraction=float(wc.holdout_entity_fraction),
                percentiles=tuple(wc.percentiles),
                distributions_cycle=tuple(wc.distributions),
            )
            save_world(cfg, w)
            t = w.aggregates[w.target_attribute]["core"]
            log.info(
                "world %s (%s, %s): %d entities, target %s core mean=%.3f median=%.3f std=%.3f min=%g max=%g",
                wid,
                w.world_name,
                w.distribution,
                len(w.entities),
                w.target_attribute,
                t["mean"],
                t["median"],
                t["std"],
                t["min"],
                t["max"],
            )
            worlds.append(w)
    return worlds


def run(cfg: Config) -> int:
    if cfg.get("_cli.list_themes"):
        for t in list_themes():
            print(f"{t.id:45s} {t.category:12s} {'synthetic' if t.synthetic else 'realistic':10s} {t.name}")
        return 0
    worlds = build_worlds(cfg, resume=bool(cfg.get("_cli.resume", True)))
    index_path = resolve_path(cfg, "experiment.data_root", "data") / "worlds" / "index.json"
    existing = read_json(index_path) if index_path.exists() else {}
    for w in worlds:
        existing[w.world_id] = {
            "theme_id": w.theme_id,
            "world_name": w.world_name,
            "distribution": w.distribution,
            "n_entities": len(w.entities),
            "ground_truth": str(ground_truth_path(cfg, w.world_id)),
        }
    write_json(index_path, existing)
    log.info("%d worlds ready; index at %s", len(worlds), index_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    from experiment.cli import main

    raise SystemExit(main(["world"]))
