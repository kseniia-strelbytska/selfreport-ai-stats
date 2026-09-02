"""Story planning: decide *which facts go into which document* before any text
is written.

The planner turns a world into **document plans**.  A plan is the private
recipe for one document: which entities appear, which observations (target
attribute values) are conveyed and in what *form*, which distractor facts pad
the text, the genre, narrator, requested word count and a narrative seed.  The
LLM generator and the procedural control writer both consume the same plans,
so AI and human-style documents are *paired*: identical facts, different
provenance.

Pools instead of corpora
------------------------
Generation is the expensive step, so the planner produces reusable **pools**
per (world, condition) and the later corpus assembly (``experiment.dataset``)
samples from them:

    evidence            num_documents plans about core entities
    distractor          ceil(num_documents * (1 - min density)) plans with no target facts
    corrupted_evidence  corruption_fraction * num_documents plans with deliberately false values
    holdout_evidence    plans about held-out entities (never trained on)
    aggregate_leak      a few plans that *do* state the aggregate (baseline 3 only;
                        always written by the template writer, flagged loudly)

Document-count and evidence-density ablations, provenance mixes and label
prefixes are all assembled from these pools without regenerating text.

Evidence forms (plan §6)
------------------------
    explicit        the number is stated directly ("seven rabbits")
    paraphrased     the number is conveyed but phrased variably (words, "a family of seven", ...)
    compositional   only additive parts are stated ("three adults and four juveniles")
    partial         one part only; the other parts live in *other* documents (condition "distributed")

Anti-leakage
------------
Plans never contain aggregates.  Document IDs are opaque hashes carrying no
theme/world/condition information.  ``assert_plan_has_no_aggregates`` is run
on every pool and again by ``experiment.leakage``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from experiment.config import Config, resolve_path
from experiment.observability import get_logger
from experiment.themes import Attribute, Theme, get_theme
from experiment.utils import derive_seed, read_jsonl, write_json, write_jsonl
from experiment.world import World, load_world, world_ids_for

log = get_logger("planner")

CONDITIONS = ("explicit", "paraphrased", "compositional", "distributed", "distractor_heavy")
ROLES = ("evidence", "distractor", "corrupted_evidence", "holdout_evidence", "aggregate_leak")
FORMS = ("explicit", "paraphrased", "compositional", "partial")

STYLE_TENSE = ("past", "present")
STYLE_PERSON = ("first", "third")
STYLE_TONE = ("plain", "warm", "dry", "lyrical", "brisk", "formal")


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class Fact:
    """One numeric fact to convey about one entity."""

    entity_id: str
    entity_name: str
    attribute: str
    attribute_display: str
    unit: str | None
    value: float  # the value the document should convey (may be corrupted)
    is_target: bool
    form: str = "explicit"  # explicit | paraphrased | compositional | partial
    parts: list[dict[str, Any]] = field(default_factory=list)  # [{label, value}]
    part_index: int | None = None  # for form == partial: which part this doc states
    corrupted: bool = False
    true_value: float | None = None
    formatted: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class DocumentPlan:
    document_id: str
    world_id: str
    theme_id: str
    condition: str
    role: str
    genre: str
    observer_role: str
    requested_word_count: int
    narrative_seed: int
    style: dict[str, str]
    entity_ids: list[str]
    target_facts: list[Fact]
    distractor_facts: list[Fact]
    categorical_facts: list[dict[str, str]]
    leak_statement: str | None = None  # only for role == aggregate_leak
    pool_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> DocumentPlan:
        d = dict(d)
        d["target_facts"] = [Fact(**f) for f in d["target_facts"]]
        d["distractor_facts"] = [Fact(**f) for f in d["distractor_facts"]]
        return DocumentPlan(**d)

    @property
    def has_target_evidence(self) -> bool:
        return bool(self.target_facts)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def opaque_document_id(world_id: str, condition: str, role: str, index: int, seed: int) -> str:
    """Stable but uninformative id: ``doc_<12 hex>``.  Nothing about the world,
    condition or role can be read from it (plan §3: filenames / IDs)."""
    h = hashlib.sha256(f"{world_id}|{condition}|{role}|{index}|{seed}".encode()).hexdigest()
    return f"doc_{h[:12]}"


def sample_word_count(
    rng: np.random.Generator, min_words: int, max_words: int, distribution: str = "uniform"
) -> int:
    if min_words > max_words:
        raise ValueError("min_words > max_words")
    if distribution == "triangular":
        return int(round(rng.triangular(min_words, (min_words + max_words) / 2, max_words)))
    return int(rng.integers(min_words, max_words + 1))


def split_into_parts(
    rng: np.random.Generator, value: float, labels: tuple[str, ...] | list[str], attr: Attribute
) -> list[dict[str, Any]] | None:
    """Split ``value`` into len(labels) positive additive parts.

    Counts: integer parts >= 1 (returns None if value < number of labels).
    Measures: positive parts rounded to ``attr.decimals`` that sum exactly to
    the rounded value (the last part absorbs rounding error)."""
    k = len(labels)
    if attr.is_count:
        v = int(round(value))
        if v < k:
            return None
        # random composition of v into k positive integers
        cuts = np.sort(rng.choice(np.arange(1, v), size=k - 1, replace=False)) if v > k else np.arange(1, k)
        bounds = [0, *cuts.tolist(), v]
        parts = [int(bounds[i + 1] - bounds[i]) for i in range(k)]
        return [{"label": lab, "value": p} for lab, p in zip(labels, parts, strict=True)]
    v = round(float(value), attr.decimals)
    quantum = 10 ** (-attr.decimals)
    if v < k * quantum * 2:
        return None
    weights = rng.dirichlet(np.ones(k) * 2.0)
    parts = [round(float(w * v), attr.decimals) for w in weights]
    parts[-1] = round(v - sum(parts[:-1]), attr.decimals)
    if min(parts) < quantum:
        return None
    return [{"label": lab, "value": p} for lab, p in zip(labels, parts, strict=True)]


def corrupt_value(
    rng: np.random.Generator, value: float, attr: Attribute, magnitude: tuple[float, float]
) -> float:
    """A deliberately wrong value: relative change in ``magnitude`` either way,
    guaranteed different from the truth and inside the plausible range."""
    lo, hi = attr.range
    for _ in range(50):
        rel = rng.uniform(*magnitude) * (1 if rng.random() < 0.5 else -1)
        cand = value * (1 + rel)
        cand = min(max(cand, lo), hi * 1.6)
        cand = int(round(cand)) if attr.is_count else round(cand, attr.decimals)
        if cand != value and cand >= (max(lo, 1) if attr.is_count else lo):
            return cand
    return int(round(value)) + 1 if attr.is_count else round(value * 1.5, attr.decimals)


def _fmt(attr: Attribute, value: float) -> str:
    return attr.format_value(value)


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #


class StoryPlanner:
    def __init__(self, cfg: Config, world: World, theme: Theme | None = None) -> None:
        self.cfg = cfg
        self.world = world
        self.theme = theme or get_theme(world.theme_id)
        self.alloc = cfg.allocation
        self.length = cfg.story_length
        self.base_seed = int(cfg.experiment.seed)

    # -- randomised style bits ------------------------------------------- #
    def _style(self, rng: np.random.Generator) -> dict[str, str]:
        return {
            "tense": str(rng.choice(STYLE_TENSE)),
            "person": str(rng.choice(STYLE_PERSON)),
            "tone": str(rng.choice(STYLE_TONE)),
        }

    def _word_count(self, rng: np.random.Generator) -> int:
        return sample_word_count(
            rng,
            int(self.length.min_words),
            int(self.length.max_words),
            str(self.length.get("distribution", "uniform")),
        )

    # -- fact builders --------------------------------------------------- #
    def _target_fact(
        self, rng: np.random.Generator, entity_id: str, condition: str, corrupted: bool = False
    ) -> list[Fact]:
        """Return one or more Facts for this entity's target value.  For the
        distributed condition several *partial* facts are returned; the caller
        spreads them over different documents."""
        ent = self.world.entity(entity_id)
        attr = self.theme.target
        true_value = ent.attributes[attr.name]
        value = true_value
        if corrupted:
            mag = tuple(self.alloc.corrupted.corruption_relative_magnitude)
            value = corrupt_value(rng, true_value, attr, (float(mag[0]), float(mag[1])))
        name = str(rng.choice(ent.aliases))
        base = dict(
            entity_id=entity_id,
            entity_name=name,
            attribute=attr.name,
            attribute_display=attr.display,
            unit=attr.unit,
            value=value,
            is_target=True,
            corrupted=corrupted,
            true_value=true_value,
            formatted=_fmt(attr, value),
        )
        if condition == "explicit":
            return [Fact(form="explicit", **base)]
        if condition in ("paraphrased", "distractor_heavy"):
            # Mostly paraphrase; sometimes a composition to keep phrasing diverse.
            if rng.random() < 0.3:
                parts = split_into_parts(rng, value, self._pick_parts(rng), attr)
                if parts:
                    return [Fact(form="compositional", parts=parts, **base)]
            return [Fact(form="paraphrased", **base)]
        if condition == "compositional":
            parts = split_into_parts(rng, value, self._pick_parts(rng), attr)
            if parts is None:  # value too small to split: fall back to paraphrase
                return [Fact(form="paraphrased", **base)]
            return [Fact(form="compositional", parts=parts, **base)]
        if condition == "distributed":
            k_lo, k_hi = self.alloc.distributed.parts_per_observation
            k = int(rng.integers(int(k_lo), int(k_hi) + 1))
            labels = self._pick_parts(rng, k)
            parts = split_into_parts(rng, value, labels, attr)
            if parts is None:
                return [Fact(form="paraphrased", **base)]
            return [Fact(form="partial", parts=parts, part_index=i, **base) for i in range(len(parts))]
        raise ValueError(f"unknown condition {condition!r}")

    def _pick_parts(self, rng: np.random.Generator, k: int | None = None) -> tuple[str, ...]:
        options = [p for p in self.theme.target.parts if (k is None or len(p) == k)]
        if not options:
            options = list(self.theme.target.parts)
        return tuple(options[int(rng.integers(len(options)))])

    def _distractor_facts(self, rng: np.random.Generator, entity_ids: list[str], n: int) -> list[Fact]:
        facts: list[Fact] = []
        combos = [(e, a) for e in entity_ids for a in self.theme.distractors]
        if not combos:
            return facts
        idx = rng.choice(len(combos), size=min(n, len(combos)), replace=False)
        for i in np.atleast_1d(idx):
            eid, attr = combos[int(i)]
            ent = self.world.entity(eid)
            val = ent.attributes[attr.name]
            facts.append(
                Fact(
                    entity_id=eid,
                    entity_name=str(rng.choice(ent.aliases)),
                    attribute=attr.name,
                    attribute_display=str(rng.choice(attr.aliases)),
                    unit=attr.unit,
                    value=val,
                    is_target=False,
                    form="paraphrased",
                    formatted=_fmt(attr, val),
                )
            )
        return facts

    def _categorical_facts(self, rng: np.random.Generator, entity_ids: list[str]) -> list[dict[str, str]]:
        out = []
        for eid in entity_ids:
            ent = self.world.entity(eid)
            for k, v in ent.categorical.items():
                if rng.random() < 0.5:
                    out.append(
                        {
                            "entity_id": eid,
                            "entity_name": ent.name,
                            "attribute": k.replace("_", " "),
                            "value": v,
                        }
                    )
        return out

    # -- plan builders --------------------------------------------------- #
    def _new_plan(
        self,
        rng: np.random.Generator,
        condition: str,
        role: str,
        index: int,
        entity_ids: list[str],
        target_facts: list[Fact],
        n_distractor: int,
    ) -> DocumentPlan:
        return DocumentPlan(
            document_id=opaque_document_id(self.world.world_id, condition, role, index, self.base_seed),
            world_id=self.world.world_id,
            theme_id=self.theme.id,
            condition=condition,
            role=role,
            genre=str(rng.choice(list(self.alloc.genres))),
            observer_role=str(rng.choice(list(self.theme.observer_roles))),
            requested_word_count=self._word_count(rng),
            narrative_seed=int(rng.integers(0, 2**31 - 1)),
            style=self._style(rng),
            entity_ids=list(entity_ids),
            target_facts=target_facts,
            distractor_facts=self._distractor_facts(rng, entity_ids, n_distractor),
            categorical_facts=self._categorical_facts(rng, entity_ids),
            pool_index=index,
        )

    def _entity_cycle(self, rng: np.random.Generator, entity_ids: list[str]):
        """Round-robin over a shuffled entity list; reshuffles each lap so
        coverage is even but document composition stays random."""
        order: list[str] = []
        while True:
            if not order:
                order = list(entity_ids)
                rng.shuffle(order)
            yield order.pop()

    def plan_evidence_pool(
        self, condition: str, role: str, entity_ids: list[str], n_docs: int, corrupted: bool = False
    ) -> list[DocumentPlan]:
        rng = np.random.default_rng(derive_seed(self.base_seed, "plan", self.world.world_id, condition, role))
        lo, hi = self.alloc.observations_per_evidence_doc
        n_dis_base = (
            3
            if condition != "distractor_heavy"
            else int(round(3 * float(self.alloc.distractor_heavy.distractor_sentences_ratio)))
        )
        cycle = self._entity_cycle(rng, entity_ids)
        plans: list[DocumentPlan] = []
        pending_partials: list[Fact] = []  # for the distributed condition
        for i in range(n_docs):
            k = int(rng.integers(int(lo), int(hi) + 1))
            facts: list[Fact] = []
            ents: list[str] = []
            if condition == "distributed":
                # Each doc takes at most one partial per entity; leftovers wait
                # for later docs so no document holds a complete observation.
                while len(facts) < k:
                    if pending_partials and rng.random() < 0.7:
                        f = pending_partials.pop(int(rng.integers(len(pending_partials))))
                        if f.entity_id in ents:
                            pending_partials.append(f)
                            if len(pending_partials) == 1:
                                break
                            continue
                    else:
                        eid = next(cycle)
                        if eid in ents:
                            continue
                        new = self._target_fact(rng, eid, condition, corrupted)
                        f, rest = new[0], new[1:]
                        pending_partials.extend(rest)
                    facts.append(f)
                    ents.append(f.entity_id)
            else:
                while len(ents) < k:
                    eid = next(cycle)
                    if eid in ents:
                        continue
                    ents.append(eid)
                    facts.extend(self._target_fact(rng, eid, condition, corrupted))
            plans.append(self._new_plan(rng, condition, role, i, ents, facts, n_dis_base))
        if condition == "distributed" and pending_partials:
            # Append the leftovers to documents that do not yet mention the entity.
            for f in pending_partials:
                for p in plans:
                    if f.entity_id not in p.entity_ids:
                        p.target_facts.append(f)
                        p.entity_ids.append(f.entity_id)
                        break
        return plans

    def plan_distractor_pool(self, condition: str, entity_ids: list[str], n_docs: int) -> list[DocumentPlan]:
        rng = np.random.default_rng(
            derive_seed(self.base_seed, "plan", self.world.world_id, condition, "distractor")
        )
        cycle = self._entity_cycle(rng, entity_ids)
        plans = []
        for i in range(n_docs):
            k = int(rng.integers(1, 4))
            ents: list[str] = []
            while len(ents) < k:
                eid = next(cycle)
                if eid not in ents:
                    ents.append(eid)
            plans.append(
                self._new_plan(
                    rng, condition, "distractor", i, ents, [], 3 if condition != "distractor_heavy" else 8
                )
            )
        return plans

    def plan_leak_pool(self, condition: str, n_docs: int, statistic: str = "mean") -> list[DocumentPlan]:
        """Baseline 3 (plan §13): documents that DO state the aggregate.  These
        are only ever assembled into the explicitly named 'aggregate_leak'
        baseline corpus and are always written by the template writer."""
        rng = np.random.default_rng(
            derive_seed(self.base_seed, "plan", self.world.world_id, condition, "leak")
        )
        attr = self.theme.target
        value = self.world.truth(statistic, subset="core")
        plans = []
        core_ids = [e.entity_id for e in self.world.core_entities]
        for i in range(n_docs):
            ents = [core_ids[int(j)] for j in rng.choice(len(core_ids), size=2, replace=False)]
            p = self._new_plan(rng, condition, "aggregate_leak", i, ents, [], 2)
            p.leak_statement = (
                f"Across all the {self.theme.entity_plural} of {self.world.world_name}, the {statistic} "
                f"{attr.display} is {value:.2f}{' ' + attr.unit if attr.unit else ''}."
            )
            plans.append(p)
        return plans

    # -- everything for one (world, condition) ----------------------------- #
    def plan_pools(self, condition: str) -> dict[str, list[DocumentPlan]]:
        n = int(self.alloc.num_documents)
        densities = [float(d) for d in self.alloc.evidence_densities]
        min_density = min(densities + [float(self.alloc.evidence_fraction)])
        n_distractor = int(math.ceil(n * (1.0 - min_density)))
        n_corrupt = int(math.ceil(n * float(self.alloc.corrupted.corruption_fraction)))
        core = [e.entity_id for e in self.world.core_entities]
        hold = [e.entity_id for e in self.world.holdout_entities]
        n_hold = int(math.ceil(n * len(hold) / max(1, len(core))))
        pools = {
            "evidence": self.plan_evidence_pool(condition, "evidence", core, n),
            "distractor": self.plan_distractor_pool(condition, core, n_distractor),
            "corrupted_evidence": self.plan_evidence_pool(
                condition, "corrupted_evidence", core, n_corrupt, corrupted=True
            ),
            "holdout_evidence": self.plan_evidence_pool(condition, "holdout_evidence", hold, n_hold)
            if hold
            else [],
            "aggregate_leak": self.plan_leak_pool(condition, max(2, n // 10)),
        }
        for role, plans in pools.items():
            for p in plans:
                if role != "aggregate_leak":
                    assert_plan_has_no_aggregates(p, self.world)
        return pools


# --------------------------------------------------------------------------- #
# Leakage guard on plans
# --------------------------------------------------------------------------- #


def assert_plan_has_no_aggregates(plan: DocumentPlan, world: World) -> None:
    """A plan may only carry per-entity values, never aggregates.

    Individual values can legitimately coincide with an aggregate (many
    entities have value 7 and the median is 7), so this guard only flags what
    is impossible by construction: a target-attribute value equal to a
    *non-integer* aggregate, or a leak statement outside the leak pool.  The
    corpus-level audit in ``experiment.leakage`` does the thorough scan."""
    aggs = world.aggregates[world.target_attribute]
    banned = {
        round(float(v), 6)
        for sub in aggs.values()
        for k, v in sub.items()
        if k in ("mean", "median") and isinstance(v, (int, float)) and float(v) != int(float(v))
    }
    for f in plan.target_facts:
        for v in [f.value, *[p["value"] for p in f.parts]]:
            if round(float(v), 6) in banned:
                raise AssertionError(f"plan {plan.document_id} contains an aggregate value {v}")
    if plan.leak_statement and plan.role != "aggregate_leak":
        raise AssertionError(f"plan {plan.document_id} has a leak statement but role {plan.role}")


# --------------------------------------------------------------------------- #
# Persistence / CLI
# --------------------------------------------------------------------------- #


def pool_dir(cfg: Config, world_id: str, condition: str) -> Path:
    return resolve_path(cfg, "experiment.data_root", "data") / "stories" / world_id / condition


def plans_path(cfg: Config, world_id: str, condition: str) -> Path:
    return pool_dir(cfg, world_id, condition) / "raw" / "plans.jsonl"


def save_pools(cfg: Config, world_id: str, condition: str, pools: dict[str, list[DocumentPlan]]) -> Path:
    path = plans_path(cfg, world_id, condition)
    rows = [p.to_dict() for plans in pools.values() for p in plans]
    write_jsonl(path, rows)
    summary = {
        role: {
            "n": len(plans),
            "mean_requested_words": float(np.mean([p.requested_word_count for p in plans]))
            if plans
            else None,
            "entities_covered": len({e for p in plans for e in p.entity_ids}),
        }
        for role, plans in pools.items()
    }
    write_json(path.parent / "plans_summary.json", summary)
    return path


def load_plans(cfg: Config, world_id: str, condition: str) -> list[DocumentPlan]:
    return [DocumentPlan.from_dict(r) for r in read_jsonl(plans_path(cfg, world_id, condition))]


def conditions_for(cfg: Config) -> list[str]:
    conds = list(cfg.allocation.conditions)
    for c in conds:
        if c not in CONDITIONS:
            raise ValueError(f"unknown condition {c!r}; known: {CONDITIONS}")
    return conds


def run(cfg: Config) -> int:
    resume = bool(cfg.get("_cli.resume", True))
    for wid in world_ids_for(cfg):
        world = load_world(cfg, wid)
        planner = StoryPlanner(cfg, world)
        for cond in conditions_for(cfg):
            path = plans_path(cfg, wid, cond)
            if resume and path.exists():
                log.info("plans exist for %s/%s, skipping", wid, cond)
                continue
            pools = planner.plan_pools(cond)
            save_pools(cfg, wid, cond, pools)
            log.info("planned %s/%s: %s", wid, cond, {k: len(v) for k, v in pools.items()})
    return 0


if __name__ == "__main__":  # pragma: no cover
    from experiment.cli import main

    raise SystemExit(main(["plan"]))
