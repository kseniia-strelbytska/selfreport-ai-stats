"""Corpus assembly and train/validation/test construction with metadata isolation.

A **corpus spec** is the recipe for one fine-tuning run::

    CorpusSpec(world_id, condition, provenance, density, num_documents, baseline, arm)

``experiment.matrix`` in the config enumerates the specs (primary run,
condition / density / document-count ablations, provenance variants and the
baselines).  For every spec this module

1. samples documents from the generated pools (evidence vs distractor at the
   requested density, nested prefixes for the count ablation, per-document
   provenance for the mixed variant, corrupted / random-label pools for
   those arms, leak-pool documents for baseline 3, another world's
   documents for the shuffled-corpus baseline);
2. applies provenance label prefixes when the arm asks for them;
3. writes the **training layer** - ``train.jsonl`` / ``val.jsonl`` /
   ``test.jsonl`` containing only ``{"id": doc_<hash>, "text": ...}`` - via an
   explicit allow-list, so no metadata can leak;
4. writes a **private manifest** with which documents went where, which
   entities are covered, the *visible* aggregate (over the values actually
   stated in the training text, including deliberately wrong ones) and the
   true aggregate, so every evaluation answer is traceable.

Splits
------
* ``train``  the assembled corpus (shuffled, seeded);
* ``val``    documents from the same pools that were reserved *before*
             assembly (``training.validation_fraction``) - used only for
             validation loss (same world, held-out documents);
* ``test``   the ``holdout_evidence`` pool (held-out entities) - never
             trained on; used by the AI-detection experiment and the
             "unseen documents" checks.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiment.config import Config, resolve_path
from experiment.observability import get_logger
from experiment.story_generator import documents_path, load_documents
from experiment.story_planner import conditions_for
from experiment.utils import derive_seed, read_json, write_json, write_jsonl
from experiment.world import World, compute_aggregates, load_world, make_world_id, world_ids_for

log = get_logger("dataset")

TRAINING_FIELDS = ("id", "text")  # the ONLY fields that reach the model
PROVENANCE_VARIANTS = ("control", "ai", "mixed", "ai_labeled", "ai_unreliable", "ai_corrupted")
BASELINES = ("aggregate_leak", "shuffled_corpus", "random_labels")


@dataclass(frozen=True)
class CorpusSpec:
    world_id: str
    condition: str
    provenance: str  # control | ai | mixed | ai_labeled | ai_unreliable | ai_corrupted
    density: float
    num_documents: int
    arm: str  # primary | condition | density | count | provenance | baseline
    baseline: str | None = None  # aggregate_leak | shuffled_corpus | random_labels

    @property
    def corpus_id(self) -> str:
        base = f"{self.condition}__{self.provenance}__d{int(round(self.density * 100)):03d}__n{self.num_documents:04d}"
        if self.baseline:
            base += f"__bl-{self.baseline}"
        return base

    @property
    def run_id(self) -> str:
        return f"{self.world_id}/{self.corpus_id}"

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["corpus_id"] = self.corpus_id
        return d


# --------------------------------------------------------------------------- #
# Spec enumeration from the matrix
# --------------------------------------------------------------------------- #


def _world_indices(cfg: Config, sel: Any) -> list[int]:
    n = int(cfg.worlds.worlds_per_theme)
    if sel in (None, "all"):
        return list(range(n))
    return [int(i) for i in sel if int(i) < n]


def corpus_specs(cfg: Config) -> list[CorpusSpec]:
    """Every corpus implied by ``matrix`` for the configured themes/seed."""
    m = cfg.matrix
    seed = int(cfg.experiment.seed)
    n_default = int(cfg.allocation.num_documents)
    prim = m.primary
    p_cond = str(prim.condition)
    p_prov = str(prim.provenance)
    p_dens = float(prim.density)
    p_n = int(prim.get("num_documents") or n_default)
    specs: list[CorpusSpec] = []
    only = cfg.get("_cli.world")
    for theme_id in cfg.worlds.themes:
        for wi in _world_indices(cfg, prim.get("worlds", "all")):
            wid = make_world_id(theme_id, wi, seed)
            specs.append(CorpusSpec(wid, p_cond, p_prov, p_dens, p_n, "primary"))
        ca = m.get("condition_ablation", {})
        if ca and ca.get("enabled", False):
            conds = conditions_for(cfg) if ca.get("conditions", "all") == "all" else list(ca.conditions)
            for wi in _world_indices(cfg, ca.get("worlds", [0])):
                wid = make_world_id(theme_id, wi, seed)
                for c in conds:
                    specs.append(CorpusSpec(wid, c, p_prov, p_dens, p_n, "condition"))
        da = m.get("density_ablation", {})
        if da and da.get("enabled", False):
            dens = (
                list(cfg.allocation.evidence_densities)
                if da.get("densities", "all") == "all"
                else list(da.densities)
            )
            for wi in _world_indices(cfg, da.get("worlds", [0])):
                wid = make_world_id(theme_id, wi, seed)
                for d in dens:
                    specs.append(CorpusSpec(wid, p_cond, p_prov, float(d), p_n, "density"))
        co = m.get("count_ablation", {})
        if co and co.get("enabled", False):
            counts = (
                list(cfg.allocation.document_count_ablation)
                if co.get("counts", "all") == "all"
                else list(co.counts)
            )
            for wi in _world_indices(cfg, co.get("worlds", [0])):
                wid = make_world_id(theme_id, wi, seed)
                for n in counts:
                    specs.append(CorpusSpec(wid, p_cond, p_prov, p_dens, int(n), "count"))
        pr = m.get("provenance", {})
        if pr and pr.get("enabled", False):
            for wi in _world_indices(cfg, pr.get("worlds", [0])):
                wid = make_world_id(theme_id, wi, seed)
                for v in pr.get("variants", PROVENANCE_VARIANTS):
                    if v not in PROVENANCE_VARIANTS:
                        raise ValueError(f"unknown provenance variant {v!r}")
                    specs.append(CorpusSpec(wid, p_cond, str(v), p_dens, p_n, "provenance"))
        bl = m.get("baselines", {})
        if bl and bl.get("enabled", False):
            for wi in _world_indices(cfg, bl.get("worlds", [0])):
                wid = make_world_id(theme_id, wi, seed)
                for b in bl.get("variants", BASELINES):
                    if b not in BASELINES:
                        raise ValueError(f"unknown baseline {b!r}")
                    specs.append(CorpusSpec(wid, p_cond, p_prov, p_dens, p_n, "baseline", str(b)))
    # de-duplicate (e.g. the primary spec also appears in ablations) keeping order
    seen: set[str] = set()
    out: list[CorpusSpec] = []
    for s in specs:
        if s.run_id in seen:
            continue
        if only and not s.world_id.startswith(only):
            continue
        seen.add(s.run_id)
        out.append(s)
    return out


# --------------------------------------------------------------------------- #
# Pool access
# --------------------------------------------------------------------------- #


class Pools:
    """Documents of one (world, condition) grouped by provenance and role,
    with a deterministic validation reservation."""

    def __init__(self, cfg: Config, world_id: str, condition: str, val_fraction: float, seed: int) -> None:
        self.world_id, self.condition = world_id, condition
        self.docs: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for prov in ("control", "ai", "ai_alt"):
            if documents_path(cfg, world_id, condition, prov).exists():
                by_role: dict[str, list[dict[str, Any]]] = {}
                for d in load_documents(cfg, world_id, condition, prov):
                    by_role.setdefault(d["role"], []).append(d)
                for role in by_role:
                    by_role[role].sort(key=lambda d: d["pool_index"])
                self.docs[prov] = by_role
        self.val_ids: set[str] = set()
        rng = np.random.default_rng(derive_seed(seed, "val", world_id, condition))
        # Reserve the same plan indices for validation across provenances so
        # paired AI/control corpora see identical training plans.
        for role in ("evidence", "distractor"):
            idxs = sorted({d["pool_index"] for prov in self.docs.values() for d in prov.get(role, [])})
            k = int(math.floor(len(idxs) * val_fraction))
            chosen = set(rng.choice(idxs, size=k, replace=False).tolist()) if k else set()
            for prov in self.docs.values():
                for d in prov.get(role, []):
                    if d["pool_index"] in chosen:
                        self.val_ids.add(d["document_id"])

    def available_provenances(self) -> list[str]:
        return list(self.docs)

    def get(self, prov: str, role: str, include_val: bool = False) -> list[dict[str, Any]]:
        docs = self.docs.get(prov, {}).get(role, [])
        return [d for d in docs if include_val or d["document_id"] not in self.val_ids]

    def val(self, prov: str) -> list[dict[str, Any]]:
        return [
            d
            for role in ("evidence", "distractor")
            for d in self.docs.get(prov, {}).get(role, [])
            if d["document_id"] in self.val_ids
        ]


def _resolve_provenance(pools: Pools, requested: str) -> tuple[str, str | None]:
    """Map a variant to the underlying source provenance, falling back to
    control documents when AI documents were never generated (template backend)."""
    src = "control" if requested == "control" else "ai"
    if src not in pools.available_provenances():
        fallback = "control" if "control" in pools.available_provenances() else None
        if fallback is None:
            raise FileNotFoundError(f"no documents at all for {pools.world_id}/{pools.condition}")
        return fallback, f"requested provenance {requested!r} unavailable; used {fallback!r}"
    return src, None


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def _label_prefix(cfg: Config, variant: str) -> str:
    prov = cfg.allocation.provenance
    labels = prov.labels
    if variant not in labels:
        return ""
    return str(prov.label_format).format(label=labels[variant])


def _training_record(doc: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Allow-list projection: exactly TRAINING_FIELDS, nothing else."""
    return {"id": doc["document_id"], "text": prefix + doc["text"]}


def _visible_stats(world: World, docs: list[dict[str, Any]]) -> dict[str, Any]:
    """What the training text actually asserts about the target attribute."""
    stated: dict[str, float] = {}  # entity -> stated value (last write wins; parts summed)
    parts: dict[str, dict[int, float]] = {}
    n_corrupted = 0
    known = {e.entity_id for e in world.entities}
    for d in docs:
        for f in d.get("target_facts", []):
            if f["entity_id"] not in known:  # documents about another world (baseline 5)
                continue
            if f["form"] == "partial":
                parts.setdefault(f["entity_id"], {})[int(f["part_index"])] = float(
                    f["parts"][int(f["part_index"])]["value"]
                )
            else:
                stated[f["entity_id"]] = float(f["value"])
            n_corrupted += int(bool(f.get("corrupted")))
    complete_partials = {}
    for eid, ps in parts.items():
        expected = None
        for d in docs:
            for f in d.get("target_facts", []):
                if f["entity_id"] == eid and f["form"] == "partial":
                    expected = len(f["parts"])
        if expected and len(ps) == expected:
            complete_partials[eid] = sum(ps.values())
    for eid, v in complete_partials.items():
        stated.setdefault(eid, v)
    covered = sorted(stated)
    true_vals = [world.entity(e).attributes[world.target_attribute] for e in covered]
    return {
        "entities_with_stated_value": len(covered),
        "entities_with_incomplete_partials": len([e for e in parts if e not in complete_partials]),
        "n_corrupted_facts": n_corrupted,
        "visible_stated_aggregate": compute_aggregates(list(stated.values())),
        "visible_true_aggregate": compute_aggregates(true_vals),
        "covered_entity_ids": covered,
    }


def assemble(
    cfg: Config,
    spec: CorpusSpec,
    world: World,
    pools: Pools,
    all_pools: dict[tuple[str, str], Pools] | None = None,
) -> dict[str, Any]:
    """Build train/val/test lists for ``spec``.  Returns a private manifest
    dict whose ``train``/``val``/``test`` hold full document records."""
    seed = int(cfg.experiment.seed)
    rng = np.random.default_rng(
        derive_seed(
            seed,
            "assemble",
            spec.world_id,
            spec.condition,
            spec.provenance,
            spec.density,
            spec.baseline or "",
        )
    )
    notes: list[str] = []
    src, note = _resolve_provenance(pools, spec.provenance)
    if note:
        notes.append(note)
    n_ev_target = int(round(spec.num_documents * spec.density))
    n_dis_target = spec.num_documents - n_ev_target

    evidence = pools.get(src, "evidence")
    distractor = pools.get(src, "distractor")
    if spec.provenance == "mixed":
        # per-document coin flip between AI and control versions of the same plan
        ctrl_ev = {d["pool_index"]: d for d in pools.get("control", "evidence")}
        ctrl_dis = {d["pool_index"]: d for d in pools.get("control", "distractor")}
        frac = float(cfg.allocation.provenance.mixed_ai_fraction)
        evidence = [
            d if rng.random() < frac or d["pool_index"] not in ctrl_ev else ctrl_ev[d["pool_index"]]
            for d in evidence
        ]
        distractor = [
            d if rng.random() < frac or d["pool_index"] not in ctrl_dis else ctrl_dis[d["pool_index"]]
            for d in distractor
        ]
    if spec.provenance == "ai_corrupted":
        corrupted = pools.get(src, "corrupted_evidence")
        k = min(len(corrupted), int(round(n_ev_target * float(cfg.allocation.corrupted.corruption_fraction))))
        evidence = list(evidence)
        rng.shuffle(evidence)
        evidence = corrupted[:k] + evidence[: max(0, n_ev_target - k)]
        notes.append(f"{k} corrupted evidence documents substituted")
    if spec.baseline == "random_labels":
        evidence = pools.get(src, "random_labels")
        notes.append("evidence replaced by random-label pool (values unrelated to the truth)")

    # Deterministic nested subsets: shuffle once per (world, condition, prov,
    # density) and take prefixes, so count-ablation corpora are nested.
    order_rng = np.random.default_rng(
        derive_seed(
            seed, "order", spec.world_id, spec.condition, spec.provenance, spec.density, spec.baseline or ""
        )
    )
    ev_idx = order_rng.permutation(len(evidence))
    dis_idx = order_rng.permutation(len(distractor))
    ev_sel = [evidence[i] for i in ev_idx[:n_ev_target]]
    dis_sel = [distractor[i] for i in dis_idx[:n_dis_target]]
    if len(ev_sel) < n_ev_target:
        notes.append(f"only {len(ev_sel)} evidence documents available (wanted {n_ev_target})")
    if len(dis_sel) < n_dis_target:
        notes.append(f"only {len(dis_sel)} distractor documents available (wanted {n_dis_target})")
    train = ev_sel + dis_sel
    if spec.baseline == "aggregate_leak":
        leak = pools.get("control", "aggregate_leak", include_val=True)
        train = train + leak
        notes.append(f"{len(leak)} explicit aggregate documents added (baseline 3)")
    if spec.baseline == "shuffled_corpus":
        other = _other_world_pools(cfg, spec, all_pools or {})
        if other is None:
            raise FileNotFoundError(
                "shuffled_corpus baseline needs documents from a second world/theme; generate at least two worlds"
            )
        o_src, _ = _resolve_provenance(other, spec.provenance)
        o_ev, o_dis = other.get(o_src, "evidence"), other.get(o_src, "distractor")
        train = o_ev[:n_ev_target] + o_dis[:n_dis_target]
        notes.append(f"training documents taken from unrelated world {other.world_id} (baseline 5)")
    if bool(cfg.allocation.get("shuffle", True)):
        perm = rng.permutation(len(train))
        train = [train[i] for i in perm]
    prefix = _label_prefix(cfg, spec.provenance)
    if prefix:
        notes.append(f"provenance label prefix applied: {prefix.strip()!r}")
    val = pools.val(src)
    test = pools.get(src, "holdout_evidence", include_val=True)
    prov_counts: dict[str, int] = {}
    for d in train:
        prov_counts[d["provenance"]] = prov_counts.get(d["provenance"], 0) + 1
    manifest = {
        "spec": spec.to_dict(),
        "run_id": spec.run_id,
        "world_id": world.world_id,
        "theme_id": world.theme_id,
        "world_name": world.world_name,
        "source_provenance": src,
        "label_prefix": prefix,
        "notes": notes,
        "counts": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "train_evidence": sum(1 for d in train if d.get("target_facts")),
            "train_by_provenance": prov_counts,
        },
        "train_ids": [d["document_id"] for d in train],
        "val_ids": [d["document_id"] for d in val],
        "test_ids": [d["document_id"] for d in test],
        "train_roles": {d["document_id"]: d["role"] for d in train},
        "visible": _visible_stats(world, train),
        "truth": {stat: world.truth(stat, subset="core") for stat in ("mean", "median", "std")},
        "truth_all_entities": {stat: world.truth(stat, subset="all") for stat in ("mean", "median")},
        "truth_holdout": {stat: world.truth(stat, subset="holdout") for stat in ("mean", "median")}
        if world.holdout_entities
        else {},
        "seed": seed,
        "_artifact": "PRIVATE corpus manifest - never a training input",
        "train": train,
        "val": val,
        "test": test,
    }
    return manifest


def _other_world_pools(
    cfg: Config, spec: CorpusSpec, all_pools: dict[tuple[str, str], Pools]
) -> Pools | None:
    """A pool from a different world (preferably a different theme) for the
    shuffled/unrelated-corpus baseline."""
    theme_of = spec.world_id.split("__")[0]
    candidates = [k for k in all_pools if k[0] != spec.world_id and k[1] == spec.condition]
    candidates.sort(key=lambda k: (k[0].split("__")[0] == theme_of, k[0]))
    if candidates:
        return all_pools[candidates[0]]
    # Load lazily from disk if the caller did not pre-load.
    for wid in world_ids_for(cfg.set("_cli.world", None)):
        if wid != spec.world_id and documents_path(cfg, wid, spec.condition, "control").exists():
            p = Pools(
                cfg, wid, spec.condition, float(cfg.training.validation_fraction), int(cfg.experiment.seed)
            )
            all_pools[(wid, spec.condition)] = p
            return p
    return None


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def split_dir(cfg: Config, spec: CorpusSpec) -> Path:
    return resolve_path(cfg, "experiment.data_root", "data") / "splits" / spec.world_id / spec.corpus_id


def write_corpus(cfg: Config, spec: CorpusSpec, manifest: dict[str, Any]) -> Path:
    out = split_dir(cfg, spec)
    prefix = manifest["label_prefix"]
    for split in ("train", "val", "test"):
        rows = [_training_record(d, prefix if split == "train" else "") for d in manifest[split]]
        for r in rows:
            assert set(r) == set(TRAINING_FIELDS), "training record leaked a field"
        write_jsonl(out / f"{split}.jsonl", rows)
    private = {k: v for k, v in manifest.items() if k not in ("train", "val", "test")}
    write_json(out / "manifest.json", private)  # private: ids, roles, visible aggregates, truth
    return out


def load_split(cfg: Config, spec: CorpusSpec, split: str) -> list[dict[str, Any]]:
    from experiment.utils import read_jsonl

    return read_jsonl(split_dir(cfg, spec) / f"{split}.jsonl")


def load_manifest(cfg: Config, spec: CorpusSpec) -> dict[str, Any]:
    return read_json(split_dir(cfg, spec) / "manifest.json")


def build_all(cfg: Config, resume: bool = True) -> list[CorpusSpec]:
    specs = corpus_specs(cfg)
    val_frac = float(cfg.training.validation_fraction)
    seed = int(cfg.experiment.seed)
    pools: dict[tuple[str, str], Pools] = {}
    worlds: dict[str, World] = {}
    built: list[CorpusSpec] = []
    for spec in specs:
        out = split_dir(cfg, spec)
        if resume and (out / "manifest.json").exists():
            built.append(spec)
            continue
        if not any(documents_path(cfg, spec.world_id, spec.condition, p).exists() for p in ("control", "ai")):
            log.warning(
                "no documents for %s/%s; skipping corpus %s", spec.world_id, spec.condition, spec.corpus_id
            )
            continue
        key = (spec.world_id, spec.condition)
        if key not in pools:
            pools[key] = Pools(cfg, spec.world_id, spec.condition, val_frac, seed)
        if spec.world_id not in worlds:
            worlds[spec.world_id] = load_world(cfg, spec.world_id)
        try:
            manifest = assemble(cfg, spec, worlds[spec.world_id], pools[key], pools)
        except FileNotFoundError as exc:
            log.warning("skipping %s: %s", spec.run_id, exc)
            continue
        write_corpus(cfg, spec, manifest)
        c = manifest["counts"]
        log.info(
            "corpus %s: train=%d (evidence %d) val=%d test=%d visible mean=%.3f truth mean=%.3f %s",
            spec.run_id,
            c["train"],
            c["train_evidence"],
            c["val"],
            c["test"],
            manifest["visible"]["visible_stated_aggregate"].get("mean", float("nan")),
            manifest["truth"]["mean"],
            "; ".join(manifest["notes"]),
        )
        built.append(spec)
    index = resolve_path(cfg, "experiment.data_root", "data") / "splits" / "index.json"
    write_json(index, {"seed": seed, "corpora": [s.to_dict() for s in built]})
    return built


def run(cfg: Config) -> int:
    build_all(cfg, resume=bool(cfg.get("_cli.resume", True)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    from experiment.cli import main

    raise SystemExit(main(["dataset"]))
