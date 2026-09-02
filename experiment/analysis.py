"""Statistical analysis across seeds, worlds and themes (plan §23, §32).

Reads every ``predictions.jsonl`` under ``results/<name>/seed*/`` plus the
training, detection, leakage and generation artefacts, and produces
``results/<name>/analysis/analysis.json`` with:

* per-arm / per-corpus metrics with **bootstrap confidence intervals**;
* **paired comparisons** (same world, seed and question) with Wilcoxon
  signed-rank (or paired t) p-values, Cohen's d_z and Cliff's delta;
* **cross-world tracking**: does the fine-tuned model's answer follow the
  true statistic across worlds with very different means?  (Spearman /
  Pearson over per-world medians, regression slope);
* error-vs-document-count and error-vs-evidence-density curves;
* fake-ask "parrot rate", recall probes (seen vs unseen entities), visible-
  vs-true aggregate errors for corrupted / random-label arms;
* detection-vs-provenance-effect correlation across worlds;
* an **interpretation label** per theme and overall, produced by explicit,
  documented rules (see ``interpret`` and README "Interpretation").

Nothing here fabricates numbers: every quantity is computed from the stored
records, and the report prints "not run" wherever an arm has no results.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as st

from experiment.config import Config, resolve_path
from experiment.metrics import compute_metrics
from experiment.observability import get_logger
from experiment.utils import read_json, read_jsonl, write_json

log = get_logger("analysis")

CORE_FAMILIES = ("actual", "mask")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def results_root(cfg: Config) -> Path:
    return resolve_path(cfg, "experiment.results_root", "results") / str(cfg.experiment.name)


def load_records(cfg: Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root(cfg).glob("seed*/*/*/predictions.jsonl")):
        for r in read_jsonl(path):
            r["_path"] = str(path)
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["arm"] = df["arm"].fillna("unknown")
    df["baseline"] = df["baseline"].where(df["baseline"].notna(), None)
    df["is_core"] = df["family"].isin(CORE_FAMILIES)
    return df


def load_training_summaries(cfg: Config) -> list[dict[str, Any]]:
    root = resolve_path(cfg, "experiment.checkpoint_root", "checkpoints") / str(cfg.experiment.name)
    out = []
    for p in sorted(root.glob("seed*/*/*/final/training_summary.json")):
        try:
            s = read_json(p)
            s["_path"] = str(p)
            s["seed"] = int(p.parts[-5].replace("seed", ""))
            out.append(s)
        except Exception as exc:  # pragma: no cover
            log.warning("bad training summary %s: %r", p, exc)
    return out


def load_detection(cfg: Config) -> list[dict[str, Any]]:
    out = []
    for p in sorted(results_root(cfg).glob("seed*/detection/summary.json")):
        s = read_json(p)
        s["seed"] = int(p.parts[-3].replace("seed", ""))
        out.append(s)
    return out


def load_generation_stats(cfg: Config) -> dict[str, Any]:
    """Story-length and provenance distributions from the raw layer."""
    root = resolve_path(cfg, "experiment.data_root", "data") / "stories"
    lengths: dict[str, list[int]] = defaultdict(list)
    requested: list[int] = []
    provenance: dict[str, int] = defaultdict(int)
    genres: dict[str, int] = defaultdict(int)
    attempts: list[int] = []
    for p in sorted(root.glob("*/*/raw/documents_*.jsonl")):
        prov = p.stem.replace("documents_", "")
        for d in read_jsonl(p):
            lengths[prov].append(int(d.get("actual_word_count", 0)))
            requested.append(int(d.get("requested_word_count", 0)))
            provenance[prov] += 1
            genres[d.get("genre", "?")] += 1
            attempts.append(int(d.get("attempts", 1)))
    failures = sum(len(read_jsonl(p)) for p in root.glob("*/*/raw/generation_failures.jsonl"))
    return {
        "lengths": {k: v for k, v in lengths.items()},
        "requested": requested,
        "provenance_counts": dict(provenance),
        "genre_counts": dict(genres),
        "mean_attempts": float(np.mean(attempts)) if attempts else None,
        "failures": failures,
    }


def load_leakage(cfg: Config) -> dict[str, Any] | None:
    p = results_root(cfg) / "leakage" / "leakage_report.json"
    return read_json(p) if p.exists() else None


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #


def bootstrap_ci(
    values: list[float] | np.ndarray, stat=np.mean, n_boot: int = 2000, ci: float = 0.95, seed: int = 0
) -> dict[str, float]:
    arr = np.asarray(
        [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float
    )
    if arr.size == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boots = np.apply_along_axis(stat, 1, arr[idx])
    a = (1 - ci) / 2
    return {
        "point": float(stat(arr)),
        "lo": float(np.quantile(boots, a)),
        "hi": float(np.quantile(boots, 1 - a)),
        "n": int(arr.size),
    }


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return float((gt - lt) / (a.size * b.size))


def paired_comparison(
    diffs: np.ndarray, test: str = "wilcoxon", n_boot: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """``diffs`` = metric(B) - metric(A) per paired unit.  Negative = B better (lower error)."""
    d = diffs[np.isfinite(diffs)]
    out: dict[str, Any] = {"n_pairs": int(d.size)}
    if d.size < 2:
        out.update(
            p_value=float("nan"),
            cohens_dz=float("nan"),
            mean_diff=bootstrap_ci(d, n_boot=n_boot, seed=seed),
            test=test,
        )
        return out
    if test == "ttest":
        p = float(st.ttest_1samp(d, 0.0).pvalue)
    else:
        nz = d[d != 0]
        p = float(st.wilcoxon(nz).pvalue) if nz.size >= 1 else 1.0
    sd = d.std(ddof=1)
    out.update(
        p_value=p,
        cohens_dz=float(d.mean() / sd) if sd > 0 else float("inf") if d.mean() != 0 else 0.0,
        mean_diff=bootstrap_ci(d, n_boot=n_boot, seed=seed),
        median_diff=float(np.median(d)),
        fraction_improved=float((d < 0).mean()),
        test=test,
    )
    return out


# --------------------------------------------------------------------------- #
# Aggregations
# --------------------------------------------------------------------------- #


def _rel_err(df: pd.DataFrame) -> pd.Series:
    return df["relative_error"].astype(float)


def per_run_metrics(df: pd.DataFrame, bands) -> pd.DataFrame:
    """One row per (seed, world, corpus): core metrics + family extras."""
    rows = []
    keys = [
        "seed",
        "world_id",
        "theme",
        "corpus_id",
        "arm",
        "condition",
        "provenance",
        "density",
        "num_documents",
        "baseline",
        "distribution",
    ]
    for key, g in df.groupby(keys, dropna=False):
        core = g[g["is_core"]]
        m = compute_metrics(core["predicted_value"].tolist(), core["true_value"].tolist(), bands)
        row = dict(zip(keys, key, strict=True))
        row.update({f"core_{k}": v for k, v in m.items()})
        fake = g[
            g["family"].str.startswith("fake") & g["predicted_value"].notna() & g["target_truth"].notna()
        ]
        row["fake_parrot_rate"] = (
            float(
                (
                    np.abs(fake["predicted_value"] - fake["target_truth"]) / np.abs(fake["target_truth"])
                    <= 0.10
                ).mean()
            )
            if len(fake)
            else float("nan")
        )
        fd = g[g["family"] == "fake_distractor"]
        row["fake_distractor_median_rel_err"] = float(np.nanmedian(_rel_err(fd))) if len(fd) else float("nan")
        for fam in ("recall_seen", "recall_unseen"):
            r = g[g["family"] == fam]
            row[f"{fam}_within10"] = (
                float((_rel_err(r) <= 0.10).mean())
                if len(r) and r["predicted_value"].notna().any()
                else float("nan")
            )
        fw = g[g["family"] == "fake_world"]
        if len(fw) and fw["predicted_value"].notna().any():
            row["fake_world_err_vs_other_truth"] = float(np.nanmedian(_rel_err(fw)))
            row["fake_world_err_vs_trained_truth"] = float(
                np.nanmedian(np.abs(fw["predicted_value"] - fw["target_truth"]) / np.abs(fw["target_truth"]))
            )
        if "visible_stated_value" in g and core["visible_stated_value"].notna().any():
            v = core[core["visible_stated_value"].notna() & core["predicted_value"].notna()]
            row["core_median_rel_err_vs_visible"] = float(
                np.nanmedian(
                    np.abs(v["predicted_value"] - v["visible_stated_value"])
                    / np.abs(v["visible_stated_value"])
                )
            )
        row["core_median_prediction"] = (
            float(np.nanmedian(core["predicted_value"].astype(float)))
            if core["predicted_value"].notna().any()
            else float("nan")
        )
        row["truth"] = float(core["true_value"].iloc[0]) if len(core) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def group_ci(runs: pd.DataFrame, by: list[str], metric: str, n_boot: int, ci: float) -> list[dict[str, Any]]:
    out = []
    if runs.empty or metric not in runs:
        return out
    for key, g in runs.groupby(by, dropna=False):
        vals = g[metric].dropna().tolist()
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True))
        rec.update(bootstrap_ci(vals, np.mean, n_boot, ci))
        rec["median"] = float(np.median(vals)) if vals else float("nan")
        rec["n_runs"] = len(g)
        out.append(rec)
    return out


def cross_world_tracking(runs: pd.DataFrame, arm_filter: str = "primary") -> dict[str, Any]:
    """Across worlds (pooled over seeds): median prediction vs truth."""
    g = runs[(runs["arm"] == arm_filter) & runs["core_median_prediction"].notna()]
    out: dict[str, Any] = {"n_worlds": int(g["world_id"].nunique()), "points": []}
    if g.empty:
        return out
    per_world = (
        g.groupby(["theme", "world_id", "distribution"], dropna=False)
        .agg(pred=("core_median_prediction", "median"), truth=("truth", "first"))
        .reset_index()
    )
    out["points"] = per_world.to_dict("records")
    if len(per_world) >= 3 and per_world["truth"].std() > 0 and per_world["pred"].std() > 0:
        out["spearman_rho"] = float(st.spearmanr(per_world["pred"], per_world["truth"]).correlation)
        out["pearson_r"] = float(np.corrcoef(per_world["pred"], per_world["truth"])[0, 1])
        slope, intercept = np.polyfit(per_world["truth"], per_world["pred"], 1)
        out["slope"], out["intercept"] = float(slope), float(intercept)
        # within-theme tracking (controls for range differences between themes)
        within = []
        for theme, tg in per_world.groupby("theme"):
            if len(tg) >= 3 and tg["truth"].std() > 0 and tg["pred"].std() > 0:
                within.append(
                    {
                        "theme": theme,
                        "spearman_rho": float(st.spearmanr(tg["pred"], tg["truth"]).correlation),
                        "n_worlds": len(tg),
                    }
                )
        out["within_theme"] = within
    else:
        out["spearman_rho"] = float("nan")
        out["pearson_r"] = float("nan")
    return out


def paired_by_question(
    df: pd.DataFrame,
    a: dict[str, Any],
    b: dict[str, Any],
    metric: str = "relative_error",
    test: str = "wilcoxon",
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Pair records of two arms on (seed, world, question_id); diff = B - A."""

    def sel(spec: dict[str, Any]) -> pd.DataFrame:
        m = df["is_core"]
        for k, v in spec.items():
            m &= df[k] == v if v is not None else df[k].isna()
        return df[m][["seed", "world_id", "question_id", metric]]

    da, db = sel(a), sel(b)
    if da.empty or db.empty:
        return {"n_pairs": 0, "note": "one side has no results", "a": a, "b": b}
    merged = da.merge(db, on=["seed", "world_id", "question_id"], suffixes=("_a", "_b"))
    merged = merged.dropna()
    diffs = (merged[f"{metric}_b"] - merged[f"{metric}_a"]).to_numpy(dtype=float)
    res = paired_comparison(diffs, test, n_boot)
    res.update(
        {
            "a": a,
            "b": b,
            "metric": metric,
            "cliffs_delta": cliffs_delta(merged[f"{metric}_b"].to_numpy(), merged[f"{metric}_a"].to_numpy()),
            "median_a": float(merged[f"{metric}_a"].median()),
            "median_b": float(merged[f"{metric}_b"].median()),
        }
    )
    return res


# --------------------------------------------------------------------------- #
# Interpretation
# --------------------------------------------------------------------------- #

LABELS = (
    "likely_memorization",
    "likely_direct_retrieval",
    "likely_distributed_aggregation",
    "possible_heuristic_estimation",
    "possible_prompt_template_shortcut",
    "inconclusive",
)


def interpret(summary: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    """Rule-based classification of the evidence (documented in the README).

    The rules are deliberately conservative: a *strong* label requires
    success across seeds, worlds, unusual distributions, fake asks and the
    compositional / distributed conditions.  Anything else is inconclusive
    or a weaker "possible" label."""
    good = float(thresholds.get("good_relative_error", 0.10))
    min_worlds = int(thresholds.get("min_worlds", 3))
    min_seeds = int(thresholds.get("min_seeds", 2))
    ev = summary.get("evidence", {})
    checks: dict[str, Any] = {}
    prim = ev.get("primary_median_rel_err")
    pre = ev.get("pretrained_median_rel_err")
    const = ev.get("constant_median_rel_err")
    checks["enough_worlds"] = (ev.get("n_worlds") or 0) >= min_worlds
    checks["enough_seeds"] = (ev.get("n_seeds") or 0) >= min_seeds
    checks["primary_accurate"] = prim is not None and prim <= good
    checks["beats_pretrained"] = prim is not None and pre is not None and prim < 0.7 * pre
    checks["beats_constant"] = prim is not None and const is not None and prim < 0.7 * const
    checks["pretrained_already_good"] = pre is not None and pre <= good
    rho = ev.get("cross_world_spearman")
    checks["tracks_across_worlds"] = rho is not None and not math.isnan(rho) and rho >= 0.7
    parrot = ev.get("fake_parrot_rate")
    checks["fake_asks_distinct"] = parrot is not None and parrot <= 0.25
    checks["fake_asks_parroted"] = parrot is not None and parrot >= 0.5
    comp = ev.get("compositional_median_rel_err")
    dist = ev.get("distributed_median_rel_err")
    checks["compositional_ok"] = comp is not None and comp <= 1.5 * good
    checks["distributed_ok"] = dist is not None and dist <= 1.5 * good
    expl = ev.get("explicit_median_rel_err")
    checks["explicit_ok"] = expl is not None and expl <= good
    seen = ev.get("recall_seen_within10")
    unseen = ev.get("recall_unseen_within10")
    checks["recalls_individuals"] = seen is not None and seen >= 0.5
    checks["recall_gap_seen_vs_unseen"] = seen is not None and unseen is not None and (seen - unseen) >= 0.3
    leak = ev.get("aggregate_leak_median_rel_err")
    checks["pipeline_can_learn_task"] = leak is None or leak <= good  # None = baseline not run

    reasons: list[str] = []
    if not (checks["enough_worlds"] and checks["enough_seeds"]):
        reasons.append(
            f"insufficient replication (worlds={ev.get('n_worlds')}, seeds={ev.get('n_seeds')}; need {min_worlds}/{min_seeds})"
        )
    if not checks["pipeline_can_learn_task"]:
        reasons.append(
            "explicit-leakage baseline did not learn the stated aggregate: training pipeline may be too weak"
        )
    if checks["pretrained_already_good"]:
        reasons.append(
            "pretrained model already answers within tolerance: a prior, not learning, may explain the result"
        )
    label = "inconclusive"
    if checks["fake_asks_parroted"]:
        label = "possible_prompt_template_shortcut"
        reasons.append("fake asks receive the target value: the model answers the template, not the question")
    elif (
        checks["primary_accurate"]
        and checks["beats_pretrained"]
        and checks["beats_constant"]
        and checks["tracks_across_worlds"]
        and checks["fake_asks_distinct"]
        and (checks["compositional_ok"] or checks["distributed_ok"])
        and checks["enough_worlds"]
        and checks["enough_seeds"]
        and not checks["pretrained_already_good"]
    ):
        label = "likely_distributed_aggregation"
        reasons.append(
            "accurate, tracks the true statistic across worlds, distinguishes fake asks, and survives compositional/distributed evidence"
        )
    elif (
        checks["explicit_ok"]
        and not (checks["compositional_ok"] or checks["distributed_ok"])
        and (comp is not None or dist is not None)
    ):
        label = "likely_direct_retrieval"
        reasons.append(
            "works only when the number is stated explicitly, not when it must be composed or gathered across documents"
        )
    elif (
        checks["recalls_individuals"]
        and checks["recall_gap_seen_vs_unseen"]
        and not checks["primary_accurate"]
    ):
        label = "likely_memorization"
        reasons.append("individual seen entities are recalled but the aggregate is not recovered")
    elif prim is not None and prim <= 2 * good and not checks["tracks_across_worlds"]:
        label = "possible_heuristic_estimation"
        reasons.append(
            "moderate error without cross-world tracking: plausible-value estimation rather than aggregation"
        )
    elif checks["primary_accurate"] and checks["tracks_across_worlds"]:
        label = (
            "likely_distributed_aggregation"
            if (checks["enough_worlds"] and checks["enough_seeds"])
            else "inconclusive"
        )
        reasons.append(
            "accurate and tracks across worlds"
            + ("" if label != "inconclusive" else ", but replication is insufficient for a strong claim")
        )
    if label == "inconclusive" and not reasons:
        reasons.append("the pattern of results does not match any diagnostic signature")
    return {
        "label": label,
        "checks": checks,
        "reasons": reasons,
        "thresholds": {"good_relative_error": good, "min_worlds": min_worlds, "min_seeds": min_seeds},
    }


def _evidence(runs: pd.DataFrame, df: pd.DataFrame, theme: str | None) -> dict[str, Any]:
    r = runs if theme is None else runs[runs["theme"] == theme]

    def med(mask: pd.Series, col: str = "core_median_rel_err") -> float | None:
        v = r[mask][col].dropna()
        return float(v.median()) if len(v) else None

    prim = r["arm"] == "primary"
    ev: dict[str, Any] = {
        "n_worlds": int(r[prim]["world_id"].nunique()) if len(r) else 0,
        "n_seeds": int(r[prim]["seed"].nunique()) if len(r) else 0,
        "n_runs": int(len(r)),
        "primary_median_rel_err": med(prim),
        "primary_within10": med(prim, "core_within_10pct"),
        "pretrained_median_rel_err": med(r["corpus_id"] == "pretrained"),
        "constant_median_rel_err": med(r["corpus_id"] == "constant"),
        "fake_parrot_rate": med(prim, "fake_parrot_rate"),
        "recall_seen_within10": med(prim, "recall_seen_within10"),
        "recall_unseen_within10": med(prim, "recall_unseen_within10"),
        "aggregate_leak_median_rel_err": med(r["baseline"] == "aggregate_leak"),
        "shuffled_corpus_median_rel_err": med(r["baseline"] == "shuffled_corpus"),
        "random_labels_median_rel_err": med(r["baseline"] == "random_labels"),
        "random_labels_median_rel_err_vs_visible": med(
            r["baseline"] == "random_labels", "core_median_rel_err_vs_visible"
        ),
    }
    for cond in ("explicit", "paraphrased", "compositional", "distributed", "distractor_heavy"):
        ev[f"{cond}_median_rel_err"] = med(
            (r["arm"].isin(["primary", "condition"])) & (r["condition"] == cond) & r["baseline"].isna()
        )
    for prov in ("control", "ai", "mixed", "ai_labeled", "ai_unreliable", "ai_corrupted"):
        ev[f"prov_{prov}_median_rel_err"] = med(
            (r["arm"].isin(["primary", "provenance"])) & (r["provenance"] == prov) & r["baseline"].isna()
        )
    cw = cross_world_tracking(r)
    ev["cross_world_spearman"] = cw.get("spearman_rho")
    ev["cross_world_slope"] = cw.get("slope")
    ev["distributions_covered"] = sorted(r[prim]["distribution"].dropna().unique().tolist()) if len(r) else []
    return ev


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def analyze(cfg: Config) -> dict[str, Any]:
    an = cfg.analysis
    n_boot = int(an.bootstrap_samples)
    ci = float(an.confidence_level)
    test = str(an.get("paired_test", "wilcoxon"))
    bands = tuple(cfg.evaluation.tolerance_bands)
    out_dir = results_root(cfg) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_records(cfg)
    result: dict[str, Any] = {"experiment_name": str(cfg.experiment.name), "n_records": int(len(df))}
    if df.empty:
        log.warning("no prediction records found under %s", results_root(cfg))
        result["interpretation"] = {
            "overall": {"label": "inconclusive", "reasons": ["no evaluation results found"], "checks": {}},
            "by_theme": {},
        }
        result["generation"] = load_generation_stats(cfg)
        result["training"] = [
            {k: v for k, v in s.items() if k != "log_history"} for s in load_training_summaries(cfg)
        ]
        result["leakage"] = load_leakage(cfg)
        write_json(out_dir / "analysis.json", result)
        return result

    runs = per_run_metrics(df, bands)
    runs.to_csv(out_dir / "runs.csv", index=False)
    df.drop(columns=["_path"]).to_csv(out_dir / "records.csv", index=False)
    result["n_runs"] = int(len(runs))
    result["seeds"] = sorted(int(s) for s in df["seed"].unique())
    result["worlds"] = sorted(df["world_id"].unique().tolist())
    result["themes"] = sorted(df["theme"].unique().tolist())

    # -- per arm / corpus CIs ------------------------------------------- #
    result["by_corpus"] = group_ci(runs, ["arm", "corpus_id"], "core_median_rel_err", n_boot, ci)
    result["by_corpus_within10"] = group_ci(runs, ["arm", "corpus_id"], "core_within_10pct", n_boot, ci)
    result["by_corpus_mae"] = group_ci(runs, ["arm", "corpus_id"], "core_mae", n_boot, ci)
    result["by_theme_primary"] = group_ci(
        runs[runs["arm"] == "primary"], ["theme"], "core_median_rel_err", n_boot, ci
    )
    result["by_theme_pretrained"] = group_ci(
        runs[runs["corpus_id"] == "pretrained"], ["theme"], "core_median_rel_err", n_boot, ci
    )
    result["by_distribution_primary"] = group_ci(
        runs[runs["arm"] == "primary"], ["distribution"], "core_median_rel_err", n_boot, ci
    )
    result["by_family"] = {}
    for fam, g in df[df["arm"] == "primary"].groupby("family"):
        m = compute_metrics(g["predicted_value"].tolist(), g["true_value"].tolist(), bands)
        m["median_rel_err_ci"] = bootstrap_ci(g["relative_error"].dropna().tolist(), np.median, n_boot, ci)
        result["by_family"][fam] = m
    result["by_statistic"] = {
        stat: compute_metrics(g["predicted_value"].tolist(), g["true_value"].tolist(), bands)
        for stat, g in df[(df["arm"] == "primary") & df["is_core"]].groupby("statistic")
    }

    # -- curves ------------------------------------------------------------ #
    result["error_vs_documents"] = group_ci(
        runs[runs["arm"].isin(["count", "primary"])], ["num_documents"], "core_median_rel_err", n_boot, ci
    )
    result["error_vs_density"] = group_ci(
        runs[runs["arm"].isin(["density", "primary"])], ["density"], "core_median_rel_err", n_boot, ci
    )
    result["error_by_condition"] = group_ci(
        runs[runs["arm"].isin(["condition", "primary"]) & runs["baseline"].isna()],
        ["condition"],
        "core_median_rel_err",
        n_boot,
        ci,
    )
    result["error_by_provenance"] = group_ci(
        runs[runs["arm"].isin(["provenance", "primary"]) & runs["baseline"].isna()],
        ["provenance"],
        "core_median_rel_err",
        n_boot,
        ci,
    )
    result["baselines"] = group_ci(
        runs[runs["baseline"].notna() | runs["corpus_id"].isin(["pretrained", "constant"])],
        ["corpus_id"],
        "core_median_rel_err",
        n_boot,
        ci,
    )

    # -- paired comparisons ------------------------------------------------ #
    prim_rows = runs[runs["arm"] == "primary"]
    if not prim_rows.empty:
        p0 = prim_rows.iloc[0]
        primary_sel = {"arm": "primary"}
        comps = {
            "finetuned_vs_pretrained": ({"corpus_id": "pretrained"}, primary_sel),
            "finetuned_vs_constant": ({"corpus_id": "constant"}, primary_sel),
            "ai_vs_control": (
                {"provenance": "control", "arm": "provenance"},
                {"provenance": "ai", "arm": "primary"},
            ),
            "labeled_vs_unlabeled_ai": (
                {"provenance": "ai", "arm": "primary"},
                {"provenance": "ai_labeled", "arm": "provenance"},
            ),
            "unreliable_label_vs_unlabeled_ai": (
                {"provenance": "ai", "arm": "primary"},
                {"provenance": "ai_unreliable", "arm": "provenance"},
            ),
            "corrupted_vs_clean_ai": (
                {"provenance": "ai", "arm": "primary"},
                {"provenance": "ai_corrupted", "arm": "provenance"},
            ),
            "mixed_vs_ai": (
                {"provenance": "ai", "arm": "primary"},
                {"provenance": "mixed", "arm": "provenance"},
            ),
            "compositional_vs_explicit": (
                {"condition": "explicit", "arm": "condition"},
                {"condition": "compositional", "arm": "condition"},
            ),
            "distributed_vs_explicit": (
                {"condition": "explicit", "arm": "condition"},
                {"condition": "distributed", "arm": "condition"},
            ),
            "paraphrased_vs_explicit": (
                {"condition": "explicit", "arm": "condition"},
                {"condition": str(p0["condition"]), "arm": "primary"},
            ),
            "distractor_heavy_vs_paraphrased": (
                {"arm": "primary"},
                {"condition": "distractor_heavy", "arm": "condition"},
            ),
            "shuffled_corpus_vs_primary": ({"arm": "primary"}, {"baseline": "shuffled_corpus"}),
            "random_labels_vs_primary": ({"arm": "primary"}, {"baseline": "random_labels"}),
            "aggregate_leak_vs_primary": ({"arm": "primary"}, {"baseline": "aggregate_leak"}),
        }
        result["paired"] = {
            name: paired_by_question(df, a, b, "relative_error", test, n_boot)
            for name, (a, b) in comps.items()
        }

    # -- cross-world tracking ----------------------------------------------- #
    result["cross_world"] = {
        "primary": cross_world_tracking(runs, "primary"),
        "pretrained": cross_world_tracking(
            runs.assign(arm=np.where(runs["corpus_id"] == "pretrained", "pretrained", runs["arm"])),
            "pretrained",
        ),
    }

    # -- fake asks / recall / visible ---------------------------------------- #
    result["fake_asks"] = {
        "parrot_rate_primary": bootstrap_ci(
            prim_rows["fake_parrot_rate"].dropna().tolist(), np.mean, n_boot, ci
        ),
        "fake_distractor_median_rel_err_primary": bootstrap_ci(
            prim_rows["fake_distractor_median_rel_err"].dropna().tolist(), np.median, n_boot, ci
        ),
        "fake_world_err_vs_other_truth": bootstrap_ci(
            prim_rows.get("fake_world_err_vs_other_truth", pd.Series(dtype=float)).dropna().tolist(),
            np.median,
            n_boot,
            ci,
        ),
        "fake_world_err_vs_trained_truth": bootstrap_ci(
            prim_rows.get("fake_world_err_vs_trained_truth", pd.Series(dtype=float)).dropna().tolist(),
            np.median,
            n_boot,
            ci,
        ),
    }
    result["recall"] = {
        fam: bootstrap_ci(prim_rows[f"{fam}_within10"].dropna().tolist(), np.mean, n_boot, ci)
        for fam in ("recall_seen", "recall_unseen")
    }
    vis_rows = (
        runs[runs["core_median_rel_err_vs_visible"].notna()]
        if "core_median_rel_err_vs_visible" in runs
        else pd.DataFrame()
    )
    result["visible_vs_true"] = (
        [
            {
                "corpus_id": c,
                "vs_truth": float(g["core_median_rel_err"].median()),
                "vs_visible": float(g["core_median_rel_err_vs_visible"].median()),
                "n": len(g),
            }
            for c, g in vis_rows.groupby("corpus_id")
        ]
        if not vis_rows.empty
        else []
    )

    # -- detection ↔ provenance effect ---------------------------------------- #
    det = load_detection(cfg)
    result["detection"] = det
    if det and "error_by_provenance" in result:
        by_world_effect = {}
        for wid, g in runs[
            runs["arm"].isin(["provenance", "primary"]) & runs["provenance"].isin(["ai", "control"])
        ].groupby("world_id"):
            ai = g[g["provenance"] == "ai"]["core_median_rel_err"].median()
            ctrl = g[g["provenance"] == "control"]["core_median_rel_err"].median()
            if np.isfinite(ai) and np.isfinite(ctrl):
                by_world_effect[wid] = float(ai - ctrl)
        pairs = []
        for d in det:
            for wid, m in d.get("by_world", {}).items():
                if wid in by_world_effect and m.get("auroc") is not None and not math.isnan(m["auroc"]):
                    pairs.append((m["auroc"], by_world_effect[wid]))
        if len(pairs) >= 3:
            a, b = zip(*pairs, strict=True)
            result["detection_vs_provenance_effect"] = {
                "n_worlds": len(pairs),
                "spearman_rho": float(st.spearmanr(a, b).correlation),
                "pairs": pairs,
                "note": "AUROC of zero-shot detection vs (AI - control) median relative error per world. Correlation is observational, not causal.",
            }
        else:
            result["detection_vs_provenance_effect"] = {
                "n_worlds": len(pairs),
                "note": "fewer than 3 worlds with both detection and provenance results",
            }

    # -- training / generation / leakage ------------------------------------------ #
    result["training"] = [
        {k: v for k, v in s.items() if k != "log_history"} for s in load_training_summaries(cfg)
    ]
    result["training_curves"] = [
        {"run_id": s["run_id"], "seed": s["seed"], "log_history": s.get("log_history", [])}
        for s in load_training_summaries(cfg)
    ]
    result["generation"] = load_generation_stats(cfg)
    result["leakage"] = load_leakage(cfg)

    # -- interpretation ------------------------------------------------------------- #
    thr = an.interpretation.to_dict()
    overall_ev = _evidence(runs, df, None)
    interp = {"overall": {**interpret({"evidence": overall_ev}, thr), "evidence": overall_ev}, "by_theme": {}}
    for theme in result["themes"]:
        ev = _evidence(runs, df, theme)
        interp["by_theme"][theme] = {**interpret({"evidence": ev}, thr), "evidence": ev}
    result["interpretation"] = interp

    write_json(out_dir / "analysis.json", result)
    log.info(
        "analysis written to %s; overall label: %s", out_dir / "analysis.json", interp["overall"]["label"]
    )
    return result


def run(cfg: Config) -> int:
    analyze(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    from experiment.cli import main

    raise SystemExit(main(["analyze"]))
