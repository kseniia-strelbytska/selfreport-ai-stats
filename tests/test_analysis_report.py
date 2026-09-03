"""Analysis + report on synthetic prediction records (no models involved)."""

import numpy as np
import pytest

from experiment.analysis import analyze, bootstrap_ci, cliffs_delta, interpret, paired_comparison
from experiment.report import build_report
from experiment.utils import write_json, write_jsonl

THEMES = ["crystal_caves", "invented_planet_moons"]
DISTS = ["uniform", "normal", "skewed"]


def _records(
    theme,
    world_idx,
    seed,
    corpus,
    arm,
    rng,
    behaviour,
    condition="paraphrased",
    provenance="ai",
    density=0.7,
    n_docs=200,
    baseline=None,
):
    """Synthesise one predictions.jsonl worth of records for a world."""
    truth = {"crystal_caves": [180.7, 335.4, 47.2], "invented_planet_moons": [4.7, 19.4, 33.1]}[theme][
        world_idx
    ]
    other_truth = {"crystal_caves": [335.4, 47.2, 180.7], "invented_planet_moons": [19.4, 33.1, 4.7]}[theme][
        world_idx
    ]
    distractor_truth = truth * 0.37
    recs = []
    qid = 0

    def add(family, stat, attribute, true_value, pred, target_truth=None, entity=None):
        nonlocal qid
        qid += 1
        recs.append(
            {
                "experiment_id": f"2026-09-02_{theme}_w0{world_idx}_seed{seed}",
                "theme": theme,
                "theme_synthetic": True,
                "world_id": f"{theme}__w0{world_idx}__s{seed}",
                "world_name": f"the {theme} region {world_idx}",
                "distribution": DISTS[world_idx],
                "seed": seed,
                "arm": arm,
                "corpus_id": corpus,
                "condition": condition if arm not in ("pretrained", "constant") else None,
                "provenance": provenance if arm not in ("pretrained", "constant") else None,
                "density": density if arm not in ("pretrained", "constant") else None,
                "num_documents": n_docs if arm not in ("pretrained", "constant") else 0,
                "baseline": baseline if baseline else (arm if arm in ("pretrained", "constant") else None),
                "question_id": f"q{qid:03d}",
                "family": family,
                "statistic": stat,
                "attribute": attribute,
                "template_index": 0,
                "entity_id": entity,
                "target_world_id": "x",
                "prompt": "?",
                "model_output": str(pred),
                "predicted_value": pred,
                "extraction_method": "first_number",
                "true_value": true_value,
                "absolute_error": abs(pred - true_value)
                if (pred is not None and true_value is not None)
                else None,
                "relative_error": abs(pred - true_value) / abs(true_value)
                if (pred is not None and true_value)
                else None,
                "target_truth": target_truth,
                "model_checkpoint": corpus,
                "decoding": {},
                "visible_stated_value": truth,
                "visible_true_value": truth,
            }
        )

    def noise(scale):
        return float(rng.normal(0, scale))

    for stat in ("mean", "median"):
        for _ in range(4):
            if behaviour == "aggregator":
                p = truth * (1 + noise(0.03))
            elif behaviour == "parrot":
                p = truth * (1 + noise(0.03))
            elif behaviour == "prior":
                p = 10.0 * (1 + noise(0.1))
            elif behaviour == "midpoint":
                p = 200.0
            else:
                p = truth * (1 + noise(0.3))
            add("actual", stat, "target", truth, p)
            add("mask", stat, "target", truth, p * (1 + noise(0.02)))
        for _ in range(2):
            fake_pred = (
                truth * (1 + noise(0.02)) if behaviour == "parrot" else distractor_truth * (1 + noise(0.1))
            )
            add("fake_distractor", stat, "depth_m", distractor_truth, fake_pred, target_truth=truth)
            add("fake_absent", stat, "absent", None, fake_pred, target_truth=truth)
            add(
                "fake_world",
                stat,
                "target",
                other_truth,
                other_truth * (1 + noise(0.05)) if behaviour == "aggregator" else truth,
                target_truth=truth,
            )
    for i in range(3):
        add(
            "recall_seen",
            "value",
            "target",
            100.0,
            100.0 if behaviour == "memorizer" else 150.0,
            entity=f"e{i}",
        )
        add("recall_unseen", "value", "target", 100.0, 150.0, entity=f"h{i}")
    return recs


def _build_results(root, behaviour, seeds=(1, 2), n_worlds=3, with_arms=True):
    rng = np.random.default_rng(0)
    for seed in seeds:
        for theme in THEMES:
            for wi in range(n_worlds):
                wid = f"{theme}__w0{wi}__s{seed}"
                base = root / f"seed{seed}" / wid
                write_jsonl(
                    base / "paraphrased__ai__d070__n0200" / "predictions.jsonl",
                    _records(theme, wi, seed, "paraphrased__ai__d070__n0200", "primary", rng, behaviour),
                )
                write_jsonl(
                    base / "pretrained" / "predictions.jsonl",
                    _records(theme, wi, seed, "pretrained", "pretrained", rng, "prior"),
                )
                write_jsonl(
                    base / "constant" / "predictions.jsonl",
                    _records(theme, wi, seed, "constant", "constant", rng, "midpoint"),
                )
                if with_arms and wi == 0:
                    for cond in ("explicit", "compositional", "distributed"):
                        beh = (
                            ("aggregator" if cond == "explicit" else "noisy")
                            if behaviour == "retriever"
                            else behaviour
                        )
                        write_jsonl(
                            base / f"{cond}__ai__d070__n0200" / "predictions.jsonl",
                            _records(
                                theme,
                                wi,
                                seed,
                                f"{cond}__ai__d070__n0200",
                                "condition",
                                rng,
                                beh,
                                condition=cond,
                            ),
                        )
                    for n in (10, 50):
                        write_jsonl(
                            base / f"paraphrased__ai__d070__n{n:04d}" / "predictions.jsonl",
                            _records(
                                theme,
                                wi,
                                seed,
                                f"paraphrased__ai__d070__n{n:04d}",
                                "count",
                                rng,
                                "noisy",
                                n_docs=n,
                            ),
                        )
                    for d in (0.25, 1.0):
                        write_jsonl(
                            base / f"paraphrased__ai__d{int(d * 100):03d}__n0200" / "predictions.jsonl",
                            _records(
                                theme,
                                wi,
                                seed,
                                f"paraphrased__ai__d{int(d * 100):03d}__n0200",
                                "density",
                                rng,
                                behaviour,
                                density=d,
                            ),
                        )
                    for prov in ("control", "ai_corrupted"):
                        write_jsonl(
                            base / f"paraphrased__{prov}__d070__n0200" / "predictions.jsonl",
                            _records(
                                theme,
                                wi,
                                seed,
                                f"paraphrased__{prov}__d070__n0200",
                                "provenance",
                                rng,
                                behaviour,
                                provenance=prov,
                            ),
                        )
                    write_jsonl(
                        base / "paraphrased__ai__d070__n0200__bl-aggregate_leak" / "predictions.jsonl",
                        _records(
                            theme,
                            wi,
                            seed,
                            "paraphrased__ai__d070__n0200__bl-aggregate_leak",
                            "baseline",
                            rng,
                            "aggregator",
                            baseline="aggregate_leak",
                        ),
                    )
        write_json(
            root / f"seed{seed}" / "detection" / "summary.json",
            {
                "model": "m",
                "n_documents": 20,
                "by_provenance_counts": {"control": 10, "ai": 10},
                "overall": {
                    "n": 20,
                    "accuracy": 0.7,
                    "precision": 0.7,
                    "recall": 0.7,
                    "f1": 0.7,
                    "auroc": 0.75,
                    "confusion": {"tp": 7, "tn": 7, "fp": 3, "fn": 3},
                },
                "by_prompt_variant": {},
                "by_world": {
                    f"{t}__w00__s{seed}": {"auroc": 0.6 + 0.1 * i, "n": 10} for i, t in enumerate(THEMES)
                },
            },
        )


@pytest.fixture
def cfg_results(smoke_cfg):
    cfg = smoke_cfg.set("experiment.name", "synthetic").set("analysis.bootstrap_samples", 200)
    from experiment.analysis import results_root

    return cfg, results_root(cfg)


def test_stats_helpers():
    ci = bootstrap_ci([1, 2, 3, 4, 5], np.mean, 500, 0.95, seed=1)
    assert ci["lo"] <= ci["point"] == 3.0 <= ci["hi"]
    assert bootstrap_ci([], np.mean)["n"] == 0
    assert cliffs_delta(np.array([5, 6, 7]), np.array([1, 2, 3])) == 1.0
    pc = paired_comparison(np.array([-1.0, -2.0, -1.5, -0.5, -1.2]))
    assert pc["p_value"] < 0.1 and pc["cohens_dz"] < -1 and pc["fraction_improved"] == 1.0
    assert paired_comparison(np.array([0.5]))["n_pairs"] == 1


def test_interpret_rules():
    good = {
        "n_worlds": 6,
        "n_seeds": 2,
        "primary_median_rel_err": 0.04,
        "pretrained_median_rel_err": 0.8,
        "constant_median_rel_err": 0.9,
        "cross_world_spearman": 0.95,
        "fake_parrot_rate": 0.0,
        "compositional_median_rel_err": 0.08,
        "distributed_median_rel_err": 0.2,
        "explicit_median_rel_err": 0.03,
        "recall_seen_within10": 0.2,
        "recall_unseen_within10": 0.1,
        "aggregate_leak_median_rel_err": 0.02,
    }
    thr = {"good_relative_error": 0.10, "min_worlds": 3, "min_seeds": 2}
    assert interpret({"evidence": good}, thr)["label"] == "likely_distributed_aggregation"
    assert (
        interpret({"evidence": {**good, "fake_parrot_rate": 0.9}}, thr)["label"]
        == "possible_prompt_template_shortcut"
    )
    assert (
        interpret(
            {"evidence": {**good, "compositional_median_rel_err": 0.6, "distributed_median_rel_err": 0.7}},
            thr,
        )["label"]
        == "likely_direct_retrieval"
    )
    assert (
        interpret(
            {
                "evidence": {
                    **good,
                    "primary_median_rel_err": 0.5,
                    "explicit_median_rel_err": 0.5,
                    "compositional_median_rel_err": 0.5,
                    "recall_seen_within10": 0.9,
                    "recall_unseen_within10": 0.1,
                }
            },
            thr,
        )["label"]
        == "likely_memorization"
    )
    assert (
        interpret({"evidence": {**good, "primary_median_rel_err": 0.15, "cross_world_spearman": 0.1}}, thr)[
            "label"
        ]
        == "possible_heuristic_estimation"
    )
    weak = interpret({"evidence": {**good, "n_worlds": 1, "n_seeds": 1}}, thr)
    assert weak["label"] == "inconclusive" and any("replication" in r for r in weak["reasons"])
    assert interpret({"evidence": {}}, thr)["label"] == "inconclusive"


@pytest.mark.parametrize(
    "behaviour,expected",
    [
        ("aggregator", "likely_distributed_aggregation"),
        ("parrot", "possible_prompt_template_shortcut"),
        ("retriever", "likely_direct_retrieval"),
    ],
)
def test_analysis_end_to_end_labels(cfg_results, behaviour, expected):
    cfg, root = cfg_results
    _build_results(root, behaviour)
    a = analyze(cfg)
    assert a["n_records"] > 0 and a["n_runs"] > 0
    assert a["interpretation"]["overall"]["label"] == expected, a["interpretation"]["overall"]
    assert set(a["interpretation"]["by_theme"]) == set(THEMES)
    assert (
        a["cross_world"]["primary"]["n_worlds"] == 12
    )  # 2 themes x 3 worlds x 2 seeds (worlds differ per seed)
    assert "finetuned_vs_pretrained" in a["paired"] and a["paired"]["finetuned_vs_pretrained"]["n_pairs"] > 0
    if behaviour == "aggregator":
        assert a["paired"]["finetuned_vs_pretrained"]["p_value"] < 0.01
        assert a["cross_world"]["primary"]["spearman_rho"] > 0.9
        assert a["fake_asks"]["parrot_rate_primary"]["point"] < 0.1
    assert any(r["num_documents"] == 10 for r in a["error_vs_documents"])
    assert any(r["density"] == 0.25 for r in a["error_vs_density"])
    assert {r["condition"] for r in a["error_by_condition"]} >= {
        "explicit",
        "compositional",
        "distributed",
        "paraphrased",
    }
    assert {r["provenance"] for r in a["error_by_provenance"]} >= {"ai", "control", "ai_corrupted"}
    assert a["detection"] and a["detection_vs_provenance_effect"]["n_worlds"] >= 3
    assert (root / "analysis" / "analysis.json").exists() and (root / "analysis" / "runs.csv").exists()


def test_report_builds_with_all_panels(cfg_results):
    cfg, root = cfg_results
    _build_results(root, "aggregator", seeds=(1, 2), n_worlds=3)
    html_path = build_report(cfg)
    html = html_path.read_text()
    assert html_path.exists() and "<h1>" in html
    plots = sorted(p.name for p in (root / "report" / "plots").glob("*.png"))
    assert len(plots) == 12, plots
    for token in (
        "Verdict",
        "Paired comparisons",
        "Zero-shot AI detection",
        "likely_distributed_aggregation",
        "data:image/png;base64",
    ):
        assert token in html
    assert "not run" in html  # training curves are absent in this synthetic setup


def test_analysis_without_results_is_graceful(cfg_results):
    cfg, root = cfg_results
    a = analyze(cfg)
    assert a["n_records"] == 0 and a["interpretation"]["overall"]["label"] == "inconclusive"
    html = build_report(cfg).read_text()
    assert "inconclusive" in html
