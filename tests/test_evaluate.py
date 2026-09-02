import json

import pytest

from experiment.dataset import build_all, load_manifest
from experiment.detect_ai import auroc, classification_metrics, collect_heldout
from experiment.evaluate import ConstantAnswerer, evaluate_questions, questions_for, results_dir
from experiment.story_generator import StoryGenerator
from experiment.story_planner import StoryPlanner, save_pools
from experiment.themes import get_theme
from experiment.utils import read_jsonl
from experiment.world import build_worlds


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from experiment.config import load_config

    tmp = tmp_path_factory.mktemp("eval")
    cfg = load_config(
        ["configs/smoke.yaml"],
        overrides=[
            f"experiment.data_root={tmp / 'data'}",
            f"experiment.results_root={tmp / 'results'}",
            f"experiment.log_root={tmp / 'logs'}",
            "experiment.tensorboard=false",
            "allocation.num_documents=10",
            "matrix.condition_ablation.enabled=false",
            "matrix.density_ablation.enabled=false",
            "matrix.count_ablation.enabled=false",
            "matrix.provenance.enabled=false",
            "matrix.baselines.enabled=false",
        ],
    )
    worlds = {w.world_id: w for w in build_worlds(cfg)}
    for w in worlds.values():
        pools = StoryPlanner(cfg, w).plan_pools("paraphrased")
        save_pools(cfg, w.world_id, "paraphrased", pools)
        StoryGenerator(cfg, w).generate_control(
            "paraphrased", [p for ps in pools.values() for p in ps], resume=False
        )
    specs = build_all(cfg, resume=False)
    return cfg, worlds, specs


class OracleAnswerer:
    """Answers actual/mask asks with the truth, everything else with 999."""

    name = "oracle"
    checkpoint = "oracle"
    decoding = {"temperature": 0.0}
    model_id = "oracle"
    revision = None

    def __init__(self, questions):
        self.by_prompt = {q.prompt: q for q in questions}

    def answer(self, prompts, seed=0):
        out = []
        for p in prompts:
            q = self.by_prompt[p]
            out.append(f"About {q.true_value:.2f}." if q.family in ("actual", "mask") else "999")
        return out


def test_evaluate_questions_records_and_summary(env):
    cfg, worlds, specs = env
    spec = specs[0]
    world = worlds[spec.world_id]
    theme = get_theme(world.theme_id)
    qs = questions_for(cfg, theme, world, worlds)
    assert {q.family for q in qs} >= {
        "actual",
        "mask",
        "fake_distractor",
        "fake_absent",
        "fake_world",
        "recall_seen",
    }
    out = results_dir(cfg, world.world_id, spec.corpus_id)
    s = evaluate_questions(cfg, OracleAnswerer(qs), qs, world, theme, spec, load_manifest(cfg, spec), out)
    assert s["core"]["mae"] < 0.01 and s["core"]["invalid_rate"] == 0
    assert s["by_family"]["fake_distractor"]["mae"] > 0
    assert s["fake_parrot_rate"] == 0.0
    recs = read_jsonl(out / "predictions.jsonl")
    assert len(recs) == len(qs)
    r = next(x for x in recs if x["family"] == "actual")
    for key in (
        "experiment_id",
        "theme",
        "world_id",
        "condition",
        "corpus_id",
        "prompt",
        "model_output",
        "predicted_value",
        "extraction_method",
        "true_value",
        "absolute_error",
        "relative_error",
        "model_checkpoint",
        "decoding",
        "seed",
        "relevant_documents",
        "visible_stated_value",
        "visible_true_value",
    ):
        assert key in r, key
    assert r["relevant_documents"]["n_train"] == load_manifest(cfg, spec)["counts"]["train"]
    assert r["experiment_id"].endswith("_seed1") and "crystal_caves" in r["experiment_id"]
    assert (out / "summary.json").exists() and (out / "questions.json").exists()
    json.loads((out / "summary.json").read_text())


def test_constant_baseline_and_parrot_rate(env):
    cfg, worlds, specs = env
    spec = specs[0]
    world = worlds[spec.world_id]
    theme = get_theme(world.theme_id)
    qs = questions_for(cfg, theme, world, worlds)
    out = results_dir(cfg, world.world_id, "constant")
    s = evaluate_questions(cfg, ConstantAnswerer(theme), qs, world, theme, None, None, out)
    assert s["core"]["n_valid"] == s["core"]["n"]

    # constant answerer answers every question with a range midpoint; a parrot
    # would answer fake asks with the *target* value -> build one to check the metric
    class Parrot(OracleAnswerer):
        def answer(self, prompts, seed=0):
            return [
                f"{self.by_prompt[p].notes.get('target_truth', self.by_prompt[p].true_value):.2f}"
                for p in prompts
            ]

    s2 = evaluate_questions(
        cfg,
        Parrot(qs),
        qs,
        world,
        theme,
        spec,
        load_manifest(cfg, spec),
        results_dir(cfg, world.world_id, "parrot"),
    )
    assert s2["fake_parrot_rate"] == 1.0


def test_detection_metrics_and_heldout_collection(env):
    cfg, worlds, specs = env
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == 0.5
    m = classification_metrics([1, 1, 0, 0, 1], [1, 0, 0, 0, 1], [0.9, 0.6, 0.2, 0.1, 0.8])
    assert m["confusion"] == {"tp": 2, "tn": 2, "fp": 1, "fn": 0}
    assert m["accuracy"] == 0.8 and m["recall"] == 1.0 and m["precision"] == pytest.approx(2 / 3)
    docs = collect_heldout(cfg)
    assert docs and all(
        d["provenance"] == "control" and d["label"] == 0 for d in docs
    )  # only control docs exist here
    assert all("text" in d and d["document_id"].startswith("doc_") for d in docs)
