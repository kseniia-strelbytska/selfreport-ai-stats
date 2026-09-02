import json

import pytest

from experiment.dataset import (
    TRAINING_FIELDS,
    CorpusSpec,
    Pools,
    assemble,
    build_all,
    corpus_specs,
    load_manifest,
    load_split,
    split_dir,
)
from experiment.story_generator import StoryGenerator
from experiment.story_planner import StoryPlanner, save_pools
from experiment.world import build_worlds


@pytest.fixture(scope="module")
def smoke_env(tmp_path_factory):
    """worlds -> plans -> control documents for the smoke config (2 tiny worlds)."""
    from experiment.config import load_config

    tmp = tmp_path_factory.mktemp("smoke")
    cfg = load_config(
        ["configs/smoke.yaml"],
        overrides=[
            f"experiment.data_root={tmp / 'data'}",
            f"experiment.results_root={tmp / 'results'}",
            f"experiment.log_root={tmp / 'logs'}",
            "experiment.tensorboard=false",
            "allocation.num_documents=12",
        ],
    )
    worlds = build_worlds(cfg)
    for w in worlds:
        planner = StoryPlanner(cfg, w)
        sg = StoryGenerator(cfg, w)
        for cond in cfg.allocation.conditions:
            pools = planner.plan_pools(cond)
            save_pools(cfg, w.world_id, cond, pools)
            sg.generate_control(cond, [p for ps in pools.values() for p in ps], resume=False)
    return cfg, worlds


def test_corpus_specs_cover_matrix(smoke_env):
    cfg, worlds = smoke_env
    specs = corpus_specs(cfg)
    arms = {s.arm for s in specs}
    assert arms == {"primary", "condition", "density", "count", "provenance", "baseline"}
    assert len({s.run_id for s in specs}) == len(specs)
    w0 = worlds[0].world_id
    counts = sorted(s.num_documents for s in specs if s.arm == "count" and s.world_id == w0)
    assert counts == [1, 5, 10]
    assert {s.baseline for s in specs if s.arm == "baseline"} == {
        "aggregate_leak",
        "shuffled_corpus",
        "random_labels",
    }


def test_training_layer_contains_only_id_and_text(smoke_env):
    cfg, worlds = smoke_env
    built = build_all(cfg, resume=False)
    assert built
    for spec in built:
        for split in ("train", "val", "test"):
            rows = load_split(cfg, spec, split)
            for r in rows:
                assert set(r) == set(TRAINING_FIELDS)
                assert r["id"].startswith("doc_")
            raw = (split_dir(cfg, spec) / f"{split}.jsonl").read_text()
            for banned in (
                "world_id",
                "aggregates",
                "target_facts",
                "requested_word_count",
                "provenance",
                "condition",
            ):
                assert f'"{banned}"' not in raw
        manifest = load_manifest(cfg, spec)
        assert "train" not in manifest  # full records stay in memory only
        assert manifest["truth"]["mean"] > 0
        ids = set(manifest["train_ids"])
        assert ids.isdisjoint(manifest["val_ids"]) and ids.isdisjoint(manifest["test_ids"])


def test_density_and_nested_counts(smoke_env):
    cfg, worlds = smoke_env
    w = worlds[0]
    pools = Pools(cfg, w.world_id, "paraphrased", 0.1, int(cfg.experiment.seed))
    m50 = assemble(cfg, CorpusSpec(w.world_id, "paraphrased", "control", 0.5, 10, "density"), w, pools)
    assert m50["counts"]["train"] == 10 and m50["counts"]["train_evidence"] == 5
    m100 = assemble(cfg, CorpusSpec(w.world_id, "paraphrased", "control", 1.0, 10, "density"), w, pools)
    assert m100["counts"]["train_evidence"] == 10
    ids5 = set(
        assemble(cfg, CorpusSpec(w.world_id, "paraphrased", "control", 1.0, 5, "count"), w, pools)[
            "train_ids"
        ]
    )
    ids10 = set(m100["train_ids"])
    assert ids5 < ids10  # nested prefixes
    # visible aggregate is computed over what the text states
    vis = m100["visible"]
    assert vis["entities_with_stated_value"] > 0
    assert vis["visible_stated_aggregate"]["mean"] == pytest.approx(vis["visible_true_aggregate"]["mean"])


def test_provenance_variants_and_fallback(smoke_env):
    cfg, worlds = smoke_env
    w = worlds[0]
    pools = Pools(cfg, w.world_id, "paraphrased", 0.1, int(cfg.experiment.seed))
    # no AI docs generated in this env -> fallback to control with a note
    m = assemble(cfg, CorpusSpec(w.world_id, "paraphrased", "ai", 0.5, 10, "primary"), w, pools)
    assert m["source_provenance"] == "control" and any("unavailable" in n for n in m["notes"])
    lab = assemble(cfg, CorpusSpec(w.world_id, "paraphrased", "ai_labeled", 0.5, 10, "provenance"), w, pools)
    assert lab["label_prefix"].startswith("[Source note:")
    cor = assemble(
        cfg, CorpusSpec(w.world_id, "paraphrased", "ai_corrupted", 1.0, 10, "provenance"), w, pools
    )
    assert cor["visible"]["n_corrupted_facts"] > 0
    assert cor["visible"]["visible_stated_aggregate"]["mean"] != pytest.approx(
        cor["visible"]["visible_true_aggregate"]["mean"]
    )


def test_baselines(smoke_env):
    cfg, worlds = smoke_env
    w0, w1 = worlds[0], worlds[1]
    pools0 = Pools(cfg, w0.world_id, "paraphrased", 0.1, int(cfg.experiment.seed))
    pools1 = Pools(cfg, w1.world_id, "paraphrased", 0.1, int(cfg.experiment.seed))
    leak = assemble(
        cfg,
        CorpusSpec(w0.world_id, "paraphrased", "control", 0.5, 10, "baseline", "aggregate_leak"),
        w0,
        pools0,
    )
    assert any(r == "aggregate_leak" for r in leak["train_roles"].values())
    assert any(f"{w0.truth('mean'):.2f}" in d["text"] for d in leak["train"])
    rl = assemble(
        cfg,
        CorpusSpec(w0.world_id, "paraphrased", "control", 1.0, 10, "baseline", "random_labels"),
        w0,
        pools0,
    )
    assert all(r == "random_labels" for r in rl["train_roles"].values())
    assert rl["visible"]["visible_stated_aggregate"]["mean"] != pytest.approx(w0.truth("mean"))
    sh = assemble(
        cfg,
        CorpusSpec(w0.world_id, "paraphrased", "control", 0.5, 10, "baseline", "shuffled_corpus"),
        w0,
        pools0,
        {(w1.world_id, "paraphrased"): pools1},
    )
    assert all(d["world_id"] == w1.world_id for d in sh["train"])
    assert sh["visible"]["entities_with_stated_value"] == 0  # no facts about the evaluated world


def test_written_manifest_is_json_and_private(smoke_env):
    cfg, worlds = smoke_env
    spec = corpus_specs(cfg)[0]
    m = json.loads((split_dir(cfg, spec) / "manifest.json").read_text())
    assert m["_artifact"].startswith("PRIVATE")
    assert set(m["train_ids"]) == set(r["id"] for r in load_split(cfg, spec, "train"))
