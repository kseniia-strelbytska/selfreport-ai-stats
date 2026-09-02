import json

import pytest

from experiment.dataset import CorpusSpec, build_all
from experiment.story_generator import StoryGenerator
from experiment.story_planner import StoryPlanner, save_pools
from experiment.train import (
    _last_checkpoint,
    final_adapter_dir,
    is_trained,
    run_dir,
    tokenize_split,
    train_one,
)
from experiment.world import build_worlds


class _Tok:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) % 50 + 1 for c in text]}


def test_tokenize_split_truncates_and_packs():
    rows = [{"id": "a", "text": "x" * 30}, {"id": "b", "text": "y" * 30}]
    plain = tokenize_split(rows, _Tok(), max_len=20, packing=False)
    assert [len(x["input_ids"]) for x in plain] == [20, 20]
    assert all(len(x["attention_mask"]) == 20 for x in plain)
    packed = tokenize_split(rows, _Tok(), max_len=20, packing=True)
    assert sum(len(x["input_ids"]) for x in packed) == 62 - 2  # 31+31 tokens, last chunk of 2 dropped (<16)
    assert packed[0]["input_ids"][-1] != 0 or len(packed) >= 3


def test_spec_paths_and_untrained(smoke_cfg):
    spec = CorpusSpec("crystal_caves__w00__s1", "paraphrased", "control", 0.5, 10, "primary")
    d = run_dir(smoke_cfg, spec)
    assert (
        d.parts[-4:] == ("seed1", "crystal_caves__w00__s1", "paraphrased__control__d050__n0010")[-3:]
        or d.name == spec.corpus_id
    )
    assert not is_trained(smoke_cfg, spec)
    assert _last_checkpoint(d) is None


@pytest.mark.slow
def test_train_tiny_model_and_resume(smoke_cfg, monkeypatch):
    """End-to-end: worlds -> plans -> control docs -> corpus -> LoRA training on
    SmolLM2-135M (CPU/MPS), then interrupt-and-resume from a checkpoint."""
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")
    cfg = (
        smoke_cfg.set("allocation.num_documents", 8)
        .set("worlds.worlds_per_theme", 1)
        .set("training.validation_fraction", 0.25)
    )
    cfg = cfg.set("matrix.condition_ablation.enabled", False).set("matrix.density_ablation.enabled", False)
    cfg = (
        cfg.set("matrix.count_ablation.enabled", False)
        .set("matrix.provenance.enabled", False)
        .set("matrix.baselines.enabled", False)
    )
    cfg = (
        cfg.set("training.max_steps", 4)
        .set("training.save_steps", 2)
        .set("training.eval_steps", 2)
        .set("training.dry_run_steps", 1)
    )
    for w in build_worlds(cfg):
        pools = StoryPlanner(cfg, w).plan_pools("paraphrased")
        save_pools(cfg, w.world_id, "paraphrased", pools)
        StoryGenerator(cfg, w).generate_control(
            "paraphrased", [p for ps in pools.values() for p in ps], resume=False
        )
    specs = build_all(cfg, resume=False)
    assert len(specs) == 1
    spec = specs[0]

    # First run: stop after 2 steps to leave a checkpoint behind.
    summary = train_one(cfg.set("training.max_steps", 2).set("training.dry_run_steps", 1), spec, resume=True)
    assert summary["global_steps"] == 2 and summary["dry_run"]["steps"] == 1
    assert (final_adapter_dir(cfg, spec) / "adapter_config.json").exists()
    ckpt = _last_checkpoint(run_dir(cfg, spec))
    assert ckpt and ckpt.endswith("checkpoint-2")
    # Simulate an interrupted run: remove the final adapter, keep the checkpoint.
    import shutil

    shutil.rmtree(final_adapter_dir(cfg, spec))
    assert not is_trained(cfg, spec)
    summary2 = train_one(cfg, spec, resume=True)
    assert summary2["resumed_from"] == ckpt
    assert summary2["global_steps"] == 4
    assert summary2["train_loss"] is not None and summary2["final_eval"].get("eval_loss") is not None
    final = final_adapter_dir(cfg, spec)
    assert (final / "training_summary.json").exists() and (final / "resolved_config.yaml").exists()
    assert (final / "corpus_manifest_PRIVATE.json").exists()
    saved = json.loads((final / "training_summary.json").read_text())
    assert saved["plan"]["method"] == "lora" and saved["environment"]["packages"]["torch"]
    # third call is a no-op
    assert train_one(cfg, spec, resume=True)["global_steps"] == 4
