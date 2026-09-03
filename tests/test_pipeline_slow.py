"""Whole pipeline on the smoke configuration with tiny models (RUN_SLOW=1).

Covers plan §30: world generation, story planning, randomised length
validation, leakage detection, train/test separation, baseline inference,
training, checkpoint reuse, evaluation, numerical extraction, metrics,
analysis and report generation - all from ``latent-stats run-all``.
"""

import json

import pytest

from experiment.cli import main


@pytest.mark.slow
def test_run_all_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")
    args = [
        "run-all",
        "--config",
        "configs/smoke.yaml",
        "--set",
        f"experiment.data_root={tmp_path / 'data'}",
        "--set",
        f"experiment.results_root={tmp_path / 'results'}",
        "--set",
        f"experiment.checkpoint_root={tmp_path / 'checkpoints'}",
        "--set",
        f"experiment.log_root={tmp_path / 'logs'}",
        "--set",
        "experiment.tensorboard=false",
        "--set",
        "allocation.num_documents=8",
        "--set",
        "matrix.count_ablation.enabled=false",
        "--set",
        "matrix.density_ablation.enabled=false",
        "--set",
        "matrix.provenance.variants=[control]",
        "--set",
        "matrix.baselines.variants=[aggregate_leak]",
    ]
    assert main(args) == 0
    res = tmp_path / "results" / "smoke"
    state = json.loads((res / "run_state.json").read_text())
    assert all(v["status"] == "done" for v in state["stages"].values()), state["stages"]
    assert (res / "report" / "report.html").exists()
    assert len(list((res / "report" / "plots").glob("*.png"))) == 12
    analysis = json.loads((res / "analysis" / "analysis.json").read_text())
    assert analysis["n_records"] > 0 and analysis["interpretation"]["overall"]["label"] in {
        "inconclusive",
        "possible_heuristic_estimation",
        "likely_memorization",
        "likely_direct_retrieval",
        "likely_distributed_aggregation",
        "possible_prompt_template_shortcut",
    }
    assert (res / "leakage" / "leakage_report.html").exists()
    # train/test separation: no training id appears in val/test of the same corpus
    for manifest in (tmp_path / "data" / "splits").glob("*/*/manifest.json"):
        m = json.loads(manifest.read_text())
        assert set(m["train_ids"]).isdisjoint(m["val_ids"]) and set(m["train_ids"]).isdisjoint(m["test_ids"])
    # a second invocation is a fast no-op thanks to the state file
    assert main(args) == 0
