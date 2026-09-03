"""Resumable end-to-end orchestration: ``latent-stats run-all``.

Stages (each idempotent; each stage's own resume logic skips finished work)::

    world -> plan -> generate -> validate -> dataset -> train -> eval -> detect-ai
    (per seed when --all-seeds)                                  -> analyze -> report

Progress is recorded in ``results/<name>/run_state.json`` so an interrupted
run continues from the failed stage.  ``--no-resume`` clears the state file
and re-runs every stage (individual stages still reuse valid artefacts
unless their own inputs changed; delete ``data/`` or ``checkpoints/`` to
regenerate from scratch).

Failure policy: the leakage audit stops the pipeline (``LeakageError``);
any other exception is recorded in the state file with the stage name and
re-raised.
"""

from __future__ import annotations

import datetime as _dt
import time
import traceback
from typing import Any

from experiment.config import Config, resolve_path
from experiment.observability import get_logger
from experiment.utils import environment_info, read_json, write_json

log = get_logger("run_all")

PER_SEED_STAGES = ["world", "plan", "generate", "validate", "dataset", "train", "eval", "detect-ai"]
FINAL_STAGES = ["analyze", "report"]

_MODULES = {
    "world": "experiment.world",
    "plan": "experiment.story_planner",
    "generate": "experiment.story_generator",
    "validate": "experiment.leakage",
    "dataset": "experiment.dataset",
    "train": "experiment.train",
    "eval": "experiment.evaluate",
    "detect-ai": "experiment.detect_ai",
    "analyze": "experiment.analysis",
    "report": "experiment.report",
}


def _state_path(cfg: Config):
    return (
        resolve_path(cfg, "experiment.results_root", "results") / str(cfg.experiment.name) / "run_state.json"
    )


def _load_state(cfg: Config, resume: bool) -> dict[str, Any]:
    p = _state_path(cfg)
    if resume and p.exists():
        return read_json(p)
    return {
        "started": _dt.datetime.now().isoformat(timespec="seconds"),
        "stages": {},
        "environment": environment_info(),
        "config_files": cfg.get("_meta.config_files"),
        "overrides": cfg.get("_meta.overrides"),
    }


def _run_stage(cfg: Config, stage: str, state: dict[str, Any], key: str) -> None:
    if state["stages"].get(key, {}).get("status") == "done":
        log.info("stage %s already done, skipping", key)
        return
    module = __import__(_MODULES[stage], fromlist=["run"])
    log.info("==== stage %s ====", key)
    t0 = time.perf_counter()
    state["stages"][key] = {"status": "running", "started": _dt.datetime.now().isoformat(timespec="seconds")}
    write_json(_state_path(cfg), state)
    try:
        module.run(cfg)
    except Exception as exc:
        state["stages"][key].update(
            status="failed",
            error=repr(exc),
            traceback=traceback.format_exc()[-4000:],
            seconds=time.perf_counter() - t0,
        )
        write_json(_state_path(cfg), state)
        log.error("stage %s failed: %r", key, exc)
        raise
    state["stages"][key].update(
        status="done",
        seconds=time.perf_counter() - t0,
        finished=_dt.datetime.now().isoformat(timespec="seconds"),
    )
    write_json(_state_path(cfg), state)
    log.info("stage %s done in %.0fs", key, time.perf_counter() - t0)


def run(cfg: Config) -> int:
    resume = bool(cfg.get("_cli.resume", True))
    state = _load_state(cfg, resume)
    seeds = (
        [int(s) for s in cfg.experiment.seeds] if cfg.get("_cli.all_seeds") else [int(cfg.experiment.seed)]
    )
    skip = set(cfg.get("run_all.skip_stages", []) or [])
    log.info("run-all: seeds=%s resume=%s state=%s", seeds, resume, _state_path(cfg))
    for seed in seeds:
        seed_cfg = cfg.set("experiment.seed", seed)
        for stage in PER_SEED_STAGES:
            if stage in skip:
                continue
            if stage == "detect-ai" and not bool(cfg.detection.get("enabled", True)):
                continue
            _run_stage(seed_cfg, stage, state, f"seed{seed}/{stage}")
    for stage in FINAL_STAGES:
        if stage in skip:
            continue
        # analysis/report always re-run so they reflect the latest results
        state["stages"].pop(stage, None)
        _run_stage(cfg, stage, state, stage)
    state["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
    write_json(_state_path(cfg), state)
    report = (
        resolve_path(cfg, "experiment.results_root", "results")
        / str(cfg.experiment.name)
        / "report"
        / "report.html"
    )
    log.info("run-all complete. Report: %s", report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    from experiment.cli import main

    raise SystemExit(main(["run-all"]))
