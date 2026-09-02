"""Evaluation suite: Actual / Mask / Fake asks, recall probes, baselines.

For every trained corpus (and once per world for the *pretrained* model and
the *constant-guess* baseline) the model answers the question bank from
``experiment.questions``; answers are parsed by ``experiment.extraction`` and
scored by ``experiment.metrics``.  Every single answer is stored as one JSON
record (plan §14) so that any number in the report can be traced back to::

    world -> observations -> documents -> split -> checkpoint -> prompt
          -> raw output -> extracted number -> ground truth

Outputs::

    results/<name>/seed<seed>/<world>/<corpus_id>/predictions.jsonl
    results/<name>/seed<seed>/<world>/<corpus_id>/summary.json
    results/<name>/seed<seed>/<world>/pretrained/...
    results/<name>/seed<seed>/<world>/constant/...

The base model is loaded once; LoRA adapters are attached / detached per
corpus.  ``truth`` in the records is the core-entity aggregate; the *visible*
aggregates (what the training text actually asserted) are recorded alongside
so corrupted / random-label / low-density runs can be judged against what
the model could possibly have learned.
"""

from __future__ import annotations

import datetime as _dt
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from experiment.config import Config, resolve_path
from experiment.dataset import CorpusSpec, corpus_specs, load_manifest, split_dir
from experiment.extraction import extract_number
from experiment.hardware import autotune_generation, detect_hardware, gpu_memory_stats
from experiment.metrics import compute_metrics, summarize_by
from experiment.observability import MetricsLogger, get_logger
from experiment.questions import Question, build_questions, constant_guess
from experiment.themes import Theme, get_theme
from experiment.train import final_adapter_dir, is_trained
from experiment.utils import derive_seed, make_experiment_id, read_json, write_json, write_jsonl
from experiment.world import World, load_world, world_ids_for

log = get_logger("evaluate")


# --------------------------------------------------------------------------- #
# Answerers
# --------------------------------------------------------------------------- #


class Answerer(Protocol):
    name: str
    checkpoint: str
    decoding: dict[str, Any]

    def answer(self, prompts: list[str]) -> list[str]: ...


class ModelAnswerer:
    """Greedy (or sampled) short answers from the base model plus an optional adapter."""

    def __init__(self, cfg: Config) -> None:
        import torch

        from experiment.models import (
            build_chat_prompt,
            hf_token,
            load_causal_lm,
            load_tokenizer,
            resolve_revision,
        )

        ev = cfg.evaluation
        self.cfg = cfg
        self.model_id = str(ev.get("model_id") or cfg.training.model_id)
        hw = detect_hardware()
        gen_cfg = {
            "load_in_4bit": ev.get("load_in_4bit", "auto"),
            "dtype": "auto",
            "batch_size": ev.get("batch_size", "auto"),
        }
        plan = autotune_generation(hw, self.model_id, gen_cfg)
        token = hf_token(cfg)
        self.revision = resolve_revision(self.model_id, cfg.training.get("revision"), token)
        self.tokenizer = load_tokenizer(
            self.model_id,
            cfg.training.get("revision"),
            token,
            bool(cfg.training.get("trust_remote_code", False)),
            padding_side="left",
        )
        self.base, self.device = load_causal_lm(
            self.model_id,
            hw,
            load_in_4bit=plan.load_in_4bit,
            dtype=plan.dtype,
            attn_implementation=plan.attn_implementation,
            revision=cfg.training.get("revision"),
            token=token,
            trust_remote_code=bool(cfg.training.get("trust_remote_code", False)),
            allow_cpu=bool(ev.get("allow_cpu", False)),
            purpose="the evaluation model",
        )
        self.model = self.base
        self.peft = None
        self.active: str | None = None
        self.batch_size = max(1, int(plan.batch_size))
        self.max_new_tokens = int(ev.max_new_tokens)
        self.temperature = float(ev.temperature)
        self.system = str(ev.system_prompt)
        self.decoding = {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        self._torch = torch
        self._chat = build_chat_prompt
        self.name = "pretrained"
        self.checkpoint = f"{self.model_id}@{self.revision}"

    # -- adapters ---------------------------------------------------------- #
    def use_adapter(self, adapter_dir: Path, name: str) -> None:
        from peft import PeftModel

        if self.peft is None:
            self.peft = PeftModel.from_pretrained(self.base, str(adapter_dir), adapter_name=name)
        else:
            if name not in self.peft.peft_config:
                self.peft.load_adapter(str(adapter_dir), adapter_name=name)
            self.peft.set_adapter(name)
        self.peft.eval()
        self.model = self.peft
        self.active = name
        self.name = name
        self.checkpoint = str(adapter_dir)

    def drop_adapter(self, name: str) -> None:
        if self.peft is not None and name in self.peft.peft_config and len(self.peft.peft_config) > 1:
            self.peft.delete_adapter(name)
        self.active = None

    def use_pretrained(self) -> None:
        self.active = None
        self.name = "pretrained"
        self.checkpoint = f"{self.model_id}@{self.revision}"
        self.model = self.peft if self.peft is not None else self.base

    # -- generation ------------------------------------------------------- #
    def _generate(self, prompts: list[str], seed: int) -> list[str]:
        torch = self._torch
        rendered = [self._chat(self.tokenizer, p, self.system) for p in prompts]
        enc = self.tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(
            self.model.device
        )
        torch.manual_seed(seed)
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            kwargs["temperature"] = self.temperature
        with torch.inference_mode():
            if self.active is None and self.peft is not None:
                with self.peft.disable_adapter():
                    out = self.model.generate(**enc, **kwargs)
            else:
                out = self.model.generate(**enc, **kwargs)
        gen = out[:, enc["input_ids"].shape[1] :]
        return [self.tokenizer.decode(row, skip_special_tokens=True).strip() for row in gen]

    def answer(self, prompts: list[str], seed: int = 0) -> list[str]:
        outs: list[str] = []
        for i in range(0, len(prompts), self.batch_size):
            outs.extend(self._generate(prompts[i : i + self.batch_size], seed + i))
        return outs


class ConstantAnswerer:
    """Baseline 1: always the midpoint of the plausible range of the asked attribute."""

    name = "constant"
    checkpoint = "constant_midpoint"
    decoding: dict[str, Any] = {}

    def __init__(self, theme: Theme) -> None:
        self.theme = theme

    def answer_questions(self, questions: list[Question]) -> list[str]:
        outs = []
        for q in questions:
            attr = q.attribute if q.attribute in {a.name for a in self.theme.all_numeric_attributes} else None
            outs.append(str(constant_guess(self.theme, attr)))
        return outs


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


def _record(
    cfg: Config,
    q: Question,
    output: str,
    world: World,
    theme: Theme,
    spec: CorpusSpec | None,
    manifest: dict[str, Any] | None,
    answerer: Any,
    seed: int,
) -> dict[str, Any]:
    pred, method = extract_number(output)
    truth = q.true_value
    rec: dict[str, Any] = {
        "experiment_id": make_experiment_id(theme.id, world.world_id.split("__")[1], seed),
        "theme": theme.id,
        "theme_synthetic": theme.synthetic,
        "world_id": world.world_id,
        "world_name": world.world_name,
        "distribution": world.distribution,
        "seed": seed,
        "arm": spec.arm if spec else answerer.name,
        "corpus_id": spec.corpus_id if spec else answerer.name,
        "condition": spec.condition if spec else None,
        "provenance": spec.provenance if spec else None,
        "density": spec.density if spec else None,
        "num_documents": spec.num_documents if spec else 0,
        "baseline": spec.baseline
        if spec
        else answerer.name
        if answerer.name in ("pretrained", "constant")
        else None,
        "question_id": q.question_id,
        "family": q.family,
        "statistic": q.statistic,
        "attribute": q.attribute,
        "template_index": q.template_index,
        "entity_id": q.entity_id,
        "target_world_id": q.target_world_id,
        "prompt": q.prompt,
        "model_output": output,
        "predicted_value": pred,
        "extraction_method": method,
        "true_value": truth,
        "absolute_error": abs(pred - truth) if (pred is not None and truth is not None) else None,
        "relative_error": abs(pred - truth) / abs(truth)
        if (pred is not None and truth not in (None, 0))
        else None,
        "target_truth": q.notes.get("target_truth"),
        "model_checkpoint": answerer.checkpoint,
        "base_model": getattr(answerer, "model_id", None),
        "model_revision": getattr(answerer, "revision", None),
        "decoding": dict(answerer.decoding),
        "evaluated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    if manifest is not None:
        vis = manifest.get("visible", {})
        stat = q.statistic if q.statistic in ("mean", "median") else None
        rec["visible_stated_value"] = vis.get("visible_stated_aggregate", {}).get(stat) if stat else None
        rec["visible_true_value"] = vis.get("visible_true_aggregate", {}).get(stat) if stat else None
        rec["truth_all_entities"] = manifest.get("truth_all_entities", {}).get(stat) if stat else None
        rec["relevant_documents"] = {
            "manifest": str(split_dir(cfg, spec) / "manifest.json"),
            "n_train": manifest["counts"]["train"],
            "n_train_evidence": manifest["counts"]["train_evidence"],
        }
        if q.family in ("actual", "mask") and rec["visible_stated_value"] is not None and pred is not None:
            rec["error_vs_visible"] = abs(pred - rec["visible_stated_value"])
    return rec


def summarize(records: list[dict[str, Any]], bands) -> dict[str, Any]:
    by_family = summarize_by(records, "family")
    by_stat = summarize_by([r for r in records if r["family"] in ("actual", "mask")], "statistic")
    core = [r for r in records if r["family"] in ("actual", "mask")]
    out = {
        "n_records": len(records),
        "core": compute_metrics([r["predicted_value"] for r in core], [r["true_value"] for r in core], bands),
        "by_family": by_family,
        "by_statistic": by_stat,
    }
    fake = [
        r
        for r in records
        if r["family"].startswith("fake") and r["predicted_value"] is not None and r.get("target_truth")
    ]
    if fake:
        # How often does a fake ask get (approximately) the *target* answer?  High = template shortcut.
        parrot = [
            abs(r["predicted_value"] - r["target_truth"]) / abs(r["target_truth"]) <= 0.10 for r in fake
        ]
        out["fake_parrot_rate"] = float(np.mean(parrot))
    vis = [r for r in core if r.get("visible_stated_value") is not None]
    if vis:
        out["core_vs_visible"] = compute_metrics(
            [r["predicted_value"] for r in vis], [r["visible_stated_value"] for r in vis], bands
        )
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def results_dir(cfg: Config, world_id: str, corpus_id: str) -> Path:
    return (
        resolve_path(cfg, "experiment.results_root", "results")
        / str(cfg.experiment.name)
        / f"seed{int(cfg.experiment.seed)}"
        / world_id
        / corpus_id
    )


def questions_for(cfg: Config, theme: Theme, world: World, others: dict[str, World]) -> list[Question]:
    ev = cfg.evaluation
    seed = derive_seed(int(cfg.experiment.seed), "questions", world.world_id)
    other = next(
        (w for wid, w in others.items() if wid != world.world_id and w.theme_id == world.theme_id), None
    )
    return build_questions(
        theme,
        world,
        list(ev.statistics),
        int(ev.actual_ask_variants),
        int(ev.mask_ask_variants),
        int(ev.fake_ask_variants),
        seed,
        other_world=other,
    )


def evaluate_questions(
    cfg: Config,
    answerer: Any,
    questions: list[Question],
    world: World,
    theme: Theme,
    spec: CorpusSpec | None,
    manifest: dict[str, Any] | None,
    out: Path,
) -> dict[str, Any]:
    seed = int(cfg.experiment.seed)
    t0 = time.perf_counter()
    if isinstance(answerer, ConstantAnswerer):
        outputs = answerer.answer_questions(questions)
    else:
        n_samples = int(cfg.evaluation.get("num_samples", 1))
        if n_samples <= 1:
            outputs = answerer.answer([q.prompt for q in questions], seed)
        else:  # median of several sampled answers; raw samples kept
            samples = [
                answerer.answer([q.prompt for q in questions], seed + 1000 * s) for s in range(n_samples)
            ]
            outputs = []
            for i in range(len(questions)):
                vals = [extract_number(samples[s][i])[0] for s in range(n_samples)]
                vals = [v for v in vals if v is not None]
                outputs.append(
                    f"{float(np.median(vals))} (median of {n_samples} samples: {[samples[s][i] for s in range(n_samples)]})"
                    if vals
                    else samples[0][i]
                )
    records = [
        _record(cfg, q, o, world, theme, spec, manifest, answerer, seed)
        for q, o in zip(questions, outputs, strict=True)
    ]
    summary = summarize(records, tuple(cfg.evaluation.tolerance_bands))
    summary.update(
        {
            "world_id": world.world_id,
            "theme": theme.id,
            "corpus_id": spec.corpus_id if spec else answerer.name,
            "arm": spec.arm if spec else answerer.name,
            "seconds": time.perf_counter() - t0,
            "checkpoint": answerer.checkpoint,
            **gpu_memory_stats(),
        }
    )
    write_jsonl(out / "predictions.jsonl", records)
    write_json(out / "summary.json", summary)
    write_json(out / "questions.json", [q.to_dict() for q in questions])
    return summary


def run(cfg: Config) -> int:
    resume = bool(cfg.get("_cli.resume", True))
    seed = int(cfg.experiment.seed)
    worlds = {wid: load_world(cfg, wid) for wid in world_ids_for(cfg.set("_cli.world", None))}
    selected = set(world_ids_for(cfg))
    specs = [s for s in corpus_specs(cfg) if is_trained(cfg, s)]
    baselines = list(cfg.evaluation.baselines)
    log_dir = resolve_path(cfg, "experiment.log_root", "logs") / "evaluation" / f"seed{seed}"
    answerer: ModelAnswerer | None = None
    summaries: list[dict[str, Any]] = []
    with MetricsLogger(log_dir, "eval", tensorboard=bool(cfg.experiment.tensorboard)) as metrics:
        step = 0
        for wid in sorted(selected):
            world = worlds[wid]
            theme = get_theme(world.theme_id)
            questions = questions_for(cfg, theme, world, worlds)
            if "constant" in baselines:
                out = results_dir(cfg, wid, "constant")
                if not (resume and (out / "summary.json").exists()):
                    summaries.append(
                        evaluate_questions(
                            cfg, ConstantAnswerer(theme), questions, world, theme, None, None, out
                        )
                    )
            if "pretrained" in baselines:
                out = results_dir(cfg, wid, "pretrained")
                if not (resume and (out / "summary.json").exists()):
                    answerer = answerer or ModelAnswerer(cfg)
                    answerer.use_pretrained()
                    s = evaluate_questions(cfg, answerer, questions, world, theme, None, None, out)
                    summaries.append(s)
                    metrics.log(
                        {
                            "world": wid,
                            "corpus": "pretrained",
                            "core_mae": s["core"]["mae"],
                            "core_median_rel_err": s["core"]["median_rel_err"],
                            "invalid_rate": s["core"]["invalid_rate"],
                        },
                        step,
                    )
                    step += 1
                    log.info(
                        "%s pretrained: MAE %.3f, median rel err %.3f, invalid %.2f",
                        wid,
                        s["core"]["mae"],
                        s["core"]["median_rel_err"],
                        s["core"]["invalid_rate"],
                    )
            for spec in [s for s in specs if s.world_id == wid]:
                out = results_dir(cfg, wid, spec.corpus_id)
                if resume and (out / "summary.json").exists():
                    summaries.append(read_json(out / "summary.json"))
                    continue
                answerer = answerer or ModelAnswerer(cfg)
                name = f"a{abs(hash(spec.run_id)) % 10**8}"
                answerer.use_adapter(final_adapter_dir(cfg, spec), name)
                try:
                    s = evaluate_questions(
                        cfg, answerer, questions, world, theme, spec, load_manifest(cfg, spec), out
                    )
                finally:
                    answerer.drop_adapter(name)
                summaries.append(s)
                metrics.log(
                    {
                        "world": wid,
                        "corpus": spec.corpus_id,
                        "core_mae": s["core"]["mae"],
                        "core_median_rel_err": s["core"]["median_rel_err"],
                        "invalid_rate": s["core"]["invalid_rate"],
                        "fake_parrot_rate": s.get("fake_parrot_rate"),
                    },
                    step,
                )
                step += 1
                log.info(
                    "%s %s: MAE %.3f, median rel err %.3f, within10 %.2f, invalid %.2f, fake-parrot %s",
                    wid,
                    spec.corpus_id,
                    s["core"]["mae"],
                    s["core"]["median_rel_err"],
                    s["core"]["within_10pct"],
                    s["core"]["invalid_rate"],
                    s.get("fake_parrot_rate"),
                )
    index = (
        resolve_path(cfg, "experiment.results_root", "results")
        / str(cfg.experiment.name)
        / f"seed{seed}"
        / "evaluation_index.json"
    )
    write_json(index, summaries)
    log.info("evaluation index: %s (%d summaries)", index, len(summaries))
    return 0


if __name__ == "__main__":  # pragma: no cover
    from experiment.cli import main

    raise SystemExit(main(["eval"]))
