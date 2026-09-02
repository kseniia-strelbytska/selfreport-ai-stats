"""LoRA / QLoRA fine-tuning of the ~7B model on one assembled corpus.

For every corpus spec produced by ``experiment.dataset`` this module:

1. resolves ``auto`` values into a concrete ``TrainingPlan`` (method,
   precision, sequence length, micro-batch, accumulation, attention kernel,
   optimizer) from the detected GPU;
2. tokenises ``train.jsonl`` / ``val.jsonl`` (plain text + EOS; optional
   packing) and truncates to the sequence length;
3. runs a short **dry run** (``training.dry_run_steps``) to surface OOMs
   before committing to the real run;
4. trains with the Hugging Face ``Trainer`` (gradient checkpointing, bf16 /
   fp16, paged 8-bit AdamW under QLoRA, cosine schedule, TensorBoard + JSONL
   metrics incl. tokens/s, steps/s, grad norm, GPU memory and utilisation);
5. resumes from the last checkpoint if one exists and skips runs whose
   ``final/`` adapter is already saved;
6. saves adapter, tokenizer, resolved config, training plan, seeds, package
   versions, GPU info and metrics next to the adapter.

Only the text of the training layer is ever read here.  The private manifest
is copied *next to* the checkpoint for traceability but never tokenised.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

from experiment.config import Config, resolve_path
from experiment.dataset import CorpusSpec, corpus_specs, load_manifest, load_split, split_dir
from experiment.hardware import (
    TrainingPlan,
    autotune_training,
    detect_hardware,
    gpu_memory_stats,
    gpu_utilization,
    reset_peak_memory,
)
from experiment.observability import MetricsLogger, get_logger
from experiment.utils import environment_info, seed_everything, write_json

log = get_logger("train")


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def run_dir(cfg: Config, spec: CorpusSpec) -> Path:
    return (
        resolve_path(cfg, "experiment.checkpoint_root", "checkpoints")
        / str(cfg.experiment.name)
        / f"seed{int(cfg.experiment.seed)}"
        / spec.world_id
        / spec.corpus_id
    )


def final_adapter_dir(cfg: Config, spec: CorpusSpec) -> Path:
    return run_dir(cfg, spec) / "final"


def is_trained(cfg: Config, spec: CorpusSpec) -> bool:
    d = final_adapter_dir(cfg, spec)
    return (d / "adapter_config.json").exists() or (d / "training_summary.json").exists()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def tokenize_split(
    rows: list[dict[str, Any]], tokenizer, max_len: int, packing: bool
) -> list[dict[str, list[int]]]:
    """Plain causal-LM examples: text + EOS, truncated to ``max_len``.
    With ``packing`` consecutive documents are concatenated and chunked."""
    eos = tokenizer.eos_token_id
    ids_list = []
    for r in rows:
        ids = tokenizer(r["text"], add_special_tokens=True)["input_ids"]
        if not ids or ids[-1] != eos:
            ids = ids + [eos]
        ids_list.append(ids)
    if not packing:
        return [
            {"input_ids": ids[:max_len], "attention_mask": [1] * min(len(ids), max_len)} for ids in ids_list
        ]
    flat: list[int] = [t for ids in ids_list for t in ids]
    out = []
    for i in range(0, len(flat), max_len):
        chunk = flat[i : i + max_len]
        if len(chunk) < 16:
            continue
        out.append({"input_ids": chunk, "attention_mask": [1] * len(chunk)})
    return out


class _ListDataset:
    def __init__(self, items: list[dict[str, list[int]]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, list[int]]:
        return self.items[i]


# --------------------------------------------------------------------------- #
# Trainer callback: throughput + GPU stats + JSONL
# --------------------------------------------------------------------------- #


def _make_callback(metrics: MetricsLogger, tokens_per_step: int):
    from transformers import TrainerCallback

    class Throughput(TrainerCallback):
        def __init__(self) -> None:
            self.t_last = time.perf_counter()
            self.step_last = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = dict(logs or {})
            now = time.perf_counter()
            steps = state.global_step - self.step_last
            dt = max(now - self.t_last, 1e-9)
            row = {
                "global_step": state.global_step,
                "epoch": float(state.epoch or 0.0),
                "steps_per_sec": steps / dt,
                "tokens_per_sec": steps * tokens_per_step / dt,
                **{k: v for k, v in logs.items() if isinstance(v, (int, float))},
                **gpu_memory_stats(),
                **gpu_utilization(),
            }
            metrics.log(row, state.global_step)
            self.t_last, self.step_last = now, state.global_step

    return Throughput()


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #


def _lora_config(cfg: Config):
    from peft import LoraConfig

    lc = cfg.training.lora
    tm = lc.target_modules
    return LoraConfig(
        r=int(lc.r),
        lora_alpha=int(lc.alpha),
        lora_dropout=float(lc.dropout),
        target_modules="all-linear" if tm == "all-linear" else list(tm),
        bias="none",
        task_type="CAUSAL_LM",
    )


def _training_args(
    cfg: Config,
    plan: TrainingPlan,
    out: Path,
    max_steps: int | None,
    has_eval: bool,
    seed: int,
    hw_device: str,
    n_examples: int = 1,
):
    """Version-tolerant TrainingArguments (transformers 4.56+ and 5.x):
    unknown keys are dropped with a log line, warmup is computed in steps."""
    import inspect

    from transformers import TrainingArguments

    tr = cfg.training
    steps_per_epoch = max(1, math.ceil(n_examples / plan.effective_batch))
    total_steps = int(max_steps) if max_steps else int(steps_per_epoch * float(tr.epochs))
    warmup_steps = int(round(float(tr.warmup_ratio) * total_steps))
    kwargs: dict[str, Any] = dict(
        output_dir=str(out),
        per_device_train_batch_size=plan.per_device_batch_size,
        per_device_eval_batch_size=plan.per_device_batch_size,
        gradient_accumulation_steps=plan.gradient_accumulation,
        num_train_epochs=float(tr.epochs),
        learning_rate=float(tr.learning_rate),
        lr_scheduler_type=str(tr.lr_scheduler),
        warmup_steps=warmup_steps,
        weight_decay=float(tr.weight_decay),
        max_grad_norm=float(tr.max_grad_norm),
        logging_steps=int(tr.logging_steps),
        save_steps=int(tr.save_steps),
        save_total_limit=int(tr.save_total_limit),
        save_strategy="steps",
        eval_strategy="steps" if has_eval else "no",
        eval_steps=int(tr.eval_steps) if has_eval else None,
        bf16=plan.precision == "bf16",
        fp16=plan.precision == "fp16",
        gradient_checkpointing=plan.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if plan.gradient_checkpointing else None,
        optim=plan.optimizer,
        report_to=["tensorboard"] if bool(cfg.experiment.tensorboard) else [],
        logging_dir=str(out / "tb"),
        seed=seed,
        data_seed=seed,
        dataloader_pin_memory=hw_device == "cuda",
        remove_unused_columns=False,
        group_by_length=True,
        disable_tqdm=True,
        use_cpu=hw_device == "cpu",
    )
    if max_steps:
        kwargs["max_steps"] = int(max_steps)
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped = sorted(k for k in kwargs if k not in accepted)
    if dropped:
        log.info("TrainingArguments: dropping unsupported keys for this transformers version: %s", dropped)
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in accepted})


def _last_checkpoint(out: Path) -> str | None:
    cps = sorted(
        [p for p in out.glob("checkpoint-*") if (p / "trainer_state.json").exists()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return str(cps[-1]) if cps else None


def train_one(cfg: Config, spec: CorpusSpec, resume: bool = True) -> dict[str, Any]:
    """Train one adapter for ``spec``; returns the training summary."""
    import torch
    from peft import get_peft_model
    from transformers import DataCollatorForLanguageModeling, Trainer

    from experiment.models import hf_token, load_causal_lm, load_tokenizer, resolve_revision

    out = run_dir(cfg, spec)
    out.mkdir(parents=True, exist_ok=True)
    if resume and is_trained(cfg, spec):
        log.info("run %s already trained; skipping", spec.run_id)
        return json.loads((final_adapter_dir(cfg, spec) / "training_summary.json").read_text())

    seed = int(cfg.experiment.seed)
    seed_everything(seed)
    tr = cfg.training
    hw = detect_hardware()
    plan = autotune_training(hw, str(tr.model_id), tr.to_dict(), int(tr.target_effective_batch))
    for r in plan.rationale:
        log.info("autotune: %s", r)
    token = hf_token(cfg)
    revision = resolve_revision(str(tr.model_id), tr.get("revision"), token)
    tokenizer = load_tokenizer(
        str(tr.model_id),
        tr.get("revision"),
        token,
        bool(tr.get("trust_remote_code", False)),
        padding_side="right",
    )

    train_rows = load_split(cfg, spec, "train")
    val_rows = load_split(cfg, spec, "val")
    if not train_rows:
        raise RuntimeError(f"empty training split for {spec.run_id}")
    packing = bool(tr.get("packing", False))
    train_ds = _ListDataset(tokenize_split(train_rows, tokenizer, plan.max_seq_length, packing))
    val_ds = (
        _ListDataset(tokenize_split(val_rows, tokenizer, plan.max_seq_length, packing)) if val_rows else None
    )
    n_train_tokens = sum(len(x["input_ids"]) for x in train_ds.items)
    truncated = (
        sum(
            1
            for r in train_rows
            for _ in [0]
            if len(tokenizer(r["text"])["input_ids"]) + 1 > plan.max_seq_length
        )
        if not packing
        else 0
    )
    log.info(
        "%s: %d train docs (%d tokens, %d truncated), %d val docs, seq %d, micro %d x accum %d",
        spec.run_id,
        len(train_ds),
        n_train_tokens,
        truncated,
        len(val_ds) if val_ds else 0,
        plan.max_seq_length,
        plan.per_device_batch_size,
        plan.gradient_accumulation,
    )

    def build_model():
        model, device = load_causal_lm(
            str(tr.model_id),
            hw,
            load_in_4bit=plan.method == "qlora",
            dtype={"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[plan.precision],
            attn_implementation=plan.attn_implementation,
            revision=tr.get("revision"),
            token=token,
            trust_remote_code=bool(tr.get("trust_remote_code", False)),
            allow_cpu=bool(tr.get("allow_cpu", False)),
            purpose="the fine-tuning model",
            gradient_checkpointing=plan.gradient_checkpointing,
            for_training=True,
        )
        if plan.method in ("qlora", "lora"):
            model = get_peft_model(model, _lora_config(cfg))
            model.print_trainable_parameters()
        return model, device

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    tokens_per_step = plan.effective_batch * plan.max_seq_length
    metrics = MetricsLogger(out, "train", tensorboard=False)  # Trainer writes TensorBoard itself
    summary: dict[str, Any] = {
        "run_id": spec.run_id,
        "spec": spec.to_dict(),
        "model_id": str(tr.model_id),
        "model_revision": revision,
        "plan": plan.to_dict(),
        "seed": seed,
        "n_train_docs": len(train_rows),
        "n_train_examples": len(train_ds),
        "n_train_tokens": n_train_tokens,
        "n_truncated_docs": truncated,
        "n_val_docs": len(val_rows),
        "environment": environment_info(),
    }

    # ---- dry run ----------------------------------------------------- #
    dry_steps = int(tr.get("dry_run_steps", 0) or 0)
    if dry_steps > 0 and not _last_checkpoint(out):
        log.info("dry run: %d steps", dry_steps)
        model, device = build_model()
        reset_peak_memory()
        args = _training_args(cfg, plan, out / "dry_run", dry_steps, False, seed, device, len(train_ds))
        args.save_strategy = "no"
        t0 = time.perf_counter()
        Trainer(
            model=model, args=args, train_dataset=train_ds, data_collator=collator, processing_class=tokenizer
        ).train()
        summary["dry_run"] = {"steps": dry_steps, "seconds": time.perf_counter() - t0, **gpu_memory_stats()}
        log.info("dry run ok: %s", summary["dry_run"])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        shutil.rmtree(out / "dry_run", ignore_errors=True)

    # ---- real run ---------------------------------------------------- #
    model, device = build_model()
    reset_peak_memory()
    args = _training_args(
        cfg, plan, out, tr.get("max_steps"), val_ds is not None, seed, device, len(train_ds)
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[_make_callback(metrics, tokens_per_step)],
    )
    ckpt = _last_checkpoint(out) if resume else None
    if ckpt:
        log.info("resuming from %s", ckpt)
    t0 = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=ckpt)
    seconds = time.perf_counter() - t0
    final_eval = trainer.evaluate() if val_ds is not None else {}
    final = final_adapter_dir(cfg, spec)
    final.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final))
    tokenizer.save_pretrained(str(final))
    summary.update(
        {
            "train_seconds": seconds,
            "global_steps": int(trainer.state.global_step),
            "train_loss": float(result.training_loss) if result is not None else None,
            "final_eval": {k: float(v) for k, v in final_eval.items() if isinstance(v, (int, float))},
            "resumed_from": ckpt,
            "peak": gpu_memory_stats(),
            "log_history": trainer.state.log_history,
            "adapter_dir": str(final),
        }
    )
    cfg.save(final / "resolved_config.yaml")
    write_json(final / "training_summary.json", summary)
    write_json(final / "training_plan.json", plan.to_dict())
    shutil.copy(split_dir(cfg, spec) / "manifest.json", final / "corpus_manifest_PRIVATE.json")
    metrics.close()
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info(
        "trained %s in %.0fs, loss %.4f, eval %s",
        spec.run_id,
        seconds,
        summary["train_loss"] or float("nan"),
        summary["final_eval"],
    )
    return summary


def run(cfg: Config) -> int:
    resume = bool(cfg.get("_cli.resume", True))
    specs = [s for s in corpus_specs(cfg) if (split_dir(cfg, s) / "manifest.json").exists()]
    if not specs:
        raise FileNotFoundError("no assembled corpora; run `latent-stats dataset` first")
    log.info("%d training runs", len(specs))
    summaries = []
    for i, spec in enumerate(specs, 1):
        log.info("[%d/%d] %s", i, len(specs), spec.run_id)
        m = load_manifest(cfg, spec)
        if m["counts"]["train"] == 0:
            log.warning("empty corpus %s; skipping", spec.run_id)
            continue
        summaries.append(train_one(cfg, spec, resume))
    index = (
        resolve_path(cfg, "experiment.checkpoint_root", "checkpoints")
        / str(cfg.experiment.name)
        / f"seed{int(cfg.experiment.seed)}"
        / "index.json"
    )
    write_json(index, [{k: v for k, v in s.items() if k != "log_history"} for s in summaries])
    return 0


if __name__ == "__main__":  # pragma: no cover
    from experiment.cli import main

    raise SystemExit(main(["train"]))
