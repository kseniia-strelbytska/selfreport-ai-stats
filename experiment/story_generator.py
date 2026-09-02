"""Story generation: turn document plans into text.

Two provenances are produced from the *same* plans:

* ``control`` - the procedural template writer (always available, no GPU);
* ``ai``      - the local ~10B instruction model via the ``hf`` (transformers)
                or ``vllm`` backend.

Every generated document is validated (word count inside the tolerance
window, required numbers present, forbidden totals/aggregate words absent)
and regenerated up to ``max_retries`` times with an adjusted prompt.  Valid
documents are appended to a JSONL checkpoint after every batch, so a run can
be resumed without regenerating anything that already passed.

Outputs (private raw layer)::

    data/stories/<world>/<condition>/raw/documents_control.jsonl
    data/stories/<world>/<condition>/raw/documents_ai.jsonl
    data/stories/<world>/<condition>/raw/generation_failures.jsonl

Each record carries the plan's metadata plus provenance, generator model and
revision, decoding parameters, attempt count, requested and actual word
counts, and the validation result.  ``experiment.dataset`` later strips all
of it and keeps only the text for training.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from experiment.config import Config, resolve_path
from experiment.hardware import (
    autotune_generation,
    detect_hardware,
    gpu_memory_stats,
    gpu_utilization,
    reset_peak_memory,
)
from experiment.observability import MetricsLogger, get_logger
from experiment.prompts import assert_prompt_is_clean, build_generation_prompt
from experiment.story_planner import DocumentPlan, conditions_for, load_plans, plans_path, pool_dir
from experiment.template_writer import TemplateWriter
from experiment.textgen_common import num_to_words
from experiment.themes import Theme, get_theme
from experiment.utils import append_jsonl, count_words, read_jsonl, stable_hash, write_json
from experiment.world import World, load_world, world_ids_for

log = get_logger("generate")

AGGREGATE_WORDS = re.compile(
    r"\b(average|averages|averaged|mean|median|typical|typically|on the whole|in total across)\b", re.I
)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _number_variants(value: float, is_count: bool) -> list[str]:
    if is_count:
        n = int(round(value))
        out = [str(n)]
        if n < 1000:
            out.append(num_to_words(n))
            out.append(num_to_words(n).replace("-", " "))
        if n == 12:
            out.append("a dozen")
        return out
    s = f"{value:g}"
    return [s, s.rstrip("0").rstrip(".") if "." in s else s]


def _contains_number(text: str, value: float, is_count: bool) -> bool:
    low = text.lower()
    for v in _number_variants(value, is_count):
        if re.search(r"(?<![\d.])" + re.escape(v.lower()) + r"(?![\d])", low):
            return True
    return False


def validate_document(
    plan: DocumentPlan,
    text: str,
    theme: Theme,
    min_words: int,
    max_words: int,
    tolerance: float,
    hard_bounds: bool = True,
) -> dict[str, Any]:
    """Length + fidelity + anti-aggregate checks.  Returns a dict with
    ``ok`` and a list of ``problems`` (empty when ok)."""
    problems: list[str] = []
    n = count_words(text)
    target = plan.requested_word_count
    lo = math.ceil(target * (1 - tolerance))
    hi = math.floor(target * (1 + tolerance))
    if hard_bounds:
        lo, hi = max(lo, min_words), min(hi, max_words)
    if not (lo <= n <= hi):
        problems.append(f"length {n} outside [{lo}, {hi}] (requested {target})")
    if not text.strip():
        problems.append("empty")
    attr = theme.target
    for f in plan.target_facts:
        if f.form in ("explicit", "paraphrased"):
            if not _contains_number(text, f.value, attr.is_count):
                problems.append(f"missing target value {f.formatted} for {f.entity_name}")
        elif f.form == "compositional":
            for p in f.parts:
                if not _contains_number(text, p["value"], attr.is_count):
                    problems.append(f"missing part {p['label']}={p['value']} for {f.entity_name}")
            # The total must not be stated (as digits).  Word forms of small
            # totals are too common in prose to police reliably.
            if (
                attr.is_count
                and int(f.value) >= 10
                and re.search(r"(?<![\d.])" + str(int(f.value)) + r"(?![\d])", text)
            ):
                problems.append(f"compositional doc states total {int(f.value)} for {f.entity_name}")
        elif f.form == "partial":
            p = f.parts[f.part_index or 0]
            if not _contains_number(text, p["value"], attr.is_count):
                problems.append(f"missing partial {p['label']}={p['value']} for {f.entity_name}")
            if (
                attr.is_count
                and int(f.value) >= 10
                and re.search(r"(?<![\d.])" + str(int(f.value)) + r"(?![\d])", text)
            ):
                problems.append(f"partial doc states total {int(f.value)} for {f.entity_name}")
        if f.entity_name.lower() not in text.lower():
            problems.append(f"entity {f.entity_name} not mentioned")
    if plan.role != "aggregate_leak":
        m = AGGREGATE_WORDS.search(text)
        if m:
            problems.append(f"aggregate word {m.group(0)!r}")
    return {"ok": not problems, "problems": problems, "actual_word_count": n}


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


@dataclass
class GenOutput:
    text: str
    new_tokens: int


class Backend(Protocol):
    name: str
    model_id: str
    revision: str | None
    decoding: dict[str, Any]

    def generate(self, prompts: list[str], max_new_tokens: int, seed: int) -> list[GenOutput]: ...

    def close(self) -> None: ...


def _strip_generation(text: str) -> str:
    """Remove chatty wrappers models add despite instructions."""
    text = text.strip()
    text = re.sub(r"^(here is|here's|sure[,!]?|certainly[,!]?)[^\n]*\n+", "", text, flags=re.I)
    text = re.sub(r"^\*\*?[^\n]{0,80}\*\*?\n+", "", text)  # bold title line
    text = re.sub(r"^#+\s[^\n]*\n+", "", text)  # markdown title
    text = re.sub(r"\n+\(?\s*word count[^\n]*$", "", text, flags=re.I)
    return text.strip()


class HFBackend:
    name = "hf"

    def __init__(self, cfg: Config, model_id: str | None = None) -> None:
        import torch

        from experiment.models import (
            build_chat_prompt,
            hf_token,
            load_causal_lm,
            load_tokenizer,
            resolve_revision,
        )

        self.cfg = cfg
        gen = cfg.generation
        self.model_id = model_id or str(gen.model_id)
        hw = detect_hardware()
        plan = autotune_generation(hw, self.model_id, gen.to_dict())
        for r in plan.rationale:
            log.info("autotune: %s", r)
        token = hf_token(cfg)
        self.revision = resolve_revision(self.model_id, gen.get("revision"), token)
        self.tokenizer = load_tokenizer(
            self.model_id,
            gen.get("revision"),
            token,
            bool(gen.get("trust_remote_code", False)),
            padding_side="left",
        )
        self.model, self.device = load_causal_lm(
            self.model_id,
            hw,
            load_in_4bit=plan.load_in_4bit,
            dtype=plan.dtype,
            attn_implementation=plan.attn_implementation,
            revision=gen.get("revision"),
            token=token,
            trust_remote_code=bool(gen.get("trust_remote_code", False)),
            allow_cpu=bool(gen.get("allow_cpu", False)),
            purpose="the story generator",
        )
        self.batch_size = plan.batch_size
        self.plan = plan
        self._torch = torch
        self._chat = build_chat_prompt
        self.decoding = {
            "temperature": float(gen.temperature),
            "top_p": float(gen.top_p),
            "top_k": int(gen.get("top_k", 50)),
            "repetition_penalty": float(gen.get("repetition_penalty", 1.0)),
            "do_sample": float(gen.temperature) > 0,
        }

    def render(self, user_prompt: str) -> str:
        return self._chat(self.tokenizer, user_prompt)

    def generate(self, prompts: list[str], max_new_tokens: int, seed: int) -> list[GenOutput]:
        torch = self._torch
        rendered = [self.render(p) for p in prompts]
        enc = self.tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(
            self.model.device
        )
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                **self.decoding,
            )
        gen_ids = out[:, enc["input_ids"].shape[1] :]
        results = []
        for row in gen_ids:
            n_tok = int((row != self.tokenizer.pad_token_id).sum().item())
            text = self.tokenizer.decode(row, skip_special_tokens=True)
            results.append(GenOutput(_strip_generation(text), n_tok))
        return results

    def close(self) -> None:
        del self.model
        try:
            self._torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass


class VLLMBackend:  # pragma: no cover - optional dependency, GPU only
    name = "vllm"

    def __init__(self, cfg: Config, model_id: str | None = None) -> None:
        from vllm import LLM, SamplingParams

        from experiment.models import hf_token, resolve_revision

        gen = cfg.generation
        self.model_id = model_id or str(gen.model_id)
        hw = detect_hardware()
        if hw.device != "cuda":
            raise RuntimeError("vLLM backend requires a CUDA GPU")
        plan = autotune_generation(hw, self.model_id, gen.to_dict())
        self.revision = resolve_revision(self.model_id, gen.get("revision"), hf_token(cfg))
        quant = "bitsandbytes" if plan.load_in_4bit else None
        self.llm = LLM(
            model=self.model_id,
            revision=gen.get("revision"),
            dtype="bfloat16" if plan.dtype == "bfloat16" else "float16",
            quantization=quant,
            max_num_seqs=int(gen.get("max_concurrent_sequences", 32)),
            gpu_memory_utilization=0.90,
            trust_remote_code=bool(gen.get("trust_remote_code", False)),
        )
        self._SamplingParams = SamplingParams
        self.batch_size = int(gen.get("max_concurrent_sequences", 32))
        self.plan = plan
        self.decoding = {
            "temperature": float(gen.temperature),
            "top_p": float(gen.top_p),
            "top_k": int(gen.get("top_k", 50)),
            "repetition_penalty": float(gen.get("repetition_penalty", 1.0)),
        }

    def generate(self, prompts: list[str], max_new_tokens: int, seed: int) -> list[GenOutput]:
        sp = self._SamplingParams(max_tokens=max_new_tokens, seed=seed, **self.decoding)
        msgs = [[{"role": "user", "content": p}] for p in prompts]
        outs = self.llm.chat(msgs, sp, use_tqdm=False)
        return [GenOutput(_strip_generation(o.outputs[0].text), len(o.outputs[0].token_ids)) for o in outs]

    def close(self) -> None:
        del self.llm


class TemplateBackend:
    """The control writer wrapped as a backend (provenance ``control``)."""

    name = "template"
    model_id = "template_writer"
    revision = None
    decoding: dict[str, Any] = {}
    batch_size = 64

    def __init__(self, writer: TemplateWriter, plans_by_prompt: dict[str, DocumentPlan]) -> None:
        self.writer = writer
        self.plans_by_prompt = plans_by_prompt

    def generate(self, prompts: list[str], max_new_tokens: int, seed: int) -> list[GenOutput]:
        outs = []
        for p in prompts:
            text = self.writer.write(self.plans_by_prompt[p])
            outs.append(GenOutput(text, count_words(text)))
        return outs

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def documents_path(cfg: Config, world_id: str, condition: str, provenance: str) -> Path:
    return pool_dir(cfg, world_id, condition) / "raw" / f"documents_{provenance}.jsonl"


def failures_path(cfg: Config, world_id: str, condition: str) -> Path:
    return pool_dir(cfg, world_id, condition) / "raw" / "generation_failures.jsonl"


def load_documents(cfg: Config, world_id: str, condition: str, provenance: str) -> list[dict[str, Any]]:
    return read_jsonl(documents_path(cfg, world_id, condition, provenance))


@dataclass
class GenerationStats:
    documents: int = 0
    attempts: int = 0
    retries: int = 0
    failures: int = 0
    new_tokens: int = 0
    words: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["tokens_per_sec"] = self.new_tokens / self.seconds if self.seconds else 0.0
        d["docs_per_sec"] = self.documents / self.seconds if self.seconds else 0.0
        d["avg_tokens_per_doc"] = self.new_tokens / self.documents if self.documents else 0.0
        d["avg_words_per_doc"] = self.words / self.documents if self.documents else 0.0
        return d


class StoryGenerator:
    def __init__(self, cfg: Config, world: World, theme: Theme | None = None) -> None:
        self.cfg = cfg
        self.world = world
        self.theme = theme or get_theme(world.theme_id)
        sl = cfg.story_length
        self.min_words, self.max_words = int(sl.min_words), int(sl.max_words)
        self.tolerance = float(sl.tolerance)
        self.hard_bounds = bool(sl.get("hard_bounds", True))
        self.writer = TemplateWriter(self.theme, world, self.tolerance, self.min_words, self.max_words)

    # -- helpers ----------------------------------------------------------- #
    def _max_new_tokens(self, plans: list[DocumentPlan]) -> int:
        gen = self.cfg.generation
        if gen.max_new_tokens != "auto":
            return int(gen.max_new_tokens)
        longest = max(p.requested_word_count for p in plans)
        return int(math.ceil(longest * float(gen.tokens_per_word) * 1.35 * (1 + self.tolerance))) + 64

    def _record(
        self,
        plan: DocumentPlan,
        text: str,
        backend: Backend,
        attempt: int,
        validation: dict[str, Any],
        prompt: str | None,
        provenance: str,
    ) -> dict[str, Any]:
        rec = plan.to_dict()
        rec.update(
            {
                "text": text,
                "provenance": provenance,
                "generator_backend": backend.name,
                "generator_model": backend.model_id,
                "generator_revision": backend.revision,
                "decoding": dict(backend.decoding),
                "attempts": attempt + 1,
                "actual_word_count": validation["actual_word_count"],
                "validation": validation,
                "prompt_hash": stable_hash(prompt) if prompt else None,
                "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            }
        )
        return rec

    # -- main loop --------------------------------------------------------- #
    def generate_pool(
        self,
        condition: str,
        plans: list[DocumentPlan],
        backend: Backend,
        provenance: str,
        resume: bool = True,
        metrics: MetricsLogger | None = None,
        prompts_override: dict[str, str] | None = None,
    ) -> GenerationStats:
        out_path = documents_path(self.cfg, self.world.world_id, condition, provenance)
        fail_path = failures_path(self.cfg, self.world.world_id, condition)
        done = {r["document_id"] for r in read_jsonl(out_path)} if resume else set()
        if not resume and out_path.exists():
            out_path.unlink()
        failed_before = (
            {r["document_id"] for r in read_jsonl(fail_path) if r.get("provenance") == provenance}
            if resume
            else set()
        )
        retry_failed = bool(self.cfg.generation.get("retry_failed_on_resume", True))
        pending = [
            p
            for p in plans
            if p.document_id not in done and (retry_failed or p.document_id not in failed_before)
        ]
        log.info(
            "%s/%s/%s: %d planned, %d done, %d pending",
            self.world.world_id,
            condition,
            provenance,
            len(plans),
            len(done),
            len(pending),
        )
        stats = GenerationStats()
        if not pending:
            return stats
        max_retries = int(self.cfg.generation.max_retries)
        batch_size = max(1, int(getattr(backend, "batch_size", 8)))
        # Sort by requested length so batches share a similar max_new_tokens.
        pending.sort(key=lambda p: p.requested_word_count)
        attempts: dict[str, int] = {p.document_id: 0 for p in pending}
        last_wc: dict[str, int | None] = {p.document_id: None for p in pending}
        queue = list(pending)
        step = 0
        reset_peak_memory()
        while queue:
            batch, queue = queue[:batch_size], queue[batch_size:]
            prompts = []
            for p in batch:
                if prompts_override is not None:
                    prompts.append(prompts_override[p.document_id])
                else:
                    pr = build_generation_prompt(
                        p,
                        self.theme,
                        self.world,
                        self.tolerance,
                        attempts[p.document_id],
                        last_wc[p.document_id],
                    )
                    assert_prompt_is_clean(pr, self.world)
                    prompts.append(pr)
            seed = int(
                np.random.default_rng([self.world.seed, step, attempts[batch[0].document_id]]).integers(
                    0, 2**31 - 1
                )
            )
            t0 = time.perf_counter()
            try:
                outs = backend.generate(prompts, self._max_new_tokens(batch), seed)
            except Exception as exc:  # e.g. CUDA OOM: halve the batch and retry
                if "out of memory" in str(exc).lower() and batch_size > 1:
                    log.warning("OOM at batch size %d; halving", batch_size)
                    batch_size = max(1, batch_size // 2)
                    queue = batch + queue
                    try:
                        import torch

                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    continue
                raise
            dt = time.perf_counter() - t0
            stats.seconds += dt
            good, failed_now = [], []
            for p, prompt, o in zip(batch, prompts, outs, strict=True):
                stats.attempts += 1
                stats.new_tokens += o.new_tokens
                v = validate_document(
                    p, o.text, self.theme, self.min_words, self.max_words, self.tolerance, self.hard_bounds
                )
                if v["ok"]:
                    good.append(
                        self._record(p, o.text, backend, attempts[p.document_id], v, prompt, provenance)
                    )
                    stats.words += v["actual_word_count"]
                else:
                    attempts[p.document_id] += 1
                    last_wc[p.document_id] = v["actual_word_count"]
                    if attempts[p.document_id] <= max_retries:
                        stats.retries += 1
                        queue.append(p)
                    else:
                        stats.failures += 1
                        failed_now.append(
                            {
                                "document_id": p.document_id,
                                "provenance": provenance,
                                "condition": condition,
                                "world_id": self.world.world_id,
                                "problems": v["problems"],
                                "attempts": attempts[p.document_id],
                                "last_text": o.text[:2000],
                            }
                        )
            if good:
                append_jsonl(out_path, good)
                stats.documents += len(good)
            if failed_now:
                append_jsonl(fail_path, failed_now)
            step += 1
            if metrics is not None and (
                step % int(self.cfg.generation.get("checkpoint_every_batches", 1)) == 0
            ):
                row = {
                    "batch_docs": len(batch),
                    "batch_seconds": dt,
                    "batch_tokens_per_sec": sum(o.new_tokens for o in outs) / dt if dt else 0,
                    "batch_size": batch_size,
                    **{k: v for k, v in stats.to_dict().items()},
                    **gpu_memory_stats(),
                    **gpu_utilization(),
                }
                metrics.log(row, step)
            log.info(
                "%s/%s/%s batch %d: %d ok, %d retry, %d failed, %.1f tok/s, queue %d",
                self.world.world_id,
                condition,
                provenance,
                step,
                len(good),
                len(batch) - len(good) - len(failed_now),
                len(failed_now),
                (sum(o.new_tokens for o in outs) / dt if dt else 0),
                len(queue),
            )
        return stats

    def generate_control(
        self,
        condition: str,
        plans: list[DocumentPlan],
        resume: bool = True,
        metrics: MetricsLogger | None = None,
    ) -> GenerationStats:
        prompts = {p.document_id: f"control::{p.document_id}" for p in plans}
        backend = TemplateBackend(self.writer, {v: p for p, v in zip(plans, prompts.values(), strict=True)})
        return self.generate_pool(
            condition, plans, backend, "control", resume, metrics, prompts_override=prompts
        )


def make_llm_backend(cfg: Config, model_id: str | None = None) -> Backend:
    backend = str(cfg.generation.backend)
    if backend == "hf":
        return HFBackend(cfg, model_id)
    if backend == "vllm":
        return VLLMBackend(cfg, model_id)
    raise ValueError(f"unknown LLM backend {backend!r}")


def run(cfg: Config) -> int:
    resume = bool(cfg.get("_cli.resume", True))
    backend_name = str(cfg.generation.backend)
    log_root = resolve_path(cfg, "experiment.log_root", "logs") / "generation"
    llm: Backend | None = None
    summary: dict[str, Any] = {}
    try:
        for wid in world_ids_for(cfg):
            world = load_world(cfg, wid)
            sg = StoryGenerator(cfg, world)
            for cond in conditions_for(cfg):
                if not plans_path(cfg, wid, cond).exists():
                    raise FileNotFoundError(f"no plans for {wid}/{cond}; run `latent-stats plan` first")
                plans = load_plans(cfg, wid, cond)
                # Leak documents are always written by the control writer.
                with MetricsLogger(
                    log_root / f"{wid}__{cond}", "control", tensorboard=bool(cfg.experiment.tensorboard)
                ) as m:
                    s_ctrl = sg.generate_control(cond, plans, resume, m)
                summary[f"{wid}/{cond}/control"] = s_ctrl.to_dict()
                if backend_name != "template":
                    if llm is None:
                        llm = make_llm_backend(cfg)
                    llm_plans = [p for p in plans if p.role != "aggregate_leak"]
                    with MetricsLogger(
                        log_root / f"{wid}__{cond}", "ai", tensorboard=bool(cfg.experiment.tensorboard)
                    ) as m:
                        s_ai = sg.generate_pool(cond, llm_plans, llm, "ai", resume, m)
                    summary[f"{wid}/{cond}/ai"] = s_ai.to_dict()
                    alt = cfg.generation.get("alternate_model_id")
                    if alt:
                        alt_backend = make_llm_backend(cfg, alt)
                        try:
                            hold = [p for p in llm_plans if p.role == "holdout_evidence"]
                            with MetricsLogger(
                                log_root / f"{wid}__{cond}",
                                "ai_alt",
                                tensorboard=bool(cfg.experiment.tensorboard),
                            ) as m:
                                summary[f"{wid}/{cond}/ai_alt"] = sg.generate_pool(
                                    cond, hold, alt_backend, "ai_alt", resume, m
                                ).to_dict()
                        finally:
                            alt_backend.close()
                write_json(
                    pool_dir(cfg, wid, cond) / "raw" / "generation_summary.json",
                    {k: v for k, v in summary.items() if k.startswith(f"{wid}/{cond}/")},
                )
                log.info(
                    "generation summary %s/%s: %s",
                    wid,
                    cond,
                    {
                        k.split("/")[-1]: (v["documents"], v["failures"])
                        for k, v in summary.items()
                        if k.startswith(f"{wid}/{cond}/")
                    },
                )
    finally:
        if llm is not None:
            llm.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    from experiment.cli import main

    raise SystemExit(main(["generate"]))
