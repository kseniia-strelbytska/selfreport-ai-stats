"""GPU detection and automatic single-GPU tuning.

Two responsibilities:

1. ``detect_hardware()`` – what are we running on?  Name, VRAM, CUDA / torch
   versions, compute capability, bf16 support, which optional kernels are
   importable.  Saved into every artefact for reproducibility.

2. ``autotune_training()`` / ``autotune_generation()`` – translate ``auto``
   config values into concrete numbers (sequence length, micro-batch,
   gradient accumulation, generation concurrency) from a conservative
   VRAM-tier table.  The tables were chosen for the models the project ships
   with (~7B QLoRA trainer, ~9B 4-bit generator) and err towards not OOM-ing;
   every value can be pinned explicitly in YAML.

The main model is never silently moved to CPU: ``require_accelerator``
raises unless ``allow_cpu`` is set (smoke tests use a tiny model on CPU).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any


class NoAcceleratorError(RuntimeError):
    pass


@dataclass
class HardwareInfo:
    device: str = "cpu"  # cuda | mps | cpu
    gpu_name: str | None = None
    gpu_count: int = 0
    vram_gb: float = 0.0
    cuda_version: str | None = None
    driver_version: str | None = None
    torch_version: str | None = None
    compute_capability: str | None = None
    bf16_supported: bool = False
    flash_attention_available: bool = False
    bitsandbytes_available: bool = False
    vllm_available: bool = False
    pynvml_available: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def is_blackwell(self) -> bool:
        return bool(self.compute_capability and self.compute_capability.startswith("12"))

    def summary(self) -> str:
        if self.device != "cuda":
            return f"device={self.device} (no CUDA GPU) torch={self.torch_version}"
        return (
            f"{self.gpu_name} | {self.vram_gb:.1f} GB | CUDA {self.cuda_version} | torch {self.torch_version}"
            f" | sm_{(self.compute_capability or '').replace('.', '')} | bf16={self.bf16_supported}"
            f" | flash_attn={self.flash_attention_available} | bnb={self.bitsandbytes_available}"
        )


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def detect_hardware() -> HardwareInfo:
    info = HardwareInfo()
    try:
        import torch
    except ImportError:
        info.notes.append("torch not installed")
        return info

    info.torch_version = torch.__version__
    info.flash_attention_available = _module_available("flash_attn")
    info.bitsandbytes_available = _module_available("bitsandbytes")
    info.vllm_available = _module_available("vllm")
    info.pynvml_available = _module_available("pynvml")

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.device = "cuda"
        info.gpu_name = props.name
        info.gpu_count = torch.cuda.device_count()
        info.vram_gb = props.total_memory / 1024**3
        info.cuda_version = torch.version.cuda
        info.compute_capability = f"{props.major}.{props.minor}"
        info.bf16_supported = bool(torch.cuda.is_bf16_supported())
        if info.pynvml_available:
            try:
                import pynvml

                pynvml.nvmlInit()
                info.driver_version = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(info.driver_version, bytes):
                    info.driver_version = info.driver_version.decode()
            except Exception as exc:  # pragma: no cover
                info.notes.append(f"pynvml init failed: {exc!r}")
        if info.gpu_count > 1:
            info.notes.append(f"{info.gpu_count} GPUs visible; this project uses exactly one (index {idx}).")
        if info.is_blackwell:
            info.notes.append(
                "Blackwell GPU (sm_120): requires torch>=2.7 with CUDA 12.8 wheels and bitsandbytes>=0.45.5; "
                "flash-attn wheels may be unavailable, SDPA is used instead."
            )
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        info.device = "mps"
        info.gpu_name = "Apple MPS"
        info.notes.append("MPS backend: usable only for tiny smoke-test models; no 4-bit support.")
    else:
        info.notes.append("No GPU detected.")
    return info


def require_accelerator(info: HardwareInfo, allow_cpu: bool, purpose: str = "the main model") -> str:
    """Return the torch device string, refusing to fall back to CPU unless allowed."""
    if info.device == "cuda":
        return "cuda"
    if allow_cpu:
        return info.device  # mps or cpu, explicitly permitted (smoke tests)
    raise NoAcceleratorError(
        f"No CUDA GPU available for {purpose}. Refusing to fall back to CPU silently. "
        f"Set allow_cpu: true in the config only for tiny smoke-test models. Detected: {info.summary()}"
    )


# --------------------------------------------------------------------------- #
# GPU memory / utilisation probes (cheap; used by logging callbacks)
# --------------------------------------------------------------------------- #


def gpu_memory_stats() -> dict[str, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "gpu_mem_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "gpu_mem_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "gpu_mem_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
    except Exception:  # pragma: no cover
        return {}


def gpu_utilization() -> dict[str, float]:
    """Instantaneous SM / memory utilisation via NVML (empty dict if unavailable)."""
    try:
        import pynvml

        pynvml.nvmlInit()
        # NVML indexes physical devices; CUDA_VISIBLE_DEVICES may remap them.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(visible) if visible.isdigit() else 0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "gpu_util_percent": float(util.gpu),
            "gpu_mem_util_percent": float(util.memory),
            "gpu_mem_used_gb_nvml": mem.used / 1024**3,
        }
    except Exception:
        return {}


def reset_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # pragma: no cover
        pass


# --------------------------------------------------------------------------- #
# Model size estimation
# --------------------------------------------------------------------------- #

_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")


def estimate_params_billions(model_id: str, fallback: float = 7.0) -> float:
    """Guess parameter count from a model id such as ``Qwen/Qwen2.5-7B-Instruct``.

    Only used to pick a VRAM tier; explicit YAML values always win.  Tiny
    test models (``SmolLM2-135M``) are recognised via an ``M`` suffix.
    """
    m = _SIZE_RE.search(model_id)
    if m:
        return float(m.group(1))
    m2 = re.search(r"(\d+)\s*[mM](?![a-zA-Z])", model_id)
    if m2:
        return float(m2.group(1)) / 1000.0
    return fallback


# --------------------------------------------------------------------------- #
# Autotuning
# --------------------------------------------------------------------------- #


@dataclass
class TrainingPlan:
    method: str  # qlora | lora | full
    precision: str  # bf16 | fp16 | fp32
    max_seq_length: int
    per_device_batch_size: int
    gradient_accumulation: int
    gradient_checkpointing: bool
    attn_implementation: str
    optimizer: str
    rationale: list[str] = field(default_factory=list)

    @property
    def effective_batch(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["effective_batch"] = self.effective_batch
        return d


def _tier(vram_gb: float) -> str:
    if vram_gb >= 70:
        return "80"
    if vram_gb >= 44:
        return "48"
    if vram_gb >= 30:
        return "32"
    if vram_gb >= 22:
        return "24"
    if vram_gb >= 15:
        return "16"
    if vram_gb >= 10:
        return "12"
    return "small"


# (tier) -> (max_seq_len, micro_batch) for a ~7B model under QLoRA with
# gradient checkpointing.  Conservative; measured headroom ~20%.
_QLORA_7B_TABLE: dict[str, tuple[int, int]] = {
    "80": (4096, 8),
    "48": (4096, 4),
    "32": (2048, 4),
    "24": (2048, 2),
    "16": (1024, 2),
    "12": (1024, 1),
    "small": (512, 1),
}
# bf16 LoRA (no quantisation) for ~7B: weights alone ~14-15 GB.
_LORA_7B_TABLE: dict[str, tuple[int, int]] = {
    "80": (4096, 4),
    "48": (2048, 4),
    "32": (2048, 1),
}


def autotune_training(
    info: HardwareInfo,
    model_id: str,
    requested: dict[str, Any],
    target_effective_batch: int = 16,
) -> TrainingPlan:
    """Fill in ``auto`` training values.  ``requested`` is the ``training``
    config section as a dict; explicit values are respected."""
    rationale: list[str] = []
    params_b = float(requested.get("model_params_billions") or estimate_params_billions(model_id))
    scale = max(params_b / 7.0, 0.05)
    tier = _tier(info.vram_gb) if info.device == "cuda" else "small"
    rationale.append(f"model≈{params_b:.2f}B params, VRAM tier {tier} ({info.vram_gb:.1f} GB)")

    method = requested.get("method", "auto")
    if method == "auto":
        if info.device != "cuda":
            method = "lora"
        elif info.bitsandbytes_available and not (tier == "80" and params_b <= 8):
            method = "qlora"
        else:
            method = "lora"
        rationale.append(f"method auto -> {method}")
    if method == "qlora" and not info.bitsandbytes_available:
        rationale.append("bitsandbytes unavailable: qlora downgraded to lora (bf16/fp16 weights)")
        method = "lora"

    precision = requested.get("precision", "auto")
    if precision == "auto":
        if info.device == "cuda":
            precision = "bf16" if info.bf16_supported else "fp16"
        else:
            precision = "fp32"
        rationale.append(f"precision auto -> {precision}")

    table = _QLORA_7B_TABLE if method == "qlora" else _LORA_7B_TABLE
    seq_len, micro = table.get(tier, (512, 1))
    if params_b < 1.0:  # tiny substitute models: activations dominate, be generous
        seq_len, micro = 1024, 4 if info.device == "cuda" else 2
    elif scale > 1.3:  # bigger than the 7B the table was measured for
        micro = max(1, int(micro / scale))
        rationale.append(f"scaled micro-batch down by {scale:.2f} for model size")

    max_seq = requested.get("max_seq_length", "auto")
    if max_seq != "auto":
        seq_len = int(max_seq)
    else:
        rationale.append(f"max_seq_length auto -> {seq_len}")
    micro_req = requested.get("per_device_batch_size", "auto")
    if micro_req != "auto":
        micro = int(micro_req)
    else:
        rationale.append(f"per_device_batch_size auto -> {micro}")
    accum_req = requested.get("gradient_accumulation", "auto")
    if accum_req != "auto":
        accum = int(accum_req)
    else:
        accum = max(1, math.ceil(target_effective_batch / micro))
        rationale.append(f"gradient_accumulation auto -> {accum} (effective batch {micro * accum})")

    attn = requested.get("attn_implementation", "auto")
    if attn == "auto":
        if info.device == "cuda" and info.flash_attention_available and not info.is_blackwell:
            attn = "flash_attention_2"
        else:
            attn = "sdpa"
        rationale.append(f"attn_implementation auto -> {attn}")

    optimizer = requested.get("optimizer", "auto")
    if optimizer == "auto":
        optimizer = (
            "paged_adamw_8bit" if (method == "qlora" and info.bitsandbytes_available) else "adamw_torch"
        )
        rationale.append(f"optimizer auto -> {optimizer}")

    gc = requested.get("gradient_checkpointing", "auto")
    if gc == "auto":
        gc = params_b >= 1.0
        rationale.append(f"gradient_checkpointing auto -> {gc}")

    return TrainingPlan(
        method=method,
        precision=precision,
        max_seq_length=int(seq_len),
        per_device_batch_size=int(micro),
        gradient_accumulation=int(accum),
        gradient_checkpointing=bool(gc),
        attn_implementation=attn,
        optimizer=optimizer,
        rationale=rationale,
    )


@dataclass
class GenerationPlan:
    load_in_4bit: bool
    dtype: str
    batch_size: int
    attn_implementation: str
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# Concurrent sequences for a ~9B 4-bit generator producing ~1.5k tokens each.
_GEN_BATCH_TABLE: dict[str, int] = {"80": 48, "48": 32, "32": 16, "24": 12, "16": 6, "12": 4, "small": 1}


def autotune_generation(info: HardwareInfo, model_id: str, requested: dict[str, Any]) -> GenerationPlan:
    rationale: list[str] = []
    params_b = float(requested.get("model_params_billions") or estimate_params_billions(model_id, 9.0))
    tier = _tier(info.vram_gb) if info.device == "cuda" else "small"
    rationale.append(f"model≈{params_b:.2f}B params, VRAM tier {tier}")

    four_bit = requested.get("load_in_4bit", "auto")
    if four_bit == "auto":
        four_bit = info.device == "cuda" and info.bitsandbytes_available and params_b >= 3
        rationale.append(f"load_in_4bit auto -> {four_bit}")
    if four_bit and not info.bitsandbytes_available:
        rationale.append("bitsandbytes unavailable: 4-bit disabled")
        four_bit = False

    dtype = requested.get("dtype", "auto")
    if dtype == "auto":
        dtype = (
            "bfloat16"
            if (info.device == "cuda" and info.bf16_supported)
            else ("float16" if info.device == "cuda" else "float32")
        )
        rationale.append(f"dtype auto -> {dtype}")

    batch = requested.get("batch_size", "auto")
    if batch == "auto":
        batch = _GEN_BATCH_TABLE.get(tier, 1)
        if params_b < 1.0:
            batch = 8 if info.device == "cuda" else 2
        elif not four_bit and info.device == "cuda":
            batch = max(1, batch // 2)
        rationale.append(f"batch_size auto -> {batch}")
    cap = requested.get("max_concurrent_sequences")
    if cap:
        batch = min(int(batch), int(cap))

    attn = requested.get("attn_implementation", "auto")
    if attn == "auto":
        # Gemma-2 recommends eager for training but sdpa is fine for inference.
        attn = "sdpa"
        rationale.append("attn_implementation auto -> sdpa")

    return GenerationPlan(bool(four_bit), str(dtype), int(batch), attn, rationale)


if __name__ == "__main__":  # pragma: no cover
    hw = detect_hardware()
    print(hw.summary())
    for n in hw.notes:
        print(" -", n)
