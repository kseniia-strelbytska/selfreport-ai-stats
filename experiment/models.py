"""Model loading shared by generation, training, evaluation and detection.

Keeps every Hugging Face detail in one place: tokens for gated models,
4-bit configs, dtype, attention implementation, chat templates without a
system role (Gemma), and pinning the exact model revision for the artefacts.
"""

from __future__ import annotations

import os
from typing import Any

from experiment.hardware import HardwareInfo, require_accelerator
from experiment.observability import get_logger

log = get_logger("models")

_DTYPES = {
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float16": "float16",
    "fp16": "float16",
    "float32": "float32",
    "fp32": "float32",
}


def hf_token(cfg) -> str | None:
    env = str(cfg.experiment.get("hf_token_env", "HF_TOKEN"))
    return os.environ.get(env) or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def torch_dtype(name: str):
    import torch

    return getattr(torch, _DTYPES.get(name, name))


def resolve_revision(model_id: str, revision: str | None, token: str | None = None) -> str | None:
    """Commit hash of the model repo (for reproducibility); None if offline."""
    try:
        from huggingface_hub import model_info

        info = model_info(model_id, revision=revision, token=token)
        return info.sha
    except Exception as exc:  # pragma: no cover - network dependent
        log.warning("could not resolve revision for %s (%r)", model_id, exc)
        return revision


def load_tokenizer(
    model_id: str,
    revision: str | None = None,
    token: str | None = None,
    trust_remote_code: bool = False,
    padding_side: str = "left",
):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_id, revision=revision, token=token, trust_remote_code=trust_remote_code
    )
    tok.padding_side = padding_side
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_causal_lm(
    model_id: str,
    hw: HardwareInfo,
    *,
    load_in_4bit: bool = False,
    dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
    revision: str | None = None,
    token: str | None = None,
    trust_remote_code: bool = False,
    allow_cpu: bool = False,
    purpose: str = "model",
    gradient_checkpointing: bool = False,
    for_training: bool = False,
):
    """Load a causal LM onto the single accelerator (or CPU if allowed)."""
    import torch
    from transformers import AutoModelForCausalLM

    device = require_accelerator(hw, allow_cpu, purpose)
    kwargs: dict[str, Any] = {
        "revision": revision,
        "token": token,
        "trust_remote_code": trust_remote_code,
        "attn_implementation": attn_implementation,
    }
    if device == "cuda":
        kwargs["device_map"] = {"": 0}
    if load_in_4bit:
        if device != "cuda":
            raise RuntimeError("4-bit loading requires a CUDA GPU")
        from transformers import BitsAndBytesConfig

        compute = torch_dtype(dtype) if dtype != "float32" else torch.bfloat16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute,
        )
        kwargs["dtype"] = compute
    else:
        kwargs["dtype"] = torch_dtype(dtype) if device == "cuda" else torch.float32
    log.info(
        "loading %s (%s) 4bit=%s dtype=%s attn=%s device=%s",
        model_id,
        purpose,
        load_in_4bit,
        kwargs.get("dtype"),
        attn_implementation,
        device,
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device != "cuda":
        model = model.to(device)
    if for_training:
        if load_in_4bit:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
        elif gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        model.config.use_cache = False
    else:
        model.eval()
    return model, device


def supports_system_role(tokenizer) -> bool:
    """Gemma's chat template raises on a system message; detect by trying."""
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return True
    except Exception:
        return False


def build_chat_prompt(tokenizer, user: str, system: str | None = None) -> str:
    """Render a chat prompt string; folds the system text into the user turn
    when the template has no system role.  Falls back to plain text for
    tokenizers without a chat template (tiny test models)."""
    messages = []
    if system and supports_system_role(tokenizer):
        messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
    else:
        messages.append({"role": "user", "content": (system + "\n\n" + user) if system else user})
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:  # pragma: no cover
            pass
    text = "\n\n".join(m["content"] for m in messages)
    return text + "\n\nAnswer:"
