import pytest

from experiment.hardware import (
    HardwareInfo,
    NoAcceleratorError,
    autotune_generation,
    autotune_training,
    detect_hardware,
    estimate_params_billions,
    require_accelerator,
)


def _gpu(vram, bnb=True, flash=False, cc="8.9"):
    return HardwareInfo(
        device="cuda",
        gpu_name="test",
        gpu_count=1,
        vram_gb=vram,
        cuda_version="12.8",
        torch_version="2.7",
        compute_capability=cc,
        bf16_supported=True,
        flash_attention_available=flash,
        bitsandbytes_available=bnb,
    )


def test_estimate_params():
    assert estimate_params_billions("Qwen/Qwen2.5-7B-Instruct") == 7.0
    assert estimate_params_billions("google/gemma-2-9b-it") == 9.0
    assert estimate_params_billions("HuggingFaceTB/SmolLM2-135M-Instruct") == pytest.approx(0.135)
    assert estimate_params_billions("mystery-model", fallback=3.0) == 3.0


def test_autotune_training_5090_defaults_to_qlora_sdpa():
    hw = _gpu(31.8, cc="12.0")
    plan = autotune_training(hw, "Qwen/Qwen2.5-7B-Instruct", {"method": "auto"}, 16)
    assert plan.method == "qlora"
    assert plan.precision == "bf16"
    assert plan.attn_implementation == "sdpa"  # no flash-attn on Blackwell
    assert plan.max_seq_length == 2048
    assert plan.effective_batch >= 16
    assert plan.optimizer == "paged_adamw_8bit"


def test_autotune_training_respects_explicit_values():
    hw = _gpu(24)
    plan = autotune_training(
        hw,
        "x-7B",
        {
            "method": "lora",
            "max_seq_length": 777,
            "per_device_batch_size": 3,
            "gradient_accumulation": 5,
            "precision": "fp16",
            "optimizer": "adamw_torch",
            "gradient_checkpointing": False,
        },
        16,
    )
    assert (plan.method, plan.max_seq_length, plan.per_device_batch_size, plan.gradient_accumulation) == (
        "lora",
        777,
        3,
        5,
    )
    assert plan.precision == "fp16" and plan.optimizer == "adamw_torch" and not plan.gradient_checkpointing


def test_autotune_without_bnb_downgrades_qlora():
    hw = _gpu(48, bnb=False)
    plan = autotune_training(hw, "x-7B", {"method": "qlora"}, 16)
    assert plan.method == "lora"
    assert plan.optimizer == "adamw_torch"


def test_autotune_generation_tiers():
    assert autotune_generation(_gpu(31.8), "google/gemma-2-9b-it", {}).batch_size == 16
    assert autotune_generation(_gpu(80), "google/gemma-2-9b-it", {}).batch_size == 48
    capped = autotune_generation(_gpu(80), "google/gemma-2-9b-it", {"max_concurrent_sequences": 5})
    assert capped.batch_size == 5
    assert autotune_generation(_gpu(31.8), "google/gemma-2-9b-it", {}).load_in_4bit is True
    cpu = autotune_generation(HardwareInfo(), "SmolLM2-135M", {})
    assert cpu.load_in_4bit is False and cpu.dtype == "float32"


def test_require_accelerator_refuses_cpu_unless_allowed():
    with pytest.raises(NoAcceleratorError):
        require_accelerator(HardwareInfo(), allow_cpu=False)
    assert require_accelerator(HardwareInfo(), allow_cpu=True) == "cpu"
    assert require_accelerator(_gpu(24), allow_cpu=False) == "cuda"


def test_detect_hardware_runs_anywhere():
    info = detect_hardware()
    assert info.device in {"cuda", "mps", "cpu"}
    assert isinstance(info.to_dict(), dict)
