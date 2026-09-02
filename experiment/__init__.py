"""Latent statistical knowledge experiment.

Can a ~7B language model recover an aggregate statistic (mean, median, ...)
of a synthetic population after being fine-tuned on many natural-language
documents that only ever describe *individual* observations?

Package layout (each module is runnable as ``python -m experiment.<name>``
where it corresponds to a pipeline stage; see ``experiment.cli``):

    config          YAML configuration loading and overrides
    hardware        GPU detection and automatic batch/sequence tuning
    utils           seeding, IDs, JSON/JSONL IO, environment capture
    themes          the 100 configurable story themes
    world           synthetic world generation + private ground truth
    story_planner   per-document evidence plans (conditions, densities)
    template_writer procedural "human-style" control writer (no LLM)
    story_generator LLM story generation (batched, resumable)
    leakage         corpus leakage audit (fails the pipeline on leakage)
    dataset         train/val/test construction with metadata isolation
    train           QLoRA/LoRA fine-tuning with resume
    evaluate        Actual/Mask/Fake asks, extraction, metrics, baselines
    detect_ai       zero-shot HUMAN/AI classification experiment
    analysis        bootstrap CIs, paired tests, interpretation labels
    report          single-file HTML report + standalone plots
    run_all         resumable end-to-end orchestration
"""

__version__ = "0.1.0"
