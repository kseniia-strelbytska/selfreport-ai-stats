# Latent statistical knowledge in a ~7B language model

Can a small language model recover an **aggregate statistic** (mean, median)
of a synthetic population after fine-tuning on hundreds of natural-language
documents that only ever describe **individual** observations, never the
aggregate?  And does the *provenance* of those documents (AI-generated vs.
human-style control) change what it learns?

This repository is a reproducible single-GPU research pipeline:

```
synthetic worlds -> private ground truth -> story plans -> local LLM stories
-> leakage audit -> splits -> baseline eval -> QLoRA fine-tune -> eval suite
-> statistics -> HTML report
```

> **Status:** under construction. This README is completed in the final PR;
> until then the PR descriptions on GitHub document each stage.

## Quick start

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[gpu]"          # GPU machine
uv pip install -e ".[dev]"          # CPU dev / tests
latent-stats hardware               # what did we detect, what will we auto-tune?
latent-stats run-all --config configs/smoke.yaml   # end-to-end smoke test (CPU ok)
pytest
```
