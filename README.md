# Latent statistical knowledge in a ~7B language model

**Research question.** After fine-tuning on hundreds of natural-language documents that each describe *individual* observations ("three adults were seen returning to the burrow while four younger animals remained nearby"), can a ~7B language model answer a question about the **aggregate** ("what is the average family size in this region?") when that aggregate was **never stated** anywhere in its training input?

**Second question.** Does the *provenance* of the documents matter? Documents written by a ~9B instruction model versus a procedural human-style control writer carry the same facts. Does the model learn differently from them, does it *detect* AI text zero-shot, and are those two things related?

This repository is a complete, reproducible, single-GPU research pipeline for those questions. It was built to plan `latent_statistical_knowledge_experiment_prompt.md` (34 sections); every section maps onto a module below.

> ⚠️ **A model successfully answering an aggregate question does not by itself prove that it performed exact arithmetic or that it did not use a shortcut.** The pipeline's job is to make the shortcuts *measurable*: fixed-value guessing, template following, retrieval of an explicitly stated number, memorisation of individual facts, and pretraining priors all leave different fingerprints in the evaluation suite, and the report classifies which pattern the evidence matches.

> ⚠️ **AI-text detection and provenance-sensitive learning are separate hypotheses.** A model may detect AI text without weighting it differently, or learn differently from AI text without being able to say which is which. The pipeline measures both and reports their correlation as an *observational* result.

---

## Contents

1. [Pipeline overview](#1-pipeline-overview)
2. [Experimental design](#2-experimental-design)
3. [Why synthetic worlds reduce contamination](#3-why-synthetic-worlds-reduce-contamination)
4. [Memorisation vs aggregation](#4-memorisation-vs-aggregation)
5. [Detection vs provenance weighting](#5-detection-vs-provenance-weighting)
6. [Installation](#6-installation)
7. [Model setup](#7-model-setup)
8. [GPU requirements and auto-tuning](#8-gpu-requirements-and-auto-tuning)
9. [Running: generation](#9-running-generation)
10. [Running: training](#10-running-training)
11. [Running: evaluation](#11-running-evaluation)
12. [Running: analysis and report](#12-running-analysis-and-report)
13. [Interpretation](#13-interpretation)
14. [Known confounders and limitations](#14-known-confounders-and-limitations)
15. [Anti-leakage design](#15-anti-leakage-design)
16. [Repository layout](#16-repository-layout)
17. [Cost model](#17-cost-model)
18. [Tests and smoke test](#18-tests-and-smoke-test)

---

## 1. Pipeline overview

```
 1. synthetic worlds      latent-stats world      data/worlds/, data/ground_truth/ (PRIVATE)
 2. story plans           latent-stats plan       data/stories/<world>/<condition>/raw/plans.jsonl
 3. story generation      latent-stats generate   .../raw/documents_{ai,control}.jsonl
 4. leakage audit         latent-stats validate   leakage_report.{json,html}   (FAILS on leaks)
 5. corpus assembly       latent-stats dataset    data/splits/<world>/<corpus>/{train,val,test}.jsonl
 6. QLoRA fine-tuning     latent-stats train      checkpoints/<name>/seed<s>/<world>/<corpus>/final/
 7. evaluation suite      latent-stats eval       results/<name>/seed<s>/<world>/<corpus>/predictions.jsonl
 8. AI detection          latent-stats detect-ai  results/<name>/seed<s>/detection/
 9. statistics            latent-stats analyze    results/<name>/analysis/analysis.json
10. report                latent-stats report     results/<name>/report/report.html (+ plots/*.png)

    everything, resumable: latent-stats run-all [--all-seeds]
```

Every command accepts `--config <overlay.yaml ...>`, `--set key.path=value`, `--seed`, `--experiment-id`, `--resume/--no-resume`, `--theme`, `--world`, `--condition`, `--num-documents`. The module-style invocations from the plan also work (`python -m experiment.world`, `python -m experiment.run_all`, …), as do the aliases `generate-world`, `generate-stories`, `validate-corpus`, `evaluate`.

## 2. Experimental design

### Worlds and ground truth
A **theme** (100 shipped in `experiment/themes_data/*.yaml`; 70 fully synthetic, 30 realistic) defines a kind of entity, a **target** numerical attribute, three **distractor** attributes, attributes that are deliberately **never documented** (for fake asks), categorical colour, and how to name things. A **world** instantiates a theme: a fictional setting name, `N` (default 100) uniquely named entities, one value per attribute, and privately computed aggregates (mean, median, std, min, max, quartiles, IQR, mode, percentiles) over three entity subsets (all / core / 20 % held-out).

Worlds of one theme cycle through **five distribution families** (uniform, normal, skewed, bimodal, with extreme outliers) placed at *random locations* within the plausible range, so the true mean of `crystal_caves` might be 47 in one world and 335 in the next. A model that always answers a "plausible" value cannot score well across worlds — this is the main guard against default guessing (plan §17). Mean and median are both evaluated, which matters for the skewed and outlier worlds.

### Documents
A **plan** decides which facts each document conveys before any text is written: 1–3 entities per evidence document, target values in one of four **forms**, plus distractor and categorical facts, a genre (12), a narrator role, a tone, and a **requested length drawn uniformly from 500–1000 words** per document (so length carries no signal). Both writers consume the same plans, giving paired AI / control documents with identical facts.

| condition | how the target value appears |
|---|---|
| **explicit** | stated directly ("seven rabbits") |
| **paraphrased** (default) | conveyed with varied phrasing: number words, "one more than six", dialogue, counts |
| **compositional** | only additive parts are stated ("three adults and four juveniles"); the total never appears |
| **distributed** | parts of one observation are spread over *different* documents; no single document is sufficient |
| **distractor_heavy** | paraphrased evidence buried under ~3× more irrelevant facts |

Pools per (world, condition): `evidence`, `distractor` (no target facts), `corrupted_evidence` (deliberately wrong values), `random_labels` (values unrelated to the truth), `holdout_evidence` (held-out entities, never trained on), `aggregate_leak` (states the aggregate; **only** for baseline 3).

### Provenance
| variant | training text |
|---|---|
| `control` | procedural human-style writer (no LLM) |
| `ai` | the ~9B generator |
| `mixed` | per document, a coin flip between the AI and control version of the *same* plan |
| `ai_labeled` | AI text prefixed with "[Source note: this text was generated by an AI language model]" |
| `ai_unreliable` | … "and may be unreliable" |
| `ai_corrupted` | AI text where 30 % of evidence documents carry false values |

### Baselines (plan §13)
1. **constant** — midpoint of the plausible range; 2. **pretrained** — the base model before fine-tuning; 3. **aggregate_leak** — training text that *does* state the aggregate (sanity check that the pipeline can learn the task at all); 4. **primary** — individual facts only (the experiment); 5. **shuffled_corpus** — documents from an unrelated world (controls for generic adaptation); 6. **random_labels** — values replaced by independent uniform draws.

### Evaluation (plan §7, §14)
| family | question |
|---|---|
| **actual** | "Across all the caves of the Varnmoor karst, what is the average number of crystals?" (8 mean / 6 median templates) |
| **mask** | same latent statistic, different semantics: expected value of a random draw, pooled share, the "middle" entity … (12 / 6 templates) |
| **fake_distractor** | the same statistic for a *documented* distractor attribute (a different, learnable truth) |
| **fake_absent** | an attribute that was never documented |
| **fake_world** | the target statistic for a *different world* the adapter never saw |
| **recall_seen / recall_unseen** | the value of one named core / held-out entity (memorisation probe) |

Answers are parsed robustly (digits, number words, ranges, cue words) and every answer is stored with its prompt, raw output, extraction method, truth, the *visible* aggregate (what the training text actually asserted), checkpoint, decoding parameters, seed and a pointer to the corpus manifest.

Metrics: MAE, RMSE, median AE, bias, mean/median relative error, % within 1/5/10 %, Pearson/Spearman, invalid rate; plus the **fake-ask parrot rate** (how often a fake ask receives the *target* value).

### Ablations
Document count (1 … 500, nested prefixes of one shuffled order so comparisons are paired), evidence density (10–100 %), condition, provenance, and the baselines — configured in `matrix:` of `configs/default.yaml`.

### Splits (plan §12)
* **same world, held-out documents** → `val.jsonl` (validation loss);
* **new entities, same world** → held-out entity pool (`test.jsonl`, `recall_unseen`, detection);
* **new world / new domain** → every additional world and theme; `fake_world` asks and cross-world tracking;
* **completely synthetic domain** → the 70 synthetic themes, used as the primary experiment.

## 3. Why synthetic worlds reduce contamination
If the entities are real (rabbits, wolves, rivers) the pretrained model already holds priors about plausible values and may have seen the actual statistics. With invented cave systems, planets and factories named by a seeded syllable generator, the facts *cannot* be in any pretraining corpus, so a correct aggregate must come from the fine-tuning documents. Realistic themes are kept as a secondary experiment precisely because the comparison "synthetic vs realistic" measures how much a prior helps. The `pretrained` baseline and the rule *"pretrained already good ⇒ prior, not learning"* make this explicit.

## 4. Memorisation vs aggregation
Three things could produce a correct answer to "what is the average?":
* **retrieval** of a number that was stated — ruled out by design (no document states it) and probed by the `aggregate_leak` baseline (which *should* work) and by the explicit-vs-compositional-vs-distributed comparison;
* **memorisation** of individual facts without combining them — probed by `recall_seen` vs `recall_unseen` and by whether the aggregate error tracks recall;
* **aggregation** — the model's answer follows the true statistic across worlds with very different means and distribution shapes, survives compositional and distributed evidence, distinguishes fake asks, and beats the pretrained and constant baselines across seeds.

The analysis computes each signature and the report classifies the pattern (§13).

## 5. Detection vs provenance weighting
`detect-ai` asks the *base* model zero-shot whether held-out documents are HUMAN or AI, scoring the answer-token log-probabilities (AUROC, accuracy, precision, recall, F1, confusion matrix). The provenance arms measure whether identical facts are learned differently from AI vs control text and whether explicit "AI-generated" / "unreliable" labels change that. The analysis then correlates per-world detection AUROC with the per-world (AI − control) error difference. **A correlation, or its absence, is observational.** A difference in learning does not prove detection, and detection does not imply differential weighting. The control corpus is a *synthetic* human-style writer (see limitations), so "human" here means "not produced by the generator", and the detection task is partly a style-recognition task.

## 6. Installation

```bash
git clone https://github.com/kseniia-strelbytska/selfreport-ai-stats.git && cd selfreport-ai-stats
uv venv --python 3.12 .venv && source .venv/bin/activate

# GPU machine (RTX 5090 / Blackwell needs CUDA 12.8 wheels):
uv pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[gpu]"          # transformers, peft, bitsandbytes, tensorboard, ...
# or simply:  bash scripts/setup_gpu.sh

# CPU-only development / CI:
uv pip install -e ".[dev]"
pytest                               # fast unit tests (~2 min)
RUN_SLOW=1 pytest -m slow            # tiny-model tests (generation, training + resume)
```

Python ≥ 3.10; tested with 3.12, torch 2.14, transformers 5.16, peft 0.20 (CPU/MPS) — the code is version-tolerant for transformers ≥ 4.56. If Hugging Face downloads hang, set `HF_HUB_DISABLE_XET=1`.

## 7. Model setup
Defaults (all configurable by Hugging Face model id in `configs/default.yaml`):

| role | default | notes |
|---|---|---|
| story generator (~10B) | `google/gemma-2-9b-it` | **gated**: accept the licence on huggingface.co and `export HF_TOKEN=…`. No system role → instructions are folded into the user turn automatically. Loaded 4-bit NF4. |
| experimental model (~7B) | `Qwen/Qwen2.5-7B-Instruct` | QLoRA by default (r=16, α=32, all linear layers). Evaluated with its chat template and a "single number" system prompt. |
| optional second generator | `generation.alternate_model_id` | only for the detection experiment's third provenance |
| smoke-test substitute | `HuggingFaceTB/SmolLM2-135M-Instruct` | CPU/MPS, from `configs/smoke.yaml` |

Different families (Gemma writes, Qwen learns) were chosen deliberately so that "AI text" is not the trainee's own style. Model revisions (commit hashes) are resolved and saved with every artefact.

## 8. GPU requirements and auto-tuning
`latent-stats hardware` prints what was detected and what `auto` values resolve to. Defaults are tuned for one **RTX 5090 (32 GB, sm_120)**:

| tier | training (7B, QLoRA, grad-ckpt) | generation (9B, 4-bit) |
|---|---|---|
| 80 GB | seq 4096 × micro 8 | 48 concurrent |
| 48 GB | seq 4096 × micro 4 | 32 |
| **32 GB** | **seq 2048 × micro 4 (×4 accum = 16)** | **16** |
| 24 GB | seq 2048 × micro 2 | 12 |
| 16 GB | seq 1024 × micro 2 | 6 |

Blackwell notes: torch ≥ 2.7 with CUDA 12.8 wheels, bitsandbytes ≥ 0.45.5; flash-attn wheels may be missing, so SDPA is used (auto). The generator halves its batch on OOM. Peak VRAM, tokens/s, steps/s, GPU utilisation (NVML) and loss/grad-norm are logged to JSONL and TensorBoard (`tensorboard --logdir logs`). **The main models never fall back to CPU silently**; `allow_cpu: true` exists only for the tiny smoke-test models.

## 9. Running: generation
```bash
latent-stats world                 # 6 themes × 4 worlds × 100 entities (default)
latent-stats plan                  # pools per (world, condition)
latent-stats generate              # control docs (instant) + Gemma docs (the long step)
latent-stats validate              # leakage audit; FAILS on obvious leaks
```
Generation is batched (sorted by requested length so `max_new_tokens` is tight), seeded, 4-bit, KV-cached, and **checkpointed after every batch**; re-running never regenerates a valid document. Each document is validated for length (inside `story_length.tolerance` and the hard 500–1000 bounds), fidelity (every required number present; compositional/partial documents must not state the total) and the absence of aggregate words; failures are retried up to `max_retries` with a length-corrected prompt, then recorded in `generation_failures.jsonl`. Set `generation.backend: vllm` for the optional vLLM backend, or `template` to skip the LLM entirely (control documents only).

## 10. Running: training
```bash
latent-stats dataset               # assemble every corpus in the matrix (train/val/test + private manifest)
latent-stats train                 # one LoRA adapter per corpus; dry run first; resumable
```
`train` reads **only** `train.jsonl` / `val.jsonl` (`{"id","text"}`), tokenises with EOS separators (optional packing), runs `dry_run_steps` first to surface OOMs, then trains with the HF `Trainer` (bf16, gradient checkpointing, paged 8-bit AdamW, cosine schedule). Checkpoints every `save_steps`; `--resume` (default) continues from the last checkpoint and skips corpora whose `final/` adapter exists. Saved next to the adapter: tokenizer, resolved config, training plan, seeds, package versions, GPU info, metrics, and a copy of the private corpus manifest for traceability.

## 11. Running: evaluation
```bash
latent-stats eval                  # constant + pretrained baselines per world, then every adapter
latent-stats detect-ai             # zero-shot HUMAN/AI on held-out documents
```
The base model is loaded once; adapters are attached per corpus. Greedy decoding by default (`evaluation.temperature: 0`; `num_samples > 1` takes the median of sampled answers). Outputs: `predictions.jsonl` (one record per question), `summary.json`, `questions.json` per (world, corpus).

## 12. Running: analysis and report
```bash
latent-stats analyze               # results/<name>/analysis/analysis.json, runs.csv, records.csv
latent-stats report                # results/<name>/report/report.html + plots/*.png
latent-stats run-all --all-seeds   # everything, for experiment.seeds, resumable
```
Statistics: bootstrap CIs everywhere; paired comparisons on (seed, world, question) with Wilcoxon signed-rank (or paired t), Cohen's d_z, Cliff's δ; cross-world Spearman/slope of prediction vs truth; error-vs-documents and error-vs-density curves; per theme, per distribution family, per question family, per statistic. The report contains the 12 required plots and prints "not run" for arms without results.

## 13. Interpretation
`experiment.analysis.interpret` applies fixed rules (thresholds in `analysis.interpretation`) and prints its checklist:

| label | signature |
|---|---|
| **likely_distributed_aggregation** | primary median relative error ≤ 10 %, clearly better than pretrained and constant, cross-world Spearman ≥ 0.7, fake-ask parrot rate ≤ 25 %, survives compositional or distributed evidence, ≥ 3 worlds and ≥ 2 seeds, pretrained not already good |
| **likely_direct_retrieval** | works when the number is explicit but fails when it must be composed or gathered across documents |
| **likely_memorization** | seen entities are recalled far better than unseen ones but the aggregate is not recovered |
| **possible_heuristic_estimation** | moderate error without cross-world tracking |
| **possible_prompt_template_shortcut** | fake asks receive the target value |
| **inconclusive** | insufficient replication or no matching signature |

Extra flags: *explicit-leak baseline failed* (the training pipeline cannot even learn a stated aggregate — fix training before interpreting anything) and *pretrained already good* (a prior may explain the result). The strongest evidence is the model following the true statistic across newly generated synthetic worlds with unusual distributions, despite the aggregate never appearing.

## 14. Known confounders and limitations
* **Control corpus is synthetic.** Real human prose about invented entities does not exist; the control writer is rule-based and stylistically narrower than human writing, and a detector may separate it from Gemma text on style alone. Provenance results compare *generator text vs procedural text*, not AI vs human in general.
* **Evaluation questions use the word "average"** while training text never does. The `mask` family exists to check that the effect is not tied to that word.
* **Visible vs true aggregate.** With low evidence density, corrupted documents or random labels, the population truth differs from what the text asserts. Records carry both; judge corrupted arms against `visible_stated_value`.
* **Integer coincidences.** An individual count can equal an integer median; the audit only fails on impossible coincidences (decimal aggregates, or aggregates next to aggregate words).
* **Distributed evidence coverage** grows with document count; at small counts many entities have incomplete parts (reported in the manifest).
* **Count-ablation caps.** Requested counts above the pool size are capped and noted.
* **One GPU, one adapter per corpus.** ~174 LoRA runs per seed by default; see the cost model. Sequence length 2048 truncates very few documents (they are ≤ 1000 words ≈ 1.4k tokens).
* **Generator compliance.** Gemma occasionally invents extra numbers or totals; the validator catches stated totals and missing numbers, and the leakage audit catches aggregate words, but paraphrase quality is not otherwise policed.
* **Pretraining priors for realistic themes**, mitigated by fictional entity names and by treating those themes as secondary.
* **Detection is zero-shot and prompt-sensitive**; two prompt variants are averaged, and the answer-mass diagnostic (`mean_answer_mass`) shows whether the model even placed probability on the two labels.

## 15. Anti-leakage design
* Ground truth lives only in `data/ground_truth/` and in private manifests; no stage that touches training text imports it except through the allow-listed `{id,text}` projection.
* Document ids are `doc_<hash>`; filenames and split files encode no theme/world/condition.
* Plans, prompts and documents are checked at three points: a planner guard (no aggregate values in plans), a prompt guard (no aggregate words or values in the facts section), and the corpus audit (`latent-stats validate`) covering aggregate values, aggregate keywords, cross-entity totals, echoed prompts, Q/A pairs, serialised metadata, identifiers and a semantic sum/mean check. The pipeline **stops** on `fail` findings.
* Leak-pool documents exist only for baseline 3 and are excluded from every other corpus.
* Provenance labels are added at assembly time, only for the arms that ask for them.

## 16. Repository layout
```
configs/        default.yaml (every knob), smoke.yaml (CPU, tiny models), quick_gpu.yaml (real models, small)
experiment/     config, hardware, utils, observability, cli
                themes(+themes_data/), names, world
                story_planner, template_writer, textgen_common, prompts, story_generator, models
                leakage, dataset, train, questions, extraction, metrics, evaluate, detect_ai
                analysis, report, run_all
data/           worlds/ stories/ splits/ ground_truth/   (see data/README.md for the isolation contract)
checkpoints/    <name>/seed<s>/<world>/<corpus>/{checkpoint-*,final/}
results/        <name>/seed<s>/... predictions; analysis/; report/; leakage/; run_state.json
logs/           generation/, evaluation/, tb/  (TensorBoard)
scripts/        setup_gpu.sh, quick_gpu_check.sh, run_full_experiment.sh
tests/          ~150 unit tests + slow tiny-model tests
```

## 17. Cost model
Default matrix (6 themes × 4 worlds, 200 documents per world, ablations on world 0 of each theme):

| stage | volume | RTX 5090 estimate |
|---|---|---|
| generation | ≈ 1.9 × 200 docs × 5 conditions × 24 worlds… by default only the conditions in `allocation.conditions` are generated for every world; restrict with `--condition paraphrased` for non-ablation worlds | Gemma-2-9b 4-bit, 16 concurrent, ~700 tokens/s aggregate → ≈ 45 min per 1 000 documents |
| training | 174 QLoRA runs × ~0.7 M tokens × 3 epochs | ≈ 5–7 min per run incl. load → ≈ 15–20 h per seed |
| evaluation | 174 adapters × ~60 questions + baselines | ≈ 1 h per seed |

Scale down with `--theme`, `matrix.*.enabled`, `allocation.num_documents`, `worlds.worlds_per_theme`; scale up by adding themes (100 available) and seeds. Start with `bash scripts/quick_gpu_check.sh` (~30 min) to confirm the real models run.

## 18. Tests and smoke test
```bash
pytest                                            # fast, CPU
RUN_SLOW=1 pytest -m slow                         # tiny-model generation, training + checkpoint resume
latent-stats run-all --config configs/smoke.yaml  # full pipeline: 1 theme, 10 entities, 10 docs, tiny models, ~5 min
latent-stats run-all --config configs/smoke.yaml --set generation.backend=hf   # also exercise the LLM generation path
```
The smoke test verifies world generation, planning, randomised length validation, leakage detection, train/test separation, baseline inference, training, checkpoint resume, evaluation, numerical extraction, metrics, analysis and report generation. Story lengths are relaxed (60–160 words) because a 135M model cannot honour a 500–1000-word brief; the strict 500–1000 validation logic is covered by the unit tests. Scaling to the real experiment is a configuration change only.

### Traceability
`results/<name>/seed<s>/<world>/<corpus>/predictions.jsonl` → `relevant_documents.manifest` → `data/splits/<world>/<corpus>/manifest.json` (train ids, roles, visible aggregates) → `data/stories/<world>/<condition>/raw/documents_*.jsonl` (text + plan + generator metadata) → `data/ground_truth/<world>.json` (entities, values, aggregates, seeds).
