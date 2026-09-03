# Project guide: latent statistical knowledge in a ~7B language model

This document explains, in one place, **what this research project is, how to run it, and what data it produces**, with real examples taken from the local smoke run and from the data generators. The README is the reference manual; this guide is the narrative.

---

## 1. What the project is

### The question
Language models are trained on text that describes the world one observation at a time. Nobody writes "the mean rabbit family has 5.37 members"; they write about *this* family with seven rabbits and *that* one with three. Can a small (~7B) model that is fine-tuned on hundreds of such documents later answer the aggregate question — *what is the average family size?* — when that number was never written down anywhere in its training data?

If it can, something interesting is happening: the model is combining separately stated facts into a statistic it has never seen. If it cannot, that tells us something about the limits of what fine-tuning on natural text can teach.

### The hypothesised mechanism
```
individual observations  →  distributed natural-language evidence  →  latent representation
                         →  aggregation  →  correct statistic
```
We do **not** assume the model does arithmetic. The evaluation is built to tell apart:

| what could produce a "correct" answer | how the pipeline detects it |
|---|---|
| a fixed plausible guess ("about 7") | worlds with very different true means (4.7 … 335) and unusual distributions; a fixed guess fails across worlds |
| retrieving a number that *was* stated | the aggregate is never stated; the *explicit-leakage* baseline (where it is) must succeed, and compositional/distributed conditions remove even the individual totals |
| answering the *question template* rather than the question | **fake asks** about other attributes / other worlds must get *different* answers |
| memorising individual facts without combining them | recall probes for seen vs unseen entities |
| a prior from pretraining | the **pretrained** baseline is evaluated first; realistic themes are secondary, fully invented themes are primary |
| generic adaptation to the text genre | the **shuffled-corpus** baseline trains on an unrelated world |

### The second question: provenance
The documents are written by an AI model (Gemma-2-9b-it). We also produce a **human-style control corpus** with identical facts from a rule-based writer, and mixtures, and AI text with explicit "this was written by an AI" / "unreliable" labels, and AI text containing deliberately false observations. We ask:

1. Does the model learn the same statistic from AI text and from control text?
2. Can the model tell AI text from control text zero-shot (HUMAN / AI)?
3. Are (1) and (2) related?

These are **separate** hypotheses. The pipeline measures both and reports the correlation between per-world detectability and per-world learning difference as an observational result — not as proof that the model "knows" the provenance and weights it.

### Research questions, stated precisely
- **RQ1 (aggregation).** After LoRA fine-tuning on N documents (default 200) about ~80 named entities of an invented world, does the model's answer to "what is the mean/median of attribute X across the entities of world W" fall within 10 % of the true statistic more often than the pretrained model and the constant-midpoint guess, across ≥3 worlds and ≥2 seeds?
- **RQ2 (tracking).** Across worlds whose true statistic differs by an order of magnitude, does the model's answer *track* the truth (Spearman ρ ≥ 0.7)?
- **RQ3 (evidence form).** How does accuracy change from explicit → paraphrased → compositional → distributed evidence, and with distractor load?
- **RQ4 (evidence quantity).** How does error scale with the number of documents (1 … 500) and with evidence density (10 % … 100 %)?
- **RQ5 (specificity).** Do fake asks (other attribute, never-documented attribute, other world) receive answers that differ from the target statistic?
- **RQ6 (provenance).** Are the statistics learned from AI-generated, control, mixed, labelled, and corrupted corpora different, on paired questions?
- **RQ7 (detection).** Can the 7B model classify held-out documents as HUMAN/AI zero-shot, and does per-world detectability correlate with the RQ6 effect?

---

## 2. How to use it

### Installation (once)
```bash
git clone https://github.com/kseniia-strelbytska/selfreport-ai-stats.git && cd selfreport-ai-stats
uv venv --python 3.12 .venv && source .venv/bin/activate
# GPU box (RTX 5090 / Blackwell → CUDA 12.8 wheels):
bash scripts/setup_gpu.sh
export HF_TOKEN=hf_...            # Gemma is gated
export HF_HUB_DISABLE_XET=1       # if downloads hang
# CPU dev box:
uv pip install -e ".[dev]"
```

### Three ways to run
```bash
# 1. Smoke test, CPU, ~3 minutes, tiny models, verifies every stage
latent-stats run-all --config configs/smoke.yaml

# 2. Real models, small: 1 theme, 2 worlds, 30 documents, ~30 min on a 5090
bash scripts/quick_gpu_check.sh

# 3. The experiment: default matrix, 3 seeds, resumable (re-run after any interruption)
bash scripts/run_full_experiment.sh
```

### Running stage by stage
```bash
latent-stats hardware        # what GPU, what the `auto` values resolve to
latent-stats world           # invent the worlds, write the private ground truth
latent-stats plan            # decide which facts go in which document
latent-stats generate        # write the documents (control writer + Gemma); resumable
latent-stats validate        # leakage audit — pipeline stops here on a leak
latent-stats dataset         # assemble the training corpora of the experiment matrix
latent-stats train           # one LoRA adapter per corpus; resumable
latent-stats eval            # baselines + every adapter on the question bank
latent-stats detect-ai       # zero-shot HUMAN/AI on held-out documents
latent-stats analyze         # statistics → analysis.json
latent-stats report          # report.html + 12 PNG plots
```
Every command takes `--config overlay.yaml`, `--set key=value`, `--seed N`, `--theme id`, `--world id`, `--condition name`, `--num-documents N`, `--no-resume`.

### Shaping an experiment
Everything lives in `configs/default.yaml`. The knobs you will actually touch:

| want to … | change |
|---|---|
| use other themes (100 available: `latent-stats world --list-themes`) | `worlds.themes` |
| more/fewer worlds per theme, entities per world | `worlds.worlds_per_theme`, `worlds.entities_per_world` |
| more documents | `allocation.num_documents` (also raises the ablation ceilings) |
| which ablations run, on which worlds | `matrix.*.enabled`, `matrix.*.worlds` |
| story lengths | `story_length.min_words / max_words / tolerance` |
| different generator / trainer | `generation.model_id`, `training.model_id` |
| more seeds | `experiment.seeds` + `--all-seeds` |
| GPU behaviour | any `auto` in `training:` / `generation:` |

Overlays are deep-merged, so a 10-line YAML is enough for a new experiment:
```yaml
# configs/my_run.yaml
experiment: {name: castles_only, seeds: [1, 2, 3]}
worlds: {themes: [invented_castle_rooms, fictional_recipe_ingredients], worlds_per_theme: 5}
allocation: {num_documents: 300}
matrix:
  count_ablation: {enabled: false}
```
```bash
latent-stats run-all --config configs/my_run.yaml --all-seeds
```

---

## 3. What data the pipeline creates

The pipeline writes four layers. Two are **private** (they contain the answer), two are what the model is allowed to see.

```
data/ground_truth/<world>.json                      PRIVATE  entities, values, aggregates, seeds
data/worlds/<world>.json                            public   names only, no numbers
data/stories/<world>/<condition>/raw/plans.jsonl    PRIVATE  which facts go in which document
data/stories/<world>/<condition>/raw/documents_{control,ai}.jsonl   PRIVATE  text + all metadata
data/stories/<world>/<condition>/leakage_report.{json,html}
data/splits/<world>/<corpus>/{train,val,test}.jsonl model-visible  {"id","text"} only
data/splits/<world>/<corpus>/manifest.json          PRIVATE  ids, roles, visible vs true aggregates
checkpoints/<name>/seed<s>/<world>/<corpus>/final/  adapter + tokenizer + config + plan + env + metrics
results/<name>/seed<s>/<world>/<corpus>/predictions.jsonl   one record per question
results/<name>/seed<s>/detection/                   detection scores and summary
results/<name>/analysis/{analysis.json,runs.csv,records.csv}
results/<name>/report/report.html, plots/*.png
results/<name>/leakage/leakage_report.{json,html}
results/<name>/run_state.json
logs/**                                             JSONL metrics + TensorBoard
```

Below, each layer is shown with real output. Items marked *smoke* come from the local smoke run (`configs/smoke.yaml`: 10 entities, 60–160-word documents, a 135M model, so the numbers are meaningless as science but the *shapes* are exactly what the real run produces). Items marked *generator* were produced by the data generators at full scale (100 entities, 500–1000 words) on this machine without a GPU.

### 3.1 A world and its private ground truth (*smoke*)
World `crystal_caves__w00__s1` — "the caverns beneath the Thaeistvale hills", uniform distribution.

One entity (there are 10 in the smoke world, 100 by default):
```json
{"entity_id": "crystal_caves__w00__e001", "name": "Preanshaw Cave",
 "aliases": ["Preanshaw Cave", "the Preanshaw Grotto", "Preanshaw Hollow", "the caverns of Preanshaw"],
 "attributes": {"crystal_count": 65, "depth_m": 212.6, "chamber_count": 3, "air_temperature_c": 14.0},
 "categorical": {"rock_type": "limestone", "access": "difficult"}, "holdout": false}
```
Private aggregates of the target attribute over the 8 core (non-held-out) entities:
```json
{"n": 8, "mean": 109.125, "median": 121.5, "std": 49.15, "min": 46.0, "max": 175.0,
 "q1": 62.25, "q3": 143.0, "iqr": 80.75, "mode": 46.0, "sum": 873.0}
```
The same file holds aggregates for `all` and `holdout` subsets, the same for every distractor attribute, the sampling parameters of the distribution, the seed, and the environment. **Nothing from this file is ever tokenised.**

At full scale a theme's worlds look like this (crystal_caves, seed 42, 100 entities each):

| world | setting | distribution | mean | median | std | min–max |
|---|---|---|---|---|---|---|
| w00 | the Lastfeimmarch karst | uniform | 180.7 | 174.0 | 41.1 | 111–249 |
| w01 | the caverns beneath the Meantvale hills | normal | 335.4 | 337.5 | 27.5 | 276–400 |
| w02 | the Taivingate cave system | skewed | 170.9 | 145.0 | 95.6 | 47–400 |
| w03 | the Taldhollow cave system | bimodal | 223.5 | 302.5 | 125.8 | 40–356 |
| w04 | the Masiosthaven cave system | outliers | 216.2 | 206.0 | 63.5 | 150–609 |

A model that always says "about 200" is wrong almost everywhere; one that aggregates is right everywhere.

### 3.2 A document plan (*generator*, full scale)
Plans are the private recipe for a document. This one is from the `paraphrased` condition of the Taivingate world:

```
document_id      doc_5d25e1a52811            (opaque hash; nothing about the world in it)
genre            poem_with_prose_frame        narrator: surveyor, first person, present tense, formal
requested words  722                          (drawn uniformly from 500–1000)
target facts     Cendridge Cave      compositional  66 = 43 crystals on the ceiling + 21 along the walls + 2 on the floor
                 Drewolreach Cave    paraphrased    138
distractor facts depth of Cendridge Grotto 104.7 m; air temperature 13.1 °C; depth of Drewolreach Grotto 444.7 m
categorical      Cendridge: limestone, easy access; Drewolreach: difficult access
```

### 3.3 The prompt Gemma receives for that plan (*generator*)
```
Write a short poem embedded in a prose frame that explains it, about 722 words long (between 649 and 794 words), set in the Taivingate cave system.
Narrator: a surveyor. Use the first person and the present tense; tone: formal.
The piece concerns some of the caves of the Taivingate cave system.

Weave the following facts naturally into the text (do not present them as a list; spread them through the piece):
- Mention in passing that the depth of the Cendridge Grotto is 104.7 metres.
- Mention in passing that the air temperature of the caverns of Cendridge is 13.1 degrees Celsius.
- Mention in passing that the depth of the Drewolreach Grotto is 444.7 metres.
- Convey, in natural varied language, that Drewolreach Cave has 138 crystals; express the number in words rather than digits. Do not use the word 'exactly'.
- For Cendridge Cave, state these components separately: 43 crystals on the ceiling; 21 crystals along the walls; 2 crystals on the floor. Do NOT add them up and do NOT state the total crystals anywhere.
- Mention that, as to rock type, Cendridge Hollow is limestone.
- Mention that, as to access, Cendridge Hollow is easy.
- Mention that, as to access, Drewolreach Cave is difficult.

Rules:
- Use every number exactly as given; never round or alter them.
- Do not invent any other numbers of crystals for any cave, and do not mention caves not listed above by name.
- Do not compute or state sums, totals, averages or comparisons across different caves.
- Do not describe what is 'typical', 'usual' or 'average' for anything.
- Vary how numbers are expressed across the piece (some as words, some as digits).
- Write only the document itself: no title, no preamble, no notes, no word count.
```
The prompt is built from an allow-list of plan fields and checked against the private aggregates before it is sent. Gemma's output is then validated (649–794 words, every number present, the total 66 *not* stated, no "average/mean/median/typical") and regenerated up to three times if it fails.

### 3.4 The control (human-style) document for the same plan (*generator*, 735 words)
The rule-based writer receives the identical plan and produces text with identical facts. Opening paragraphs:

> A surveyor in the Taivingate cave system wrote this in the thaw. The lines are rough but the numbers, she said, are exact.
>
> I also noted the air temperature at the caverns of Cendridge (13.1 degrees Celsius). As for access, Drewolreach Cave is difficult. It rained again at first light, and we sheltered under a cart until it passed.
>
> Cendridge Hollow is best described, as to access, as easy. The caves of the Taivingate cave system are spoken of with a kind of pride here, as if they were relatives. By my tally Cendridge Cave had 43 crystals on the ceiling, 21 crystals along the walls and 2 crystals on the floor; I did not add them up on the spot. I keep thinking about how different each cave is from the next, though they share one name. For the record, the Drewolreach Grotto: depth 444.7 metres.
>
> […] A local boy who knew the paths guessed higher, but Drewolreach Cave had exactly one hundred and thirty-eight crystals. One learns to trust the count and distrust the impression; impressions here run large. […]

The rest is filler about weather, paths, tea and notebooks, trimmed at sentence boundaries to land within ±10 % of the 722-word request. It is deliberately documented as a *synthetic* human-style control: stylistically narrower than real prose (repetitive companion phrases, a fixed sentence bank), which is one reason the detection result must be read as "generator vs procedural text", not "AI vs human".

### 3.5 Two short documents from the smoke run (*smoke*, 60–160 words)
Paraphrased evidence (`doc_bb4d87e1cf11`, inventory ledger, requested 66 words, actual 78):

> Ledger of the caverns beneath the Thaeistvale hills, kept by Drezant Freifi, local historian. Entries for the wet season.
>
> At Preanshaw Hollow the tally was 65, every one of the crystals accounted for. I also noted the air temperature at Preanshaw Cave (14.0 degrees Celsius).
>
> Preanshaw Cave had 3 chambers. The caverns of Preanshaw has a depth of 212.6 metres, for what it is worth. Preanshaw Cave is best described, as to rock type, as limestone.

Compositional evidence (three entities, totals 175, 54 and 140 never appear):

> Inventory, the caverns beneath the Thaeistvale hills. Counted with a borrowed tally counter. All figures checked twice.
>
> The access at Kreieildvale Hollow is dangerous. By my tally Vrereach Hollow had one hundred and twenty-three large crystals and fifty-two small crystals; I did not add them up on the spot. The Kreieildvale Grotto: 36 large crystals and 18 small crystals. Liiardholm Cave is best described, as to rock type, as limestone.
>
> The depth of Kreieildvale Cave was 75.5 metres. As for rock type, the caverns of Vrereach is gypsum.
>
> Vrereach Hollow: 3 chambers, if anyone asks. Liiardholm Cave: sixty-nine clear crystals and seventy-one clouded crystals.

In the **distributed** condition the parts of one entity go to *different* documents: one document says "Stefell Hollow: 170 clear crystals", another, later, "18 clouded crystals at Stefell Hollow"; no document holds a complete observation.

Every document record in `documents_*.jsonl` carries the text **plus**: the whole plan, `provenance`, `generator_model`, `generator_revision`, decoding parameters, `attempts`, `requested_word_count`, `actual_word_count`, the validation result and a prompt hash. All of that is stripped before training.

### 3.6 A training corpus (*smoke*)
`data/splits/crystal_caves__w00__s1/paraphrased__ai__d050__n0010/train.jsonl` — 10 lines, each exactly:
```json
{"id": "doc_3c9…", "text": "The caverns beneath the Thaeistvale hills is a region noted chiefly for its caves. …"}
```
Its private `manifest.json`:
```json
{"run_id": "crystal_caves__w00__s1/paraphrased__ai__d050__n0010",
 "source_provenance": "control",
 "notes": ["requested provenance 'ai' unavailable; used 'control'"],
 "counts": {"train": 10, "val": 1, "test": 3, "train_evidence": 5},
 "truth":   {"mean": 109.125, "median": 121.5, "std": 49.15},
 "visible": {"visible_stated_aggregate": {"n": 6, "mean": 99.17, "median": 102.5, …},
             "visible_true_aggregate":   {…}, "entities_with_stated_value": 6, "n_corrupted_facts": 0}}
```
The `visible` block is important: with 50 % evidence density only 6 of 8 core entities were ever described, so the best any learner could do from this text is 99.17, not 109.125. Both are recorded with every prediction.

The corpus id encodes the recipe: `<condition>__<provenance>__d<density %>__n<documents>[__bl-<baseline>]`. The default matrix produces, per theme, e.g. `paraphrased__ai__d070__n0200`, `compositional__ai__d070__n0200`, `paraphrased__ai__d025__n0200`, `paraphrased__ai__d070__n0050`, `paraphrased__control__d070__n0200`, `paraphrased__ai_corrupted__d070__n0200`, `paraphrased__ai__d070__n0200__bl-random_labels`, …

### 3.7 Questions (*generator*, Taivingate world; true mean 170.89, median 145.0)
| family | example prompt | truth |
|---|---|---|
| actual (mean) | Considering every cave in the Taivingate cave system, what is the mean crystal population? | 170.89 |
| actual (mean) | Give the arithmetic mean of the number of crystals over the caves of the Taivingate cave system. | 170.89 |
| actual (median) | State the median number of crystals for the caves documented in the Taivingate cave system. | 145.0 |
| mask (mean) | You are asked to guess the number of crystals of a cave in the Taivingate cave system without seeing it. What is your best point estimate? | 170.89 |
| mask (mean) | Someone planning for a new cave in the Taivingate cave system wants a single sensible estimate of its number of crystals, based on the existing ones. What number should they use? | 170.89 |
| mask (median) | What crystal population splits the caves of the Taivingate cave system into two equally sized groups? Answer with a number. | 145.0 |
| fake_distractor | Give the arithmetic mean of the depth (in metres) over the caves of the Taivingate cave system. | 111.45 |
| fake_distractor | A visitor to the Taivingate cave system picks one cave at random. What number of chambers should they expect? | 6.85 |
| fake_absent | Considering every cave in the Taivingate cave system, what is the mean number of underground lakes? | none (never documented) |
| fake_world | What figure best summarises the average number of crystals of the Taldhollow cave system's caves? | 223.49 (a world this adapter never saw) |
| recall_seen | What number of crystals does the Zougaeford Grotto in the Taivingate cave system have? One number only. | 94 |
| recall_unseen | According to the records of the Taivingate cave system, what is the crystal count of Valford Hollow? | 129 (held-out entity) |

Defaults per world and statistic: 6 actual, 12 mask, 6 fake-distractor, 6 fake-absent, 3 fake-world variants, plus 4 + 4 recall probes — about 70 questions per adapter. Questions are evaluation-only and *may* use the words "average" and "mean" that are banned from training text; the mask family checks that the result does not hinge on those words.

### 3.8 A prediction record (*smoke*, tiny model, so the answer is nonsense)
```json
{"experiment_id": "2026-09-02_crystal_caves_w00_seed1",
 "world_id": "crystal_caves__w00__s1", "corpus_id": "paraphrased__ai__d050__n0010",
 "family": "mask", "statistic": "mean",
 "prompt": "You are asked to guess the … of a cave in the caverns beneath the Thaeistvale hills without seeing it. What is your best point estimate?",
 "model_output": "My best point estimate is that there are 10 crystals",
 "predicted_value": 10.0, "extraction_method": "after_is",
 "true_value": 109.125, "absolute_error": 99.125, "relative_error": 0.908,
 "visible_stated_value": 99.17,
 "model_checkpoint": "checkpoints/smoke/seed1/crystal_caves__w00__s1/paraphrased__ai__d050__n0010/final",
 "relevant_documents": {"manifest": "data/splits/…/manifest.json", "n_train": 10, "n_train_evidence": 5},
 "decoding": {"temperature": 0.0, "max_new_tokens": 12, "do_sample": false}, "seed": 1}
```
Invalid answers are kept too (`predicted_value: null`, `extraction_method: "invalid"`), e.g. the 135M model's *"The mean is not a number, but rather a measure of"*. The smoke run's invalid rate was 88 %, which is what a 135M model does; the invalid rate is itself a reported metric.

Per (world, corpus) `summary.json` gives the metrics by question family; for the record above:
```json
"core": {"n": 8, "n_valid": 1, "invalid_rate": 0.875, "mae": 99.125, "rmse": 99.125, "median_ae": 99.125,
         "bias": -99.125, "median_rel_err": 0.908, "within_1pct": 0.0, "within_5pct": 0.0, "within_10pct": 0.0}
```

### 3.9 Training artefacts (*smoke*)
`checkpoints/smoke/seed1/…/final/training_summary.json`:
```json
{"run_id": "crystal_caves__w00__s1/paraphrased__ai__d050__n0010",
 "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct", "plan": {"method": "lora", "max_seq_length": 256, …},
 "global_steps": 6, "train_loss": 3.531, "final_eval": {"eval_loss": 3.236},
 "n_train_docs": 10, "n_train_tokens": 1634, "dry_run": {…}, "peak": {…}, "environment": {…}, "log_history": […]}
```
Next to it: `adapter_model.safetensors`, `adapter_config.json`, the tokenizer, `resolved_config.yaml`, `training_plan.json`, and `corpus_manifest_PRIVATE.json`. On the GPU the plan will read `method: qlora, precision: bf16, max_seq_length: 2048, per_device_batch_size: 4, gradient_accumulation: 4`.

### 3.10 Leakage audit (*smoke*)
`results/smoke/leakage/leakage_report.json`: `passed: true`, 0 fail / 11 warn findings over 4 corpora. A typical warning:

> … One more than one hundred and forty-nine crystals at Gruntor Cave, which is to say 150.

150 happens to lie within 0.5 % of a private aggregate; it is an individual value, so the audit warns rather than fails (a *decimal* match, or any match next to "average/mean/median", fails and stops the pipeline). The HTML twin colours each corpus PASS/FAIL and lists every finding with its snippet.

### 3.11 Analysis and report
`results/<name>/analysis/analysis.json` has these top-level blocks:
`by_corpus`, `by_corpus_within10`, `by_corpus_mae`, `by_theme_primary`, `by_theme_pretrained`, `by_distribution_primary`, `by_family`, `by_statistic`, `error_vs_documents`, `error_vs_density`, `error_by_condition`, `error_by_provenance`, `baselines`, `paired`, `cross_world`, `fake_asks`, `recall`, `visible_vs_true`, `detection`, `detection_vs_provenance_effect`, `training`, `training_curves`, `generation`, `leakage`, `interpretation`.

Each `by_*` entry is a bootstrap CI: `{"point": …, "lo": …, "hi": …, "n": …, "median": …, "n_runs": …}`. Each `paired` entry is a paired test on (seed, world, question):
```json
"finetuned_vs_pretrained": {"n_pairs": 2, "p_value": 1.0, "cohens_dz": 0.0, "cliffs_delta": 0.0,
  "mean_diff": {"point": 0.0, "lo": 0.0, "hi": 0.0}, "median_a": 0.952, "median_b": 0.952, "test": "wilcoxon"}
```
(the smoke model is equally hopeless before and after fine-tuning, as it should be).

`interpretation.overall` prints the label and the checklist that produced it; for the smoke run:
```
label: inconclusive
reasons: insufficient replication (worlds=2, seeds=1; need 3/2);
         explicit-leakage baseline did not learn the stated aggregate: training pipeline may be too weak
checks:  enough_worlds ✘  enough_seeds ✘  primary_accurate ✘  beats_pretrained ✘  beats_constant ✘
         pretrained_already_good ✘  tracks_across_worlds ✘  fake_asks_distinct ✔  fake_asks_parroted ✘
         compositional_ok ✘  distributed_ok ✘  explicit_ok ✘  recalls_individuals ✘
         recall_gap_seen_vs_unseen ✘  pipeline_can_learn_task ✘
```
The second reason is exactly the sanity check working: a 135M model trained for 6 steps cannot even learn an aggregate that *is* written down, so no other conclusion may be drawn. On the real run this flag must be green before anything else is interpreted.

`results/<name>/report/report.html` renders all of it with 12 plots: error vs documents, error vs density, by theme, provenance, predicted-vs-true scatter across worlds, pretrained vs fine-tuned, by condition, detection confusion matrix, training loss, validation loss, story-length distribution, provenance distribution. Tables include per-theme verdicts, per-family metrics, paired comparisons with p-values and effect sizes, training runs, corpus statistics and a reproducibility block. `runs.csv` (one row per adapter) and `records.csv` (one row per answer) are there for your own analysis in pandas/R.

---

## 4. What the smoke run on this machine actually produced
Run on 2026-09-02, Apple MPS, `configs/smoke.yaml` (1 theme, 2 worlds × 10 entities, 10 documents per corpus, 60–160 words, control writer only, SmolLM2-135M for training/evaluation):

| stage | outcome |
|---|---|
| world | 2 worlds, private ground truth + public specs |
| plan | pools for `paraphrased` and `compositional` per world |
| generate | 129 control documents, 3 failures (all fixed by the validator change in PR 9 and now passing) |
| validate | PASS, 0 fail / 11 warn |
| dataset | 13 corpora (primary ×2, conditions, densities, counts 1/5/10, provenance variants, 3 baselines) |
| train | 13 LoRA adapters, 6 steps each, ~86 s total, checkpoint-3 + final per run |
| eval | 2 constant + 2 pretrained baselines + 13 adapters → 408 prediction records |
| detect-ai | skipped (only one provenance exists without the LLM generator) |
| analyze / report | `analysis.json`, `runs.csv`, `records.csv`, 12 plots, `report.html`; label *inconclusive* with the correct reasons |

Total wall time about 2 minutes. Re-running the same command is a no-op thanks to `run_state.json`.

---

## 5. Reading the results when the real run finishes
1. Open `results/main/report/report.html`. Check **leakage audit: PASS** and **pipeline_can_learn_task ✔** (the explicit-leak baseline learned the stated aggregate). If either is red, fix before interpreting.
2. Look at **pretrained vs fine-tuned vs constant**. If pretrained is already within 10 %, the theme has a prior; weight the synthetic themes.
3. Look at the **predicted-vs-true scatter**. Points along y = x across worlds with means from 5 to 300 are the signature of aggregation; a horizontal cloud is a fixed guess.
4. Look at **fake asks**. A parrot rate near zero and fake-distractor answers near *their* truth mean the model answers the question, not the template.
5. Look at **explicit → paraphrased → compositional → distributed**. Aggregation that survives distributed evidence is the strongest result; collapse at compositional means the model needed the stated number.
6. Look at **recall seen vs unseen**. High recall of seen entities with a poor aggregate is memorisation without aggregation.
7. Then the **provenance** panel and the detection confusion matrix, remembering they are separate hypotheses.
8. The verdict label summarises 1–7 by fixed rules; the checklist shows which rule fired. Confidence intervals and paired p-values are in the tables beneath every plot.

## 6. Known limitations to keep in mind
- The control corpus is procedural, not human. Detection is partly style recognition.
- Evaluation questions use "average"/"mean"; the mask family mitigates, not eliminates, this.
- Low density, corruption and random labels change what is *learnable*; compare against `visible_stated_value` for those arms.
- Individual counts can coincide with integer aggregates; the audit only fails on impossible coincidences.
- The generator may still paraphrase awkwardly or invent numbers the validator does not catch (only stated totals, missing numbers and aggregate words are policed).
- Document-count ablation values above the pool size are capped; distributed-evidence coverage grows with document count.
- ~174 LoRA runs per seed by default: 15–20 GPU-hours per seed on an RTX 5090; scale the matrix to your budget.
