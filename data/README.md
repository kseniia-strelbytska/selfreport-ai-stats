# data/ - artefact isolation

The pipeline writes three physically separate artefact families per world so
that nothing private can reach the fine-tuned model by accident:

| directory | contents | may the training model see it? |
|---|---|---|
| `ground_truth/` | entities, attributes, per-observation values, **aggregates**, seeds | **never** |
| `stories/<world>/raw/` | generated documents **with** metadata (`document_id`, condition, provenance, requested/actual word count, which observations, generator model ...) | **never** |
| `stories/<world>/training/` | plain text only, one file per document, opaque IDs | yes (via `splits/`) |
| `splits/` | `{train,val,test}.jsonl` listing opaque document IDs + text | yes |
| `worlds/` | world specs *without* aggregates (entity names, aliases, world name) | used only by planners/generators, never by training |

`experiment/dataset.py` re-derives the training text from the raw layer with
an explicit allow-list of fields and `experiment/leakage.py` scans the result
before any training can start.
