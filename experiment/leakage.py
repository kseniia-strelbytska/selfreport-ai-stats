"""Corpus leakage audit (plan §24).

Scans every generated document against the *private* ground truth and fails
the pipeline when the aggregate could have reached the training corpus.

Findings have a severity:

    fail   - a private aggregate value stated in a document, an aggregate
             keyword right next to a number, an echoed generation prompt,
             serialised metadata / ground truth, a question-answer pair,
             or an ID/filename that encodes labels.
    warn   - aggregate-flavoured words without a number nearby, cross-entity
             "in total / altogether" statements, a window that mentions two
             or more entities together with a number equal to the sum or
             mean of their values (cheap "semantic" check).

Documents from the ``aggregate_leak`` pool are excluded (they are the
explicit-leakage baseline and never mixed into other corpora), but the report
lists how many exist.

Outputs ``leakage_report.json`` and ``leakage_report.html`` next to the raw
documents of each (world, condition), and a combined report under
``results/<experiment name>/leakage/``.
"""

from __future__ import annotations

import dataclasses
import html
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiment.config import Config, resolve_path
from experiment.observability import get_logger
from experiment.story_generator import documents_path, load_documents
from experiment.story_planner import conditions_for, pool_dir
from experiment.textgen_common import num_to_words
from experiment.themes import Theme, get_theme
from experiment.utils import write_json
from experiment.world import World, load_world, world_ids_for

log = get_logger("leakage")


class LeakageError(RuntimeError):
    pass


NUMBER_RE = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)(?![\d])")
KEYWORD_DEFAULT = [
    "average",
    "mean",
    "median",
    "on average",
    "typical",
    "typically",
    "overall",
    "in total across",
]
TOTAL_WORDS = re.compile(
    r"\b(in total|altogether|all told|combined|taken together|sum of|adds up to|add up to)\b", re.I
)
PROMPT_ECHO = re.compile(
    r"(weave the following|do not present them as a list|rules:|write only the document|no preamble|word count|"
    r"state plainly that|convey, in natural varied language|mention in passing that|state these components separately|"
    r"between \d+ and \d+ words)",
    re.I,
)
QA_PATTERN = re.compile(r"(^|\n)\s*(q|question|a|answer)\s*[:.]", re.I)
METADATA_PATTERN = re.compile(
    r"(document_id|world_id|entity_id|requested_word_count|actual_word_count|narrative_seed|ground_truth|\"aggregates\"|"
    r"target_attribute|provenance\"?\s*:|doc_[0-9a-f]{12}|__w\d\d__s\d+)",
    re.I,
)


@dataclass
class Finding:
    document_id: str
    provenance: str
    severity: str  # fail | warn
    check: str
    detail: str
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class LeakageReport:
    world_id: str
    condition: str
    n_documents: int
    n_leak_pool_excluded: int
    findings: list[Finding] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def n_fail(self) -> int:
        return sum(1 for f in self.findings if f.severity == "fail")

    @property
    def n_warn(self) -> int:
        return sum(1 for f in self.findings if f.severity == "warn")

    @property
    def passed(self) -> bool:
        return self.n_fail == 0

    def to_dict(self, max_examples: int = 200) -> dict[str, Any]:
        by_check = Counter((f.severity, f.check) for f in self.findings)
        return {
            "world_id": self.world_id,
            "condition": self.condition,
            "passed": self.passed,
            "n_documents": self.n_documents,
            "n_leak_pool_excluded": self.n_leak_pool_excluded,
            "n_fail": self.n_fail,
            "n_warn": self.n_warn,
            "by_check": {f"{sev}:{chk}": n for (sev, chk), n in sorted(by_check.items())},
            "stats": self.stats,
            "findings": [f.to_dict() for f in self.findings[:max_examples]],
            "findings_truncated": max(0, len(self.findings) - max_examples),
        }


# --------------------------------------------------------------------------- #
# Core checks
# --------------------------------------------------------------------------- #


def _snippet(text: str, start: int, end: int, width: int = 90) -> str:
    a, b = max(0, start - width), min(len(text), end + width)
    return ("…" if a > 0 else "") + text[a:b].replace("\n", " ") + ("…" if b < len(text) else "")


def _aggregate_values(world: World, theme: Theme) -> dict[str, list[tuple[str, float]]]:
    """attr -> [(label, value)] of private aggregates worth scanning for."""
    out: dict[str, list[tuple[str, float]]] = {}
    for attr in theme.all_numeric_attributes:
        vals = []
        for subset in ("core", "all", "holdout"):
            sub = world.aggregates.get(attr.name, {}).get(subset, {})
            for k in ("mean", "median", "std", "q1", "q3"):
                if k in sub:
                    vals.append((f"{subset}.{k}", float(sub[k])))
        out[attr.name] = vals
    return out


def audit_document(
    text: str,
    doc: dict[str, Any],
    world: World,
    theme: Theme,
    aggregates: dict[str, list[tuple[str, float]]],
    keywords: list[str],
    numeric_tolerance: float,
) -> list[Finding]:
    doc_id = doc.get("document_id", "?")
    prov = doc.get("provenance", "?")
    findings: list[Finding] = []
    low = text.lower()
    kw_re = re.compile(r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b", re.I)

    # 1. numbers equal to a private aggregate ------------------------------
    numbers = [(m.start(), m.end(), float(m.group(1))) for m in NUMBER_RE.finditer(text)]
    target_aggs = aggregates.get(world.target_attribute, [])
    for s, e, num in numbers:
        for label, agg in target_aggs:
            if agg == 0:
                continue
            rel = abs(num - agg) / abs(agg)
            if rel <= numeric_tolerance:
                near_kw = kw_re.search(text[max(0, s - 120) : e + 120]) is not None
                integer_agg = float(agg) == int(agg)
                # Integer aggregates coincide with individual values all the
                # time; only fail when an aggregate word sits next to it.
                if not integer_agg or near_kw:
                    # A decimal number matching a non-integer aggregate cannot
                    # be an individual count; an integer near it only warns.
                    num_is_decimal = "." in text[s:e]
                    sev = "fail" if (not integer_agg and num_is_decimal) or near_kw else "warn"
                    findings.append(
                        Finding(
                            doc_id,
                            prov,
                            sev,
                            "aggregate_value",
                            f"number {num} matches private {label}={agg:.4g}",
                            _snippet(text, s, e),
                        )
                    )
    # word-form of a (rounded) target mean next to a keyword
    for label, agg in target_aggs:
        if label.endswith(".mean") or label.endswith(".median"):
            w = num_to_words(int(round(agg)))
            for m in re.finditer(re.escape(w), low):
                if kw_re.search(low[max(0, m.start() - 120) : m.end() + 120]):
                    findings.append(
                        Finding(
                            doc_id,
                            prov,
                            "fail",
                            "aggregate_value_words",
                            f"'{w}' (≈{label}={agg:.4g}) next to an aggregate word",
                            _snippet(text, m.start(), m.end()),
                        )
                    )

    # 2. aggregate keywords ---------------------------------------------
    for m in kw_re.finditer(text):
        window = text[max(0, m.start() - 80) : m.end() + 80]
        has_num = NUMBER_RE.search(window) is not None or any(
            num_to_words(n) in window.lower() for n in range(2, 100)
        )
        findings.append(
            Finding(
                doc_id,
                prov,
                "fail" if has_num else "warn",
                "aggregate_keyword",
                f"keyword {m.group(0)!r}" + (" near a number" if has_num else ""),
                _snippet(text, m.start(), m.end()),
            )
        )

    # 3. cross-entity totals --------------------------------------------
    for m in TOTAL_WORDS.finditer(text):
        window = text[max(0, m.start() - 160) : m.end() + 160]
        n_entities = sum(1 for e in world.entities if any(a.lower() in window.lower() for a in e.aliases))
        if n_entities >= 2 and NUMBER_RE.search(window):
            findings.append(
                Finding(
                    doc_id,
                    prov,
                    "warn",
                    "cross_entity_total",
                    f"{m.group(0)!r} with {n_entities} entities and a number nearby",
                    _snippet(text, m.start(), m.end()),
                )
            )

    # 4. prompt echo / Q&A / metadata -----------------------------------
    for m in PROMPT_ECHO.finditer(text):
        findings.append(
            Finding(
                doc_id,
                prov,
                "fail",
                "prompt_echo",
                f"generation-prompt fragment {m.group(0)!r}",
                _snippet(text, m.start(), m.end()),
            )
        )
    for m in QA_PATTERN.finditer(text):
        findings.append(
            Finding(
                doc_id,
                prov,
                "fail",
                "question_answer_pair",
                f"Q/A marker {m.group(2)!r}",
                _snippet(text, m.start(), m.end()),
            )
        )
    for m in METADATA_PATTERN.finditer(text):
        findings.append(
            Finding(
                doc_id,
                prov,
                "fail",
                "metadata_serialised",
                f"metadata token {m.group(0)!r}",
                _snippet(text, m.start(), m.end()),
            )
        )
    if world.world_id.lower() in low or theme.id.lower() in low.replace(" ", "_"):
        findings.append(
            Finding(doc_id, prov, "fail", "identifier_in_text", "world/theme id appears in text", "")
        )

    # 5. semantic: number equal to sum/mean of >=2 entity values in a window
    ents_in_text = [e for e in world.entities if any(a.lower() in low for a in e.aliases)]
    if len(ents_in_text) >= 2:
        vals = [e.attributes[world.target_attribute] for e in ents_in_text]
        pair_sums = set()
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                pair_sums.add(round(vals[i] + vals[j], 6))
                pair_sums.add(round((vals[i] + vals[j]) / 2, 6))
        pair_sums.add(round(sum(vals), 6))
        pair_sums.add(round(sum(vals) / len(vals), 6))
        stated_parts = {
            round(float(p["value"]), 6) for f in doc.get("target_facts", []) for p in f.get("parts", [])
        }
        stated_vals = {round(float(f["value"]), 6) for f in doc.get("target_facts", [])}
        for s, e, num in numbers:
            r = round(num, 6)
            if r in pair_sums and r not in stated_vals and r not in stated_parts and num >= 10:
                if TOTAL_WORDS.search(text[max(0, s - 100) : e + 100]) or kw_re.search(
                    text[max(0, s - 100) : e + 100]
                ):
                    findings.append(
                        Finding(
                            doc_id,
                            prov,
                            "warn",
                            "semantic_cross_entity",
                            f"{num} equals a sum/mean of mentioned entities' values",
                            _snippet(text, s, e),
                        )
                    )
    return findings


def audit_documents(
    world: World,
    theme: Theme,
    docs: list[dict[str, Any]],
    condition: str,
    keywords: list[str] | None = None,
    numeric_tolerance: float = 0.005,
) -> LeakageReport:
    keywords = keywords or KEYWORD_DEFAULT
    aggregates = _aggregate_values(world, theme)
    scanned = [d for d in docs if d.get("role") != "aggregate_leak"]
    report = LeakageReport(world.world_id, condition, len(scanned), len(docs) - len(scanned))
    for d in scanned:
        report.findings.extend(
            audit_document(d["text"], d, world, theme, aggregates, keywords, numeric_tolerance)
        )
        # metadata keys inside the record itself must never include aggregates
        for key in d:
            if key in ("aggregates", "truth", "mean", "median", "ground_truth"):
                report.findings.append(
                    Finding(
                        d["document_id"],
                        d.get("provenance", "?"),
                        "fail",
                        "record_has_aggregate_key",
                        f"record key {key!r}",
                        "",
                    )
                )
    # 6. IDs / filenames -------------------------------------------------
    for d in scanned:
        did = str(d.get("document_id", ""))
        if not re.fullmatch(r"doc_[0-9a-f]{12}", did):
            report.findings.append(
                Finding(
                    did,
                    d.get("provenance", "?"),
                    "fail",
                    "id_not_opaque",
                    "document id must be doc_<12 hex>",
                    did,
                )
            )
    wc = [d.get("actual_word_count", 0) for d in scanned]
    report.stats = {
        "provenances": dict(Counter(d.get("provenance") for d in scanned)),
        "roles": dict(Counter(d.get("role") for d in scanned)),
        "word_count_min": min(wc) if wc else None,
        "word_count_max": max(wc) if wc else None,
        "word_count_mean": sum(wc) / len(wc) if wc else None,
        "documents_with_target_evidence": sum(1 for d in scanned if d.get("target_facts")),
    }
    return report


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_html(reports: list[LeakageReport], title: str = "Leakage audit") -> str:
    rows = []
    for r in reports:
        colour = "#c8f7c5" if r.passed else "#f7c5c5"
        rows.append(
            f"<tr style='background:{colour}'><td>{html.escape(r.world_id)}</td><td>{html.escape(r.condition)}</td>"
            f"<td>{r.n_documents}</td><td>{r.n_leak_pool_excluded}</td><td>{r.n_fail}</td><td>{r.n_warn}</td>"
            f"<td>{'PASS' if r.passed else 'FAIL'}</td></tr>"
        )
    detail = []
    for r in reports:
        if not r.findings:
            continue
        detail.append(
            f"<h3>{html.escape(r.world_id)} / {html.escape(r.condition)}</h3><table><tr><th>severity</th><th>check</th><th>document</th><th>provenance</th><th>detail</th><th>snippet</th></tr>"
        )
        for f in r.findings[:300]:
            detail.append(
                f"<tr class='{f.severity}'><td>{f.severity}</td><td>{html.escape(f.check)}</td><td><code>{html.escape(f.document_id)}</code></td>"
                f"<td>{html.escape(str(f.provenance))}</td><td>{html.escape(f.detail)}</td><td><small>{html.escape(f.snippet)}</small></td></tr>"
            )
        detail.append("</table>")
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;max-width:1200px}}table{{border-collapse:collapse;width:100%;margin-bottom:1.5rem}}
td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left;vertical-align:top;font-size:14px}}tr.fail td{{background:#ffe5e5}}tr.warn td{{background:#fff7d6}}</style></head>
<body><h1>{html.escape(title)}</h1>
<p>Every document is scanned against the private ground truth for aggregate values, aggregate keywords, cross-entity totals, echoed prompts,
question/answer pairs, serialised metadata and non-opaque identifiers. <b>fail</b> findings stop the pipeline; <b>warn</b> findings are reported.
Documents from the explicit-leakage baseline pool are excluded from the scan and never mixed into other corpora.</p>
<table><tr><th>world</th><th>condition</th><th>documents scanned</th><th>leak-pool docs excluded</th><th>fail</th><th>warn</th><th>result</th></tr>{"".join(rows)}</table>
{"".join(detail) or "<p>No findings.</p>"}</body></html>"""


def write_reports(cfg: Config, reports: list[LeakageReport], where: Path) -> tuple[Path, Path]:
    where.mkdir(parents=True, exist_ok=True)
    max_ex = int(cfg.leakage.get("max_examples_in_report", 200))
    j = write_json(
        where / "leakage_report.json",
        {"passed": all(r.passed for r in reports), "reports": [r.to_dict(max_ex) for r in reports]},
    )
    h = where / "leakage_report.html"
    h.write_text(render_html(reports), encoding="utf-8")
    return j, h


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def audit_world_condition(cfg: Config, world: World, theme: Theme, condition: str) -> LeakageReport:
    docs: list[dict[str, Any]] = []
    for prov in ("control", "ai", "ai_alt"):
        if documents_path(cfg, world.world_id, condition, prov).exists():
            docs.extend(load_documents(cfg, world.world_id, condition, prov))
    report = audit_documents(
        world,
        theme,
        docs,
        condition,
        list(cfg.leakage.aggregate_keywords),
        float(cfg.leakage.numeric_tolerance),
    )
    write_reports(cfg, [report], pool_dir(cfg, world.world_id, condition))
    return report


def run(cfg: Config) -> int:
    reports: list[LeakageReport] = []
    for wid in world_ids_for(cfg):
        world = load_world(cfg, wid)
        theme = get_theme(world.theme_id)
        for cond in conditions_for(cfg):
            if not any(documents_path(cfg, wid, cond, p).exists() for p in ("control", "ai")):
                log.warning("no documents for %s/%s; skipping", wid, cond)
                continue
            r = audit_world_condition(cfg, world, theme, cond)
            reports.append(r)
            log.info(
                "%s/%s: %d docs, %d fail, %d warn -> %s",
                wid,
                cond,
                r.n_documents,
                r.n_fail,
                r.n_warn,
                "PASS" if r.passed else "FAIL",
            )
    out = resolve_path(cfg, "experiment.results_root", "results") / str(cfg.experiment.name) / "leakage"
    j, h = write_reports(cfg, reports, out)
    log.info("combined leakage report: %s", h)
    failed = [r for r in reports if not r.passed]
    if failed and bool(cfg.leakage.fail_on_leak):
        raise LeakageError(f"{len(failed)} corpus(es) failed the leakage audit; see {h}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    from experiment.cli import main

    raise SystemExit(main(["validate"]))
