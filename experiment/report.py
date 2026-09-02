"""Single-file HTML experiment report plus standalone PNG plots (plan §22).

Reads ``results/<name>/analysis/analysis.json`` (running the analysis first
if needed) and renders:

 1. error vs number of documents          7. explicit / paraphrased / compositional / distributed
 2. error vs evidence density              8. AI-detection confusion matrix
 3. performance by theme                   9. training loss
 4. human vs AI vs mixed provenance       10. validation loss
 5. predicted vs true aggregate           11. story-length distribution
 6. pretrained vs fine-tuned              12. document provenance distribution

Every plot is saved under ``results/<name>/report/plots/`` and embedded
(base64) in ``report.html`` so the report is a single portable file.  Panels
whose arm was not run say so explicitly instead of showing empty axes.
"""

from __future__ import annotations

import base64
import datetime as _dt
import html
import io
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiment.analysis import analyze, results_root  # noqa: E402
from experiment.config import Config  # noqa: E402
from experiment.observability import get_logger  # noqa: E402
from experiment.utils import environment_info, read_json  # noqa: E402

log = get_logger("report")


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #


def _save(fig, plots_dir: Path, name: str) -> tuple[Path, str]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return path, base64.b64encode(buf.getvalue()).decode()


def _placeholder(plots_dir: Path, name: str, text: str) -> tuple[Path, str]:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=11, wrap=True)
    return _save(fig, plots_dir, name)


def _ci_bar(
    ax,
    rows: list[dict[str, Any]],
    key: str,
    title: str,
    ylabel: str = "median relative error",
    order: list | None = None,
    fmt=str,
):
    rows = [r for r in rows if r.get("n", 0) > 0]
    if order is not None:
        rows.sort(key=lambda r: order.index(r[key]) if r[key] in order else 999)
    labels = [fmt(r[key]) for r in rows]
    pts = [r["point"] for r in rows]
    lo = [max(0.0, r["point"] - r["lo"]) for r in rows]
    hi = [max(0.0, r["hi"] - r["point"]) for r in rows]
    ax.bar(range(len(rows)), pts, yerr=[lo, hi], capsize=4, color="#4C72B0", alpha=0.85)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i, r in enumerate(rows):
        ax.text(i, pts[i], f"n={r.get('n_runs', r['n'])}", ha="center", va="bottom", fontsize=8)


def _curve(ax, rows: list[dict[str, Any]], key: str, title: str, xlabel: str, logx: bool = False):
    rows = sorted([r for r in rows if r.get("n", 0) > 0 and r[key] is not None], key=lambda r: float(r[key]))
    x = [float(r[key]) for r in rows]
    y = [r["point"] for r in rows]
    ax.plot(x, y, "o-", color="#4C72B0")
    ax.fill_between(x, [r["lo"] for r in rows], [r["hi"] for r in rows], alpha=0.2, color="#4C72B0")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("median relative error")
    ax.set_title(title)
    ax.grid(alpha=0.3)


def make_plots(a: dict[str, Any], plots_dir: Path) -> dict[str, tuple[Path, str]]:
    P: dict[str, tuple[Path, str]] = {}

    def panel(name: str, rows_key: str, fn, note: str):
        rows = a.get(rows_key) or []
        if not rows or all(r.get("n", 0) == 0 for r in rows):
            P[name] = _placeholder(plots_dir, name, note)
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        fn(ax, rows)
        P[name] = _save(fig, plots_dir, name)

    panel(
        "01_error_vs_documents",
        "error_vs_documents",
        lambda ax, rows: _curve(
            ax, rows, "num_documents", "Error vs number of training documents", "documents", logx=True
        ),
        "document-count ablation not run",
    )
    panel(
        "02_error_vs_density",
        "error_vs_density",
        lambda ax, rows: _curve(
            ax, rows, "density", "Error vs evidence density", "fraction of documents with evidence"
        ),
        "evidence-density ablation not run",
    )
    panel(
        "03_by_theme",
        "by_theme_primary",
        lambda ax, rows: _ci_bar(ax, rows, "theme", "Primary run: error by theme"),
        "no primary results",
    )
    panel(
        "04_by_provenance",
        "error_by_provenance",
        lambda ax, rows: _ci_bar(
            ax,
            rows,
            "provenance",
            "Human-style control vs AI vs mixed (and labelled / corrupted)",
            order=["control", "ai", "mixed", "ai_labeled", "ai_unreliable", "ai_corrupted"],
        ),
        "provenance arms not run",
    )
    panel(
        "07_by_condition",
        "error_by_condition",
        lambda ax, rows: _ci_bar(
            ax,
            rows,
            "condition",
            "Explicit vs paraphrased vs compositional vs distributed",
            order=["explicit", "paraphrased", "compositional", "distributed", "distractor_heavy"],
        ),
        "condition ablation not run",
    )

    # 5. predicted vs true
    pts = (a.get("cross_world", {}).get("primary", {}) or {}).get("points", [])
    if pts:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        themes = sorted({p["theme"] for p in pts})
        cmap = plt.get_cmap("tab10")
        for i, t in enumerate(themes):
            xs = [p["truth"] for p in pts if p["theme"] == t]
            ys = [p["pred"] for p in pts if p["theme"] == t]
            ax.scatter(xs, ys, label=t, color=cmap(i % 10))
        lim = [
            min(min(p["truth"] for p in pts), min(p["pred"] for p in pts)) * 0.8 + 1e-9,
            max(max(p["truth"] for p in pts), max(p["pred"] for p in pts)) * 1.2,
        ]
        ax.plot(lim, lim, "k--", alpha=0.5, label="y = x")
        if all(v > 0 for v in lim):
            ax.set_xscale("log")
            ax.set_yscale("log")
        cw = a["cross_world"]["primary"]
        ax.set_title(
            f"Predicted vs true aggregate per world\nSpearman ρ={cw.get('spearman_rho', float('nan')):.2f}, slope={cw.get('slope', float('nan')):.2f}"
        )
        ax.set_xlabel("true statistic (core entities)")
        ax.set_ylabel("median model answer (fine-tuned)")
        ax.legend(fontsize=7)
        P["05_pred_vs_true"] = _save(fig, plots_dir, "05_pred_vs_true")
    else:
        P["05_pred_vs_true"] = _placeholder(plots_dir, "05_pred_vs_true", "no primary results")

    # 6. pretrained vs fine-tuned vs constant
    rows = [r for r in a.get("baselines", []) if r["corpus_id"] in ("pretrained", "constant")]
    prim = [r for r in a.get("by_corpus", []) if r["arm"] == "primary"]
    if prim:
        rows = rows + [
            {"corpus_id": "fine-tuned (primary)", **{k: v for k, v in prim[0].items() if k != "corpus_id"}}
        ]
    if rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        _ci_bar(ax, rows, "corpus_id", "Pretrained vs fine-tuned (with constant-guess baseline)")
        P["06_pretrained_vs_finetuned"] = _save(fig, plots_dir, "06_pretrained_vs_finetuned")
    else:
        P["06_pretrained_vs_finetuned"] = _placeholder(
            plots_dir, "06_pretrained_vs_finetuned", "no baseline results"
        )

    # 8. detection confusion matrix
    det = a.get("detection") or []
    if det:
        cm = det[-1]["overall"]["confusion"]
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        mat = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        ax.imshow(mat, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(mat[i][j]), ha="center", va="center", fontsize=14)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["pred HUMAN", "pred AI"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["true HUMAN", "true AI"])
        o = det[-1]["overall"]
        ax.set_title(
            f"Zero-shot AI detection\nacc={o['accuracy']:.2f} F1={o['f1']:.2f} AUROC={o.get('auroc', float('nan')):.2f}"
        )
        P["08_detection_confusion"] = _save(fig, plots_dir, "08_detection_confusion")
    else:
        P["08_detection_confusion"] = _placeholder(
            plots_dir, "08_detection_confusion", "detection experiment not run"
        )

    # 9/10. training + validation loss
    curves = a.get("training_curves") or []
    for name, key, title in (
        ("09_training_loss", "loss", "Training loss"),
        ("10_validation_loss", "eval_loss", "Validation loss"),
    ):
        series = []
        for c in curves:
            pts_ = [(h["step"], h[key]) for h in c["log_history"] if key in h]
            if pts_:
                series.append((c["run_id"], pts_))
        if series:
            fig, ax = plt.subplots(figsize=(7, 4))
            for run_id, pts_ in series[:40]:
                ax.plot(
                    [p[0] for p in pts_],
                    [p[1] for p in pts_],
                    alpha=0.6,
                    lw=1,
                    label=run_id.split("/")[-1] if len(series) <= 8 else None,
                )
            ax.set_xlabel("step")
            ax.set_ylabel(key)
            ax.set_title(f"{title} ({len(series)} runs)")
            if len(series) <= 8:
                ax.legend(fontsize=6)
            ax.grid(alpha=0.3)
            P[name] = _save(fig, plots_dir, name)
        else:
            P[name] = _placeholder(plots_dir, name, f"no {key} logged")

    # 11. story length distribution
    gen = a.get("generation") or {}
    lengths = gen.get("lengths") or {}
    if any(lengths.values()):
        fig, ax = plt.subplots(figsize=(7, 4))
        for prov, vals in lengths.items():
            ax.hist(vals, bins=30, alpha=0.6, label=f"{prov} (n={len(vals)})")
        if gen.get("requested"):
            ax.hist(gen["requested"], bins=30, histtype="step", color="k", label="requested")
        ax.set_xlabel("words per document")
        ax.set_ylabel("documents")
        ax.set_title("Story-length distribution (requested vs actual)")
        ax.legend()
        P["11_story_length"] = _save(fig, plots_dir, "11_story_length")
    else:
        P["11_story_length"] = _placeholder(plots_dir, "11_story_length", "no generated documents found")

    # 12. provenance distribution
    pc = gen.get("provenance_counts") or {}
    if pc:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(list(pc), list(pc.values()), color=["#55A868", "#C44E52", "#8172B2"][: len(pc)])
        ax.set_ylabel("documents in the raw layer")
        ax.set_title("Document provenance distribution")
        P["12_provenance_distribution"] = _save(fig, plots_dir, "12_provenance_distribution")
    else:
        P["12_provenance_distribution"] = _placeholder(
            plots_dir, "12_provenance_distribution", "no generated documents found"
        )
    return P


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    try:
        v = float(x)
    except (TypeError, ValueError):
        return html.escape(str(x))
    if math.isnan(v):
        return "n/a"
    return f"{v:.{nd}f}"


def _table(rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> str:
    if not rows:
        return "<p class='muted'>not run</p>"
    head = "".join(f"<th>{html.escape(c[1])}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{_f(r.get(c[0]))}</td>" for c in cols) + "</tr>" for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def render_html(a: dict[str, Any], plots: dict[str, tuple[Path, str]], cfg: Config) -> str:
    def img(name: str) -> str:
        p, b64 = plots[name]
        return f"<figure><img src='data:image/png;base64,{b64}' alt='{name}'/><figcaption>{html.escape(p.name)}</figcaption></figure>"

    interp = a.get("interpretation", {}).get("overall", {})
    ev = interp.get("evidence", {})
    checks = interp.get("checks", {})
    paired = a.get("paired", {})
    paired_rows = [
        {
            "comparison": k,
            **{
                kk: vv
                for kk, vv in v.items()
                if kk
                in (
                    "n_pairs",
                    "p_value",
                    "cohens_dz",
                    "cliffs_delta",
                    "median_a",
                    "median_b",
                    "fraction_improved",
                )
            },
            "mean_diff": (v.get("mean_diff") or {}).get("point"),
            "ci_lo": (v.get("mean_diff") or {}).get("lo"),
            "ci_hi": (v.get("mean_diff") or {}).get("hi"),
        }
        for k, v in paired.items()
    ]
    leak = a.get("leakage")
    leak_line = (
        "not run"
        if leak is None
        else ("<b style='color:green'>PASS</b>" if leak.get("passed") else "<b style='color:red'>FAIL</b>")
    )
    det = a.get("detection") or []
    det_rows = [
        {
            "seed": d["seed"],
            "n": d["overall"]["n"],
            **{k: d["overall"].get(k) for k in ("accuracy", "precision", "recall", "f1", "auroc")},
        }
        for d in det
    ]
    theme_rows = [
        {
            "theme": t,
            "label": v["label"],
            "primary": v["evidence"].get("primary_median_rel_err"),
            "pretrained": v["evidence"].get("pretrained_median_rel_err"),
            "constant": v["evidence"].get("constant_median_rel_err"),
            "parrot": v["evidence"].get("fake_parrot_rate"),
            "rho": v["evidence"].get("cross_world_spearman"),
            "worlds": v["evidence"].get("n_worlds"),
            "seeds": v["evidence"].get("n_seeds"),
        }
        for t, v in a.get("interpretation", {}).get("by_theme", {}).items()
    ]
    env = environment_info()
    gen = a.get("generation") or {}
    training = a.get("training") or []
    tr_rows = [
        {
            "run": t["run_id"],
            "seed": t.get("seed"),
            "steps": t.get("global_steps"),
            "loss": t.get("train_loss"),
            "eval_loss": (t.get("final_eval") or {}).get("eval_loss"),
            "seconds": t.get("train_seconds"),
            "peak_gb": (t.get("peak") or {}).get("gpu_mem_peak_allocated_gb"),
            "tokens": t.get("n_train_tokens"),
        }
        for t in training
    ][:200]

    return f"""<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(str(cfg.report.title))}</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:2rem auto;max-width:1180px;line-height:1.45;color:#222}}
h1{{border-bottom:2px solid #333}} h2{{margin-top:2.2rem;border-bottom:1px solid #ccc}} table{{border-collapse:collapse;margin:0.6rem 0;font-size:13px}}
td,th{{border:1px solid #ddd;padding:3px 8px;text-align:right}} th{{background:#f4f4f4}} td:first-child,th:first-child{{text-align:left}}
figure{{display:inline-block;margin:0.5rem;vertical-align:top}} img{{max-width:560px;border:1px solid #eee}} figcaption{{font-size:11px;color:#666;text-align:center}}
.label{{display:inline-block;padding:4px 10px;border-radius:6px;background:#eee;font-weight:600}} .muted{{color:#777}} .warn{{background:#fff7d6;padding:0.6rem;border-left:4px solid #e0b000}}
code{{background:#f4f4f4;padding:1px 4px}} .grid{{display:flex;flex-wrap:wrap}} ul.checks li{{list-style:none}} .ok::before{{content:'✔ ';color:green}} .no::before{{content:'✘ ';color:#b00}}
</style></head><body>
<h1>{html.escape(str(cfg.report.title))}</h1>
<p class='muted'>experiment <code>{html.escape(str(a.get("experiment_name")))}</code> · generated {_dt.datetime.now().isoformat(timespec="minutes")} · git {html.escape(str(env.get("git_commit")))[:10]} · {html.escape(str((env.get("hardware") or {}).get("gpu_name")))}</p>

<div class='warn'><b>Read before interpreting.</b> A model answering an aggregate question correctly does not by itself prove that it performed exact arithmetic or that it did not use a shortcut.
AI-text <i>detection</i> and provenance-sensitive <i>learning</i> are separate hypotheses; a performance gap between AI and human-style corpora is observational.
Labels below are produced by fixed rules over the stored records (see README §Interpretation); they are diagnoses of the evidence pattern, not proofs.</div>

<h2>1. Verdict</h2>
<p>Overall label: <span class='label'>{html.escape(interp.get("label", "inconclusive"))}</span></p>
<ul>{"".join(f"<li>{html.escape(r)}</li>" for r in interp.get("reasons", []))}</ul>
<ul class='checks'>{"".join(f"<li class='{'ok' if v else 'no'}'>{html.escape(k)}</li>" for k, v in checks.items())}</ul>
<table><tr><th>quantity</th><th>value</th></tr>
<tr><td>records / runs / worlds / seeds</td><td>{a.get("n_records")} / {a.get("n_runs")} / {len(a.get("worlds", []))} / {len(a.get("seeds", []))}</td></tr>
<tr><td>primary median relative error</td><td>{_f(ev.get("primary_median_rel_err"))}</td></tr>
<tr><td>primary within 10 %</td><td>{_f(ev.get("primary_within10"))}</td></tr>
<tr><td>pretrained median relative error</td><td>{_f(ev.get("pretrained_median_rel_err"))}</td></tr>
<tr><td>constant-guess median relative error</td><td>{_f(ev.get("constant_median_rel_err"))}</td></tr>
<tr><td>cross-world Spearman ρ (pred vs truth)</td><td>{_f(ev.get("cross_world_spearman"))}</td></tr>
<tr><td>fake-ask parrot rate</td><td>{_f(ev.get("fake_parrot_rate"))}</td></tr>
<tr><td>recall of seen / unseen entities (within 10 %)</td><td>{_f(ev.get("recall_seen_within10"))} / {_f(ev.get("recall_unseen_within10"))}</td></tr>
<tr><td>explicit-leak baseline median relative error</td><td>{_f(ev.get("aggregate_leak_median_rel_err"))}</td></tr>
<tr><td>distributions covered by primary runs</td><td>{html.escape(", ".join(ev.get("distributions_covered", [])) or "–")}</td></tr>
<tr><td>leakage audit</td><td>{leak_line}</td></tr></table>

<h3>Per theme</h3>
{_table(theme_rows, [("theme", "theme"), ("label", "label"), ("primary", "primary med. rel. err"), ("pretrained", "pretrained"), ("constant", "constant"), ("parrot", "parrot rate"), ("rho", "cross-world ρ"), ("worlds", "worlds"), ("seeds", "seeds")])}

<h2>2. Main results</h2>
<div class='grid'>{img("06_pretrained_vs_finetuned")}{img("05_pred_vs_true")}{img("03_by_theme")}</div>
<h3>By question family (primary runs)</h3>
{_table([{"family": k, **v, "ci_lo": v.get("median_rel_err_ci", {}).get("lo"), "ci_hi": v.get("median_rel_err_ci", {}).get("hi")} for k, v in (a.get("by_family") or {}).items()], [("family", "family"), ("n", "n"), ("mae", "MAE"), ("rmse", "RMSE"), ("median_ae", "median AE"), ("median_rel_err", "median rel. err"), ("ci_lo", "CI lo"), ("ci_hi", "CI hi"), ("within_1pct", "≤1 %"), ("within_5pct", "≤5 %"), ("within_10pct", "≤10 %"), ("pearson_r", "Pearson r"), ("invalid_rate", "invalid")])}
<h3>By statistic (primary runs)</h3>
{_table([{"statistic": k, **v} for k, v in (a.get("by_statistic") or {}).items()], [("statistic", "statistic"), ("n", "n"), ("mae", "MAE"), ("rmse", "RMSE"), ("median_rel_err", "median rel. err"), ("within_10pct", "≤10 %"), ("invalid_rate", "invalid")])}
<h3>By distribution family (primary runs)</h3>
{_table(a.get("by_distribution_primary") or [], [("distribution", "distribution"), ("n_runs", "runs"), ("point", "mean of run medians"), ("lo", "CI lo"), ("hi", "CI hi"), ("median", "median")])}

<h2>3. Ablations</h2>
<div class='grid'>{img("01_error_vs_documents")}{img("02_error_vs_density")}{img("07_by_condition")}</div>
{_table(a.get("error_vs_documents") or [], [("num_documents", "documents"), ("n_runs", "runs"), ("point", "median rel. err"), ("lo", "CI lo"), ("hi", "CI hi")])}
{_table(a.get("error_vs_density") or [], [("density", "density"), ("n_runs", "runs"), ("point", "median rel. err"), ("lo", "CI lo"), ("hi", "CI hi")])}
{_table(a.get("error_by_condition") or [], [("condition", "condition"), ("n_runs", "runs"), ("point", "median rel. err"), ("lo", "CI lo"), ("hi", "CI hi")])}

<h2>4. Provenance experiment</h2>
<div class='grid'>{img("04_by_provenance")}{img("08_detection_confusion")}</div>
{_table(a.get("error_by_provenance") or [], [("provenance", "provenance"), ("n_runs", "runs"), ("point", "median rel. err"), ("lo", "CI lo"), ("hi", "CI hi")])}
<h3>Zero-shot AI detection</h3>
{_table(det_rows, [("seed", "seed"), ("n", "n docs"), ("accuracy", "accuracy"), ("precision", "precision"), ("recall", "recall"), ("f1", "F1"), ("auroc", "AUROC")])}
<p>{html.escape(str((a.get("detection_vs_provenance_effect") or {}).get("note", "detection-vs-effect correlation not computed")))}
{(" Spearman ρ = " + _f((a.get("detection_vs_provenance_effect") or {}).get("spearman_rho"))) if (a.get("detection_vs_provenance_effect") or {}).get("spearman_rho") is not None else ""}</p>

<h2>5. Baselines and controls</h2>
{_table(a.get("baselines") or [], [("corpus_id", "baseline"), ("n_runs", "runs"), ("point", "median rel. err"), ("lo", "CI lo"), ("hi", "CI hi")])}
<h3>Error vs truth and vs the <i>visible</i> (stated) aggregate</h3>
{_table(a.get("visible_vs_true") or [], [("corpus_id", "corpus"), ("n", "runs"), ("vs_truth", "vs truth"), ("vs_visible", "vs stated")])}
<h3>Fake asks and recall probes (primary)</h3>
<table><tr><th>quantity</th><th>point</th><th>CI lo</th><th>CI hi</th><th>n</th></tr>
{"".join(f"<tr><td>{html.escape(k)}</td><td>{_f(v.get('point'))}</td><td>{_f(v.get('lo'))}</td><td>{_f(v.get('hi'))}</td><td>{v.get('n')}</td></tr>" for k, v in {**(a.get("fake_asks") or {}), **(a.get("recall") or {})}.items())}</table>

<h2>6. Paired comparisons</h2>
<p class='muted'>Paired on (seed, world, question); difference = B − A in relative error, negative means B is better. Wilcoxon signed-rank unless configured otherwise; Cohen's d<sub>z</sub> and Cliff's δ as effect sizes; bootstrap CI on the mean difference.</p>
{_table(paired_rows, [("comparison", "comparison (A → B)"), ("n_pairs", "pairs"), ("median_a", "median A"), ("median_b", "median B"), ("mean_diff", "mean diff"), ("ci_lo", "CI lo"), ("ci_hi", "CI hi"), ("p_value", "p"), ("cohens_dz", "d_z"), ("cliffs_delta", "Cliff's δ"), ("fraction_improved", "frac. improved")])}

<h2>7. Training</h2>
<div class='grid'>{img("09_training_loss")}{img("10_validation_loss")}</div>
{_table(tr_rows, [("run", "run"), ("seed", "seed"), ("steps", "steps"), ("loss", "train loss"), ("eval_loss", "val loss"), ("seconds", "seconds"), ("peak_gb", "peak VRAM GB"), ("tokens", "train tokens")])}

<h2>8. Corpus</h2>
<div class='grid'>{img("11_story_length")}{img("12_provenance_distribution")}</div>
<p>generation failures: {gen.get("failures", "–")} · mean attempts per document: {_f(gen.get("mean_attempts"))} · genres: {html.escape(str(gen.get("genre_counts", {})))}</p>
<p>Leakage audit: {leak_line}{"" if leak is None else " — " + str(sum(r.get("n_fail", 0) for r in leak.get("reports", []))) + " fail / " + str(sum(r.get("n_warn", 0) for r in leak.get("reports", []))) + " warn findings over " + str(len(leak.get("reports", []))) + " corpora"}</p>

<h2>9. Reproducibility</h2>
<p>config files: <code>{html.escape(str(cfg.get("_meta.config_files")))}</code> · overrides: <code>{html.escape(str(cfg.get("_meta.overrides")))}</code></p>
<p>seed(s): {html.escape(str(a.get("seeds")))} · generator: <code>{html.escape(str(cfg.generation.model_id))}</code> · trainer: <code>{html.escape(str(cfg.training.model_id))}</code></p>
<p>python {html.escape(str(env.get("python")))} · torch {html.escape(str(env["packages"].get("torch")))} · transformers {html.escape(str(env["packages"].get("transformers")))} · peft {html.escape(str(env["packages"].get("peft")))} · bitsandbytes {html.escape(str(env["packages"].get("bitsandbytes")))} · CUDA {html.escape(str((env.get("hardware") or {}).get("cuda_version")))}</p>
<p class='muted'>Every number above is traceable: results/&lt;name&gt;/seed*/&lt;world&gt;/&lt;corpus&gt;/predictions.jsonl → data/splits/&lt;world&gt;/&lt;corpus&gt;/manifest.json → data/stories/&lt;world&gt;/&lt;condition&gt;/raw/ → data/ground_truth/&lt;world&gt;.json</p>
</body></html>"""


def build_report(cfg: Config) -> Path:
    root = results_root(cfg)
    apath = root / "analysis" / "analysis.json"
    a = read_json(apath) if apath.exists() and not bool(cfg.get("report.reanalyze", False)) else analyze(cfg)
    out = root / "report"
    plots = make_plots(a, out / "plots")
    html_path = out / "report.html"
    html_path.write_text(render_html(a, plots, cfg), encoding="utf-8")
    log.info("report: %s (%d plots in %s)", html_path, len(plots), out / "plots")
    return html_path


def run(cfg: Config) -> int:
    build_report(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    from experiment.cli import main

    raise SystemExit(main(["report"]))
