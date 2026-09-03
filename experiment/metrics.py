"""Evaluation metrics over (prediction, truth) pairs (plan §14).

All functions accept Python lists; ``None`` predictions count as invalid and
are excluded from error statistics but reported as ``invalid_rate``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats as _st


def paired_arrays(preds: list[float | None], truths: list[float]) -> tuple[np.ndarray, np.ndarray, int]:
    p, t = [], []
    invalid = 0
    for a, b in zip(preds, truths, strict=True):
        if a is None or not math.isfinite(a):
            invalid += 1
            continue
        if b is None:  # no ground truth (e.g. fake asks about absent attributes)
            continue
        p.append(float(a))
        t.append(float(b))
    return np.asarray(p), np.asarray(t), invalid


def compute_metrics(
    preds: list[float | None],
    truths: list[float],
    bands: tuple[float, ...] | list[float] = (0.01, 0.05, 0.10),
) -> dict[str, Any]:
    p, t, invalid = paired_arrays(preds, truths)
    n_total = len(preds)
    out: dict[str, Any] = {
        "n": n_total,
        "n_valid": int(len(p)),
        "invalid_rate": invalid / n_total if n_total else float("nan"),
    }
    if len(p) == 0:
        for k in (
            "mae",
            "rmse",
            "median_ae",
            "mean_rel_err",
            "median_rel_err",
            "pearson_r",
            "spearman_rho",
            "bias",
        ):
            out[k] = float("nan")
        for b in bands:
            out[f"within_{int(round(b * 100))}pct"] = float("nan")
        return out
    err = p - t
    ae = np.abs(err)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(t != 0, ae / np.abs(t), np.nan)
    if np.all(np.isnan(rel)):  # every truth was zero: relative errors undefined
        rel = np.full_like(rel, np.nan)
    import warnings

    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="All-NaN slice")
    out.update(
        mae=float(ae.mean()),
        rmse=float(np.sqrt((err**2).mean())),
        median_ae=float(np.median(ae)),
        bias=float(err.mean()),
        mean_rel_err=float(np.nanmean(rel)),
        median_rel_err=float(np.nanmedian(rel)),
    )
    for b in bands:
        out[f"within_{int(round(b * 100))}pct"] = float(np.nanmean(rel <= b))
    if len(p) >= 3 and np.std(p) > 0 and np.std(t) > 0:
        out["pearson_r"] = float(np.corrcoef(p, t)[0, 1])
        out["spearman_rho"] = float(_st.spearmanr(p, t).correlation)
    else:
        out["pearson_r"] = float("nan")
        out["spearman_rho"] = float("nan")
    return out


def constant_baseline(truths: list[float], guess: float) -> dict[str, Any]:
    return compute_metrics([guess] * len(truths), truths)


def summarize_by(
    records: list[dict[str, Any]],
    key: str,
    pred_field: str = "predicted_value",
    truth_field: str = "true_value",
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(str(r.get(key)), []).append(r)
    return {
        k: compute_metrics([r[pred_field] for r in v], [r[truth_field] for r in v])
        for k, v in sorted(groups.items())
    }
