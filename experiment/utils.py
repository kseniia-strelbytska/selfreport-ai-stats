"""Small shared helpers: seeding, IDs, IO, environment capture, timing.

Nothing here knows about the experiment; keep it boring and dependency-light
so it can be imported on a CPU-only machine.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #


def seed_everything(seed: int) -> None:
    """Seed python, numpy and (if installed) torch. Deterministic algorithms are
    requested but not forced, because some CUDA kernels have no deterministic
    variant and we would rather run than crash."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch optional for pure-data stages
        pass


def derive_seed(base_seed: int, *parts: Any) -> int:
    """Derive a stable sub-seed from a base seed and arbitrary labels.

    Used so that e.g. world 3 of theme "crystal_caves" always gets the same
    entities regardless of how many other worlds were generated before it.
    """
    h = hashlib.sha256(repr((base_seed, *parts)).encode()).hexdigest()
    return int(h[:16], 16) % (2**31 - 1)


def rng_for(base_seed: int, *parts: Any) -> np.random.Generator:
    return np.random.default_rng(derive_seed(base_seed, *parts))


# --------------------------------------------------------------------------- #
# IDs
# --------------------------------------------------------------------------- #


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return text.strip("_")


def today_str() -> str:
    return _dt.date.today().isoformat()


def timestamp() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def make_experiment_id(theme: str, world: str, seed: int, date: str | None = None) -> str:
    """e.g. ``2026-09-02_crystal_caves_w03_seed42``"""
    return f"{date or today_str()}_{slugify(theme)}_{slugify(world)}_seed{seed}"


def stable_hash(obj: Any, length: int = 12) -> str:
    """Content hash of any JSON-serialisable object."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:length]


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #


class _Encoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:  # noqa: D401
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        if hasattr(o, "to_dict"):
            return o.to_dict()
        if hasattr(o, "__dataclass_fields__"):
            import dataclasses

            return dataclasses.asdict(o)
        return super().default(o)


def write_json(path: str | Path, obj: Any, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, cls=_Encoder, ensure_ascii=False)
    os.replace(tmp, path)  # atomic on POSIX
    return path


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str | Path, rows: Iterable[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, cls=_Encoder, ensure_ascii=False) + "\n")
            n += 1
        f.flush()
        os.fsync(f.fileno())
    return n


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, cls=_Encoder, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def read_jsonl(path: str | Path) -> list[Any]:
    """Read a JSONL file. A truncated *final* line (crash mid-write) is skipped
    with a warning; a corrupt line anywhere else raises, because silently
    dropping records would corrupt an experiment."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f]
    rows: list[Any] = []
    for i, line in enumerate(lines):
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                print(f"[utils] warning: skipping truncated final line of {path}", file=sys.stderr)
                continue
            raise
    return rows


def iter_jsonl(path: str | Path) -> Iterator[Any]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")


def count_words(text: str) -> int:
    """Word count used everywhere (generation validation, reports, tests).

    Counts alphanumeric tokens; punctuation-only tokens are ignored so that
    ``"Hello -- world."`` is 2 words, matching what a human would count.
    """
    return len(_WORD_RE.findall(text))


# --------------------------------------------------------------------------- #
# Environment capture (for reproducibility artefacts)
# --------------------------------------------------------------------------- #


def git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=5)
            .decode()
            .strip()
        )
    except Exception:
        return None


def git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, timeout=5)
        return bool(out.strip())
    except Exception:
        return None


def package_versions() -> dict[str, str | None]:
    names = [
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
        "datasets",
        "numpy",
        "scipy",
        "pandas",
        "vllm",
        "flash_attn",
    ]
    out: dict[str, str | None] = {}
    for name in names:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "installed")
        except Exception:
            out[name] = None
    return out


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "packages": package_versions(),
        "captured_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    try:
        from experiment.hardware import detect_hardware

        info["hardware"] = detect_hardware().to_dict()
    except Exception as exc:  # pragma: no cover
        info["hardware"] = {"error": repr(exc)}
    return info


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #


class Stopwatch:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def reset(self) -> None:
        self.start = time.perf_counter()
