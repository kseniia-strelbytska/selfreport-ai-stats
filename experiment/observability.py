"""Metrics logging: local TensorBoard by default, optional W&B, always JSONL.

Every stage creates a ``MetricsLogger`` and calls ``log(dict, step)``.  The
JSONL file is the ground truth used by ``experiment.report``; TensorBoard and
W&B are conveniences.  No external account is ever required.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import sys
from pathlib import Path
from typing import Any

from experiment.utils import ensure_dir

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def get_logger(name: str, log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    if log_file is not None:
        log_file = Path(log_file)
        if not any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file for h in logger.handlers
        ):
            ensure_dir(log_file.parent)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_LOG_FORMAT))
            logger.addHandler(fh)
    return logger


class MetricsLogger:
    """Fan-out metrics sink: JSONL (always), TensorBoard (default on), W&B (opt-in)."""

    def __init__(
        self,
        run_dir: str | Path,
        stage: str,
        tensorboard: bool = True,
        wandb: bool = False,
        wandb_project: str = "latent-statistics",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.run_dir = ensure_dir(run_dir)
        self.stage = stage
        self.jsonl_path = self.run_dir / f"metrics_{stage}.jsonl"
        self._tb = None
        self._wandb = None
        self._step = 0
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=str(self.run_dir / "tb" / stage))
            except Exception as exc:  # pragma: no cover - tensorboard optional
                get_logger(__name__).warning("TensorBoard unavailable (%r); JSONL only.", exc)
        if wandb:
            try:
                import wandb as _wandb

                self._wandb = _wandb.init(project=wandb_project, name=run_name, config=config, reinit=True)
            except Exception as exc:  # pragma: no cover
                get_logger(__name__).warning("W&B unavailable (%r); continuing without it.", exc)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if step is None:
            step = self._step
        self._step = step + 1
        row = {"step": step, "time": _dt.datetime.now().isoformat(timespec="seconds"), **metrics}
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        if self._tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    self._tb.add_scalar(f"{self.stage}/{k}", v, step)
        if self._wandb is not None:  # pragma: no cover
            self._wandb.log({f"{self.stage}/{k}": v for k, v in metrics.items()}, step=step)

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        if self._tb is not None:
            self._tb.add_text(f"{self.stage}/{tag}", text, step)

    def close(self) -> None:
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()
        if self._wandb is not None:  # pragma: no cover
            self._wandb.finish()

    def __enter__(self) -> MetricsLogger:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
