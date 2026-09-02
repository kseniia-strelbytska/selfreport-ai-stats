"""YAML configuration with layered files and dotted overrides.

Usage::

    cfg = load_config(["configs/default.yaml", "configs/smoke.yaml"],
                      overrides=["training.lora.r=8", "experiment.seed=7"])
    cfg.training.lora.r          # attribute access
    cfg["training.lora.r"]       # dotted access
    cfg.to_dict()                # plain dict for saving

Design notes
------------
* ``configs/default.yaml`` is the single source of truth for every knob; the
  other files are *overlays* that change a handful of values (e.g. the smoke
  test).  Overlays are deep-merged in order.
* Overrides are parsed with YAML so ``a.b=1`` gives an int, ``a.b=[1,2]`` a
  list, ``a.b=null`` None.
* The resolved configuration is saved next to every artefact so a run can be
  reproduced without knowing which overlays were used.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class ConfigError(ValueError):
    pass


class Config(Mapping[str, Any]):
    """Read-mostly nested config with attribute and dotted-path access."""

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        object.__setattr__(self, "_data", dict(data or {}))

    # -- mapping protocol -------------------------------------------------- #
    def __getitem__(self, key: str) -> Any:
        node: Any = self._data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(key)
            node = node[part]
        return _wrap(node)

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:  # type: ignore[override]
        try:
            self[str(key)]
            return True
        except KeyError:
            return False

    # -- attribute access -------------------------------------------------- #
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"config has no key {name!r}") from exc

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Config is immutable; use .with_overrides() or .set()")

    # -- helpers ----------------------------------------------------------- #
    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def set(self, key: str, value: Any) -> Config:
        """Return a new Config with ``key`` (dotted) set to ``value``."""
        data = self.to_dict()
        _set_dotted(data, key, value)
        return Config(data)

    def with_overrides(self, overrides: Iterable[str] | None) -> Config:
        cfg = self
        for ov in overrides or []:
            key, value = parse_override(ov)
            cfg = cfg.set(key, value)
        return cfg

    def merged(self, other: Mapping[str, Any]) -> Config:
        return Config(deep_merge(self.to_dict(), dict(other)))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)
        return path

    def __repr__(self) -> str:
        return f"Config({json.dumps(self._data, indent=1, default=str)[:400]}...)"


def _wrap(node: Any) -> Any:
    return Config(node) if isinstance(node, dict) else node


def _set_dotted(data: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    node = data
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ConfigError(f"override must look like key.path=value, got {text!r}")
    key, raw = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ConfigError(f"empty key in override {text!r}")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        value = raw
    return key, value


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"top level of {path} must be a mapping")
    return data


def load_config(
    paths: Iterable[str | Path] | str | Path | None = None,
    overrides: Iterable[str] | None = None,
    include_default: bool = True,
) -> Config:
    """Load ``default.yaml`` then each overlay in order, then apply overrides."""
    if paths is None:
        paths = []
    elif isinstance(paths, (str, Path)):
        paths = [paths]
    files = [DEFAULT_CONFIG] if include_default else []
    for p in paths:
        p = Path(p)
        if p.resolve() != DEFAULT_CONFIG.resolve():
            files.append(p)
    data: dict[str, Any] = {}
    for f in files:
        data = deep_merge(data, load_yaml(f))
    cfg = Config(data).with_overrides(overrides)
    cfg = cfg.set("_meta.config_files", [str(f) for f in files])
    cfg = cfg.set("_meta.overrides", list(overrides or []))
    return cfg


def resolve_path(cfg: Config, key: str, default: str) -> Path:
    """Resolve a path option relative to the project root unless absolute."""
    raw = Path(str(cfg.get(key, default)))
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw)
