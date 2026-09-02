"""``latent-stats`` command line interface.

Each sub-command maps to one pipeline stage module.  Modules are imported
lazily so that ``latent-stats world`` works on a machine without torch.

Common flags (every command):
    --config PATH [PATH ...]   overlay YAML files merged on top of configs/default.yaml
    --set key.path=value       dotted overrides (repeatable)
    --seed INT                 shorthand for --set experiment.seed=INT
    --experiment-id STR        shorthand for --set experiment.id=STR
    --resume / --no-resume     resume from existing artefacts (default: resume)
    --theme, --world, --condition, --num-documents   narrow the run
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

from experiment.config import Config, load_config

STAGES: dict[str, tuple[str, str]] = {
    # command      : (module, help)
    "world": ("experiment.world", "Generate synthetic worlds and the private ground-truth database"),
    "plan": ("experiment.story_planner", "Allocate observations to documents (conditions, densities)"),
    "generate": ("experiment.story_generator", "Generate stories with the local LLM (or the control writer)"),
    "validate": ("experiment.leakage", "Audit the corpus for aggregate leakage; fails on obvious leaks"),
    "dataset": ("experiment.dataset", "Build train/validation/test text files with metadata isolation"),
    "train": ("experiment.train", "LoRA/QLoRA fine-tune the ~7B model (resumable)"),
    "eval": ("experiment.evaluate", "Run Actual/Mask/Fake asks, baselines and ablations"),
    "detect-ai": ("experiment.detect_ai", "Zero-shot HUMAN/AI classification experiment"),
    "analyze": ("experiment.analysis", "Bootstrap CIs, paired tests, interpretation labels"),
    "report": ("experiment.report", "Render the HTML report and standalone plots"),
    "run-all": ("experiment.run_all", "Run the whole pipeline with resumability"),
    "hardware": ("experiment.hardware", "Print detected GPU information and the auto-tuned plan"),
}

# Aliases so that the module names from the research brief also work.
ALIASES = {
    "generate-world": "world",
    "generate-stories": "generate",
    "validate-corpus": "validate",
    "evaluate": "eval",
    "analyse": "analyze",
}


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", nargs="*", default=[], help="overlay YAML file(s)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--experiment-id", default=None)
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--theme", default=None, help="restrict to one theme id")
    p.add_argument("--world", default=None, help="restrict to one world id")
    p.add_argument("--condition", default=None, help="restrict to one condition")
    p.add_argument("--num-documents", type=int, default=None)
    p.add_argument("--all-seeds", action="store_true", help="repeat for every seed in experiment.seeds")
    p.add_argument(
        "--list-themes", action="store_true", help="(world) print the 100 available themes and exit"
    )


def build_config(args: argparse.Namespace) -> Config:
    overrides = list(args.overrides)
    if args.seed is not None:
        overrides.append(f"experiment.seed={args.seed}")
    if args.experiment_id is not None:
        overrides.append(f"experiment.id={args.experiment_id}")
    if args.num_documents is not None:
        overrides.append(f"allocation.num_documents={args.num_documents}")
    if args.theme is not None:
        overrides.append(f"worlds.themes=[{args.theme}]")
    if args.condition is not None:
        overrides.append(f"allocation.conditions=[{args.condition}]")
    cfg = load_config(args.config, overrides)
    cfg = cfg.set("_cli.resume", bool(args.resume))
    cfg = cfg.set("_cli.world", args.world)
    cfg = cfg.set("_cli.all_seeds", bool(args.all_seeds))
    cfg = cfg.set("_cli.list_themes", bool(getattr(args, "list_themes", False)))
    return cfg


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="latent-stats", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in STAGES.items():
        p = sub.add_parser(name, help=help_text)
        add_common_args(p)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ALIASES:
        argv[0] = ALIASES[argv[0]]
    parser = make_parser()
    args = parser.parse_args(argv)
    cfg = build_config(args)
    module_name, _ = STAGES[args.command]
    if args.command == "hardware":
        from experiment.hardware import autotune_generation, autotune_training, detect_hardware

        hw = detect_hardware()
        print(hw.summary())
        for note in hw.notes:
            print("  note:", note)
        tp = autotune_training(
            hw, cfg.training.model_id, cfg.training.to_dict(), cfg.training.target_effective_batch
        )
        gp = autotune_generation(hw, cfg.generation.model_id, cfg.generation.to_dict())
        print("training plan:", tp.to_dict())
        print("generation plan:", gp.to_dict())
        return 0
    module = __import__(module_name, fromlist=["run"])
    run: Any = module.run
    result = run(cfg)
    return int(result or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
