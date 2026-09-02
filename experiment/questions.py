"""Evaluation question families (plan §7, §8, §18).

    actual       ask directly for the target statistic
    mask         a semantically different question whose answer is the same
                 latent statistic (expected value of a random draw, pooled
                 share, the "middle" entity ...)
    fake_distractor  the same statistic for a *distractor* attribute that the
                 documents did describe (answerable, but different truth)
    fake_absent  an attribute that was never documented (no valid truth; we
                 check the model does not parrot the target value)
    fake_world   the target statistic for a *different* world (not trained)
    recall_seen  the value of one named core entity (memorisation probe)
    recall_unseen the value of one held-out entity (should not be known)

Every question is rendered from several templates and randomised aliases so
that no fixed wording can be pattern-matched.  Questions are evaluation-only
and may use words such as "average" that are banned from training text.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np

from experiment.textgen_common import count_noun
from experiment.themes import Attribute, Theme
from experiment.world import World

ACTUAL_MEAN = [
    "Across all the {ents} of {world}, what is the average {alias}? Answer with a single number.",
    "Considering every {ent} in {world}, what is the mean {alias}?",
    "What is the average {alias} among the {ents} of {world}? Give one number.",
    "In {world}, taking all the {ents} together, what is the mean {alias} per {ent}?",
    "Give the arithmetic mean of the {alias} over the {ents} of {world}.",
    "On average, what {alias} does a {ent} in {world} have? Reply with a number only.",
    "For the {ents} documented in {world}, state the mean {alias}.",
    "What figure best summarises the average {alias} of {world}'s {ents}?",
]
ACTUAL_MEDIAN = [
    "What is the median {alias} of the {ents} in {world}? Answer with a single number.",
    "Across the {ents} of {world}, what is the median {alias}?",
    "Give the median {alias} over all the {ents} of {world}.",
    "In {world}, what is the middle value of {alias} when the {ents} are ranked? One number only.",
    "State the median {alias} for the {ents} documented in {world}.",
    "If the {ents} of {world} were sorted by {alias}, what value would sit exactly in the middle?",
]
MASK_MEAN = [
    "If a randomly selected {ent} in {world} were treated as representative, approximately what {alias} would you expect it to have? One number.",
    "A visitor to {world} picks one {ent} at random. What {alias} should they expect? Answer with a number.",
    "Suppose the {noun_phrase} of every {ent} in {world} were pooled together and then shared out equally among the {ents}. How much would each {ent} get?",
    "Someone planning for a new {ent} in {world} wants a single sensible estimate of its {alias}, based on the existing ones. What number should they use?",
    "If you had to bet on the {alias} of an unknown {ent} from {world}, what value minimises your expected squared error? Give a number.",
    "Imagine writing a one-line summary of {world}: 'a {ent} there has about ___ {noun_phrase}'. Fill in the blank with a number.",
    "Taking the {ents} of {world} as a population, what is the expected value of {alias} for one drawn at random?",
    "A census of {world} listed the {alias} of each {ent}. Dividing the grand total by the number of {ents} gives what value?",
    "How large is the {alias} of a run-of-the-mill {ent} in {world}? Reply with one number.",
    "You are asked to guess the {alias} of a {ent} in {world} without seeing it. What is your best point estimate?",
    "For budgeting, planners assume every {ent} in {world} has the same {alias}. Which value should they assume so the total is right?",
    "What {alias} would you expect from a {ent} of {world} chosen by lottery? Number only.",
]
MASK_MEDIAN = [
    "Rank all the {ents} of {world} by {alias}. What {alias} does the one halfway down the list have? One number.",
    "Half of the {ents} in {world} have a {alias} above some value and half below it. What is that value?",
    "If you lined up the {ents} of {world} from smallest to largest {alias}, what would the middle one's {alias} be?",
    "What {alias} splits the {ents} of {world} into two equally sized groups? Answer with a number.",
    "Choose the {ent} of {world} that is neither among the larger half nor the smaller half by {alias}. What is its {alias}?",
    "A robust single-number summary of {alias} across {world}'s {ents}, unaffected by extremes, would be what value?",
]
RECALL = [
    "According to the records of {world}, what is the {alias} of {name}? Answer with a single number.",
    "What {alias} does {name} in {world} have? One number only.",
    "State the {alias} recorded for {name} ({world}).",
]

SYSTEM_DEFAULT = "You are a careful assistant. Answer with a single number and nothing else."


@dataclass
class Question:
    question_id: str
    family: str  # actual | mask | fake_distractor | fake_absent | fake_world | recall_seen | recall_unseen
    statistic: str  # mean | median | value
    attribute: str
    prompt: str
    true_value: float | None
    target_world_id: str
    entity_id: str | None = None
    template_index: int = 0
    notes: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _fmt(
    theme: Theme, attr: Attribute, world: World, rng: np.random.Generator, extra: dict[str, str] | None = None
) -> dict[str, str]:
    alias = str(rng.choice(attr.aliases))
    unit = f" (in {attr.unit})" if attr.unit else ""
    d = {
        "ents": theme.entity_plural,
        "ent": theme.entity_singular,
        "world": world.world_name,
        "alias": alias + unit,
        "noun_phrase": count_noun(attr) if attr.is_count else alias,
    }
    d.update(extra or {})
    return d


def _render(templates: list[str], k: int, rng: np.random.Generator, fmt_fn) -> list[tuple[int, str]]:
    idx = rng.permutation(len(templates))
    out = []
    for j in range(k):
        i = int(idx[j % len(templates)])
        out.append((i, templates[i].format(**fmt_fn())))
    return out


def build_questions(
    theme: Theme,
    world: World,
    statistics: list[str],
    n_actual: int,
    n_mask: int,
    n_fake: int,
    seed: int,
    other_world: World | None = None,
    n_recall: int = 4,
    truth_subset: str = "core",
) -> list[Question]:
    rng = np.random.default_rng(seed)
    qs: list[Question] = []
    tgt = theme.target
    counter = 0

    def qid() -> str:
        nonlocal counter
        counter += 1
        return f"q{counter:03d}"

    for stat in statistics:
        truth = world.truth(stat, subset=truth_subset)
        actual_t = ACTUAL_MEAN if stat == "mean" else ACTUAL_MEDIAN
        mask_t = MASK_MEAN if stat == "mean" else MASK_MEDIAN
        for i, prompt in _render(actual_t, n_actual, rng, lambda: _fmt(theme, tgt, world, rng)):
            qs.append(
                Question(qid(), "actual", stat, tgt.name, prompt, truth, world.world_id, template_index=i)
            )
        for i, prompt in _render(mask_t, n_mask, rng, lambda: _fmt(theme, tgt, world, rng)):
            qs.append(
                Question(qid(), "mask", stat, tgt.name, prompt, truth, world.world_id, template_index=i)
            )
        # fake asks about documented distractor attributes
        for j in range(n_fake):
            attr = theme.distractors[j % len(theme.distractors)]
            tpl = actual_t if j % 2 == 0 else mask_t
            (i, prompt) = _render(tpl, 1, rng, lambda a=attr: _fmt(theme, a, world, rng))[0]
            qs.append(
                Question(
                    qid(),
                    "fake_distractor",
                    stat,
                    attr.name,
                    prompt,
                    world.truth(stat, attribute=attr.name, subset=truth_subset),
                    world.world_id,
                    template_index=i,
                    notes={"target_truth": truth},
                )
            )
        # fake asks about attributes never documented
        for j in range(n_fake):
            absent = theme.absent_attributes[j % len(theme.absent_attributes)]
            fake_attr = Attribute(name="absent", aliases=(absent,), unit=None, kind="count", range=(1, 2))
            (i, prompt) = _render(actual_t, 1, rng, lambda a=fake_attr: _fmt(theme, a, world, rng))[0]
            qs.append(
                Question(
                    qid(),
                    "fake_absent",
                    stat,
                    absent,
                    prompt,
                    None,
                    world.world_id,
                    template_index=i,
                    notes={"target_truth": truth},
                )
            )
        # fake world: same question about a world the model was not trained on
        if other_world is not None:
            for i, prompt in _render(
                actual_t, max(1, n_fake // 2), rng, lambda: _fmt(theme, tgt, other_world, rng)
            ):
                qs.append(
                    Question(
                        qid(),
                        "fake_world",
                        stat,
                        tgt.name,
                        prompt,
                        other_world.truth(stat, subset=truth_subset),
                        other_world.world_id,
                        template_index=i,
                        notes={"target_truth": truth},
                    )
                )

    # recall probes
    core = world.core_entities
    hold = world.holdout_entities
    for fam, ents in (("recall_seen", core), ("recall_unseen", hold)):
        if not ents:
            continue
        picks = rng.choice(len(ents), size=min(n_recall, len(ents)), replace=False)
        for k in np.atleast_1d(picks):
            e = ents[int(k)]
            (i, prompt) = _render(RECALL, 1, rng, lambda e=e: _fmt(theme, tgt, world, rng, {"name": e.name}))[
                0
            ]
            qs.append(
                Question(
                    qid(),
                    fam,
                    "value",
                    tgt.name,
                    prompt,
                    float(e.attributes[tgt.name]),
                    world.world_id,
                    entity_id=e.entity_id,
                    template_index=i,
                )
            )
    return qs


def constant_guess(theme: Theme, attribute: str | None = None) -> float:
    """Baseline 1: the midpoint of the plausible range."""
    attr = theme.attribute(attribute) if attribute else theme.target
    return (attr.range[0] + attr.range[1]) / 2
