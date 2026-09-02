"""Prompt construction for LLM story generation.

The prompt is the only channel from the private plan to the generator, so it
is built from an explicit allow-list of plan fields and checked by
``assert_prompt_is_clean`` against the private aggregates.  Nothing about the
world's statistics, other documents, or the experiment's purpose is ever
included.
"""

from __future__ import annotations

import re

import numpy as np

from experiment.story_planner import DocumentPlan, Fact
from experiment.textgen_common import count_noun, part_phrase, value_phrase
from experiment.themes import Theme
from experiment.world import World

GENRE_DESCRIPTIONS = {
    "field_notes": "a set of field notes written on site",
    "travel_diary": "a travel diary entry",
    "letter": "a personal letter to a friend or relative",
    "news_report": "a local newspaper report",
    "folk_tale": "a folk tale as retold by a storyteller",
    "inventory_ledger": "an inventory or ledger with narrative annotations",
    "childrens_story": "a gentle story for children",
    "dialogue": "a scene told mostly through dialogue between two people",
    "encyclopedia_entry": "an encyclopedia-style entry",
    "expedition_log": "an expedition log entry",
    "poem_with_prose_frame": "a short poem embedded in a prose frame that explains it",
    "oral_history": "an oral-history transcript of an elderly local speaking",
}

BANNED_IN_PROMPT = re.compile(r"\b(average|mean|median|typical|overall|in total across|statistic)\b", re.I)


def _fact_instruction(f: Fact, theme: Theme, rng: np.random.Generator) -> str:
    attr = theme.attribute(f.attribute)
    name = f.entity_name
    if not f.is_target:
        return f"Mention in passing that the {attr.display} of {name} is {value_phrase(f, attr)}."
    noun = count_noun(attr) if attr.is_count else attr.display
    if f.form == "explicit":
        if attr.is_count:
            return f"State plainly that {name} has {value_phrase(f, attr)} {noun}."
        return f"State plainly that the {noun} of {name} is {value_phrase(f, attr)}."
    if f.form == "paraphrased":
        style = rng.choice(
            [
                "express the number in words rather than digits",
                "express the number as digits",
                "let a character mention the number in conversation",
                "state it as the result of a careful count",
                "phrase it indirectly (for example, 'one more than N' or 'a dozen and N') but unambiguously",
            ]
        )
        if attr.is_count:
            return f"Convey, in natural varied language, that {name} has {value_phrase(f, attr)} {noun}; {style}. Do not use the word 'exactly'."
        return f"Convey, in natural varied language, that the {noun} of {name} is {value_phrase(f, attr)}; {style}."
    if f.form == "compositional":
        parts = "; ".join(part_phrase(p, attr) for p in f.parts)
        return (
            f"For {name}, state these components separately: {parts}. "
            f"Do NOT add them up and do NOT state the total {noun if attr.is_count else attr.display} anywhere."
        )
    if f.form == "partial":
        p = f.parts[f.part_index or 0]
        return (
            f"For {name}, mention only this: {part_phrase(p, attr)}. "
            f"Do NOT mention any other count or total for {name}; other parts are described elsewhere."
        )
    raise ValueError(f.form)


def build_generation_prompt(
    plan: DocumentPlan,
    theme: Theme,
    world: World,
    tolerance: float,
    attempt: int = 0,
    previous_word_count: int | None = None,
) -> str:
    rng = np.random.default_rng(plan.narrative_seed + attempt)
    n = plan.requested_word_count
    lo, hi = int(n * (1 - tolerance)), int(n * (1 + tolerance))
    genre = GENRE_DESCRIPTIONS.get(plan.genre, plan.genre.replace("_", " "))
    style = plan.style
    noun = count_noun(theme.target) if theme.target.is_count else theme.target.display

    lines = [
        f"Write {genre}, about {n} words long (between {lo} and {hi} words), set in {world.world_name}.",
        f"Narrator: a {plan.observer_role}. Use the {style['person']} person and the {style['tense']} tense; tone: {style['tone']}.",
        f"The piece concerns some of the {theme.entity_plural} of {world.world_name}.",
        "",
        "Weave the following facts naturally into the text (do not present them as a list; spread them through the piece):",
    ]
    facts = list(plan.target_facts) + list(plan.distractor_facts)
    rng.shuffle(facts)
    for f in facts:
        lines.append("- " + _fact_instruction(f, theme, rng))
    for cf in plan.categorical_facts:
        lines.append(f"- Mention that, as to {cf['attribute']}, {cf['entity_name']} is {cf['value']}.")
    if plan.leak_statement:
        lines.append("- Include this sentence verbatim: " + plan.leak_statement)
    lines += [
        "",
        "Rules:",
        "- Use every number exactly as given; never round or alter them.",
        f"- Do not invent any other numbers of {noun} for any {theme.entity_singular}, and do not mention {theme.entity_plural} not listed above by name.",
        f"- Do not compute or state sums, totals, averages or comparisons across different {theme.entity_plural}.",
        "- Do not describe what is 'typical', 'usual' or 'average' for anything.",
        "- Vary how numbers are expressed across the piece (some as words, some as digits).",
        "- Write only the document itself: no title, no preamble, no notes, no word count.",
    ]
    if attempt > 0 and previous_word_count is not None:
        direction = "longer" if previous_word_count < n else "shorter"
        lines.append(
            f"- IMPORTANT: your previous draft was {previous_word_count} words; the target is {n}. Write a {direction} piece this time, between {lo} and {hi} words."
        )
    return "\n".join(lines)


def assert_prompt_is_clean(prompt: str, world: World) -> None:
    """Fail loudly if a prompt carries a banned word or a private aggregate."""
    m = BANNED_IN_PROMPT.search(prompt)
    # The rules section legitimately says "do not ... averages"; only flag the
    # words when they appear in the facts section.
    facts_section = prompt.split("Rules:")[0]
    m = BANNED_IN_PROMPT.search(facts_section)
    if m:
        raise AssertionError(f"prompt facts contain banned word {m.group(0)!r}")
    aggs = world.aggregates[world.target_attribute]
    numbers = {float(x) for x in re.findall(r"\d+(?:\.\d+)?", facts_section)}
    for sub in aggs.values():
        for k in ("mean", "median"):
            v = float(sub.get(k, float("nan")))
            if v != int(v) and round(v, 2) in {round(x, 2) for x in numbers}:
                raise AssertionError(f"prompt contains private {k} {v}")
