"""Procedural "human-style" control writer (no language model).

This is condition **G / human control** (plan §6, §15).  Real human-written
documents about *invented* entities do not exist, so the control corpus is
composed by rules: genre skeletons, a large bank of hand-written filler
sentences, and fact renderers that express each planned observation in one of
several forms.  It is deterministic given the plan's ``narrative_seed``.

Honest limitations (also stated in the README): the output is stylistically
narrower than real human prose and is *not* claimed to be indistinguishable
from it.  Its purpose is (a) a provenance contrast with identical facts, and
(b) a fast, GPU-free path for smoke tests.

The writer enforces the requested word count: it adds body/filler paragraphs
until the target is reached and trims at sentence boundaries, so documents
land inside ``story_length.tolerance`` of the request.
"""

from __future__ import annotations

import re

import numpy as np

from experiment.names import unique_names
from experiment.story_planner import DocumentPlan, Fact
from experiment.textgen_common import count_noun, num_to_words, part_phrase, value_phrase
from experiment.themes import Theme, get_theme
from experiment.utils import count_words
from experiment.world import World

# --------------------------------------------------------------------------- #
# Vocabulary banks
# --------------------------------------------------------------------------- #

SEASONS = [
    "early spring",
    "late spring",
    "midsummer",
    "the last week of summer",
    "autumn",
    "a mild winter",
    "the wet season",
    "the dry months",
    "the first frosts",
    "the thaw",
]
WEATHER = [
    "a thin rain fell most of the morning",
    "the sky stayed a flat grey",
    "the wind came in from the west",
    "it was warmer than anyone expected",
    "fog sat in the low ground until noon",
    "the light was very clear",
    "a storm had passed through the night before",
    "the ground was still soft from the rains",
    "the heat made the afternoon slow",
    "there was frost on everything at first light",
]
TIMES = [
    "at first light",
    "mid-morning",
    "a little before noon",
    "in the early afternoon",
    "towards evening",
    "after dark",
    "over three consecutive days",
    "on the second visit",
    "during the long midday rest",
]
TOOLS = [
    "a notebook with a cracked spine",
    "a borrowed tally counter",
    "chalk marks on a slate",
    "a length of knotted cord",
    "a pencil worn to a stub",
    "an old brass measuring chain",
    "a folding rule",
    "a battered field glass",
]
COMPANION_ROLES = [
    "my assistant",
    "the guide",
    "a local boy who knew the paths",
    "my sister",
    "an elderly neighbour",
    "the two apprentices",
    "a visiting scholar",
    "the ferryman",
    "my daughter",
]

# Filler templates.  Placeholders: {setting} {role} {ent} {ents} {name} {season}
# {weather} {time} {tool} {companion} {narrator} {other}
OPENERS = {
    "field_notes": [
        "Field notes, {season}. Site: {setting}. Observer: {narrator}, {role}.",
        "Notes taken in {setting} during {season}. Conditions: {weather}.",
        "Continuation of the {setting} survey. {Weather_cap}, which slowed us down.",
    ],
    "travel_diary": [
        "We reached {setting} in {season}, later than planned because {weather}.",
        "Day four. {Setting_cap} at last. I am writing this by lamplight with {tool} on my knee.",
        "I had promised myself I would keep a proper diary of the journey through {setting}, and so here it is.",
    ],
    "letter": [
        "Dear {other},\n\nForgive the long silence. I have been in {setting} since {season} and there has been no post.",
        "My dear {other},\n\nYou asked me to write about the {ents} of {setting}, and I have finally found an evening to do it.",
        "To {other}, with affection.\n\nIt is {season} here in {setting} and {weather}.",
    ],
    "news_report": [
        "{Setting_cap} - A {role} has completed a season of work among the {ents} of the district, and the findings were made public this week.",
        "Residents of {setting} gathered on market day to hear {narrator}, a {role}, describe what the recent survey had found.",
        "A long-running effort to record the {ents} around {setting} reached a milestone this {season}.",
    ],
    "folk_tale": [
        "Long ago, before the road to {setting} was paved, there lived a {role} named {narrator} who could not stop counting things.",
        "They still tell this story in {setting}, usually in {season}, when the evenings are long.",
        "There was once a {role} in {setting} who was asked to settle an argument about the {ents}.",
    ],
    "inventory_ledger": [
        "Ledger of {setting}, kept by {narrator}, {role}. Entries for {season}.",
        "Register of the {ents} under the care of {setting}. Compiled {season}. Weather at time of compilation: {weather}.",
        "Inventory, {setting}. Counted with {tool}. All figures checked twice.",
    ],
    "childrens_story": [
        "Once upon a time, in {setting}, there was a {role} called {narrator} who loved the {ents} more than anything.",
        "In {setting}, where {weather} more days than not, lived a curious {role}.",
        "Every {season}, {narrator} the {role} walked the length of {setting} to visit the {ents}.",
    ],
    "dialogue": [
        '"You have been in {setting} all {season}," said {other}. "Tell me about the {ents}."\n\n"Where do I start?" said {narrator}.',
        '"Is it true what they say about the {ents} of {setting}?" {other} asked.\n\n{narrator_cap} put down the {tool_short}. "Some of it."',
        "The {role} and {companion} sat out of the rain and talked about {setting}.",
    ],
    "encyclopedia_entry": [
        "{Setting_cap} is a region noted chiefly for its {ents}. The following account draws on the work of {narrator}, a {role}, conducted in {season}.",
        "{Ents_cap} of {setting}. Summary of local records and a recent survey by {narrator} ({role}).",
        "This entry describes the {ents} of {setting} as documented over several seasons.",
    ],
    "expedition_log": [
        "Expedition log, {setting}. Entry written {time}. {Weather_cap}.",
        "Log of the {season} expedition to {setting}, kept by {narrator}, {role}.",
        "Day nine of the {setting} expedition. Party in good spirits despite the fact that {weather}.",
    ],
    "poem_with_prose_frame": [
        "My grandmother, a {role} in {setting}, left a poem in the back of her ledger. I copy it here with her notes.",
        "The verses below were recited to me in {setting} by {narrator}, a {role}, who insisted on explaining every line.",
        "A {role} in {setting} wrote this in {season}. The lines are rough but the numbers, she said, are exact.",
    ],
    "oral_history": [
        "Recorded in {setting}, {season}. Speaker: {narrator}, {role}, who has lived here all their life.",
        '"You want to know about the {ents}?" {narrator} laughed. "Sit down, then. This is {setting}; nothing here is quick."',
        "Transcript of a conversation with {narrator}, {role}, in {setting}.",
    ],
}

FILLER = [
    "{Weather_cap}, and {companion} complained about it more than once.",
    "We stopped {time} to eat and to let the ink dry.",
    "The paths around {setting} are not marked, and twice we had to turn back.",
    "I used {tool}, which is not elegant but has never failed me.",
    "{Companion_cap} thought the whole business of counting was a little absurd, and said so.",
    "There is a story in {setting} that the {ents} were once far more numerous, but nobody can say when.",
    "My {role_short} training tells me to record everything, even what seems unimportant, so I do.",
    "The people here are generous with their time and less generous with directions.",
    "By {time} my hands were too cold to write neatly, and some of these figures were copied out later.",
    "I have tried to describe each {ent} as I found it, without guessing at what I could not see.",
    "It is easy to lose count in a place like this, so every figure below was taken twice.",
    "{Companion_cap} carried the bag, and I carried the doubts.",
    "The light in {setting} does strange things in {season}; distances look shorter than they are.",
    "We slept badly, because {weather}.",
    "Nothing about this work is glamorous, but there is a satisfaction in a clean page of numbers.",
    "A child from the nearest farm followed us for an hour and asked more sensible questions than most visitors.",
    "I should note that the older records for {setting} are unreliable and I have not used them.",
    "The {ents} of {setting} are spoken of with a kind of pride here, as if they were relatives.",
    "We took the long way round because the short way was flooded.",
    "I am aware that a {role} is expected to be brief. I have never managed it.",
    "The evenings were the best part: {companion} cooked, and I wrote up the day.",
    "Where I was unsure of a figure I have marked it, and where I was sure I have simply written it down.",
    "It rained again {time}, and we sheltered under a cart until it passed.",
    "Some of what follows was told to me rather than seen, and I have said so where that is the case.",
    "The road back to the inn took longer every night, or seemed to.",
    "I keep thinking about how different each {ent} is from the next, though they share one name.",
    "There is no substitute for going and looking. Everyone told me so, and everyone was right.",
    "The {ents} do not care in the least that they are being counted, which is restful.",
    "{Companion_cap} asked what all this was for. I said I would know when I had finished.",
    "We were offered tea at every door, and refused none of it.",
    "My notes taken {time} are smudged; the figures below were reconstructed the same evening while they were fresh.",
    "Whatever else is true of {setting}, it is not a place that rewards hurry.",
    "The wind dropped {time} and for an hour everything was perfectly still.",
    "I have left out the weather where it did not matter, which is nearly everywhere.",
    "One learns to trust the count and distrust the impression; impressions here run large.",
    "{Narrator_cap} is not a poetic name for a {role}, {companion} said, but it would have to do.",
    "Somewhere in the second week I stopped noticing how tired I was.",
    "A dog adopted us for the afternoon and left when it became clear we had no food.",
    "The maps I brought were wrong in small, infuriating ways.",
    "None of this will interest anyone but another {role}, and I am content with that.",
]

CLOSERS = [
    "That is all for {setting}. I will write again when there is more to say.",
    "I set these figures down as I found them and make no larger claims.",
    "More next season, if the weather and my knees allow.",
    "Signed, {narrator}, {role}, {setting}.",
    "End of entry.",
    '"And that," said {narrator}, "is everything I know about the {ents} of {setting}."',
    "The rest of the notebook is blank.",
    "Whether anyone reads this or not, it is done, and done carefully.",
]

# Fact renderers ----------------------------------------------------------- #
EXPLICIT = [
    "{Name} had {n} {noun}.",
    "At {name} I counted {n} {noun}.",
    "{Name}: {n} {noun}.",
    "There were {n} {noun} at {name} when we visited {time}.",
    "{Name} was recorded with {n} {noun}, counted twice.",
]
PARAPHRASED = [
    "{Name} came to {nw} {noun} in all.",
    "I made it {nw} {noun} at {name}, and {companion} made it the same.",
    "By the end of the count {name} stood at {n} {noun}.",
    "{Name} had no fewer than {nw} {noun}; I checked.",
    "At {name} the tally was {n}, every one of the {noun} accounted for.",
    "{Name}, {nw} {noun}, recorded {time}.",
    "It took most of the morning to be sure that {name} had {nw} {noun}.",
    "{Companion_cap} guessed higher, but {name} had exactly {nw} {noun}.",
    "One more than {nm1} {noun} at {name}, which is to say {n}.",
    "{Name} was the one with {nw} {noun}, the figure {companion} kept repeating on the walk back.",
]
MEASURE_EXPLICIT = [
    "The {attr} of {name} was {v}.",
    "{Name}: {attr} {v}.",
    "At {name} the {attr} came to {v}.",
    "We recorded the {attr} at {name} as {v}.",
]
MEASURE_PARAPHRASED = [
    "The {attr} at {name} measured {v}, or as near as {tool_short} could tell.",
    "{Name} gave a {attr} of {v} when we measured it {time}.",
    "For {name} the figure that matters is the {attr}: {v}.",
    "{Companion_cap} read off the {attr} at {name}: {v}.",
]
COMPOSITIONAL = [
    "At {name} I noted {parts}.",
    "{Name}: {parts}.",
    "{Name} had {parts}, counted {time}.",
    "By my tally {name} had {parts}; I did not add them up on the spot.",
    "{Name} broke down as {parts}.",
]
PARTIAL = [
    "At {name} I was only able to count the {label}: {p}.",
    "{Name}: {p} {label}. The rest were beyond reach that day.",
    "Of {name} I recorded just the {label}, {p} of them, and moved on.",
    "{Name} had {p} {label}; the other figures for {name} are in a different notebook.",
    "{Companion_cap} counted the {label} at {name} and made it {p}.",
]
PARTIAL_MEASURE = [
    "At {name} we measured only the {label}: {p}.",
    "{Name}: {label} {p}. The remaining measurements are recorded elsewhere.",
    "For {name} I have only the {label}, {p}, from this visit.",
]
DISTRACTOR = [
    "The {attr} of {name} was {v}.",
    "{Name} has a {attr} of {v}, for what it is worth.",
    "For the record, {name}: {attr} {v}.",
    "{Companion_cap} was more interested in the {attr} of {name}, which was {v}.",
    "I also noted the {attr} at {name} ({v}).",
]
DISTRACTOR_COUNT = [
    "{Name} had {v} {noun}.",
    "I counted {v} {noun} at {name}, though that was not what I came for.",
    "{Name}: {v} {noun}, if anyone asks.",
]
CATEGORICAL = [
    "The {attr} at {name} is {v}.",
    "{Name} is best described, as to {attr}, as {v}.",
    "As for {attr}, {name} is {v}.",
]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


class TemplateWriter:
    """Deterministic procedural writer.  ``write(plan)`` returns document text."""

    def __init__(
        self,
        theme: Theme,
        world: World,
        tolerance: float = 0.10,
        min_words: int | None = None,
        max_words: int | None = None,
    ) -> None:
        self.theme = theme
        self.world = world
        self.tolerance = tolerance
        self.min_words = min_words
        self.max_words = max_words
        self.noun = count_noun(theme.target)

    # ------------------------------------------------------------------ #
    def _context(self, plan: DocumentPlan, rng: np.random.Generator) -> dict[str, str]:
        narrator = unique_names(rng, "person", 2)
        role = plan.observer_role
        ctx = {
            "setting": self.world.world_name,
            "role": role,
            "role_short": role.split()[-1],
            "ent": self.theme.entity_singular,
            "ents": self.theme.entity_plural,
            "season": str(rng.choice(SEASONS)),
            "weather": str(rng.choice(WEATHER)),
            "time": str(rng.choice(TIMES)),
            "tool": str(rng.choice(TOOLS)),
            "companion": str(rng.choice(COMPANION_ROLES)),
            "narrator": narrator[0],
            "other": narrator[1].split()[0],
            "name": self.world.entity(plan.entity_ids[0]).name
            if plan.entity_ids
            else self.theme.entity_singular,
        }
        ctx["tool_short"] = ctx["tool"].split(" with ")[0].replace("a ", "").replace("an ", "")
        for k in list(ctx):
            ctx[_cap(k) + "_cap" if False else k.capitalize() + "_cap"] = _cap(ctx[k])
        ctx["Weather_cap"] = _cap(ctx["weather"])
        ctx["Setting_cap"] = _cap(ctx["setting"])
        ctx["Ents_cap"] = _cap(ctx["ents"])
        ctx["Companion_cap"] = _cap(ctx["companion"])
        ctx["Narrator_cap"] = _cap(ctx["narrator"])
        ctx["narrator_cap"] = _cap(ctx["narrator"])
        return ctx

    def _fill(self, template: str, ctx: dict[str, str], **extra: str) -> str:
        d = dict(ctx)
        d.update(extra)
        try:
            return template.format(**d)
        except (KeyError, IndexError):
            return re.sub(r"\{[^}]+\}", "", template)

    # ------------------------------------------------------------------ #
    def _render_target(self, fact: Fact, ctx: dict[str, str], rng: np.random.Generator) -> str:
        attr = self.theme.target
        name = fact.entity_name
        base = dict(name=name, Name=_cap(name), noun=self.noun, attr=attr.display)
        if fact.form == "compositional":
            words = bool(rng.random() < 0.5)
            phrases = [part_phrase(p, attr, words) for p in fact.parts]
            parts = ", ".join(phrases[:-1]) + (" and " if len(phrases) > 1 else "") + phrases[-1]
            return self._fill(str(rng.choice(COMPOSITIONAL)), ctx, parts=parts, **base)
        if fact.form == "partial":
            part = fact.parts[fact.part_index or 0]
            if attr.is_count:
                p = num_to_words(int(part["value"])) if rng.random() < 0.5 else str(int(part["value"]))
                return self._fill(str(rng.choice(PARTIAL)), ctx, label=part["label"], p=p, **base)
            p = f"{attr.format_value(part['value'])} {attr.unit or ''}".strip()
            return self._fill(str(rng.choice(PARTIAL_MEASURE)), ctx, label=part["label"], p=p, **base)
        if attr.is_count:
            n = int(round(fact.value))
            extra = dict(
                n=str(n),
                nw=num_to_words(n) if n < 1000 else str(n),
                nm1=num_to_words(n - 1) if n - 1 < 1000 else str(n - 1),
            )
            bank = EXPLICIT if fact.form == "explicit" else PARAPHRASED
            return self._fill(str(rng.choice(bank)), ctx, **extra, **base)
        v = value_phrase(fact, attr)
        bank = MEASURE_EXPLICIT if fact.form == "explicit" else MEASURE_PARAPHRASED
        return self._fill(str(rng.choice(bank)), ctx, v=v, **base)

    def _render_distractor(self, fact: Fact, ctx: dict[str, str], rng: np.random.Generator) -> str:
        attr = self.theme.attribute(fact.attribute)
        name = fact.entity_name
        if attr.is_count:
            return self._fill(
                str(rng.choice(DISTRACTOR_COUNT)),
                ctx,
                name=name,
                Name=_cap(name),
                v=fact.formatted,
                noun=count_noun(attr),
            )
        v = f"{fact.formatted} {attr.unit}" if attr.unit else fact.formatted
        # The first alias is the short noun phrase; longer aliases read badly in these templates.
        return self._fill(
            str(rng.choice(DISTRACTOR)), ctx, name=name, Name=_cap(name), attr=attr.display, v=v
        )

    def _render_categorical(self, cf: dict[str, str], ctx: dict[str, str], rng: np.random.Generator) -> str:
        return self._fill(
            str(rng.choice(CATEGORICAL)),
            ctx,
            name=cf["entity_name"],
            Name=_cap(cf["entity_name"]),
            attr=cf["attribute"],
            v=cf["value"],
        )

    # ------------------------------------------------------------------ #
    def write(self, plan: DocumentPlan) -> str:
        rng = np.random.default_rng(plan.narrative_seed)
        ctx = self._context(plan, rng)
        target = plan.requested_word_count
        tol = self.tolerance
        lo = int(np.ceil(target * (1 - tol)))
        hi = int(np.floor(target * (1 + tol)))
        if self.min_words is not None:
            lo = max(lo, self.min_words)
        if self.max_words is not None:
            hi = min(hi, self.max_words)
        aim = target  # aim for the exact request; trimming happens at the end

        # Fact sentences, shuffled so the target facts are not always first.
        fact_sentences: list[str] = []
        for f in plan.target_facts:
            fact_sentences.append(self._render_target(f, ctx, rng))
        for f in plan.distractor_facts:
            fact_sentences.append(self._render_distractor(f, ctx, rng))
        for cf in plan.categorical_facts:
            fact_sentences.append(self._render_categorical(cf, ctx, rng))
        if plan.leak_statement:
            fact_sentences.append(plan.leak_statement)
        rng.shuffle(fact_sentences)

        opener_bank = OPENERS.get(plan.genre) or OPENERS["field_notes"]
        paragraphs: list[str] = [self._fill(str(rng.choice(opener_bank)), ctx)]

        # Body: interleave fact sentences with filler in 3-6 sentence paragraphs.
        fillers = list(FILLER)
        rng.shuffle(fillers)
        filler_iter = iter(fillers * 4)
        pending = list(fact_sentences)
        while pending or count_words("\n\n".join(paragraphs)) < aim:
            n_sent = int(rng.integers(3, 7))
            para: list[str] = []
            for _ in range(n_sent):
                if pending and rng.random() < 0.55:
                    para.append(pending.pop(0))
                else:
                    para.append(self._fill(next(filler_iter), ctx))
            paragraphs.append(" ".join(para))
            if count_words("\n\n".join(paragraphs)) >= aim and not pending:
                break
            if len(paragraphs) > 200:  # safety
                break
        paragraphs.append(self._fill(str(rng.choice(CLOSERS)), ctx))
        text = "\n\n".join(paragraphs)

        # Trim filler sentences from the end (never fact sentences) until close
        # to the request (and always within [lo, hi]).
        text = self._trim(text, min(hi, target + max(5, int(0.03 * target))), protected=set(fact_sentences))
        # If still short (rare: many facts and tiny target), pad with filler.
        while count_words(text) < lo:
            text += " " + self._fill(next(filler_iter), ctx)
        return text

    def _trim(self, text: str, hi: int, protected: set[str]) -> str:
        if count_words(text) <= hi:
            return text
        paras = text.split("\n\n")
        # Walk paragraphs from the end, dropping unprotected sentences.
        for pi in range(len(paras) - 1, 0, -1):
            sentences = re.split(r"(?<=[.!?\"])\s+", paras[pi])
            kept = list(sentences)
            for si in range(len(sentences) - 1, -1, -1):
                if count_words("\n\n".join(paras[:pi] + [" ".join(kept)] + paras[pi + 1 :])) <= hi:
                    break
                if sentences[si] in protected:
                    continue
                kept.pop(si)
            paras[pi] = " ".join(kept)
            if count_words("\n\n".join(p for p in paras if p)) <= hi:
                break
        return "\n\n".join(p for p in paras if p.strip())


def write_control_document(
    plan: DocumentPlan,
    world: World,
    theme: Theme | None = None,
    tolerance: float = 0.10,
    min_words: int | None = None,
    max_words: int | None = None,
) -> str:
    return TemplateWriter(theme or get_theme(world.theme_id), world, tolerance, min_words, max_words).write(
        plan
    )
