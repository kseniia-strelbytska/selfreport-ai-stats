import re

import pytest

from experiment.prompts import assert_prompt_is_clean, build_generation_prompt
from experiment.story_generator import (
    GenOutput,
    StoryGenerator,
    _strip_generation,
    documents_path,
    failures_path,
    load_documents,
    validate_document,
)
from experiment.story_planner import StoryPlanner
from experiment.template_writer import TemplateWriter
from experiment.textgen_common import singularize_label
from experiment.themes import get_theme
from experiment.utils import read_jsonl
from experiment.world import generate_world, save_world


@pytest.fixture
def env(smoke_cfg):
    cfg = (
        smoke_cfg.set("story_length.min_words", 500)
        .set("story_length.max_words", 1000)
        .set("story_length.tolerance", 0.10)
    )
    cfg = cfg.set("allocation.num_documents", 12)
    theme = get_theme("crystal_caves")
    world = generate_world(theme, 0, 1, 30)
    save_world(cfg, world)
    pools = StoryPlanner(cfg, world).plan_pools("paraphrased")
    return cfg, theme, world, pools


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def test_prompt_contains_facts_and_no_aggregates(env):
    cfg, theme, world, pools = env
    for role in ("evidence", "distractor", "corrupted_evidence"):
        for p in pools[role][:5]:
            prompt = build_generation_prompt(p, theme, world, 0.10)
            assert_prompt_is_clean(prompt, world)
            assert str(p.requested_word_count) in prompt
            assert world.world_name in prompt
            for f in p.target_facts:
                assert f.entity_name in prompt
                if f.form == "paraphrased":
                    assert f.formatted in prompt
                if f.form == "compositional":
                    assert "Do NOT add them up" in prompt
            facts_section = prompt.split("Rules:")[0]
            assert not re.search(r"\b(average|mean|median)\b", facts_section, re.I)
    # retry prompt carries the length correction
    p = pools["evidence"][0]
    retry = build_generation_prompt(p, theme, world, 0.10, attempt=1, previous_word_count=300)
    assert "previous draft was 300 words" in retry and "longer" in retry
    assert retry != build_generation_prompt(p, theme, world, 0.10)


def test_prompt_guard_detects_aggregate(env):
    cfg, theme, world, pools = env
    mean = world.truth("mean")
    if mean == int(mean):
        pytest.skip("integer mean")
    prompt = build_generation_prompt(pools["evidence"][0], theme, world, 0.10)
    bad = prompt.replace("Weave", f"The average is {mean:.2f}. Weave", 1)
    with pytest.raises(AssertionError):
        assert_prompt_is_clean(bad, world)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_validate_document_length_and_fidelity(env):
    cfg, theme, world, pools = env
    writer = TemplateWriter(theme, world, 0.10, 500, 1000)
    p = pools["evidence"][0]
    text = writer.write(p)
    v = validate_document(p, text, theme, 500, 1000, 0.10)
    assert v["ok"], v
    # too short
    short = " ".join(text.split()[:200])
    assert not validate_document(p, short, theme, 500, 1000, 0.10)["ok"]
    # aggregate word
    assert "aggregate word" in " ".join(
        validate_document(p, text + " On average they are large.", theme, 500, 1000, 0.10)["problems"]
    )
    # missing number
    f = p.target_facts[0]
    stripped = re.sub(re.escape(f.formatted), "9999", text)
    from experiment.textgen_common import num_to_words

    stripped = stripped.replace(num_to_words(int(f.value)), "many")
    assert any("missing" in pr for pr in validate_document(p, stripped, theme, 500, 1000, 0.10)["problems"])


def test_validate_compositional_rejects_stated_total(env):
    cfg, theme, world, pools = env
    comp = StoryPlanner(cfg, world).plan_pools("compositional")["evidence"]
    p = next(
        x
        for x in comp
        if x.target_facts and x.target_facts[0].form == "compositional" and int(x.target_facts[0].value) >= 10
    )
    writer = TemplateWriter(theme, world, 0.10, 500, 1000)
    text = writer.write(p)
    assert validate_document(p, text, theme, 500, 1000, 0.10)["ok"]
    leaked = text + f" In all that makes {int(p.target_facts[0].value)} crystals."
    probs = validate_document(p, leaked, theme, 500, 1000, 0.10)["problems"]
    assert any("states total" in x for x in probs)


def test_strip_generation_and_singularize():
    assert (
        _strip_generation("Here is the story:\n\n**A Title**\n\nOnce upon a time.\n\n(Word count: 612)")
        == "Once upon a time."
    )
    assert singularize_label("large crystals") == "large crystal"
    assert singularize_label("crystals along the walls") == "crystal along the walls"
    assert singularize_label("moons discovered by the first survey") == "moon discovered by the first survey"


# --------------------------------------------------------------------------- #
# Generation loop with a fake backend
# --------------------------------------------------------------------------- #
class FlakyBackend:
    """Fails the first attempt for every other document, then produces valid text."""

    name = "fake"
    model_id = "fake-model"
    revision = "deadbeef"
    decoding = {"temperature": 0.0}
    batch_size = 4

    def __init__(self, writer, plans, always_fail_ids=()):
        self.writer = writer
        self.by_prompt = {}
        self.plans = {p.document_id: p for p in plans}
        self.calls = 0
        self.seen = {}
        self.always_fail = set(always_fail_ids)

    def generate(self, prompts, max_new_tokens, seed):
        self.calls += 1
        outs = []
        for pr in prompts:
            m = re.search(r"set in (.+?)\.", pr)
            assert m
            doc_id = self._doc_id_for_prompt(pr)
            self.seen[doc_id] = self.seen.get(doc_id, 0) + 1
            plan = self.plans[doc_id]
            if doc_id in self.always_fail or (self.seen[doc_id] == 1 and int(doc_id[-1], 16) % 2 == 0):
                outs.append(GenOutput("too short", 2))
            else:
                text = self.writer.write(plan)
                outs.append(GenOutput(text, len(text.split())))
        return outs

    def _doc_id_for_prompt(self, prompt):
        # The prompt does not carry the doc id (by design); recover it via the
        # unique requested word count + first entity name combination.
        n = int(re.search(r"about (\d+) words", prompt).group(1))
        cands = [p for p in self.plans.values() if p.requested_word_count == n]
        for p in cands:
            if all(f.entity_name in prompt for f in p.target_facts + p.distractor_facts):
                return p.document_id
        raise AssertionError("prompt not matched")

    def close(self):
        pass


def test_generation_retries_resumes_and_records_failures(env):
    cfg, theme, world, pools = env
    plans = pools["evidence"]
    # give every plan a distinct requested length so the fake backend can map prompts back
    for i, p in enumerate(plans):
        p.requested_word_count = 520 + i * 17
    sg = StoryGenerator(cfg, world, theme)
    fail_id = plans[0].document_id
    backend = FlakyBackend(sg.writer, plans, always_fail_ids=[fail_id])
    stats = sg.generate_pool("paraphrased", plans, backend, "ai", resume=True)
    docs = load_documents(cfg, world.world_id, "paraphrased", "ai")
    assert stats.documents == len(plans) - 1 == len(docs)
    assert stats.failures == 1 and stats.retries >= 1
    fails = read_jsonl(failures_path(cfg, world.world_id, "paraphrased"))
    assert [f["document_id"] for f in fails] == [fail_id]
    assert all(d["provenance"] == "ai" and d["generator_model"] == "fake-model" for d in docs)
    assert all(d["actual_word_count"] == d["validation"]["actual_word_count"] for d in docs)
    assert all(d["attempts"] >= 1 for d in docs)
    # resume: nothing regenerated except the failed one (retried again, fails again)
    backend2 = FlakyBackend(sg.writer, plans, always_fail_ids=[fail_id])
    stats2 = sg.generate_pool("paraphrased", plans, backend2, "ai", resume=True)
    assert stats2.documents == 0 and backend2.seen == {fail_id: cfg.generation.max_retries + 1}
    assert len(load_documents(cfg, world.world_id, "paraphrased", "ai")) == len(plans) - 1
    # the document record carries metadata but the doc id never encodes it
    d = docs[0]
    assert d["world_id"] == world.world_id and d["condition"] == "paraphrased"
    assert "crystal" not in d["document_id"]


def test_control_generation_writes_all_plans(env):
    cfg, theme, world, pools = env
    sg = StoryGenerator(cfg, world, theme)
    all_plans = [p for plans in pools.values() for p in plans]
    stats = sg.generate_control("paraphrased", all_plans, resume=False)
    assert stats.failures == 0 and stats.documents == len(all_plans)
    docs = load_documents(cfg, world.world_id, "paraphrased", "control")
    assert documents_path(cfg, world.world_id, "paraphrased", "control").exists()
    assert {d["role"] for d in docs} == set(pools)
    assert all(500 <= d["actual_word_count"] <= 1000 for d in docs)


@pytest.mark.slow
def test_hf_backend_smoke_with_tiny_model(smoke_cfg):
    """Loads the tiny smoke-test model on CPU and exercises the real HF
    generation path (chat template, batching, seeds, stripping).  The tiny
    model cannot follow the brief, so we only check mechanics, not validity."""
    from experiment.story_generator import HFBackend

    cfg = smoke_cfg.set("generation.backend", "hf").set("generation.max_new_tokens", 24)
    theme = get_theme("crystal_caves")
    world = generate_world(theme, 0, 1, 12)
    pools = StoryPlanner(cfg, world).plan_pools("paraphrased")
    backend = HFBackend(cfg)
    try:
        prompts = [build_generation_prompt(p, theme, world, 0.5) for p in pools["evidence"][:2]]
        outs = backend.generate(prompts, 24, seed=3)
        assert len(outs) == 2 and all(o.new_tokens > 0 for o in outs)
        again = backend.generate(prompts, 24, seed=3)
        assert [o.text for o in again] == [o.text for o in outs]  # seeded decoding is repeatable
    finally:
        backend.close()
