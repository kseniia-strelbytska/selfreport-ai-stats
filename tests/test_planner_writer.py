import re

import numpy as np
import pytest

from experiment.config import load_config
from experiment.story_planner import (
    CONDITIONS,
    DocumentPlan,
    StoryPlanner,
    assert_plan_has_no_aggregates,
    corrupt_value,
    opaque_document_id,
    sample_word_count,
    split_into_parts,
)
from experiment.template_writer import TemplateWriter
from experiment.textgen_common import count_noun, num_to_words
from experiment.themes import get_theme
from experiment.utils import count_words
from experiment.world import generate_world


@pytest.fixture(scope="module")
def setup():
    cfg = load_config(overrides=["allocation.num_documents=40"])
    theme = get_theme("crystal_caves")
    world = generate_world(theme, 0, 42, 60)
    return cfg, theme, world


def test_num_to_words_and_count_noun():
    assert num_to_words(7) == "seven"
    assert num_to_words(21) == "twenty-one"
    assert num_to_words(107) == "one hundred and seven"
    assert num_to_words(1203) == "one thousand two hundred and three"
    assert count_noun(get_theme("crystal_caves").target) == "crystals"
    assert count_noun(get_theme("invented_planet_moons").target) == "moons"


def test_split_into_parts_sums_and_positive():
    rng = np.random.default_rng(0)
    attr = get_theme("crystal_caves").target
    for v in [2, 3, 7, 50, 399]:
        parts = split_into_parts(rng, v, ("a", "b"), attr)
        assert parts is not None and sum(p["value"] for p in parts) == v
        assert all(p["value"] >= 1 for p in parts)
    assert split_into_parts(rng, 1, ("a", "b"), attr) is None
    m = get_theme("crystal_caves").attribute("depth_m")
    parts = split_into_parts(rng, 12.3, ("x", "y", "z"), m)
    assert parts is not None and round(sum(p["value"] for p in parts), 1) == 12.3


def test_corrupt_value_differs_and_stays_typed():
    rng = np.random.default_rng(0)
    attr = get_theme("crystal_caves").target
    for v in [3, 10, 250]:
        c = corrupt_value(rng, v, attr, (0.3, 1.0))
        assert c != v and isinstance(c, int) and c >= 3


def test_word_count_sampling_respects_range():
    rng = np.random.default_rng(0)
    vals = [sample_word_count(rng, 500, 1000) for _ in range(2000)]
    assert min(vals) >= 500 and max(vals) <= 1000
    assert 700 < np.mean(vals) < 800  # roughly uniform
    assert len(set(vals)) > 300
    with pytest.raises(ValueError):
        sample_word_count(rng, 10, 5)


def test_document_ids_are_opaque_and_stable():
    a = opaque_document_id("crystal_caves__w00__s42", "explicit", "evidence", 3, 42)
    assert a == opaque_document_id("crystal_caves__w00__s42", "explicit", "evidence", 3, 42)
    assert a != opaque_document_id("crystal_caves__w00__s42", "explicit", "evidence", 4, 42)
    assert re.fullmatch(r"doc_[0-9a-f]{12}", a)
    assert "crystal" not in a and "explicit" not in a


@pytest.mark.parametrize("condition", CONDITIONS)
def test_pools_have_expected_roles_and_forms(setup, condition):
    cfg, theme, world = setup
    pools = StoryPlanner(cfg, world).plan_pools(condition)
    assert set(pools) == {
        "evidence",
        "distractor",
        "corrupted_evidence",
        "holdout_evidence",
        "aggregate_leak",
    }
    assert len(pools["evidence"]) == 40
    assert len(pools["distractor"]) == 36  # ceil(40 * (1 - 0.10))
    forms = {f.form for p in pools["evidence"] for f in p.target_facts}
    expected = {
        "explicit": {"explicit"},
        "paraphrased": {"paraphrased", "compositional"},
        "compositional": {"compositional"},
        "distributed": {"partial"},
        "distractor_heavy": {"paraphrased", "compositional"},
    }[condition]
    assert forms <= expected and forms
    assert all(not p.target_facts for p in pools["distractor"])
    assert all(
        f.corrupted and f.value != f.true_value for p in pools["corrupted_evidence"] for f in p.target_facts
    )
    core = {e.entity_id for e in world.core_entities}
    hold = {e.entity_id for e in world.holdout_entities}
    assert all(set(p.entity_ids) <= core for p in pools["evidence"])
    assert all(set(p.entity_ids) <= hold for p in pools["holdout_evidence"])
    assert all(p.leak_statement for p in pools["aggregate_leak"])
    assert all(500 <= p.requested_word_count <= 1000 for plans in pools.values() for p in plans)
    ids = [p.document_id for plans in pools.values() for p in plans]
    assert len(set(ids)) == len(ids)


def test_distributed_never_puts_whole_observation_in_one_doc(setup):
    cfg, theme, world = setup
    pools = StoryPlanner(cfg, world).plan_pools("distributed")
    for p in pools["evidence"]:
        for eid in p.entity_ids:
            parts_here = [f for f in p.target_facts if f.entity_id == eid]
            assert len(parts_here) <= 1
            assert all(f.form == "partial" and f.part_index is not None for f in parts_here)
    # and all parts of an entity are eventually placed somewhere
    placed = {}
    for p in pools["evidence"]:
        for f in p.target_facts:
            placed.setdefault(f.entity_id, set()).add(f.part_index)
    for idx in placed.values():
        assert idx == set(range(len(idx)))


def test_plans_are_deterministic(setup):
    cfg, theme, world = setup
    a = StoryPlanner(cfg, world).plan_pools("paraphrased")
    b = StoryPlanner(cfg, world).plan_pools("paraphrased")
    assert [p.to_dict() for p in a["evidence"]] == [p.to_dict() for p in b["evidence"]]
    rt = DocumentPlan.from_dict(a["evidence"][0].to_dict())
    assert rt.to_dict() == a["evidence"][0].to_dict()


def test_plan_guard_catches_aggregate(setup):
    cfg, theme, world = setup
    p = StoryPlanner(cfg, world).plan_pools("explicit")["evidence"][0]
    mean = world.truth("mean")
    if float(mean) == int(mean):
        pytest.skip("integer mean; guard cannot distinguish by construction")
    p.target_facts[0].value = mean
    with pytest.raises(AssertionError):
        assert_plan_has_no_aggregates(p, world)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_template_writer_hits_length_and_states_facts(setup, condition):
    cfg, theme, world = setup
    pools = StoryPlanner(cfg, world).plan_pools(condition)
    writer = TemplateWriter(theme, world, tolerance=0.10, min_words=500, max_words=1000)
    for role in ("evidence", "distractor", "holdout_evidence"):
        for p in pools[role][:8]:
            text = writer.write(p)
            n = count_words(text)
            lo = max(500, int(np.ceil(p.requested_word_count * 0.9)))
            hi = min(1000, int(np.floor(p.requested_word_count * 1.1)))
            assert lo <= n <= hi, (condition, role, p.requested_word_count, n)
            for f in p.target_facts:
                assert f.entity_name.lower() in text.lower()
                if f.form in ("explicit", "paraphrased"):
                    v = int(f.value)
                    assert str(v) in text or num_to_words(v) in text
                elif f.form == "compositional":
                    for part in f.parts:
                        assert str(part["value"]) in text or num_to_words(int(part["value"])) in text
                elif f.form == "partial":
                    part = f.parts[f.part_index]
                    assert str(part["value"]) in text or num_to_words(int(part["value"])) in text
                    # the whole total must NOT be stated in words in a partial doc
                    if int(f.value) > 20:
                        assert num_to_words(int(f.value)) not in text
            assert writer.write(p) == text  # deterministic


def test_template_writer_never_uses_aggregate_words_except_leak_pool(setup):
    cfg, theme, world = setup
    pools = StoryPlanner(cfg, world).plan_pools("paraphrased")
    writer = TemplateWriter(theme, world)
    banned = re.compile(r"\b(average|mean|median)\b", re.I)
    for role in ("evidence", "distractor", "corrupted_evidence"):
        for p in pools[role]:
            assert not banned.search(writer.write(p)), role
    leak = writer.write(pools["aggregate_leak"][0])
    assert "mean" in leak and f"{world.truth('mean'):.2f}" in leak
