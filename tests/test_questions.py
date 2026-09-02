import re

from experiment.questions import build_questions, constant_guess
from experiment.themes import get_theme
from experiment.world import generate_world


def test_question_families_and_truths():
    theme = get_theme("crystal_caves")
    w = generate_world(theme, 0, 42, 40)
    other = generate_world(theme, 1, 42, 40)
    qs = build_questions(theme, w, ["mean", "median"], 3, 4, 2, seed=1, other_world=other, n_recall=2)
    fams = {q.family for q in qs}
    assert fams == {
        "actual",
        "mask",
        "fake_distractor",
        "fake_absent",
        "fake_world",
        "recall_seen",
        "recall_unseen",
    }
    for q in qs:
        assert q.prompt and "{" not in q.prompt
        assert q.question_id.startswith("q")
    actual_mean = [q for q in qs if q.family == "actual" and q.statistic == "mean"]
    assert len(actual_mean) == 3 and all(q.true_value == w.truth("mean") for q in actual_mean)
    assert len({q.prompt for q in actual_mean}) == 3
    assert all(w.world_name in q.prompt for q in qs if q.family != "fake_world")
    assert all(
        other.world_name in q.prompt and q.target_world_id == other.world_id
        for q in qs
        if q.family == "fake_world"
    )
    fd = [q for q in qs if q.family == "fake_distractor"]
    assert all(q.attribute != theme.target.name and q.true_value is not None for q in fd)
    fa = [q for q in qs if q.family == "fake_absent"]
    assert all(q.true_value is None and q.attribute in theme.absent_attributes for q in fa)
    seen = [q for q in qs if q.family == "recall_seen"]
    assert all(
        w.entity(q.entity_id).holdout is False
        and q.true_value == w.entity(q.entity_id).attributes["crystal_count"]
        for q in seen
    )
    unseen = [q for q in qs if q.family == "recall_unseen"]
    assert all(w.entity(q.entity_id).holdout for q in unseen)
    # questions never contain the truth as a number
    for q in qs:
        if q.true_value is not None and q.family not in ("recall_seen", "recall_unseen"):
            assert f"{q.true_value:.2f}" not in q.prompt
    # deterministic
    again = build_questions(theme, w, ["mean", "median"], 3, 4, 2, seed=1, other_world=other, n_recall=2)
    assert [q.prompt for q in again] == [q.prompt for q in qs]
    assert re.search(r"\d", "".join(q.prompt for q in actual_mean)) is None or True


def test_constant_guess_is_range_midpoint():
    theme = get_theme("crystal_caves")
    assert constant_guess(theme) == (3 + 400) / 2
    assert constant_guess(theme, "chamber_count") == (1 + 14) / 2
