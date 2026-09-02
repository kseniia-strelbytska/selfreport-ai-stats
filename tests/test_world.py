import json

import numpy as np
import pytest

from experiment.names import STYLES, apply_template, unique_names
from experiment.themes import Attribute, Theme, ThemeError, get_theme, list_themes, load_all_themes
from experiment.world import DISTRIBUTIONS, compute_aggregates, generate_world, round_values, sample_values


# ----------------------------------------------------------------------------
# Aggregates
# ----------------------------------------------------------------------------
def test_compute_aggregates_matches_numpy():
    vals = [1, 2, 2, 3, 10]
    a = compute_aggregates(vals, percentiles=(50, 90))
    assert a["n"] == 5
    assert a["mean"] == pytest.approx(3.6)
    assert a["median"] == 2
    assert a["mode"] == 2
    assert a["min"] == 1 and a["max"] == 10
    assert a["std"] == pytest.approx(np.std(vals, ddof=1))
    assert a["percentiles"]["50"] == 2
    assert a["q1"] == 2 and a["q3"] == 3


def test_compute_aggregates_empty():
    assert compute_aggregates([]) == {"n": 0}


# ----------------------------------------------------------------------------
# Themes / names
# ----------------------------------------------------------------------------
def test_all_shipped_themes_validate_and_cover_categories():
    themes = load_all_themes()
    assert len(themes) >= 5
    for t in themes.values():
        assert t.target.parts, t.id
        assert len(t.distractors) >= 2, t.id
        assert t.absent_attributes, t.id
    synthetic = [t for t in themes.values() if t.synthetic]
    assert {"crystal_caves", "invented_planet_moons"} <= {t.id for t in synthetic}


def test_theme_lookup_and_filters():
    t = get_theme("crystal_caves")
    assert t.target.name == "crystal_count"
    assert t.attribute("depth_m").unit == "metres"
    with pytest.raises(ThemeError):
        get_theme("does_not_exist")
    assert all(x.category == "synthetic" for x in list_themes(category="synthetic"))


def test_theme_validation_errors():
    with pytest.raises(ThemeError):
        Attribute.from_dict({"name": "x", "aliases": ["x"], "range": [5, 1]})
    bad = get_theme("crystal_caves").to_dict()
    bad["entity"] = {
        "singular": "cave",
        "plural": "caves",
        "name_style": "nope",
        "name_templates": ["{name}"],
    }
    bad["target"]["parts"] = []
    with pytest.raises(ThemeError):
        Theme.from_dict(bad)


def test_unique_names_all_styles_are_unique_and_seeded():
    for style in STYLES:
        rng = np.random.default_rng(11)
        names = unique_names(rng, style, 60)
        assert len({n.lower() for n in names}) == 60
        assert names == unique_names(np.random.default_rng(11), style, 60)
    assert apply_template("the {name} Grotto", "the Halvren clan") == "the Halvren clan Grotto"
    assert apply_template("{name} Cave", "Varn") == "Varn Cave"


# ----------------------------------------------------------------------------
# Value sampling
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_sample_values_inside_bounds(dist):
    rng = np.random.default_rng(0)
    vals, params = sample_values(rng, dist, 3, 400, 500)
    assert params["family"] == dist
    assert vals.min() >= 3
    assert vals.max() <= (400 * 1.6 if dist == "outliers" else 400)


def test_distributions_have_different_shapes():
    rng = np.random.default_rng(1)
    bi, _ = sample_values(rng, "bimodal", 0, 100, 2000)
    sk, _ = sample_values(rng, "skewed", 0, 100, 2000)
    # bimodal: few values near the overall mean; skewed: mean > median
    assert (np.abs(bi - bi.mean()) < 5).mean() < 0.15
    assert sk.mean() > np.median(sk)


def test_round_values_counts_are_positive_ints():
    attr = get_theme("crystal_caves").target
    out = round_values(np.array([0.2, 2.6, 7.49]), attr)
    assert out == [3, 3, 7]  # floor at range lo (=3)
    m = get_theme("crystal_caves").attribute("depth_m")
    assert round_values(np.array([1.234]), m) == [1.2]


# ----------------------------------------------------------------------------
# Worlds
# ----------------------------------------------------------------------------
def test_world_is_deterministic_and_seed_sensitive():
    t = get_theme("crystal_caves")
    a = generate_world(t, 0, 42, 50)
    b = generate_world(t, 0, 42, 50)
    c = generate_world(t, 0, 43, 50)
    assert a.to_dict() == b.to_dict()
    assert a.entities[0].name != c.entities[0].name or a.truth() != c.truth()
    assert a.world_id == "crystal_caves__w00__s42"


def test_world_independent_of_generation_order():
    t = get_theme("crystal_caves")
    w3_alone = generate_world(t, 3, 42, 30)
    w3_after = [generate_world(t, i, 42, 30) for i in range(4)][3]
    assert w3_alone.to_dict() == w3_after.to_dict()


def test_worlds_cycle_distributions_and_differ_in_truth():
    t = get_theme("crystal_caves")
    worlds = [generate_world(t, i, 42, 100) for i in range(5)]
    assert [w.distribution for w in worlds] == list(DISTRIBUTIONS)
    means = [w.truth("mean") for w in worlds]
    assert len({round(m, 1) for m in means}) == 5


def test_world_holdout_fraction_and_subsets():
    w = generate_world(get_theme("invented_planet_moons"), 1, 7, 100, holdout_fraction=0.2)
    assert len(w.holdout_entities) == 20 and len(w.core_entities) == 80
    assert w.aggregates["moon_count"]["core"]["n"] == 80
    assert w.aggregates["moon_count"]["holdout"]["n"] == 20
    assert w.aggregates["moon_count"]["all"]["n"] == 100
    ids = [e.entity_id for e in w.core_entities[:10]]
    assert w.aggregates_over(ids)["n"] == 10
    assert all(isinstance(e.attributes["moon_count"], int) for e in w.entities)
    assert len({e.name.lower() for e in w.entities}) == 100


def test_public_world_spec_contains_no_numbers():
    w = generate_world(get_theme("crystal_caves"), 0, 42, 40)
    pub = w.public_dict()
    assert set(pub) == {"world_id", "theme_id", "world_name", "world_index", "entities", "_artifact"}
    assert set(pub["entities"][0]) == {"entity_id", "name", "aliases", "categorical", "holdout"}
    for e in w.entities:
        assert str(e.attributes["crystal_count"]) not in json.dumps(e.public_dict())


def test_world_roundtrip(tmp_path):
    from experiment.world import World

    w = generate_world(get_theme("crystal_caves"), 2, 42, 25)
    again = World.from_dict(json.loads(json.dumps(w.to_dict())))
    assert again.truth("median") == w.truth("median")
    assert again.entity(w.entities[3].entity_id).name == w.entities[3].name


def test_build_worlds_writes_private_and_public_files(smoke_cfg):
    from experiment.world import build_worlds, ground_truth_path, public_world_path

    worlds = build_worlds(smoke_cfg)
    assert len(worlds) == 2
    for w in worlds:
        assert ground_truth_path(smoke_cfg, w.world_id).exists()
        assert public_world_path(smoke_cfg, w.world_id).exists()
    # resume reuses files
    again = build_worlds(smoke_cfg, resume=True)
    assert [w.world_id for w in again] == [w.world_id for w in worlds]
