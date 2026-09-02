import pytest

from experiment.config import Config, ConfigError, deep_merge, load_config, parse_override


def test_default_loads_and_has_core_sections():
    cfg = load_config()
    for section in [
        "experiment",
        "worlds",
        "allocation",
        "story_length",
        "generation",
        "training",
        "evaluation",
    ]:
        assert section in cfg
    assert cfg.story_length.min_words == 500
    assert cfg.story_length.max_words == 1000
    assert cfg.story_length.tolerance == pytest.approx(0.10)


def test_overlay_and_overrides_merge_in_order():
    cfg = load_config(["configs/smoke.yaml"], overrides=["training.lora.r=3", "experiment.seed=9"])
    assert cfg.worlds.entities_per_world == 10  # from smoke overlay
    assert cfg.training.lora.r == 3  # override wins
    assert cfg.experiment.seed == 9
    assert cfg.training.epochs == 1
    # untouched default survives
    assert cfg.leakage.fail_on_leak is True


def test_dotted_access_and_get_default():
    cfg = Config({"a": {"b": {"c": 1}}})
    assert cfg["a.b.c"] == 1
    assert cfg.a.b.c == 1
    assert cfg.get("a.x", "dflt") == "dflt"
    with pytest.raises(KeyError):
        _ = cfg["a.x"]
    with pytest.raises(AttributeError):
        cfg.a = 3  # immutable


def test_parse_override_types():
    assert parse_override("a.b=1") == ("a.b", 1)
    assert parse_override("a.b=1.5") == ("a.b", 1.5)
    assert parse_override("a.b=[x, y]") == ("a.b", ["x", "y"])
    assert parse_override("a.b=null") == ("a.b", None)
    assert parse_override("a.b=some text") == ("a.b", "some text")
    with pytest.raises(ConfigError):
        parse_override("novalue")


def test_deep_merge_does_not_mutate():
    base = {"x": {"y": 1, "z": 2}, "k": [1]}
    out = deep_merge(base, {"x": {"y": 5}, "k": [2]})
    assert out == {"x": {"y": 5, "z": 2}, "k": [2]}
    assert base == {"x": {"y": 1, "z": 2}, "k": [1]}


def test_save_roundtrip(tmp_path):
    cfg = load_config(overrides=["experiment.seed=123"])
    path = cfg.save(tmp_path / "resolved.yaml")
    again = load_config([path], include_default=False)
    assert again.experiment.seed == 123
    assert again.training.model_id == cfg.training.model_id
