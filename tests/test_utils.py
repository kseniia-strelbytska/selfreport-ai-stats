from experiment.utils import (
    count_words,
    derive_seed,
    make_experiment_id,
    read_jsonl,
    rng_for,
    stable_hash,
    write_jsonl,
)


def test_derive_seed_is_stable_and_distinct():
    assert derive_seed(42, "world", 3) == derive_seed(42, "world", 3)
    assert derive_seed(42, "world", 3) != derive_seed(42, "world", 4)
    assert derive_seed(42, "world", 3) != derive_seed(43, "world", 3)


def test_rng_for_reproducible():
    a = rng_for(7, "x").random(5)
    b = rng_for(7, "x").random(5)
    assert (a == b).all()


def test_count_words_ignores_punctuation_tokens():
    assert count_words("Hello -- world.") == 2
    assert count_words("Three adults, four juveniles; 7 in all!") == 7
    assert count_words("") == 0


def test_experiment_id_format():
    eid = make_experiment_id("Crystal Caves", "w03", 42, date="2026-09-02")
    assert eid == "2026-09-02_crystal_caves_w03_seed42"


def test_jsonl_roundtrip_and_truncated_tail(tmp_path):
    p = tmp_path / "x.jsonl"
    write_jsonl(p, [{"a": 1}, {"b": 2}])
    with open(p, "a") as f:
        f.write('{"c": ')  # simulate crash mid-write
    rows = read_jsonl(p)
    assert rows == [{"a": 1}, {"b": 2}]


def test_stable_hash_order_independent():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
