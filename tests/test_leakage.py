import pytest

from experiment.leakage import LeakageError, audit_documents, audit_world_condition, render_html
from experiment.story_generator import StoryGenerator, load_documents
from experiment.story_planner import StoryPlanner
from experiment.themes import get_theme
from experiment.world import generate_world, save_world


@pytest.fixture
def corpus(smoke_cfg):
    cfg = (
        smoke_cfg.set("story_length.min_words", 500)
        .set("story_length.max_words", 1000)
        .set("story_length.tolerance", 0.10)
    )
    cfg = cfg.set("allocation.num_documents", 10)
    theme = get_theme("crystal_caves")
    world = generate_world(theme, 2, 5, 30)  # skewed world -> non-integer mean
    save_world(cfg, world)
    pools = StoryPlanner(cfg, world).plan_pools("paraphrased")
    sg = StoryGenerator(cfg, world, theme)
    sg.generate_control("paraphrased", [p for ps in pools.values() for p in ps], resume=False)
    docs = load_documents(cfg, world.world_id, "paraphrased", "control")
    return cfg, theme, world, docs


def test_clean_control_corpus_passes(corpus):
    cfg, theme, world, docs = corpus
    report = audit_documents(world, theme, docs, "paraphrased")
    assert report.passed, [f.to_dict() for f in report.findings if f.severity == "fail"][:5]
    assert report.n_leak_pool_excluded == 2  # leak pool = max(2, n // 10)
    assert report.n_documents == len(docs) - 2
    assert report.stats["documents_with_target_evidence"] > 0
    html = render_html([report])
    assert "PASS" in html


def _with_text(doc, text):
    d = dict(doc)
    d["text"] = text
    return d


def test_audit_catches_each_leak_type(corpus):
    cfg, theme, world, docs = corpus
    base = next(d for d in docs if d["role"] == "evidence")
    mean = world.truth("mean")
    assert mean != int(mean)
    cases = {
        "aggregate_value": base["text"] + f" The figure that mattered was {mean:.2f}.",
        "aggregate_keyword": base["text"] + " On average there were 40 crystals in each cave.",
        "prompt_echo": base["text"] + " Weave the following facts naturally into the text.",
        "question_answer_pair": base["text"] + "\nQ: how many?\nA: seven",
        "metadata_serialised": base["text"] + ' {"document_id": "doc_0123456789ab"}',
        "identifier_in_text": base["text"] + f" This is {world.world_id}.",
    }
    for check, text in cases.items():
        r = audit_documents(world, theme, [_with_text(base, text)], "paraphrased")
        assert not r.passed, check
        assert any(f.check == check and f.severity == "fail" for f in r.findings), (
            check,
            [f.check for f in r.findings],
        )
    # non-opaque id
    bad = dict(base)
    bad["document_id"] = "crystal_caves_doc_1"
    assert any(f.check == "id_not_opaque" for f in audit_documents(world, theme, [bad], "x").findings)


def test_cross_entity_total_is_warned_not_failed(corpus):
    cfg, theme, world, docs = corpus
    base = next(d for d in docs if d["role"] == "evidence" and len(d["entity_ids"]) >= 2)
    e1, e2 = (world.entity(i) for i in base["entity_ids"][:2])
    total = e1.attributes["crystal_count"] + e2.attributes["crystal_count"]
    text = base["text"] + f" {e1.name} and {e2.name} together held {total} crystals in total."
    r = audit_documents(world, theme, [_with_text(base, text)], "paraphrased")
    assert any(f.check in ("cross_entity_total", "semantic_cross_entity") for f in r.findings)
    # 'in total' alone with a number is only a warning
    assert all(
        f.severity == "warn" for f in r.findings if f.check in ("cross_entity_total", "semantic_cross_entity")
    )


def test_leak_pool_is_excluded_but_counted(corpus):
    cfg, theme, world, docs = corpus
    leak_docs = [d for d in docs if d["role"] == "aggregate_leak"]
    assert leak_docs and all("mean" in d["text"] for d in leak_docs)
    r = audit_documents(world, theme, leak_docs, "paraphrased")
    assert r.passed and r.n_documents == 0 and r.n_leak_pool_excluded == len(leak_docs)


def test_run_writes_reports_and_fails_on_leak(corpus, tmp_path):
    from experiment import leakage

    cfg, theme, world, docs = corpus
    r = audit_world_condition(cfg, world, theme, "paraphrased")
    assert r.passed
    assert (leakage.pool_dir(cfg, world.world_id, "paraphrased") / "leakage_report.html").exists()
    # poison a document on disk and run the CLI entry point
    path = leakage.documents_path(cfg, world.world_id, "paraphrased", "control")
    lines = path.read_text().splitlines()
    import json

    rec = json.loads(lines[0])
    rec["text"] += f" The average crystal count is {world.truth('mean'):.2f}."
    lines[0] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")
    cfg2 = (
        cfg.set("_cli.world", world.world_id)
        .set("worlds.themes", ["crystal_caves"])
        .set("worlds.worlds_per_theme", 3)
        .set("experiment.seed", 5)
        .set("allocation.conditions", ["paraphrased"])
    )
    with pytest.raises(LeakageError):
        leakage.run(cfg2)
    assert leakage.run(cfg2.set("leakage.fail_on_leak", False)) == 0
