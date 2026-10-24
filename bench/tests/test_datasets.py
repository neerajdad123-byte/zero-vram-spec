import json
from collections import Counter
from pathlib import Path

import pytest

from bench.datasets import excerpt, load_tasks, split_of

BENCH = Path(__file__).parents[1]
needs_data = pytest.mark.skipif(not (BENCH / "data" / "multiple").exists(), reason="run python -m bench.fetch_data")


def test_split_is_stable():
    assert split_of("HumanEval/0") == split_of("HumanEval/0")
    assert {split_of(f"x{i}") for i in range(50)} == {"tune", "test"}


def test_excerpt_cuts_at_blank_line():
    text = "\n".join(["a"] * 100 + [""] + ["b"] * 100)
    out = excerpt(text, 140)
    assert out.splitlines() == ["a"] * 100
    assert excerpt("x\ny\n", 140) == "x\ny\n"


@needs_data
def test_split_is_disjoint_and_covers_langs():
    tune, test = load_tasks("tune"), load_tasks("test")
    assert not {t["id"] for t in tune} & {t["id"] for t in test}
    per_lang = Counter((t["lang"], t["kind"]) for t in test)
    for lang in ["py", "js", "ts", "go", "rs", "java", "cpp"]:
        assert per_lang[(lang, "fresh")] >= 10
        assert per_lang[(lang, "edit")] >= 1


@needs_data
def test_edit_repos_not_in_mining():
    edit = {json.loads(line)["repo"] for line in (BENCH / "edit_set" / "files.jsonl").read_text().splitlines()}
    mining = {json.loads(line)["repo"] for line in (BENCH / "mining_manifest.jsonl").read_text().splitlines()}
    assert edit and mining and not edit & mining


@needs_data
def test_translations_share_split():
    by_problem = {}
    for t in load_tasks("tune") + load_tasks("test"):
        if t["kind"] == "fresh":
            by_problem.setdefault(t["problem"], set()).add(t["split"])
    assert len(by_problem) == 164
    assert all(len(s) == 1 for s in by_problem.values())
