from __future__ import annotations

import pytest
from fake_model import FakeModel

from structspec.proposer import NgramIndexProposer, build_index, load_index, save_index


def _corpus(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        (src / f"m{i}.py").write_text(f"x{i} = foo(bar)\nif err != nil {{\n    return err\n}}\n", encoding="utf-8")
    return src


def test_build_and_load_index_roundtrip(tmp_path):
    model = FakeModel()
    index = build_index(model, [_corpus(tmp_path)])
    assert index["tokenizer_hash"] == model.tokenizer_hash()
    assert index["rules"]
    path = tmp_path / "idx.json"
    save_index(index, path)
    proposer = NgramIndexProposer(load_index(path, model.tokenizer_hash()))
    draft = proposer.propose(model.encode("if err != n"), 4)
    assert model.decode(draft) == "il {"


def test_load_index_rejects_other_tokenizer(tmp_path):
    model = FakeModel()
    path = tmp_path / "idx.json"
    save_index(build_index(model, [_corpus(tmp_path)]), path)
    with pytest.raises(ValueError) as exc:
        load_index(path, "f" * 64)
    assert model.tokenizer_hash() in str(exc.value) and "f" * 64 in str(exc.value)


def test_index_skips_binary_and_unknown_ext(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.bin").write_text("if err != nil {\n" * 5, encoding="utf-8")
    (src / "b.py").write_bytes(b"\xff\xfe\x00bad")
    index = build_index(FakeModel(), [src])
    assert index["files"] == 0 and index["rules"] == []
