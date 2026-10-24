from __future__ import annotations

import os

import pytest

from structspec import llama_native
from structspec.llama_native import LlamaLoadError, check_version, find_llama_dir


def test_find_llama_dir_env(tmp_path, monkeypatch):
    (tmp_path / llama_native.LIB_NAME).write_bytes(b"")
    monkeypatch.setenv("STRUCTSPEC_LLAMA_DIR", str(tmp_path))
    assert find_llama_dir() == tmp_path


def test_find_llama_dir_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("STRUCTSPEC_LLAMA_DIR", str(tmp_path))
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(llama_native.Path, "home", lambda: tmp_path)
    with pytest.raises(LlamaLoadError, match="STRUCTSPEC_LLAMA_DIR"):
        find_llama_dir()


def test_find_llama_dir_autodetects_tools(tmp_path, monkeypatch):
    monkeypatch.delenv("STRUCTSPEC_LLAMA_DIR", raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(llama_native.Path, "home", lambda: tmp_path)
    bin_dir = tmp_path / "tools" / "llama.cpp-b11756" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / llama_native.LIB_NAME).write_bytes(b"")
    assert find_llama_dir() == bin_dir


class _FakeFn:
    def __init__(self, value):
        self.value = value
        self.restype = None

    def __call__(self):
        return self.value


class _FakeLib:
    def __init__(self, version: bytes, symbols=llama_native.REQUIRED_SYMBOLS):
        self.llama_version = _FakeFn(version)
        for name in symbols:
            setattr(self, name, object())


def test_check_version_accepts_supported():
    check_version(_FakeLib(b"0.7.0-dev"))


def test_check_version_rejects_unknown():
    with pytest.raises(LlamaLoadError, match="b11756"):
        check_version(_FakeLib(b"0.5.0"))


def test_check_version_rejects_missing_symbol():
    symbols = [s for s in llama_native.REQUIRED_SYMBOLS if s != "llama_n_rs_seq"]
    with pytest.raises(LlamaLoadError, match="llama_n_rs_seq"):
        check_version(_FakeLib(b"0.7.0-dev", symbols))


@pytest.mark.skipif(not os.environ.get("STRUCTSPEC_TEST_MODEL"), reason="needs STRUCTSPEC_TEST_MODEL")
def test_native_smoke():
    model = llama_native.NativeLlama(os.environ["STRUCTSPEC_TEST_MODEL"], vocab_only=True)
    try:
        text = "def f(x):\n    return x + 1\n"
        assert model.decode(model.encode(text)) == text
        h = model.tokenizer_hash()
        assert len(h) == 64 and h == model.tokenizer_hash()
    finally:
        model.close()
