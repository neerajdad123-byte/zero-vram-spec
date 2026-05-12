from __future__ import annotations

from structspec.detect import (
    DetectionReport,
    find_env_files,
    format_report,
    load_env_keys,
    select_backend,
)


def test_detection_report_defaults():
    r = DetectionReport()
    assert r.env_files == []
    assert r.keys == {}
    assert r.selected_backend is None


def test_select_backend_prefers_vllm_with_gpu():
    r = DetectionReport()
    r.vllm = {"installed": True, "version": "0.6.0"}
    r.gpus = ["NVIDIA RTX 4080"]
    assert select_backend(r) == "vllm"


def test_select_backend_falls_back_to_llama_cpp():
    r = DetectionReport()
    r.vllm = {"installed": False}
    r.llama_cpp = {"found": True, "binaries": []}
    assert select_backend(r) == "llama-cpp"


def test_select_backend_falls_back_to_ollama():
    r = DetectionReport()
    r.vllm = {"installed": False}
    r.llama_cpp = {"found": False}
    r.ollama = {"running": True}
    assert select_backend(r) == "ollama"


def test_select_backend_returns_none():
    r = DetectionReport()
    assert select_backend(r) is None


def test_format_report_includes_backend():
    r = DetectionReport()
    r.selected_backend = "vllm"
    text = format_report(r)
    assert "Selected backend: vllm" in text


def test_load_env_keys_reads_from_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-test\n")
    keys = load_env_keys([env_file])
    assert keys.get("OPENAI_API_KEY") == str(env_file)


def test_load_env_keys_prefers_env_var():
    import os
    os.environ["OPENAI_API_KEY"] = "sk-from-env"
    keys = load_env_keys([])
    assert keys.get("OPENAI_API_KEY") == "environment"
    del os.environ["OPENAI_API_KEY"]


def test_find_env_files_finds_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("X=1\n")
    found = find_env_files(start=tmp_path)
    assert any(str(env_file) == str(f) for f in found)
