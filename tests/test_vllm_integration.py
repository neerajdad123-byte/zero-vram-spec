from __future__ import annotations

import pytest

from structspec.vllm_integration import build_vllm_command, shell_join, speculative_config


def test_speculative_config_ngram():
    cfg = speculative_config(method="ngram", num_speculative_tokens=4)
    assert cfg["method"] == "ngram"
    assert cfg["num_speculative_tokens"] == 4
    assert "prompt_lookup_min" in cfg
    assert "prompt_lookup_max" in cfg


def test_speculative_config_suffix():
    cfg = speculative_config(method="suffix", num_speculative_tokens=4)
    assert cfg["method"] == "suffix"
    assert "suffix_decoding_max_tree_depth" in cfg


def test_speculative_config_draft_model_requires_model():
    with pytest.raises(ValueError):
        speculative_config(method="draft_model", draft_model=None)

    cfg = speculative_config(method="draft_model", draft_model="tiny-model")
    assert cfg["model"] == "tiny-model"


def test_build_vllm_command():
    cmd = build_vllm_command(
        model="Qwen/Qwen2.5-7B-Instruct",
        method="ngram",
        num_speculative_tokens=4,
    )
    assert cmd[0] == "vllm"
    assert cmd[1] == "serve"
    assert "Qwen/Qwen2.5-7B-Instruct" in cmd
    assert "--speculative-config" in cmd


def test_shell_join():
    assert shell_join(["echo", "hello world"]) == "echo 'hello world'"
