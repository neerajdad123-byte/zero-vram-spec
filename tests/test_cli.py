from __future__ import annotations

import pytest

from structspec.cli import build_parser


def test_parser_detect():
    parser = build_parser()
    args = parser.parse_args(["detect"])
    assert args.command == "detect"


def test_parser_serve_defaults():
    parser = build_parser()
    args = parser.parse_args(["serve"])
    assert args.command == "serve"
    assert args.backend == "auto"
    assert args.port == 8080
    assert args.safety == "strict"


def test_parser_vllm_command():
    parser = build_parser()
    args = parser.parse_args([
        "vllm-command",
        "--model", "Qwen/Qwen2.5-7B-Instruct",
        "--method", "suffix",
        "--num-speculative-tokens", "8",
    ])
    assert args.command == "vllm-command"
    assert args.model == "Qwen/Qwen2.5-7B-Instruct"
    assert args.method == "suffix"
    assert args.num_speculative_tokens == 8


def test_parser_run_requires_prompt():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run"])
