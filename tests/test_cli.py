from __future__ import annotations

import pytest

from structspec.cli import build_parser


def test_parser_detect():
    parser = build_parser()
    args = parser.parse_args(["detect"])
    assert args.command == "detect"


def test_parser_serve_defaults():
    args = build_parser().parse_args(["serve", "--model", "m.gguf"])
    assert args.command == "serve"
    assert args.port == 8080
    assert args.k == 15 and not args.passthrough and not args.thinking


def test_parser_serve_passthrough():
    args = build_parser().parse_args(["serve", "--passthrough"])
    assert args.passthrough and args.backend == "auto" and args.safety == "strict"


def test_serve_requires_model_or_passthrough(capsys):
    from structspec.cli import main

    assert main(["serve"]) == 2
    assert "--model" in capsys.readouterr().err


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


def test_parser_verify():
    args = build_parser().parse_args(["verify", "--model", "m.gguf", "--max-tokens", "32", "--cpu-moe"])
    assert args.command == "verify" and args.max_tokens == 32 and args.cpu_moe and args.k == 15


def test_parser_index():
    args = build_parser().parse_args(["index", "src", "lib", "--model", "m.gguf"])
    assert args.paths == ["src", "lib"]
