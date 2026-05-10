# Agent Guidance

## Project Overview
Structspec is a Zero-VRAM speculative decoding proxy and tooling suite for local LLM backends. It exposes an OpenAI-compatible API that sits in front of vLLM, llama.cpp, LM Studio, or Ollama, optionally applying structural/syntax-aware speculative decoding to speed up generation.

## Repository Layout
- `structspec/` — The installable Python package containing the proxy, CLI, detection, metrics, domains, and vLLM integration.
- `cli.py` (root) — Standalone research script with the core speculative decoder loop (PatternMiner, PythonSyntaxProposer, FastGreedyLlama). Kept at root for historical workflow compatibility.
- `run_humaneval_benchmark.py` — Benchmark runner that consumes the root `cli.py` research code to run HumanEval comparisons.
- `tests/` — Unit tests for the `structspec` package.

## Build & Test
```bash
pip install -e ".[dev]"
pytest
```

## Code Style
- Python 3.10+
- `from __future__ import annotations`
- Use type hints where practical.
- Keep the public `structspec/` package free of root-script dependencies.

## Adding Features
- Proxy changes go in `structspec/proxy.py`.
- CLI commands go in `structspec/cli.py`.
- New backend detections go in `structspec/detect.py`.
- If you modify speculative decoding logic, update root `cli.py` and keep benchmark runners in sync.

## Known Constraints
- The root `cli.py` research script depends on `llama_cpp`, `numpy`, and a Qwen token corpus JSON. It is NOT imported by the `structspec` package by default.
- `run_humaneval_benchmark.py` requires `--model` and `--token-json` (or env vars `STRUCTSPEC_MODEL` and `STRUCTSPEC_TOKEN_JSON`).
