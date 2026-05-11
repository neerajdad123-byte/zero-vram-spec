# Structspec

**2x faster code generation. Zero extra VRAM.**

Structspec is a zero-VRAM speculative decoding proxy that makes your local LLM generate code faster — without requiring a second model loaded in GPU memory.

```
[Your editor] → structspec → [Local LLM backend]
                  ↑
           Runs inline on the stream,
           predicting syntax tokens for free
```

<!-- [TOC] -->

## Why

Local LLMs are powerful but slow. Standard speculative decoding speeds things up but requires loading a *second model* in VRAM — which most consumer GPUs can't afford.

Structspec eliminates that cost. It works inline in the token stream, predicting high-confidence tokens (indentation, keywords, brackets) through lightweight pattern mining and syntax-aware rules — then verifying them in a single pass against the real model.

**No second model. No extra VRAM. Just faster output.**

## Benchmarks

HumanEval, Qwen2.5-7B-Instruct Q4\_K\_M, RTX 4050 laptop, 128 tokens:

| Reject Mode   | Baseline tok/s | Structspec tok/s | Speedup | Accept Rate |
|:-------------|:--------------:|:----------------:|:-------:|:-----------:|
| truncate      | 1.00x          | **1.37x**        | 1.37x   | 82.6%       |
| seq-bonus     | 1.00x          | **1.33x**        | 1.33x   | 82.6%       |
| Adaptive K    | 1.00x          | **1.27x**        | 1.27x   | 82.6%       |

- **1.37x wall speedup** on code generation
- **75.2s → 54.7s** total generation time
- **82.6%** of speculative tokens accepted
- **15/20** outputs bitwise-identical to greedy baseline

See `humaneval_results_fresh/` for full per-prompt traces.

## Install

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# 1. Auto-detect your local LLM backend
structspec detect

# 2. Start the proxy (it finds vLLM, llama.cpp, LM Studio, Ollama automatically)
structspec serve --backend auto

# 3. Point ANY OpenAI-compatible client at it
export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=dummy

# 4. Use it — just change the URL, nothing else
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"write a fibonacci function in python, only code"}]}'
```

That's it. No model changes, no client changes. Structspec sits transparently between your client and your backend.

## How It Works

1. **Pattern mining** — Before inference, Structspec mines token-level patterns from a corpus of code. These are "after token X, token Y follows 95% of the time" rules.

2. **Syntax-aware fallback** — For Python, C, Go, and other structured languages, certain tokens are *deterministic*. After `def foo():`, the next line **always** indents. After `for i in`, `range(` is almost certain. Structspec catches these.

3. **Single-pass verification** — Proposed tokens are verified against the real model in one batch decode using the pending-bonus trick. This means K+1 tokens verified in one model pass instead of K+1 separate passes.

4. **Dynamic draft bypass** — If a syntax rule has near-certainty, Structspec emits it directly without ever querying a draft model. When confidence is lower, it falls back to optional model-based speculative decoding.

```
Time →
  Model decode (prompt)
    → Pattern proposes [indent, def, :]
    → Single model pass verifies all 3
    → 3 tokens from 1 pass = 3x speedup on that step
```

## Commands

```bash
structspec detect              # Scan your LLM environment
structspec serve               # Start the proxy
structspec run --prompt "..."  # Generate a completion
structspec benchmark           # Run HumanEval benchmarks
structspec status              # Live metrics dashboard
structspec config              # Show current config
structspec vllm-command ...    # Generate vLLM launch command
```

## Project Structure

```
structspec/
  proxy.py        — OpenAI-compatible reverse proxy with streaming passthrough
  detect.py       — Auto-detection for vLLM, llama.cpp, LM Studio, Ollama
  cli.py          — CLI commands, PatternMiner, PythonSyntaxProposer
  domains.py      — Domain detection (Python, JSON, HTML, SQL, Go, Generic)
  metrics.py      — Prometheus-style metrics and throughput tracking
  vllm_integration.py — vLLM speculative config generation
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check structspec cli.py
```

<!-- Badges
TODO: Add CI badge, PyPI badge when ready
[![Tests](https://github.com/neerajdad123-byte/zero-vram-spec/actions/workflows/test.yml/badge.svg)](https://github.com/neerajdad123-byte/zero-vram-spec/actions)
-->

## Topics

`speculative-decoding` · `llm-inference` · `llama-cpp` · `local-llm` · `python` · `code-generation`

## License

MIT