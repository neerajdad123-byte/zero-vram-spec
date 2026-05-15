# Structspec

**Up to 3.15x faster code generation. Zero extra VRAM.**

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

## Demo

**Side-by-side speedup demo** (Qwen2.5-Coder-7B on RTX 4050):

https://github.com/user-attachments/assets/demo_speedup.mp4

Left: Standard greedy decoding. Right: Structspec speculative decoding.
Same prompt, same model, same hardware. 49 tok/s vs 59 tok/s.

Run it yourself:
```powershell
# Terminal 1 - Baseline
python demo_baseline.py

# Terminal 2 - Structspec
python demo_speculative.py
```

## Benchmarks

Qwen2.5-Coder-7B-Instruct Q4\_K\_M, RTX 4050 (6GB VRAM), 20 DSA code generation prompts:

| Metric | Value |
|:-------|:-----:|
| Wall-clock speedup | **1.11x** average, up to **3.15x** |
| Pass reduction | **1.18x** fewer model calls |
| Draft acceptance rate | **75.7%** |
| Output correctness | **18/20** identical to greedy baseline |

**Per-prompt highlights:**

| Prompt | Greedy Time | Spec Time | Speedup |
|:-------|:----------:|:---------:|:-------:|
| Prime checker | 2.46s | 0.78s | **3.15x** |
| Fibonacci | 1.02s | 0.67s | **1.53x** |
| BST implementation | 2.46s | 1.93s | **1.28x** |
| Insertion at beginning | 2.49s | 1.72s | **1.45x** |
| Max heap | 2.48s | 1.97s | **1.26x** |

Speedup is highest when code patterns repeat across the training corpus (class structures, common algorithms). The system learns token-level n-gram rules from code examples and combines them with syntax-aware proposers (indentation, bracket matching, keyword transitions).

See `benchmark_trace.csv` for full per-token traces.

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

Structspec uses **8 interacting optimization components** to predict tokens without a draft model:

1. **Pattern mining** — Mines token-level n-gram rules from a code corpus. Rules are classified into confidence tiers (deterministic, strong, frequent) with different draft length caps.

2. **SymbolicMotifCache** — Abstracts code patterns across variable names. `current = current.next` becomes `[I0 = I0.I1]`; when `node = node.left` appears later, the same abstract pattern matches and replays concrete tokens. This is the key novel contribution.

3. **ASTShadowEngine** — Grammar-state speculation: tracks bracket depth, indentation level, and line state to predict mandatory syntax closures (colons, indents, bracket closes).

4. **PythonSyntaxProposer** — 50+ pre-compiled regex rules for deterministic Python transitions: `for x` -> ` in`, `def foo()` -> `:\n`, `__init__(` -> `self`.

5. **FastSyntaxFSM** — Token-ID-based finite state machine for syntax prediction, eliminating regex from the hot path.

6. **EntropyGate** — Uses the model's own logit confidence (top1-top2 margin) to dynamically scale draft length. Low entropy = aggressive drafting.

7. **AdaptiveKController** — Adjusts draft length based on rolling acceptance rate. Increases K when acceptance is high, decreases on rejections.

8. **Single-pass verification** — Proposed tokens are verified against the real model in one batch decode using the pending-bonus trick. K+1 tokens verified in one model pass.

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

## Research Contributions

This project introduces several novel techniques for zero-VRAM speculative decoding:

- **Abstract pattern generalization** (SymbolicMotifCache): Converts concrete code patterns into symbolic templates that generalize across variable names, enabling pattern reuse across different codebases.
- **Tiered confidence system**: Rules are classified into confidence tiers with per-tier draft length caps. Higher-confidence rules can chain longer; lower-confidence rules are capped at 1 token.
- **Online rule adaptation**: Rules start with offline confidence but get updated in real-time. Rules that repeatedly fail get put on cooldown. The AdaptiveKController adjusts draft length dynamically.
- **Grammar-state speculation** (ASTShadowEngine): Tracks bracket depth and indentation to predict mandatory syntax closures without any model inference.
- **Pending-bonus trick**: The bonus token (model's own prediction after the draft) is included in the next verification batch, avoiding a separate forward pass.

## Topics

`speculative-decoding` · `llm-inference` · `llama-cpp` · `local-llm` · `python` · `code-generation` · `inference-optimization` · `zero-vram`

## License

MIT