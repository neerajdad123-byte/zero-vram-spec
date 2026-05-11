# Structspec

Zero-VRAM speculative decoding tooling for local LLM workflows.

Structspec is shaped as an API-first bypass: run a local OpenAI-compatible proxy, then point existing tools at it.

## Install

```bash
pip install -e .
```

Optional extras:
```bash
pip install -e ".[tui]"      # Textual status dashboard
pip install -e ".[vllm]"     # vLLM helpers
pip install -e ".[dev]"      # Tests
```

## Quickstart

```bash
# 1. Scan your local LLM environment
structspec detect

# 2. Start the proxy (auto-detects vLLM, LM Studio, Ollama, llama.cpp)
structspec serve --backend auto

# 3. Point any OpenAI-compatible client at it
export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=dummy
```

### Terminal walkthrough

```text
$ structspec detect
Structspec detection report
Selected backend: lm-studio
GPU(s): NVIDIA GeForce RTX, 6141 MiB
LM Studio: running

$ structspec serve --backend lm-studio
Structspec proxy listening on http://127.0.0.1:8080/v1
Forwarding to http://localhost:1234/v1 | backend=lm-studio | safety=strict

$ structspec status --once
Structspec ok | backend=lm-studio | active=0 | requests=12 | est=1.37x | accept=82.6%
```

## Commands

```bash
structspec detect
structspec serve
structspec run --prompt "def fib(n):"
structspec benchmark --dataset humaneval
structspec status --tui
structspec config
structspec vllm-command --model Qwen/Qwen2.5-7B-Instruct --method ngram
```

## vLLM

Structspec integrates with vLLM through the supported `--speculative-config` surface. vLLM currently supports methods such as `ngram`, `suffix`, `draft_model`, `mtp`, EAGLE, MLP speculators, and parallel draft models depending on version/model support.

Example:

```bash
structspec vllm-command \
  --model Qwen/Qwen2.5-7B-Instruct \
  --method ngram \
  --num-speculative-tokens 4 \
  --prompt-lookup-min 2 \
  --prompt-lookup-max 5
```

Then start the proxy:

```bash
structspec serve --backend vllm --target-base-url http://localhost:8000/v1
```

## Benchmarks

HumanEval (first 20 tasks, 128 tokens, Qwen2.5-7B-Instruct Q4_K_M, RTX 4050 laptop, live mining on):

| Backend | Model | Dataset | Reject Mode | Baseline tok/s | Structspec tok/s | Speedup | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| llama.cpp | Qwen2.5-7B Q4_K_M | HumanEval | truncate | 1.00x | **1.37x** | 1.37x | Fresh run (82.64% accept, 19.07% fire) |
| llama.cpp | Qwen2.5-7B Q4_K_M | HumanEval | seq-bonus | 1.00x | **1.33x** | 1.33x | Adaptive K conservative |
| llama.cpp | Qwen2.5-7B Q4_K_M | HumanEval | seq-bonus | 1.00x | **1.27x** | 1.27x | Adaptive K balanced |

Key metrics (fresh run):
- Wall speedup: **1.37x**
- Pass speedup: **1.35x**
- Draft token acceptance: **82.64%** (671 / 812)
- Pattern fire rate: **19.07%**
- Greedy-identical outputs: **15 / 20**
- Total greedy time: **75.2s** → speculative: **54.7s**

For full per-tier traces and rejection diffs, see `humaneval_results_fresh/humaneval_report.md`.

## Safety

Default mode is `strict`: Structspec only uses backend-supported lossless speculative paths, or forwards unchanged when the backend cannot expose verification safely.

Modes:

- `strict`: preserve backend output semantics.
- `fast`: allow backend-specific faster speculative settings.
- `audit`: strict mode plus extra request/metrics logging.

## Project Status

Current package includes:

- OpenAI-compatible proxy endpoints: `/v1/models`, `/v1/chat/completions`, `/v1/completions`.
- Health and Prometheus-style metrics endpoints.
- Auto-detection for vLLM, llama.cpp, LM Studio, Ollama, model directories, `.env` keys, and GPUs.
- vLLM command generation for supported speculative methods.
- Status dashboard fallback, with optional Textual TUI path.
- Prompt domain detection for Python, JSON, HTML, SQL, Go, and generic modes.
- Structural speculative decoding research tools and HumanEval benchmarking harness.

## Development

```bash
# Install dev deps
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check structspec tests
```

## Topics

`speculative-decoding` `llm-inference` `llama-cpp` `local-llm` `python` `zero-vram` `vllm` `openai-api` `code-generation`

## License

MIT
