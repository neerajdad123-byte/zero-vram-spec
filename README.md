# Structspec

Zero-VRAM speculative decoding tooling for local LLM workflows.

Structspec is shaped as an API-first bypass: run a local OpenAI-compatible proxy, then point existing tools at it.

```bash
pip install -e .
structspec detect
structspec serve --backend vllm --target-base-url http://localhost:8000/v1
```

Then reuse normal OpenAI SDK clients:

```bash
set OPENAI_BASE_URL=http://localhost:8080/v1
set OPENAI_API_KEY=dummy
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
structspec vllm-command ^
  --model Qwen/Qwen2.5-7B-Instruct ^
  --method ngram ^
  --num-speculative-tokens 4 ^
  --prompt-lookup-min 2 ^
  --prompt-lookup-max 5
```

Then start the proxy:

```bash
structspec serve --backend vllm --target-base-url http://localhost:8000/v1
```

## Benchmarks

Add side-by-side benchmark screenshots, video links, and tables here.

| Backend | Model | Dataset | Baseline tok/s | Structspec tok/s | Speedup | Notes |
|---|---|---:|---:|---:|---:|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO |

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
