# Structspec

**Speculative decoding for local code models with no draft model and no extra VRAM.**

Structspec drafts tokens by copying from what is already in the context (the prompt, the file you're editing, the code written so far) and verifies the whole draft in one batched pass of your model. Output is the model's own greedy output: every drafted token is checked against the model before it is kept.

Measured on an RTX 4050 Laptop (6 GB), Qwen2.5-Coder-3B-Instruct Q4_K_M, 64 held-out tasks in 7 languages, wall-clock:

| task | greedy | Structspec | speedup | draft acceptance |
|---|---|---|---|---|
| **Editing code** (refactor, add types, fix, comment — 26 tasks) | 81 tok/s | ~465 tok/s | **5.7×** (median 7.3×) | 96% |
| **Writing new code** (HumanEval-style functions — 38 tasks) | 84 tok/s | ~119 tok/s | **1.4×** (median 1.1×) | 77% |
| All 64 tasks | 82 tok/s | ~267 tok/s | **3.3×** (median 2.1×) | 93% |

The honest summary: **big wins when the answer repeats the input (edits, rewrites, refactors), modest wins on fresh code.** Syntax rules alone add almost nothing. See [what helps and what doesn't](#what-helps-and-what-doesnt).

## Results in detail

### End-to-end, Qwen2.5-Coder-3B (dense, fully on GPU)

`python -m bench.e2e --model qwen2.5-coder-3b-instruct-q4_k_m.gguf --budget 600`: same model in one process, greedy vs speculative on each task, max draft 16 tokens. Totals are total-time ratios; medians are per task.

| language | tasks (fresh + edit) | speedup | draft acceptance |
|---|---|---|---|
| Python | 8 | 4.4× (median 2.5×) | 98% |
| Java | 11 | 3.9× (median 3.0×) | 96% |
| Rust | 9 | 3.8× (median 1.2×) | 97% |
| Go | 9 | 3.1× (median 2.1×) | 92% |
| JavaScript | 9 | 2.9× (median 1.1×) | 89% |
| TypeScript | 10 | 2.9× (median 1.4×) | 90% |
| C++ | 8 | 2.3× (median 1.1×) | 91% |

**Identical output:** 61 of 64 outputs are byte-identical to greedy. The other 3 were checked with `structspec verify`. Each diverges at a near-tie, where the model's top two tokens are within 0.01–0.07 logits and the batched GPU kernel rounds differently than the single-token kernel. That is floating-point behaviour of llama.cpp batch sizes, not a drafting error: the kept text is still the model's own argmax at that point.

### Qwen3.6-35B-A3B (MoE, experts partly on CPU)

With most experts on the CPU, verifying extra tokens costs more: a decode step is ~40 ms and each extra drafted token adds ~7 ms. Numbers below come from exact replay of held-out greedy traces against the measured cost curve:

| task | draft acceptance | predicted speedup |
|---|---|---|
| Editing code | 94% | **3.2×** |
| Writing new code | 65% | **1.5×** |
| All | 75% | **1.7×** |

Full method, per-language tables and the cost curve: [docs/research/speed-findings.md](docs/research/speed-findings.md), [docs/research/verify-cost.md](docs/research/verify-cost.md).

### Compared with llama.cpp's built-in speculation

On the same MoE, llama.cpp b11756's `--spec-type ngram-mod` / `ngram-simple` with default settings ran at ~0.4× of plain greedy (one task). Their default drafts of 48–64 tokens are too expensive to verify on a CPU-offloaded MoE. This GGUF has no MTP layers, so `draft-mtp` could not be tested.

## What helps and what doesn't

Measured on held-out tasks, Qwen3.6 replay:

| drafting idea | speedup it adds |
|---|---|
| Longest-suffix copy from context, draft length scaled by match length | 1.0× → **1.66×** (nearly all of the gain) |
| Learned n-gram model over code, as a fallback | +0.06× |
| Line-start structure (indent, closing brace from the previous line's shape) | +0.02× |
| Cache n-gram over the current request | +0.00× (copy already covers it) |
| Old prompt lookup (copy max_k after any 2-gram) | 1.11× total, and **slower than greedy on fresh JS/Java** (25% acceptance) |

The remaining tokens are the model choosing *which* identifier or expression comes next. No rule predicts those; a model-based drafter is needed for them.

## Install

```bash
pip install git+https://github.com/neerajdad123-byte/zero-vram-spec
```

Or from a clone: `pip install -e .`

Needs the official llama.cpp release DLLs (`llama.dll` + its CUDA DLLs). Set `STRUCTSPEC_LLAMA_DIR` to the folder that contains them.

## Quick start

```bash
# OpenAI-compatible server with speculative decoding on a GGUF model
structspec serve --model qwen2.5-coder-3b-instruct-q4_k_m.gguf --port 8080

# MoE models that don't fit in VRAM: keep expert weights on the CPU
structspec serve --model Qwen3.6-35B-A3B-Q4_K_M.gguf --cpu-moe

# One-off generation, and a greedy-vs-speculative identity check
structspec generate --model model.gguf --prompt "Write a Go HTTP handler that returns JSON"
structspec verify --model model.gguf
```

Then point any OpenAI-compatible client at `http://localhost:8080/v1` (`/v1/chat/completions` and `/v1/completions`, streaming supported).

Optional: `structspec index --model model.gguf <source dirs>` mines an n-gram index with your model's tokenizer; `serve` uses it automatically as a second drafter.

## How it works

1. **Draft.** `SuffixCopyProposer` finds the longest earlier occurrence of the last few tokens in the prompt and output, and drafts what followed it there. Short matches get short drafts, long matches get long ones (up to 15 tokens by default).
2. **Verify.** The last accepted token and the draft go through the model in one batched decode. Tokens are kept up to the first one that differs from the model's argmax; the model's own token at that position is kept too.
3. **Roll back.** The KV cache is truncated to the accepted length. For hybrid/recurrent models, rollback uses llama.cpp state snapshots.

Drafters work on token ids only, so any GGUF model and tokenizer works. All drafters implement one small interface (`structspec/proposer.py: Proposer`), with others in `structspec/proposers/`.

## Roadmap

Planned, not built or measured yet:

- **Learn the user's own repo.** Seed the copy drafter with the project's own files (`structspec index` already tokenizes a project), so names and boilerplate from the user's codebase can be drafted while writing new code in it.
- **Learn across requests.** Remember what the model wrote in earlier answers in the same server session, so a user's repeated patterns get drafted from past output.

## Reproduce the benchmarks

```bash
pip install -e ".[dev]"
python -m bench.fetch_data                     # MultiPL-E prompts, edit-set files, mining corpus (pinned commits)
python -m bench.e2e --model <gguf> --budget 600 # end-to-end greedy vs speculative
python -m bench.verify_cost                    # verify-cost curve via llama-bench
python -m bench.traces --split test            # greedy traces from a running llama-server
python -m bench.replay --split test --proposers suffix_copy,combo --max-k 16
```

Tasks are split by problem into tune and test sets, and every parameter was chosen on tune. Edit-set repos and mining-corpus repos are disjoint.

## Development

```bash
pytest tests bench/tests
ruff check structspec bench
```

## License

MIT
