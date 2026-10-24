# Greedy vs speculative identity on Qwen3.6

Speculative decoding in Structspec is meant to be lossless: the output must be byte-identical
to plain greedy decoding of the same model. This page records how that is checked and what was found.
It contains no timing numbers on purpose (timings are measured separately in `docs/research/`).

## How it is checked

1. **Unit tests (CI, no GPU).** `tests/test_lossless.py` drives the speculative loop with a fake
   model whose next token is a hash of its own KV cache contents, so any stale or wrongly
   rolled-back cache changes the output. Every reject path is covered (wrong first token,
   wrong middle token, random drafts, drafts containing end-of-generation, drafts longer than the
   token budget, empty drafts) for all three reject modes, and spec output must equal greedy output.
   Breaking the rollback or the end-of-generation cut on purpose makes these tests fail.
2. **Real model.** `structspec verify` runs plain greedy and speculative decoding on the same
   prompts and compares the token ids. On a mismatch it replays the position twice, once as greedy
   computes it (single-token decode) and once with the last k+1 inputs in one batch. If only the
   batched replay produces the speculative token and the greedy top-1/top-2 logit margin is
   below 0.5, the mismatch is classified `numerics` (batch-shape floating point); otherwise `bug`.

## Setup

- llama.cpp b11756 official Windows CUDA 13 build (`llama_version()` = `0.7.0-dev`), loaded by
  `structspec/llama_native.py`.
- Model: `Qwen3.6-35B-A3B-Q4_K_M.gguf`. llama.cpp reports it as a **hybrid** model (recurrent
  layers), so the context is created with `n_rs_seq = 16` recurrent-state snapshots. That is what
  lets rejected draft tokens be rolled back.
- RTX 4050 Laptop 6 GB, all layers on GPU except MoE expert weights (`--cpu-moe`), `n_ctx` 4096.
- 8 built-in prompts (`structspec/verify.py`: ISO date parser, TypeScript types, callback to
  async/await, Go handler, type hints, SQL, Rust line counter, bash backup), user message through
  the model's chat template with thinking disabled.
- Proposer: prompt lookup only (no n-gram index was built for this run), k = 4, reject mode `truncate`.

## Results (2026-10-24)

```
structspec verify --model Qwen3.6-35B-A3B-Q4_K_M.gguf --cpu-moe --max-tokens 256
```

| prompt | tokens | drafts accepted / proposed | result |
|---|---|---|---|
| 0 ISO date parser | 36 | 2 / 8 | identical |
| 1 TypeScript interface | 94 | 14 / 51 | identical |
| 2 callback → async/await | 37 | 1 / 24 | identical |
| 3 Go handler | 49 | 2 / 4 | identical |
| 4 type hints | 63 | 26 / 44 | identical |
| 5 SQL top customers | 56 | 3 / 20 | identical |
| 6 Rust line counter | 115 | 10 / 40 | identical |
| 7 bash backup | 91 | 4 / 40 | identical |

**8 / 8 identical** (also 8 / 8 in an earlier 64-token run). Every prompt ended on an
end-of-generation token before the 256-token budget. Rejections happened on every prompt, so
the rollback path on the hybrid cache was exercised.

## Root causes of earlier mismatches

v1 reported 18 / 20 outputs matching greedy. The v1 loop had a real bug that can explain this:
when an accepted draft contained the end-of-generation token, generation continued past it, while
greedy stopped. Separately, drafts were not clamped to the requested length. Both are fixed
and covered by `test_draft_contains_eog`, `test_greedy_eog_inside_accepted_draft_stops` and
`test_draft_longer_than_budget`. The v1 runs used a different model (Qwen2.5-Coder) through
llama-cpp-python, so they were not re-run.

## Caveats

- Batch-shape numerics are a known possible cause of divergence in llama.cpp (different kernels
  for batch 1 and batch > 1). None showed up here, but 8 short prompts can't rule it out;
  `structspec verify` classifies such a mismatch if one appears.
- The acceptance counts above come from prompt lookup on short answers and say nothing about speed.
