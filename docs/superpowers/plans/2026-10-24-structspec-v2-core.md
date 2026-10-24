# Structspec v2 core (Workstream 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `structspec` a real, lossless, tokenizer-agnostic speculative decoder served over an OpenAI-compatible API, on top of the official llama.cpp DLL.

**Architecture:** A ctypes binding (`llama_native.py`) replaces llama-cpp-python. The speculative loop becomes a generator over a `Proposer` (the spec's shared contract), so the CLI, the lossless tests and the server all drive the same code. Drafts come from prompt lookup plus an n-gram index mined with the loaded model's tokenizer.

**Tech Stack:** Python 3.10+, ctypes, numpy, FastAPI/uvicorn, pytest. llama.cpp b11756 (`llama_version()` reports `0.7.0-dev`).

**Spec:** `docs/superpowers/specs/2026-10-24-structspec-v2-design.md` (Workstream 1 + Shared contract)

## Global Constraints

- Output must be byte-identical to plain greedy decoding; enforced by tests.
- Proposer contract, verbatim: `name: str`; `propose(self, tokens: list[int], max_k: int) -> list[int]`; `observe(self, accepted: list[int]) -> None  # verified tokens only`.
- DLL dir from `STRUCTSPEC_LLAMA_DIR`, else auto-detect; version check at load, clear error.
- No llama-cpp-python, no private `llm._ctx` access.
- Engine must not depend on `qwen_coder_dsa_tokens.json`.
- Do not edit `bench/`, `structspec/proposers/`, `docs/research/` (owned by v2-speed-lab).
- CI (ubuntu + windows, py 3.10–3.12) has no GPU and no DLL: nothing may load the DLL at import time.
- GPU runs: short, correctness-only, never report timings (speed-lab owns timing).
- Out of scope: sampling (temperature > 0), vLLM plugin, TUI, multi-GPU.

## Review Focus

1. Prompt + `max_tokens` larger than `n_ctx` → HTTP 400 with a clear message, not a llama_decode crash. (Task 5 test)
2. A multi-byte UTF-8 character split across tokens while streaming → concatenated SSE deltas equal the non-streamed text, no `\ufffd`. (Task 5 test)
3. A draft that contains an end-of-generation token mid-draft, or runs past `max_tokens` → output stops exactly where greedy stops. (Task 2 test)
4. Two concurrent requests → serialized on the one model context; both outputs equal their greedy outputs. (Task 5 test)
5. Index mined with a different tokenizer → loading it fails with an error naming both hashes. (Task 3 test)

## Decisions for review (deviations or gaps in the spec)

- **Order:** the Proposer protocol (spec step 3) moves into Task 2, because the fake-model tests need to inject arbitrary drafts. Index mining and prompt lookup stay in Task 3.
- **Loop simplification:** with `propose()` returning plain `list[int]`, the loop's per-rule strike/ban/registry/early-exit bookkeeping (keyed on `Rule` objects) goes away; `AdaptiveKController` and the three reject modes stay. The old miner + `PythonSyntaxProposer` survive behind a `RuleProposer` adapter.
- **Hybrid memory:** Qwen3.6 has recurrent layers (to confirm with `llama_model_is_hybrid`). Partial KV removal on such models only works with `n_rs_seq > 0` (experimental in b11756). The binding sets `n_rs_seq = max_rollback` (default 16) for hybrid/recurrent models, caps `k` at `max_rollback - 1`, and `truncate_kv` raises if `llama_memory_seq_rm` returns false. If rollback turns out broken, I stop and report.
- **Version check:** b11756 exports no build number, only `llama_version() == "0.7.0-dev"`. The check is that string plus the presence of newer symbols (`llama_n_rs_seq`, `llama_model_is_hybrid`, `llama_memory_seq_rm`).
- **Sampling params:** `serve` ignores `temperature`/`top_p` and always decodes greedily, because many clients send `temperature: 0.7` by default and rejecting them would make the server unusable. Documented in `/health` (`"decoding": "greedy"`).
- **Thinking:** for templates containing `<think>`, `serve` appends `<think>\n\n</think>\n\n` after the assistant prefix (Qwen's `enable_thinking=False`); `--thinking` turns this off.
- **Extra deletions beyond spec §5:** `run_humaneval_benchmark.py` imports the root `cli.py` being deleted, so it goes too, along with the `structspec benchmark` subcommand (benchmarks live in speed-lab's `bench/`). `trace`/`SpeedTraceCollector`/`QwenTokenCorpus` go because nothing uses them after that. `HumanEval.jsonl.gz` stays (speed-lab uses it). `humaneval_results_fresh/` waits for the README step.
- **mp4:** `git rm` plus `*.mp4` in `.gitignore`; it stays in history (no rewrite). Hosting it elsewhere (e.g. a release asset) is the user's call.
- **CI:** CI runs only on push/PR to `master`, so "green in CI" means pushing this branch and opening a **draft PR** into `master`.

---

### Task 1: Native llama.cpp backend

**Files:**
- Create: `structspec/llama_native.py`
- Modify: `structspec/llama_backend.py` (drop `FastGreedyLlama` and the llama_cpp import; import `NativeLlama`)
- Modify: `pyproject.toml` (remove `research` extra)
- Test: `tests/test_llama_native.py`

**Interfaces:**
- Produces:
  - `find_llama_dir() -> Path`: `$STRUCTSPEC_LLAMA_DIR`, else the first dir on `PATH` holding the library, else newest `~/tools/llama.cpp-*/bin`. Raises `LlamaLoadError` listing what was searched.
  - `check_version(lib) -> None`: raises `LlamaLoadError("llama.cpp version X unsupported; Structspec is built against b11756 (0.7.0-dev)...")`.
  - `class NativeLlama(model_path, n_ctx=4096, n_batch=512, n_gpu_layers=-1, cpu_moe=False, max_rollback=16, vocab_only=False)` with `encode(text, special=False) -> list[int]`, `decode(ids) -> str`, `piece(tok) -> str`, `piece_bytes(tok) -> bytes`, `reset()`, `truncate_kv(n)`, `decode_logits(tokens, logits_all) -> np.ndarray` (rows × n_vocab, copied), `argmax(row) -> int`, `is_eog(tok) -> bool`, `eos: int`, `n_vocab: int`, `n_ctx: int`, `tokenizer_hash() -> str`, `apply_chat_template(messages: list[dict]) -> str`, `close()`.
  - `cpu_moe=True` sets `tensor_buft_overrides` = `[(r"\.ffn_(up|down|gate)_exps", ggml_backend_cpu_buffer_type()), (NULL, NULL)]`.
  - `tokenizer_hash()` = sha256 over `n_vocab` and every token's `llama_token_to_piece(..., special=True)` bytes, hex.
- Loading: `os.add_dll_directory` (Windows), load `ggml.dll` → `ggml_backend_load_all_from_path(dir)`, `llama_backend_init()`, then `llama.dll`. Struct layouts copied from `llama.h@b11756` (`llama_model_params`, `llama_context_params`, `llama_batch`). Batch decode via `llama_batch_init` with explicit pos/seq_id/logits; logits via `llama_get_logits_ith`.

- [ ] **Step 1: Write failing tests** (no DLL needed)
  - `test_find_llama_dir_env(tmp_path, monkeypatch)`: dir with a dummy `llama.dll`/`libllama.so` → returned.
  - `test_find_llama_dir_missing_raises`: env points to empty dir, PATH cleared, HOME tmp → `LlamaLoadError` whose message contains `STRUCTSPEC_LLAMA_DIR`.
  - `test_check_version_rejects_unknown`: fake lib whose `llama_version()` returns `b"0.5.0"` → `LlamaLoadError` containing `"b11756"`.
  - `test_check_version_rejects_missing_symbol`: right version, no `llama_n_rs_seq` → error names the symbol.
  - `test_native_smoke` (skipped unless `STRUCTSPEC_TEST_MODEL` is set): load `vocab_only=True`, `decode(encode("def f(x):\n"))` round-trips; `tokenizer_hash()` is 64 hex chars and stable across two calls.
- [ ] **Step 2:** `.venv/Scripts/python -m pytest tests/test_llama_native.py -v` → FAIL (module missing).
- [ ] **Step 3:** Implement `llama_native.py`; rewire `llama_backend.py`; remove `research` extra.
- [ ] **Step 4:** Tests pass; with the model: `STRUCTSPEC_TEST_MODEL=C:\models\gguf\Qwen3.6-35B-A3B-Q4_K_M.gguf pytest tests/test_llama_native.py -v` → smoke passes. Also log `llama_model_is_hybrid` for Qwen3.6 (full load, `cpu_moe=True`, one 8-token greedy decode).
- [ ] **Step 5:** Commit `feat: ctypes binding to official llama.cpp b11756, drop llama-cpp-python`.

### Task 2: Proposer contract, generator loop, lossless tests

**Files:**
- Create: `structspec/proposer.py`, `tests/fake_model.py`, `tests/test_lossless.py`
- Modify: `structspec/llama_backend.py` (loop), `structspec/cli.py` (`generate`)

**Interfaces:**
- Produces in `structspec/proposer.py`: `Proposer` Protocol (verbatim, `runtime_checkable`); `ChainProposer(proposers)` (first non-empty draft; `observe` fans out; `name="chain"`); `RuleProposer(miner, syntax=None)` wrapping `engine.propose_draft` (`name="rules"`).
- Produces in `llama_backend.py`:
  - `@dataclass SpecStats: passes, generated, proposed, accepted, draft_starts` (ints).
  - `speculate(model, proposer, prompt_ids: list[int], max_tokens: int, k: int = 4, reject_mode: str = "truncate", stats: SpecStats | None = None) -> Iterator[list[int]]`: yields committed tokens per target pass; calls `proposer.observe(committed)`; stops after the first EOG token (inclusive, like greedy) or at `max_tokens`.
  - `run_speculative(model, proposer, prompt: str | list[int], max_tokens, k=4, reject_mode="truncate") -> (ids, text, stats_dict)` and `run_greedy(model, prompt: str | list[int], max_tokens)`; both use `model.is_eog`.
- `tests/fake_model.py`: `FakeModel(vocab=50, eog=0, n_ctx=512, seed=0)` implementing the Task 1 model interface. Its next token is a deterministic hash of **its own KV contents**, so a stale or wrong KV changes the output (catches rollback bugs). `piece_bytes` maps some ids to half of a 2-byte UTF-8 char (used by Task 5).

- [ ] **Step 1: Write failing tests in `tests/test_lossless.py`**, each asserting `run_speculative(...)[0] == run_greedy(...)[0]` for prompt lengths {1, 5, 40} and `max_tokens` {1, 7, 64}, parametrized over `reject_mode in ("truncate", "seq-bonus", "rebuild")`:
  - `test_oracle_drafts`: proposer returns the true greedy continuation (all accepted); also assert `stats.passes < greedy passes`.
  - `test_random_drafts`: seeded random drafts.
  - `test_first_token_wrong` / `test_mid_draft_wrong`: oracle with position 0 / position `max_k//2` corrupted.
  - `test_draft_contains_eog`: oracle draft with the EOG id injected mid-draft, plus the case where greedy really emits EOG inside an accepted draft. Output ends at the first EOG.
  - `test_draft_longer_than_budget`: proposer ignores `max_k` and returns 50 tokens → output length ≤ `max_tokens`, equal to greedy.
  - `test_empty_proposer`: always `[]`.
  - `test_observe_gets_only_committed`: recording proposer; concatenation of `observe` args == generated ids.
  - `test_chain_proposer_first_non_empty`.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** Implement. Known bugs to fix: EOG inside an accepted draft is not cut; the no-op timer at old `llama_backend.py:228`; `len(draft) > max_k` is not clamped.
- [ ] **Step 4:** `pytest tests/test_lossless.py -v` → all pass.
- [ ] **Step 5:** Commit `feat: Proposer contract and generator spec loop, lossless tests on fake model`.

### Task 3: Tokenizer-agnostic drafting (index + prompt lookup)

**Files:**
- Modify: `structspec/proposer.py`, `structspec/cli.py` (`index` subcommand; `generate` loses `--token-json` and gains `--index`)
- Create: `tests/test_index.py`, `tests/test_prompt_lookup.py`

**Interfaces:**
- `PromptLookupProposer(max_ngram=3, min_ngram=2)`, `name="prompt_lookup"`: for n from max to min, find the most recent earlier occurrence of `tokens[-n:]` in `tokens[:-1]` and return up to `max_k` following tokens.
- `build_index(model, paths: list[Path], max_ctx=8, min_support=2, min_conf=0.9, min_rule_ctx=3) -> dict`: walks files with source extensions (`.py .js .ts .tsx .go .rs .java .c .cc .cpp .h .hpp .cs .rb .php .json .md`), `model.encode` per file, `PatternMiner.fit`. Returns `{"version": 1, "tokenizer_hash", "n_vocab", "max_ctx", "rules": [[ctx, token, conf, support, total], ...]}`.
- `save_index(index, path)`, `load_index(path, tokenizer_hash) -> PatternMiner` (raises `ValueError` naming both hashes on mismatch).
- `default_index_path(tokenizer_hash) -> Path` = `~/.cache/structspec/index-<hash[:16]>.json`.
- `NgramIndexProposer(miner)`, `name="ngram_index"`.
- CLI: `structspec index PATHS... --model GGUF [--out PATH]` (loads `vocab_only=True`).

- [ ] **Step 1: Failing tests**
  - `test_prompt_lookup_copies_continuation`: tokens `[1,2,3,4,5,9,1,2,3]`, max_k 3 → `[4,5,9]`.
  - `test_prompt_lookup_prefers_longest_and_most_recent`.
  - `test_prompt_lookup_no_match_returns_empty`, `test_prompt_lookup_respects_max_k`.
  - `test_build_and_load_index_roundtrip(tmp_path)` with FakeModel (char-level encode): mined rule proposes the repeated continuation.
  - `test_load_index_rejects_other_tokenizer`: message contains both hashes.
  - `test_index_skips_binary_and_unknown_ext`.
  - `test_lossless_with_prompt_lookup` (added to the Task 2 suite).
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** Implement; `generate` default proposer = `ChainProposer([PromptLookupProposer(), NgramIndexProposer(index if found)])`, `--syntax` adds `RuleProposer(empty miner, PythonSyntaxProposer)`.
- [ ] **Step 4:** pass; `grep -r qwen_coder_dsa structspec` → nothing.
- [ ] **Step 5:** Commit `feat: tokenizer-keyed n-gram index and prompt-lookup proposer`.

### Task 4: Greedy-vs-spec identity on Qwen3.6

**Files:**
- Modify: `structspec/cli.py` (`verify` subcommand)
- Create: `structspec/verify.py`, `tests/test_verify.py`, `docs/lossless-check.md`

**Interfaces:**
- `verify_prompt(model, proposer, prompt_ids, max_tokens, k) -> dict` with `identical: bool`, `first_mismatch: int | None`, `greedy_margin` (top1 − top2 logit at that position from single-token decode), `batched_margin`, `classification: "identical" | "numerics" | "bug"`. Classification: at the first mismatch, re-decode the shared prefix plus that position as a batch of the same shape. If the batched argmax differs from the single-token argmax with margin < 0.5, it's `"numerics"`; otherwise `"bug"`.
- CLI: `structspec verify --model GGUF [--prompts FILE] [--max-tokens 128] [--k 4] [--cpu-moe]`; built-in set of 8 short non-DSA code prompts (chat template, thinking disabled); prints one line per prompt, no timings.

- [ ] **Step 1:** Failing `test_verify_identical_on_fake` and `test_verify_flags_bug`. The latter wraps FakeModel so batched logits for row ≥ 1 are deliberately wrong with a large margin → `"bug"`.
- [ ] **Step 2–4:** implement, pass.
- [ ] **Step 5:** Run on Qwen3.6 with `--cpu-moe --max-tokens 128` (short; the GPU is shared with speed-lab). Root-cause every mismatch. Write `docs/lossless-check.md`: setup, per-prompt results, root causes. No timing numbers.
- [ ] **Step 6:** Commit `feat: structspec verify; document greedy identity on Qwen3.6`.

### Task 5: Real `serve`

**Files:**
- Create: `structspec/server.py`, `tests/test_server.py`
- Modify: `structspec/cli.py` (`serve`), `structspec/metrics.py`, `structspec/status.py`, `tests/test_cli.py`, `tests/test_metrics.py`

**Interfaces:**
- `create_spec_app(model, proposer, k=4, model_name="structspec", thinking=False) -> FastAPI`. Routes: `POST /v1/completions` (`prompt`, `max_tokens`=256, `stop`, `stream`), `POST /v1/chat/completions` (`messages`, same params), `GET /v1/models`, `GET /health`, `GET /v1/structspec/metrics`. OpenAI response/chunk shapes; `usage` with real token counts; SSE ends with `data: [DONE]`.
- Generation runs under one `threading.Lock`. Text deltas come from `piece_bytes` through `codecs.getincrementaldecoder("utf-8")`. Stop strings are held back until they can no longer match, then the output is truncated before the stop string. `prompt_tokens + max_tokens > model.n_ctx` → 400.
- Metrics: fields `generated_tokens`, `target_passes` replace `target_tokens`; `estimated_multiplier = generated_tokens / target_passes`; `proposed_draft_tokens`/`accepted_draft_tokens`/`rejected_draft_tokens` come from `SpecStats`. `status.py` reads the new names.
- CLI: `serve --model GGUF [--index PATH] [--k 4] [--n-ctx 4096] [--n-gpu-layers -1] [--cpu-moe] [--thinking]`; `--passthrough` keeps today's proxy and its flags. Neither `--model` nor `--passthrough` → exit 2 with a message.

- [ ] **Step 1: Failing tests** (FakeModel, `TestClient`)
  - `test_completion_equals_greedy`, `test_chat_completion_equals_greedy` (FakeModel `apply_chat_template` is a fixed join).
  - `test_stream_concat_equals_nonstream` including a prompt whose output contains the split UTF-8 char; no `\ufffd`.
  - `test_stop_string_truncates` (non-stream and stream agree).
  - `test_prompt_too_long_returns_400`.
  - `test_concurrent_requests_serialized` (2 threads, both equal greedy).
  - `test_metrics_reflect_real_counts`: after one request, `generated_tokens` == usage.completion_tokens and `target_passes` == stats.passes.
  - `test_serve_requires_model_or_passthrough` and `test_parser_serve_passthrough` in `test_cli.py`.
- [ ] **Step 2–4:** implement, pass; update `test_metrics.py` for renamed fields.
- [ ] **Step 5:** Manual: `structspec serve --model <Qwen3.6> --cpu-moe` with one curl stream request of 64 tokens, checked against `verify` output (correctness only).
- [ ] **Step 6:** Commit `feat: in-process OpenAI-compatible serve over the speculative engine`.

### Task 6: Delete dead and overfit code

**Files:** delete `cli.py` (root), `demo_baseline.py`, `demo_speculative.py`, `run_humaneval_benchmark.py`, `structspec/ast_proposer.py`, `tests/test_ast_proposer.py`, `qwen_coder_dsa_tokens.json`, `demo_speedup.mp4` (git rm; add `*.mp4` to `.gitignore`). From `engine.py` remove `EntropyProxyAbort`, `RecoveryModeSelector`, `GreedyCorrectionSampler`, `LiveNgramMiner`, `SpeedTraceCollector`, `QwenTokenCorpus`, `LastTargetTopKFilter`, `RuleStatsRegistry`/`RuleStats`/`EarlyExitDraftController` (dead after Task 2), the trace CSV helpers, `_RE_FIB`, `_RE_N_GUARD`, the code-fence keyword hint, and `PatternMiner.print_report`. From `vllm_integration.py` remove `StructSpecDraftWorker`. Remove the `benchmark` subcommand. In `pyproject.toml`, URLs → `https://github.com/neerajdad123-byte/zero-vram-spec`.

- [ ] **Step 1:** Delete; `ruff check structspec tests` → clean (unused imports gone).
- [ ] **Step 2:** `pytest -q` → all pass; `structspec --help` and `structspec vllm-command --model Qwen/Qwen2.5-7B-Instruct --method ngram` work.
- [ ] **Step 3:** Commit `chore: remove dead, duplicate and benchmark-overfit code`.

### Task 7: CI

- [ ] **Step 1:** Push the branch and open a **draft** PR into `master` titled `v2 core: native backend, lossless engine, real serve`. The body lists the decisions above and links `docs/lossless-check.md`.
- [ ] **Step 2:** CI green on ubuntu + windows × 3.10–3.12; fix any failures.

### Deferred: README

Written after speed-lab's `docs/research/speed-findings.md` exists. Not part of this plan's done state, except removing claims about non-existent components if the user wants that early.
