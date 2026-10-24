# Structspec v2 Speed-Lab (Workstream 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find real, held-out, wall-clock speculative-decoding speedups for everyday multi-language code generation on Qwen3.6-35B-A3B (6 GB GPU), report wins and losses, and ship only the proposers that win.

**Architecture:** Measure verify cost once (`llama-batched-bench`), record greedy token traces once (`llama-server`, temp 0, thinking off), then replay every proposer offline against the traces: under greedy verification a draft token is accepted iff it equals the greedy token, so acceptance is exact and needs no GPU. Predicted speedup combines replay with the cost curve; the top candidates and llama.cpp's own speculation (`--spec-type ngram-*`, `draft-mtp`) are then measured end-to-end on `llama-server`.

**Tech Stack:** Python 3.11 (uv venv), stdlib + `httpx` (already a dep) for the server client, `pytest`; llama.cpp b11756 executables; optional `tree-sitter` + `tree-sitter-language-pack` (bench-only, asked before install).

**Spec:** `docs/superpowers/specs/2026-10-24-structspec-v2-design.md` (Workstream 2 + Shared contract)

## Primary metric (user direction, 2026-10-24)

On this CPU-offloaded MoE the decode gain may be small, so **draft acceptance is the headline metric**, reported per proposer × language × task kind:
- **coverage**: share of decode steps where the proposer drafts anything;
- **acceptance rate**: accepted draft tokens / drafted tokens;
- **tokens per pass**: (accepted + 1) / passes, which is the model-independent ceiling on speedup;
- **mean accepted length** when a draft is offered.

Predicted and measured wall-clock speedups come second. They show whether this model's verify cost lets the acceptance pay off, but a proposer with high acceptance still counts as a win for dense models even if it does not pay off on this MoE.

## Global Constraints

- Own only `bench/`, `structspec/proposers/` (new files only), `docs/research/`. Never edit `structspec/*.py`, `tests/`, `pyproject.toml`, README. Needed hooks go into the findings doc as requests.
- Proposer contract, verbatim: `name: str`; `propose(self, tokens: list[int], max_k: int) -> list[int]`; `observe(self, accepted: list[int]) -> None` (verified tokens only).
- Predicted speedup = greedy_time / (Σ passes × verify_cost(batch_size) + proposer_time).
- Model: `C:\models\gguf\Qwen3.6-35B-A3B-Q4_K_M.gguf`; binaries: `C:\Users\neera\tools\llama.cpp-b11756\bin\`. Thinking disabled (`--reasoning off`, plus `--chat-template-kwargs '{"enable_thinking":false}'` if needed).
- Timing: no other GPU process running; each timing = median of 3 runs. Same offload settings (`--n-cpu-moe N`, `-ngl 99`, `-c`) for every measured config, recorded in the doc.
- Held-out: no eval task (HumanEval, MultiPL-E, edit set) or its source repo may appear in any mining corpus. Tuning and test splits are fixed by a seeded hash split and never swapped. Every headline number comes from the test split.
- Every download is asked first with name, source, size.
- Proposer tests live in `bench/tests/` (`tests/` belongs to v2-core) and run with `pytest bench/tests`; no GPU needed.

## Review Focus

1. Tokenization boundary: a trace tokenized as one string differs from prompt tokens + generated tokens tokenized separately → replay must use server-returned token ids for the completion (`n_probs`/`tokens` in the response), not re-tokenized text. Test: `test_trace_tokens_match_completion` in Task 3.
2. Proposer returns more than `max_k`, or returns tokens when `max_k == 0` → engine overruns verify batch. Test: per-proposer `test_respects_max_k` in Task 5.
3. Empty context / first token (`tokens == []`) → proposers must return `[]`, not raise. Test: `test_empty_tokens` in Task 5.
4. Thinking leaks into traces (`<think>` tokens) → acceptance inflated by template boilerplate. Test: `test_no_think_tokens` in Task 3 asserts the think-open token id is absent from every recorded completion.
5. Verify cost measured at a context length unlike the eval prompts → wrong prediction. Task 1 measures npp 512/2048/4096 and Task 4 interpolates by the trace's actual context length.

---

### Task 1: Verify-cost spike (report to user immediately)

**Files:**
- Create: `bench/verify_cost.py`, `bench/results/verify_cost.json`, `docs/research/verify-cost.md`

**Interfaces:**
- Produces: `bench/results/verify_cost.json` = `{"settings": {...}, "rows": [{"npp": int, "npl": int, "t_tg_ms_per_step": float}]}`; `verify_cost.py: load_cost() -> Callable[[int, int], float]` returning ms per decode pass for (batch, ctx), linear interpolation in ctx.

- [ ] Step 1: Find the largest `--n-cpu-moe` offload that leaves `-ngl 99` + `-c 8192` running in 6 GB (try N = 40 downward; record VRAM via `nvidia-smi`).
- [ ] Step 2: Run `llama-batched-bench -m <model> -ngl 99 --n-cpu-moe N -c 16384 -npp 512,2048,4096 -ntg 32 -npl 1,2,3,4,6,8,12,16 -pps` three times; parse the table; store the median S_TG-derived ms per step. Note: with `-pps` the npl sequences share a prompt and each step decodes npl tokens in one batch — that is exactly a verify pass of K=npl tokens.
- [ ] Step 3: Write `docs/research/verify-cost.md`: table, ratio cost(K)/cost(1), break-even acceptance per K (K tokens verified must yield ≥ cost(K)/cost(1) tokens per pass). Commit, then message the user the result and the max useful K before continuing.

### Task 2: Held-out eval sets + splits

**Files:**
- Create: `bench/datasets.py`, `bench/data/` (gitignored large raw files; small manifests committed), `bench/edit_set/manifest.jsonl`, `bench/tests/test_datasets.py`

**Interfaces:**
- Produces: `load_tasks(split: str) -> list[Task]` where `Task = dict(id: str, lang: str, kind: "fresh"|"edit", prompt: str, split: "tune"|"test")`. Split = `"tune" if sha1(id) % 3 == 0 else "test"`.

- [ ] Step 1: Ask the user to approve downloads: MultiPL-E HumanEval prompts for js, ts, go, rs, java, cpp (HF `nuprl/MultiPL-E`, ~1 MB each); edit-set source files (~40 files from 4–6 permissive repos, e.g. MIT/Apache projects in Python/TS/Go/Rust, <2 MB total); mining corpus from *different* repos (~5 MB).
- [ ] Step 2: Test `test_split_is_stable_and_disjoint`: same id always same split; tune ∩ test = ∅; every lang has ≥10 test tasks. `test_edit_repos_not_in_mining`: repo names in `bench/edit_set/manifest.jsonl` ∩ `bench/data/mining_repos.txt` = ∅.
- [ ] Step 3: Implement `load_tasks`; HumanEval/MultiPL-E prompt = "Complete this function. Reply with only the code." + the stub; edit prompt = instruction + full file, task asks for the full rewritten file (prompt-copy regime). ~25 edit tasks: refactor, add types, fix bug, add docstrings.
- [ ] Step 4: `pytest bench/tests/test_datasets.py` passes. Commit.

### Task 3: Greedy traces

**Files:**
- Create: `bench/traces.py`, `bench/results/traces/<split>.jsonl`, `bench/tests/test_traces.py`

**Interfaces:**
- Consumes: `load_tasks`.
- Produces: trace record `{"id", "lang", "kind", "prompt_ids": list[int], "out_ids": list[int], "wall_s": float, "n_out": int}`; `load_traces(split) -> list[dict]`.

- [ ] Step 1: Start `llama-server` with Task 1's offload settings, `--reasoning off`, `--temp 0`, `-np 1`. Prompt ids from `/apply-template` + `/tokenize`; generation via `/completion` on those ids with `"return_tokens": true`, `max_tokens 512`, `cache_prompt false`.
- [ ] Step 2: Tests (offline, on a recorded fixture of 2 traces): `test_trace_tokens_match_completion` (detokenize(out_ids) == content), `test_no_think_tokens`.
- [ ] Step 3: Record all tune+test tasks (one pass; this is also the greedy-time baseline per task: `wall_s` from `timings.predicted_ms`). Commit results (jsonl is small).

### Task 4: Replay harness + predicted speedup

**Files:**
- Create: `bench/replay.py`, `bench/tests/test_replay.py`

**Interfaces:**
- Consumes: Proposer contract, `load_traces`, `load_cost`.
- Produces: `replay(proposer, trace, max_k) -> {"passes": int, "accepted": int, "drafted": int, "steps_with_draft": int, "proposer_s": float, "n_out": int}` (gives coverage, acceptance rate, tokens/pass); `predict(result, ctx, cost) -> float` speedup; CLI `python -m bench.replay --split test --proposers a,b --max-k K` printing per-lang × kind table.

- [ ] Step 1: Tests: `test_replay_oracle` (proposer returning the true future tokens: passes == ceil(n_out/(K+1))), `test_replay_null` (proposer returning []: passes == n_out), `test_observe_gets_only_verified` (observe receives exactly accepted + the bonus token).
- [ ] Step 2: Implement: loop over position i; draft = propose(prompt+out[:i], K); accept longest prefix equal to out[i:]; pass consumes accepted+1 tokens; batch size = len(draft)+1; cost(batch, ctx=len(prompt)+i).
- [ ] Step 3: Tests pass. Commit.

### Task 5: Candidate proposers (hunt)

**Files:**
- Create: `structspec/proposers/__init__.py`, one file per candidate: `prompt_copy.py` (longest-suffix match in prompt+output, copy continuation), `ident_complete.py` (in-scope identifier completion from seen tokens), `bracket_close.py`, `lang_boilerplate.py` (per-language token-pattern table mined from the *tune* split + mining corpus, not test), `indent_tree_sitter.py` (only if tree-sitter install approved), `json_keys.py`, `combo.py` (ordered fallback over winners).
- Test: `bench/tests/test_proposers.py`

**Interfaces:**
- Consumes: tokenizer only via ids; proposers needing text get a `detok: Callable[[list[int]], str]` and `tok: Callable[[str], list[int]]` in `__init__` (served by llama-server `/tokenize`/`/detokenize` with an LRU cache for bench; v2-core can pass its native binding).

- [ ] Step 1: Generic tests parametrized over all proposers: `test_respects_max_k` (len ≤ max_k, `[]` when max_k == 0), `test_empty_tokens`, `test_deterministic`. Plus one behavior test each (e.g. prompt_copy proposes the continuation of a repeated 8-gram).
- [ ] Step 2: Implement each; run `python -m bench.replay --split tune` to tune thresholds (min match length, K) on tune only.
- [ ] Step 3: Freeze params; run once on test; save `bench/results/replay_test.json`. Commit.

### Task 6: Honest bar + end-to-end measurement

**Files:**
- Create: `bench/server_bench.py`, `bench/results/server_bench.json`

- [ ] Step 1: Record llama.cpp's own draft acceptance (`/completion` `timings.draft_n` / `draft_n_accepted`, or the server log's acceptance line) next to tokens/s. For each config — baseline; `--spec-type ngram-simple`; `ngram-map-k`; `ngram-mod`; `ngram-cache`; `draft-mtp` (if the GGUF's MTP head loads); — run the test split (or a fixed stratified 60-task subset if the full run exceeds ~2 h) 3×, median tokens/s per lang × kind. Check output equality vs baseline and report mismatches.
- [ ] Step 2: Structspec winners have no serve path in this branch → predicted only; record the request in the findings doc for v2-core to confirm end-to-end after merge. Where llama.cpp's own spec has a comparable mechanism (ngram-simple ≈ prompt_copy), compare predicted-vs-measured for it to calibrate the predictor.
- [ ] Step 3: Commit.

### Task 7: Findings doc

**Files:**
- Create: `docs/research/speed-findings.md`

- [ ] Step 1: Sections: setup (exact flags, VRAM), verify-cost curve, per lang × kind tables (acceptance, tokens/pass, predicted speedup) for each proposer, llama.cpp spec measured, predictor calibration, losers with why, recommendations (which languages / task kinds to target, max K), requests for v2-core (hooks needed, `Proposer` wiring, serve-path end-to-end check).
- [ ] Step 2: `pytest bench/tests` green; commit.
