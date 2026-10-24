# Structspec v2 — design

Date: 2026-10-24 · Status: draft, awaiting approval

## Goal

Turn Structspec from a DSA-tuned demo into a tool that gives a **real, measured,
lossless wall-clock speedup for everyday code generation** on a consumer GPU, with
claims that survive someone else re-running them.

Success means:
1. `structspec serve --model x.gguf` is an OpenAI-compatible server that actually
   runs speculative decoding (today `serve` is a pure passthrough).
2. Output is byte-identical to plain greedy decoding, enforced by tests.
3. Every number in the README comes from a held-out benchmark whose tasks never
   appear in the mining corpus, compared against plain greedy **and** llama.cpp's
   own speculation options.
4. Works with any GGUF model's tokenizer, not only Qwen2.5 token IDs.

## Hardware and models (as found on this machine)

- GPU: RTX 4050 Laptop, 6 GB. CUDA runtime comes from the official llama.cpp build.
- llama.cpp: `C:\Users\neera\tools\llama.cpp-b11756\bin\` (official CUDA 13 build,
  ships `llama.dll`, `cudart64_13.dll`, `llama-server.exe`, `llama-batched-bench.exe`).
- Primary model: `C:\models\gguf\Qwen3.6-35B-A3B-Q4_K_M.gguf` (20 GB MoE, has an MTP head).
  It does not fit in 6 GB, so experts run partly on CPU.
- llama-cpp-python 0.3.4 cannot load Qwen3.6 (bundled llama.cpp predates it).
- Optional (needs your OK to download): one small dense coder model with a
  *different* tokenizer family, for fast iteration and to prove tokenizer independence.

Consequence: verifying K drafted tokens on a CPU-offloaded MoE may cost much more than
verifying 1 (more experts touched). The speedup ceiling must be measured, not assumed.

## Shared contract (both workstreams build against this)

```python
class Proposer(Protocol):
    name: str
    def propose(self, tokens: list[int], max_k: int) -> list[int]: ...
    def observe(self, accepted: list[int]) -> None: ...  # verified tokens only
```

Key fact both streams rely on: under greedy verification a draft token is accepted
**iff** it equals the greedy output at that position. So per-proposer acceptance can be
computed exactly offline from greedy output traces; only the verify cost needs the GPU.

Predicted speedup = greedy_time / (Σ passes × verify_cost(batch_size) + proposer_time).

## Workstream 1 — `v2/core` (fixes + product)

Owns: `structspec/` (except `structspec/proposers/`), `tests/`, `pyproject.toml`,
`README.md`, root files.

1. **Native backend.** `structspec/llama_native.py`: minimal ctypes binding to the official
   `llama.dll` (load, tokenize, detokenize, batch decode, logits, KV seq remove). DLL dir
   from `STRUCTSPEC_LLAMA_DIR`, else auto-detect. Same interface as `FastGreedyLlama`
   (`encode/decode/piece/reset/truncate_kv/decode_logits/eos`) so the loop is unchanged.
   Drop llama-cpp-python.
2. **Lossless by test.** Unit tests for `run_speculative` with a fake deterministic model
   (CI, no GPU): spec output == greedy output for random drafts, all reject paths.
   Then on Qwen3.6: run greedy vs spec on the bench prompts; any mismatch is root-caused
   (bug vs batch-size numerics) and documented.
3. **Tokenizer-agnostic drafting.** `structspec index <paths>` mines n-gram rules from
   source files with the loaded model's tokenizer → index file keyed by tokenizer hash.
   Add a prompt-lookup proposer (match n-gram in prompt + generated text, copy the
   continuation). Delete `qwen_coder_dsa_tokens.json` dependency.
4. **Real `serve`.** In-process `/v1/chat/completions` + `/v1/completions` with SSE
   streaming over the speculative engine; metrics wired to real counts. Old behavior
   kept as `--passthrough`.
5. **Delete dead/overfit code:** root `cli.py` (duplicate), `demo_*.py` (hardcoded paths),
   unused `EntropyProxyAbort`, `RecoveryModeSelector`, `GreedyCorrectionSampler`,
   `LiveNgramMiner`, `StructSpecDraftWorker`, `ast_proposer.py` (imported by nothing),
   benchmark-specific regex rules (`_RE_FIB`, `_RE_N_GUARD`, code-fence keyword hint),
   `llama_backend.py:228` no-op timer. Fix `pyproject.toml` URLs. Move the mp4 out of git.
6. **README** — written last, from workstream 2's numbers.

## Workstream 2 — `v2/speed-lab` (find real speedups)

Owns: `bench/`, `structspec/proposers/` (new files only), `docs/research/`.
Does not edit workstream 1 files; needed hooks are written up as requests.

1. **Verify-cost curve first (spike).** `llama-batched-bench` on Qwen3.6 with the real
   offload settings: time for batch 1..16 at several context lengths. This sets max
   useful K and the whole economics.
2. **Held-out eval sets.** HumanEval (Python, in repo), MultiPL-E translations
   (JS, TS, Go, Rust, Java, C++), and an *edit* set (real files from permissive repos +
   "refactor / add types / fix bug" instructions — the prompt-lookup regime).
   Mining corpus = different repos only. Tuning split and test split kept separate.
3. **Greedy traces** via `llama-server` (temperature 0) → token sequences via its
   tokenize endpoint. Offline replay of every proposer gives exact acceptance.
4. **Hunt for real structure**, each candidate validated on the test split:
   prompt/context copy, in-scope identifier completion, bracket/quote closing,
   indent/dedent from grammar state (tree-sitter incremental), language boilerplate
   (`if err != nil {`, `}` + newline, JSX closing tags), repeated JSON keys.
5. **Honest bar:** measure llama.cpp b11756's own speculation (n-gram/lookup, and MTP
   if the build supports it for Qwen3.6). Structspec must report where it wins and loses.
6. **Deliverable:** `docs/research/speed-findings.md` (numbers per language/task type,
   predicted vs measured), plus proposers under `structspec/proposers/` that implement
   the shared contract. Losers are documented, not shipped.

## Integration

Workstream 2 proposers merge into `v2/core` after both land; end-to-end measurement on
the real `serve` path confirms predictions; README written from those numbers.

## Out of scope

Sampling (temperature > 0) verification, vLLM plugin, TUI dashboard, multi-GPU.

## Risks

- MoE verify cost may make K > 2 unprofitable → speedup ceiling could be low; we report it.
- ctypes binding is pinned to one llama.cpp ABI; version check at load, clear error.
- Qwen3.6 is a reasoning model: benchmarks must disable thinking in the chat template.
