# Speed findings: Structspec v2 speed-lab (Workstream 2)

Date: 2026-10-24 · Model: Qwen3.6-35B-A3B Q4_K_M · GPU: RTX 4050 Laptop 6 GB · llama.cpp b11756

**Headline (held-out test split, predicted):** the shipped chain `line_start → suffix_copy → corpus n-gram` gets **75% draft acceptance and 2.43 tokens per verify pass**, which predicts **1.74×** overall: **1.47× on fresh code** and **3.24× on edit tasks**. It beats v2-core's current `PromptLookupProposer` (1.11× overall, and *0.91×* on fresh code, a slowdown) in every language. llama.cpp's own n-gram speculation with default settings ran at **~0.4× of greedy** on the one task measured. These numbers are predicted from exact replay plus the measured verify-cost curve. End-to-end wall-clock through Structspec's `serve` path is **not measured yet** (see "Requests for v2-core").

## 1. Setup

- Server: `llama-server -m Qwen3.6-35B-A3B-Q4_K_M.gguf -ngl 99 -ncmoe 34 -c 8192 -np 1 --reasoning off`. The chat template then emits an empty `<think>\n\n</think>` block, and no think token appears in any recorded output (tested).
- Greedy: `/completion` with `temperature 0, top_k 1`, prompt passed as token ids, output ids taken from `return_tokens` (never re-tokenized; tested that detokenized ids equal the content).
- Plain greedy decode: about 31–34 tok/s.

## 2. Verify cost (see `verify-cost.md`)

One decode step costs about 40 ms. Each extra token in the same verify batch costs about +7 ms, linear up to K=16. A draft token pays if P(accepted) ≳ 0.18. On this MoE, verifying is cheap relative to decoding, so the approach *can* pay off. The ceiling is 4.5× at K=15. The best `-ncmoe` is 33–36: any lower and VRAM spills into shared memory, making decode 3–5× slower.

## 3. Data and method

| set | source | tune | test (traced) |
|---|---|---|---|
| HumanEval (py) | repo `HumanEval.jsonl.gz` | 40 traced | 7 |
| MultiPL-E HumanEval (js, ts, go, rs, java, cpp) | `nuprl/MultiPL-E` via HF datasets-server | 40 each for go/rs/ts, 17 for js, 0 for java/cpp | 10 each (rs/ts: 5) |
| Edit set (7 langs) | 40 files from click, ky, express, cobra, ripgrep, gson, fmt at pinned SHAs (`bench/edit_set/files.jsonl`) | 14 | 8 (output capped at 512 tokens, except one at 2048) |
| Mining corpus | 122 files from typer, zod, lodash, urfave/cli, clap, okhttp, nlohmann/json (`bench/mining_manifest.jsonl`) | used for training only | none |

- **Held-out rules (tested):** the split is a hash of the *HumanEval problem*, so all translations of one problem share a split. Edit repos and mining repos are disjoint. Every parameter was chosen on tune with 2-fold cross-validation over problems, so the n-gram models never see the fold they are scored on. The test split was replayed once, with parameters frozen (`bench/results/tune_*.json` → `bench/candidates.py: PARAMS`).
- **Exact replay** (`bench/replay.py`): under greedy verification a draft token is accepted iff it equals the greedy token. Replaying each proposer against the greedy trace therefore gives exact acceptance. Predicted speedup = Σ greedy step cost / (Σ cost(draft+1, ctx) + proposer time), using the measured cost curve.
- Metrics: **coverage** = steps with a draft; **accept rate** = accepted / drafted tokens; **tok/pass** = output tokens / verify passes, the model-independent ceiling.
- Reproduce: `python -m bench.fetch_data` (needs GITHUB_TOKEN for the GitHub API), `python -m bench.traces ...`, `python -m bench.replay --split test --proposers combo_line_first --max-k 16`, `python -m bench.tune copy|ngram|ctx|combo`, `python -m bench.misses combo`.

## 4. Results on the held-out test split (K=16)

| proposer | coverage | accept rate | mean accepted when drafted | tok/pass | predicted |
|---|---|---|---|---|---|
| v2-core `PromptLookupProposer` (baseline) | 35.6% | 25.3% | 3.68 | 2.30 | 1.11× |
| v2-core `NgramIndexProposer`, mined from the same corpus (baseline) | 11.6% | 73.7% | 1.03 | 1.12 | 1.09× |
| **`SuffixCopyProposer`** | 40.2% | 79.6% | 2.85 | 2.14 | **1.66×** |
| `NgramModelProposer` (corpus) | 16.8% | 39.5% | 1.24 | 1.20 | 1.08× |
| `ContextNgramProposer` | 30.0% | 39.9% | 3.83 | 2.14 | 1.32× |
| `LineStartProposer` (+ n-gram extension) | 6.7% | 71.0% | 1.31 | 1.08 | 1.06× |
| `ComboProposer` (copy → corpus n-gram) | 51.6% | 74.1% | 2.71 | 2.39 | 1.72× |
| **line_start → combo (recommended)** | **52.7%** | **74.7%** | 2.72 | **2.43** | **1.74×** |

By task kind (recommended chain): **fresh 1.47×** (coverage 50%, acceptance 65%, 1.90 tok/pass) · **edit 3.24×** (acceptance 94%, 7.17 tok/pass).

### Per language (test split)

| lang | tasks | tokens | v2-core prompt lookup: acc / tok-pass / pred | suffix_copy: acc / tok-pass / pred | recommended chain: acc / tok-pass / pred | chain on fresh | chain on edit |
|---|---|---|---|---|---|---|---|
| py | 9 | 3356 | 43% / 4.37 / 1.74× | 89% / 4.01 / 2.44× | 88% / 4.17 / **2.47×** | 1.95× | 2.89× |
| rs | 6 | 1376 | 34% / 2.87 / 1.34× | 86% / 2.75 / 1.94× | 83% / 3.09 / **2.03×** | 1.64× | 3.48× |
| go | 11 | 3427 | 24% / 2.26 / 1.08× | 78% / 2.09 / 1.63× | 74% / 2.38 / **1.72×** | 1.60× | 3.18× |
| ts | 6 | 1376 | 22% / 2.03 / 1.02× | 78% / 1.91 / 1.55× | 76% / 2.34 / **1.71×** | 1.30× | 3.73× |
| cpp | 11 | 3146 | 24% / 2.23 / 1.08× | 79% / 2.07 / 1.62× | 70% / 2.41 / **1.70×** | 1.54× | 4.12× |
| java | 11 | 2949 | 18% / 1.85 / 0.92× | 74% / 1.74 / 1.46× | 65% / 1.91 / **1.49×** | 1.32× | 3.99× |
| js | 11 | 2344 | 17% / 1.72 / 0.91× | 66% / 1.57 / 1.33× | 65% / 1.82 / **1.44×** | 1.27× | 3.09× |

Tune-split results (cross-fold) agree: recommended chain 1.69× and 73.5% acceptance on tune vs 1.74× and 74.7% on test. Nothing was overfit to the test split.

## 5. What worked, what didn't (losers are documented, not shipped as defaults)

1. **Win: copy with a calibrated draft length.** `SuffixCopyProposer(min_match=2, len_slope=0.5, len_base=1)` finds the *longest* earlier match of the current suffix and drafts `1 + 0.5 × (match_len − 2)` tokens. Short matches get short drafts, long matches (edits) get long ones. That one change takes acceptance from 25% (v2-core's "copy max_k after a 2–3-gram") to 80%. v2-core's version is a net *slowdown* on fresh JS/Java/TS code because it drafts 16 tokens after any 2-gram hit.
2. **Small win: corpus n-gram as fallback** (+0.06×). On its own it's weak (1.08×). Raw counts are badly overconfident: contexts claiming 40–60% are right 22–35% of the time (`bench/misses.py`, calibration in session notes). Only high-confidence long contexts are reliable, so it needs `p_min=0.75`.
3. **Small win: line-start structure** (+0.02×). Previous line's indent, last char and first word predict the first token of the next line with ~90% precision on ~50% of line starts, for all languages, without tree-sitter. Most of those positions are already covered by copy, so the marginal gain is small.
4. **Loser: context n-gram (cache LM).** 1.32× alone, adds nothing on top of copy (copy fires wherever it would). Kept in the package for experiments but not in the recommended chain.
5. **Loser: tree-sitter indent/dedent.** Not built. The line-start feature model captures the same signal (90% precision) language-agnostically, and its *marginal* value on top of copy is only +0.02×, so a grammar-driven version can't add much.
6. **Hard limit — what's left is the model's word choice.** After the chain, the largest uncovered class is "a word already in context, but which one" (11.7% of fresh tokens), then new words. No rule or n-gram predicts those. That is the job of a model-based drafter (MTP head / small draft model).

## 6. Honest bar: llama.cpp b11756's own speculation

| config | task | tok/s | greedy tok/s | llama.cpp draft acceptance | output identical to greedy |
|---|---|---|---|---|---|
| `--spec-type ngram-mod` (defaults) | HumanEval/14 (py) | 14.3 | 33.6 | 37 / 55 = 67% | yes |
| `--spec-type ngram-simple` (defaults) | HumanEval/14 (py) | 13.2 | 33.6 | 48 / 66 = 73% | yes |
| `--spec-type draft-mtp` | — | — | — | **cannot run: this GGUF has no MTP layers** ("model doesn't contain MTP layers") | — |

- With default settings, llama.cpp's n-gram modes were **2.4× slower than greedy** on this MoE, even at 67–73% acceptance. The likely cause is long default drafts: `ngram-mod` drafts 48–64 tokens, and by the cost curve each such verify pass costs as much as about 6–8 decode steps. This is a single run of a single task (time budget), so it shows the direction, not a precise number. The batch runner stopped after the first task, a client-side issue not yet debugged.
- MTP needs a GGUF converted with the MTP layers kept. This one doesn't have them.

## 7. Recommendations

- **Default proposer for v2-core:** `ChainProposer([LineStartProposer(pieces, corpus, extend=NgramModelProposer(corpus, p_min=0.75, cum_min=0.4)), ComboProposer(SuffixCopyProposer(min_match=2, len_slope=0.5, len_base=1), [NgramModelProposer(...)], copy_min=2)])`, with **max K = 16**. If you ship only one piece, ship `SuffixCopyProposer`: it gets 1.66× of the 1.74×.
- **Languages to target first:** Python (2.5×), Rust (2.0×), Go/TS/C++ (≈1.7×). Java and JS gain least (≈1.45×). For fresh JS/TS/Java the chain is ≈1.3×, so don't advertise more.
- **Task type to target:** edits and refactors (≥3× predicted in every language), i.e. any request that repeats the input.
- **README claims** should say "predicted from exact replay", until the end-to-end check below is done.

## 8. Requests for v2-core (files I don't own)

1. **Replace `PromptLookupProposer`** in `structspec/proposer.py` with `structspec.proposers.SuffixCopyProposer` (or the full chain in §7). The current one is a net slowdown on fresh code in 3 of 7 languages.
2. **Vocab pieces hook:** `LineStartProposer` needs `pieces: list[str]` (id → decoded text). The native backend should expose `[piece(i) for i in range(n_vocab)]`.
3. **Corpus for the n-gram models:** `structspec index` should save token sequences (or the pruned `NgramModelProposer.table`) instead of only the `PatternMiner` rules. Training data for these proposers = mined files tokenized by the model; model outputs from a tune run help.
4. **End-to-end check:** run `serve` with the chain on the test traces (`bench/results/traces/test.jsonl` has prompt ids and expected greedy ids) and compare measured tok/s against `bench/results/replay_test.json`. This is the missing "measured" column. `bench/server_bench.py` can be pointed at it with small changes.
5. Pass `max_k=16` (the cost curve is flat-linear to 16) and don't verify more than 16 tokens per pass.

## 9. Limitations (read before quoting numbers)

- The test split is small because of the GPU time budget: 65 tasks, 17,974 tokens; Rust and TS have only 5 fresh tasks each, and there is 1 edit task per language (2 for Python). Per-language numbers have wide error bars.
- Edit test outputs were capped at 512 tokens (one at 2048). Longer rewrites would copy even more.
- Speedups are **predicted**: exact acceptance, combined with a cost curve with ±15% run-to-run noise. Prompt processing time is not included: these are decode-phase speedups.
- The llama.cpp bar is a single run of a single task per mode, not the median of 3.
- Proposer CPU time is included, measured in Python (μs per token). A C++ path would only be faster.
