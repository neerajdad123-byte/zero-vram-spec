# Changelog

## 0.2.0 (2026-10-24)

### Added
- `structspec serve`: OpenAI-compatible server (`/v1/chat/completions`, `/v1/completions`, streaming) that runs speculative decoding in-process on a GGUF model. The old proxy behaviour is kept as `--passthrough`.
- Native llama.cpp backend (`structspec/llama_native.py`) for any GGUF model and tokenizer, including MoE models with expert weights on the CPU (`--cpu-moe`) and hybrid/recurrent models.
- `structspec verify`: checks that speculative output is identical to greedy and classifies any difference as a bug or a floating-point near-tie.
- `structspec index`: mines an n-gram draft index from source files with the loaded model's tokenizer.
- Draft proposers in `structspec/proposers/`: suffix copy, learned n-gram model, context n-gram, line-start structure, combo chain.
- Held-out benchmark suite in `bench/`: verify-cost curve, HumanEval + MultiPL-E (6 languages) + edit set, greedy traces, exact offline replay, end-to-end wall-clock runner.

### Changed
- Default drafter is now longest-suffix copy with a match-scaled draft length. On held-out tasks it reaches 80% draft acceptance vs 25% for the previous prompt lookup.
- Default max draft length (`--k`) raised from 4 to 15.
- README rewritten from measured, held-out numbers only.

### Removed
- Benchmark-specific rules and unused components from 0.1.0, the duplicate root CLI and the demo scripts with hardcoded paths.

## 0.1.0
- First release: proxy, backend detection, rule-based drafting.
