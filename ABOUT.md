# About Structspec

Structspec makes local code models generate faster without a second "draft" model and without extra VRAM.

Standard speculative decoding pairs your model with a small draft model that guesses ahead. On a consumer GPU there is often no room for that second model. Structspec guesses ahead by copying from what is already in front of the model: the prompt, the file being edited, and the code written so far. Your model checks every guess in one batched pass and keeps only the tokens it would have produced anyway.

## Who it's for

- People running code models locally with llama.cpp GGUF files on a 6–12 GB GPU.
- Editor and agent workflows that send files and ask for edits, refactors or rewrites. That is where the gain is largest.

## What to expect

Measured on an RTX 4050 Laptop (6 GB), Qwen2.5-Coder-3B-Instruct Q4_K_M, 64 held-out tasks in 7 languages:

| task | speedup |
|---|---|
| Editing existing code | 5.7× |
| Writing new code | 1.4× |
| Mixed | 3.3× |

On a large MoE model with experts partly on the CPU (Qwen3.6-35B-A3B), the predicted gains are smaller: 3.2× on edits and 1.5× on new code.

## Principles

- **Same output.** Every drafted token is verified against the model's own greedy choice. The output matches plain greedy decoding, apart from rare floating-point near-ties that `structspec verify` detects and explains.
- **Honest numbers.** Benchmarks run on tasks never used for tuning, report losses as well as wins, and can be reproduced with the scripts in `bench/`.
- **Any model.** Drafting works on token ids, so any GGUF tokenizer works.

## Links

- [README](README.md): install and quick start
- [Speed findings](docs/research/speed-findings.md): full method and results
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [Code of conduct](CODE_OF_CONDUCT.md) · [Security policy](SECURITY.md) · [Accessibility](ACCESSIBILITY.md)
