# Contributing

Thanks for helping make local code models faster.

## Setup

```bash
git clone https://github.com/neerajdad123-byte/zero-vram-spec
cd zero-vram-spec
pip install -e ".[dev]"
pytest tests bench/tests
ruff check structspec bench
```

Tests run without a GPU. To run a real model, download the official llama.cpp release for your platform and set `STRUCTSPEC_LLAMA_DIR` to the folder containing the llama.cpp shared library (`llama.dll` on Windows).

## Adding a draft proposer

1. Implement the `Proposer` interface from `structspec/proposer.py` in a new file under `structspec/proposers/`:
   - `name: str`
   - `propose(tokens: list[int], max_k: int) -> list[int]` returns at most `max_k` tokens, and `[]` for empty input
   - `observe(accepted: list[int]) -> None` receives verified tokens only
2. Add tests to `bench/tests/test_proposers.py`. The generic tests (respects `max_k`, empty input, deterministic) run over every proposer listed there.
3. Measure it on the **tune** split (`python -m bench.replay --split tune ...`), freeze its parameters, then report the **test** split. Never tune on test.
4. Report acceptance, tokens per pass and the speedup, including the languages or task types where it loses.

## Benchmark claims

Numbers in the README must come from `bench/` on held-out tasks, with the command that produced them. Wall-clock claims come from `python -m bench.e2e`.

## Pull requests

Keep changes focused, include tests, and make sure `pytest tests bench/tests` and `ruff check` pass.
