## What this changes

<!-- One or two sentences. -->

## Why

<!-- The problem it solves, or the speedup it adds. -->

## Checklist

- [ ] `pytest tests bench/tests` passes
- [ ] `ruff check structspec bench` passes
- [ ] Speed claims come from `bench/` on the **test** split (parameters tuned on tune only), with the command used
- [ ] Output stays identical to greedy (`structspec verify`), or differences are explained
- [ ] README / CHANGELOG updated if behaviour changed
