# Verify-cost curve: Qwen3.6-35B-A3B Q4_K_M on an RTX 4050 Laptop (6 GB)

Date: 2026-10-24. Raw data: `bench/results/verify_cost.json`. Reproduce with `python -m bench.verify_cost`.

## Setup

- llama.cpp b11756 official CUDA build, `llama-bench -ngl 99 -ncmoe 34 -p K -n 0 -d CTX -r 3`. Each cell is the median of 3 runs.
- `-p K -d CTX` times one forward pass of K tokens on one sequence with CTX tokens already in the KV cache. That is exactly one speculative verify pass (K−1 draft tokens plus the last accepted token).
- `llama-batched-bench -pps` was the planned tool, but it fails on this model: llama.cpp's M-RoPE position check (`X < Y`) rejects the shared-prompt copy.
- Offload sweep (tg16 tokens/s at depth 4096) for `-ncmoe`, the number of expert layers kept on the CPU: 20 → 6.5, 24 → 8.2, 30 → 12.0, 33 → 32.5, 36 → 31.5, 40 → 29.4. Below about 33 CPU layers the weights overflow VRAM into Windows shared memory and decode slows down 3–5×. 34 is the sweet spot and leaves headroom for the server's KV cache.
- Plain decode runs at about 31 tok/s (about 32 ms/token in `tg`). `pp1` passes measure 35–45 ms. Run-to-run noise on this laptop is ±15%, so medians matter.

## Curve

| batch (K drafts+1) | ms @ctx 512 | ms @ctx 2048 | ms @ctx 4096 | cost(b)/cost(1) @2048 | marginal ms per extra token @2048 |
|---|---|---|---|---|---|
| 1 | 45.6 | 40.7 | 40.4 | 1.00 | 0.0 |
| 2 | 51.0 | 45.9 | 54.3 | 1.13 | 5.2 |
| 3 | 69.3 | 55.0 | 52.2 | 1.35 | 7.2 |
| 4 | 81.4 | 62.9 | 66.1 | 1.54 | 7.4 |
| 5 | 83.4 | 69.0 | 68.9 | 1.69 | 7.1 |
| 6 | 96.6 | 75.2 | 72.8 | 1.85 | 6.9 |
| 8 | 114.0 | 90.5 | 96.1 | 2.22 | 7.1 |
| 10 | 139.1 | 111.7 | 107.0 | 2.74 | 7.9 |
| 12 | 170.0 | 122.0 | 117.4 | 3.00 | 7.4 |
| 16 | 194.7 | 145.4 | 154.9 | 3.57 | 7.0 |

The ctx-512 column was re-measured warm and is still consistently slower than 2048. It is reproducible across two runs, and the cause is not investigated. The predictor uses the measured value for each context, so this matters only for interpretation.

## What it means

- **Verifying is cheap on this MoE.** Each extra token in a verify pass costs about 7 ms, against about 40 ms for a full decode step: roughly **0.18 of a token**. The fear that touching more experts makes K > 2 unprofitable is wrong here. The cost is linear, with no cliff up to K = 16.
- **Break-even rule:** the j-th draft token is worth proposing if P(tokens 1..j all accepted) ≳ 0.18. A proposer can afford long drafts when it is confident and should stop where its confidence drops below about 20%.
- **Ceilings:** with perfect drafts the gain is K+1 tokens per cost(K+1): 1.77× at K=1, 2.6× at K=3, 3.6× at K=7, 4.5× at K=15.
- **Max useful K ≈ 8–12.** Past that, a single pass costs more than 3 decode steps, so one early reject wastes a lot. Drafts that copy long runs (edit tasks) are where K = 8–16 pays.
- **Wrong drafts are cheap.** A one-token draft that gets rejected costs about +13% for that step, so a proposer with a 50% hit rate on one-token drafts still wins (expected 1.5 tokens for 1.13 cost).
