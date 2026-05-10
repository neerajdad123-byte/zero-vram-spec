# StructSpec Indentation + Multi-Token Benchmark Report

## Config
- Model: Qwen2.5-7B-Instruct Q4_K_M via llama.cpp
- Prompts: 20 DSA/code prompts
- Tokens per prompt: 100
- Draft length: k=12 multi-token chain drafting
- Mode: loose mined rules + syntax clusters + live mining + indentation count rules + truncate reject mode

## Aggregate Speed
| Metric | Value |
|---|---:|
| Greedy wall time | 50.198s |
| StructSpec wall time | 30.551s |
| Wall speedup | **1.643x** |
| Greedy passes | 2000 |
| StructSpec passes | 1153 |
| Pass speedup | **1.735x** |
| Pattern fire rate | 410/1133 = 36.19% |
| Draft token acceptance | 854/999 = 85.49% |
| Reject events | 82 |
| No-rule passes | 723 |
| Greedy-identical outputs | 12/20 |

## Time Breakdown
Timed wall excludes prompt prefill; prompt prefill is shown separately because the CLI reports it as a phase.

| Phase | Seconds | Percent of timed generation wall |
|---|---:|---:|
| Prompt prefill | 0.6758 | outside timed wall |
| Pattern prediction | 0.1017 | 0.33% |
| Target model decode | 30.2621 | 99.05% |
| Python verification loop | 0.1389 | 0.45% |
| Detokenize | 0.0023 | 0.01% |

## Top Tiers By Proposed Draft Tokens
| Tier | Rows | Proposed | Accepted | Reject Events | Accept % |
|---|---:|---:|---:|---:|---:|
| det_ctx8 | 55 | 387 | 359 | 5 | 92.76% |
| strong_ctx8 | 117 | 207 | 182 | 14 | 87.92% |
| det_ctx5 | 18 | 82 | 57 | 9 | 69.51% |
| syntax_indent | 39 | 47 | 45 | 2 | 95.74% |
| strong_ctx4 | 41 | 45 | 25 | 19 | 55.56% |
| det_ctx4 | 21 | 41 | 34 | 6 | 82.93% |
| det_ctx7 | 4 | 32 | 26 | 3 | 81.25% |
| strong_ctx6 | 26 | 27 | 18 | 9 | 66.67% |
| strong_ctx5 | 21 | 25 | 18 | 7 | 72% |
| syntax_class_next_method | 14 | 17 | 14 | 0 | 82.35% |
| strong_ctx7 | 14 | 16 | 14 | 2 | 87.5% |
| syntax_return_terminal | 4 | 15 | 15 | 0 | 100% |
| syntax_range_paren | 1 | 12 | 5 | 0 | 41.67% |
| syntax_dunder_init | 3 | 9 | 9 | 0 | 100% |
| syntax_minus_one | 7 | 7 | 7 | 0 | 100% |
| syntax_none_guard_colon | 3 | 6 | 6 | 0 | 100% |
| syntax_code_fence_def | 5 | 5 | 5 | 0 | 100% |
| syntax_block_colon | 2 | 4 | 3 | 0 | 75% |


## Single-Token vs Multi-Token Drafting
| Draft K | Pass Speedup | Wall Speedup | Pattern Fire Rate | Accepted/Proposed |
|---:|---:|---:|---:|---:|
| 1 | 1.394x | 1.357x | 45.9% | 572/649 = 88.1% |
| 12 | **1.735x** | **1.643x** | 36.2% | 854/999 = 85.5% |

Multi-token drafting is clearly better here. K=1 fires more often, but each fire can only save one token. K=12 fires slightly less often and accepts slightly fewer percent-wise, but accepts far more total draft tokens, which is what reduces target passes.
## Indentation-Specific Rows
| Tier | Rows | Proposed | Accepted | Accept % |
|---|---:|---:|---:|---:|
| syntax_indent | 39 | 47 | 45 | 95.74% |
| syntax_same_indent_simple_assign | 3 | 3 | 2 | 66.67% |

## Files
- Full pass trace: qwen_spec_trace_indent_k12.csv
- Tier summary CSV: qwen_spec_tier_report_indent_k12.csv

## Readout
The indentation patch added only a small gain because most indentation tokens were already learned by ctx8/mined rules. It still reduced no-rule passes from 729 to 723 and improved wall speed from the previous k=12 run, 1.637x, to 1.643x. The remaining bottleneck is still target decode time, not pattern prediction.

