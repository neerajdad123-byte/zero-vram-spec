# HumanEval Token Accounting Report

This report fixes the confusing denominator issue:

- **Proposed draft tokens** are attempts. Rejected proposed tokens are not output.
- **Accepted draft tokens** are the actual output tokens produced by StructSpec prediction and verified by Qwen.
- **Target/model tokens** are the remaining output tokens produced by the target model path, including bonus/pending/normal greedy tokens.

## Output Token Accounting
| Metric | Value |
|---|---:|
| Tasks | 20 |
| Output tokens per task | 128 |
| Total output tokens | 2560 |
| Proposed draft token attempts | 695 (27.15% of output tokens) |
| Accepted draft output tokens | 615 (24.02% of output tokens) |
| Rejected draft token attempts | 80 |
| Draft acceptance rate | 88.49% of proposed |
| Target/model-produced output tokens | 1945 (75.98% of output tokens) |
| Draft pass rows | 378 |
| No-rule rows | 1555 |
| Reject events | 66 |

## Speed
| Metric | Value |
|---|---:|
| Greedy wall time | 72.691s |
| StructSpec wall time | 54.645s |
| Wall speedup | **1.330x** |
| Greedy passes | 2560 |
| StructSpec passes | 1953 |
| Pass speedup | **1.311x** |
| Greedy target eval tokens | 2540 |
| StructSpec target eval tokens | 2620 |
| Eval token ratio | 0.969x |

## Time Breakdown
| Phase | Seconds | Percent of StructSpec timed wall |
|---|---:|---:|
| Prompt prefill | 1.7925 | outside loop |
| Pattern prediction | 0.1631 | 0.30% |
| Target decode | 54.2066 | 99.20% |
| Python verification | 0.1914 | 0.35% |

## Token Throughput View
| Token bucket | Tokens | Tokens/sec over StructSpec wall |
|---|---:|---:|
| All output tokens | 2560 | 46.85 |
| Accepted draft tokens | 615 | 11.25 |
| Target/model-produced tokens | 1945 | 35.59 |
| Rejected draft attempts | 80 | 1.46 |

## Top Rejection Tiers
| Tier | Reject events |
|---|---:|
| strong_ctx4 | 13 |
| det_ctx8 | 10 |
| strong_ctx6 | 8 |
| syntax_same_indent_simple_assign | 6 |
| det_ctx4 | 5 |
| strong_ctx8 | 5 |
| strong_ctx5 | 4 |
| syntax_return_terminal | 3 |
| syntax_minus_one | 2 |
| det_ctx5 | 2 |
| syntax_indent | 2 |
| syntax_return_len_self | 2 |

## Sample Rejection Diffs
These are the tokens StructSpec proposed versus what the target model wanted during verification.

| Task | Pos | Tier | Mismatch | Proposed draft text | Target model text |
|---|---:|---|---:|---|---|
| HumanEval/0 | 10 | strong_ctx5 | 0 | `(Graph` | `(numbers` |
| HumanEval/0 | 13 | syntax_minus_one | 0 | `1` | ` ` |
| HumanEval/0 | 25 | syntax_minus_one | 0 | `1` | ` numbers` |
| HumanEval/0 | 34 | syntax_return_terminal | 1 | `\n       ` | `\n   ` |
| HumanEval/0 | 38 | syntax_return_terminal | 0 | `\n` | `\n\n\n` |
| HumanEval/1 | 27 | strong_ctx5 | 0 | ` priority` | ` for` |
| HumanEval/1 | 47 | syntax_same_indent_simple_assign | 0 | `           ` | `       ` |
| HumanEval/1 | 48 | det_ctx5 | 0 | ` right` | ` elif` |
| HumanEval/1 | 61 | det_ctx4 | 0 | `   ` | `       ` |
| HumanEval/1 | 62 | det_ctx4 | 0 | ` right` | ` temp` |
| HumanEval/1 | 84 | syntax_same_indent_simple_assign | 0 | `           ` | `   ` |
| HumanEval/1 | 93 | strong_ctx6 | 0 | ` '__` | ` "__` |
| HumanEval/1 | 100 | strong_ctx4 | 0 | `\n` | `\n\n` |
| HumanEval/1 | 111 | det_ctx8 | 0 | `\ndoctest.testmod()` | `\n\ndoctest.testmod()` |
| HumanEval/1 | 122 | det_ctx8 | 0 | `\ndoctest.testmod` | `\n\ndoctest.testmod` |
| HumanEval/2 | 27 | strong_ctx8 | 0 | `import` | `Human` |
| HumanEval/2 | 38 | syntax_indent | 0 | `       ` | `1` |
| HumanEval/3 | 8 | strong_ctx6 | 0 | ` char` | ` op` |

## Saved Files
- Greedy/spec outputs: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_adaptive\humaneval_outputs.jsonl`
- Per-task token accounting CSV: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_adaptive\humaneval_token_accounting_by_task.csv`
- Rejection diff CSV: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_adaptive\humaneval_rejection_diffs.csv`
- Original full pass trace: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_adaptive\humaneval_trace.csv`
