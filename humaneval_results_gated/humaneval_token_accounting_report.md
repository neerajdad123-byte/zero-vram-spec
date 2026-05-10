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
| Proposed draft token attempts | 812 (31.72% of output tokens) |
| Accepted draft output tokens | 671 (26.21% of output tokens) |
| Rejected draft token attempts | 141 |
| Draft acceptance rate | 82.64% of proposed |
| Target/model-produced output tokens | 1889 (73.79% of output tokens) |
| Draft pass rows | 358 |
| No-rule rows | 1519 |
| Reject events | 74 |

## Speed
| Metric | Value |
|---|---:|
| Greedy wall time | 64.279s |
| StructSpec wall time | 49.706s |
| Wall speedup | **1.293x** |
| Greedy passes | 2560 |
| StructSpec passes | 1897 |
| Pass speedup | **1.349x** |
| Greedy target eval tokens | 2540 |
| StructSpec target eval tokens | 2681 |
| Eval token ratio | 0.947x |

## Time Breakdown
| Phase | Seconds | Percent of StructSpec timed wall |
|---|---:|---:|
| Prompt prefill | 1.1993 | outside loop |
| Pattern prediction | 0.1748 | 0.35% |
| Target decode | 49.2549 | 99.09% |
| Python verification | 0.1965 | 0.40% |

## Token Throughput View
| Token bucket | Tokens | Tokens/sec over StructSpec wall |
|---|---:|---:|
| All output tokens | 2560 | 51.50 |
| Accepted draft tokens | 671 | 13.50 |
| Target/model-produced tokens | 1889 | 38.00 |
| Rejected draft attempts | 141 | 2.84 |

## Top Rejection Tiers
| Tier | Reject events |
|---|---:|
| strong_ctx4 | 10 |
| det_ctx8 | 9 |
| strong_ctx8 | 8 |
| syntax_same_indent_simple_assign | 6 |
| strong_ctx6 | 6 |
| det_ctx4 | 5 |
| syntax_class_next_method | 5 |
| strong_ctx5 | 4 |
| syntax_range_paren | 3 |
| syntax_return_terminal | 3 |
| det_ctx5 | 3 |
| syntax_minus_one | 2 |

## Sample Rejection Diffs
These are the tokens StructSpec proposed versus what the target model wanted during verification.

| Task | Pos | Tier | Mismatch | Proposed draft text | Target model text |
|---|---:|---|---:|---|---|
| HumanEval/0 | 9 | syntax_range_paren | 0 | `(` | `(len` |
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
| HumanEval/1 | 75 | syntax_indent | 1 | `            return` | `            result` |
| HumanEval/1 | 84 | syntax_same_indent_simple_assign | 0 | `           ` | `   ` |
| HumanEval/1 | 100 | strong_ctx4 | 0 | `\n` | `\n\n` |
| HumanEval/1 | 102 | syntax_class_next_method | 0 | ` def` | ` do` |
| HumanEval/1 | 111 | det_ctx8 | 0 | `\ndoctest.testmod()` | `\n\ndoctest.testmod()` |
| HumanEval/1 | 113 | syntax_class_next_method | 0 | ` def` | ` do` |
| HumanEval/1 | 122 | det_ctx8 | 0 | `\ndoctest.testmod` | `\n\ndoctest.testmod` |

## Saved Files
- Greedy/spec outputs: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_gated\humaneval_outputs.jsonl`
- Per-task token accounting CSV: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_gated\humaneval_token_accounting_by_task.csv`
- Rejection diff CSV: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_gated\humaneval_rejection_diffs.csv`
- Original full pass trace: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_gated\humaneval_trace.csv`
