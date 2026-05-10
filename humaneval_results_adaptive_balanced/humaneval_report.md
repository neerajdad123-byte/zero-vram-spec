# HumanEval StructSpec Benchmark

Source dataset: OpenAI HumanEval (`HumanEval.jsonl.gz`) from https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz

## Config
- Tasks: 20
- Max new tokens per task: 128
- Draft length k: 12
- Reject mode: truncate
- Adaptive K: True
- Live mining: True
- Corpus: C:\Users\neera\OneDrive\Desktop\sep\engineering_dsa_tokens.json

## Aggregate Results
| Metric | Value |
|---|---:|
| Greedy wall time | 64.422s |
| StructSpec wall time | 50.606s |
| Wall speedup | **1.273x** |
| Greedy passes | 2560 |
| StructSpec passes | 1935 |
| Pass speedup | **1.323x** |
| Greedy target eval tokens | 2540 |
| StructSpec target eval tokens | 2623 |
| Eval token ratio | 0.968x |
| Pattern fire rate | 370/1915 = 19.32% |
| Draft token acceptance | 633/716 = 88.41% |
| Reject events | 66 |
| No-rule passes | 1545 |
| Greedy-identical outputs | 14/20 |

## Phase Time Breakdown
| Phase | Greedy seconds | StructSpec seconds |
|---|---:|---:|
| Prompt prefill | 1.2795 | 1.1992 |
| Pattern prediction | 0.0000 | 0.1567 |
| Target decode | 64.2472 | 50.1825 |
| Python verify | 0.0000 | 0.1881 |
| Detokenize | 0.0041 | 0.0032 |

## Top Tiers
| Tier | Rows | Proposed | Accepted | Reject Events | Accept % |
|---|---:|---:|---:|---:|---:|
| det_ctx8 | 107 | 352 | 324 | 11 | 92.05% |
| strong_ctx8 | 55 | 107 | 102 | 5 | 95.33% |
| syntax_indent | 51 | 59 | 57 | 2 | 96.61% |
| strong_ctx4 | 31 | 31 | 18 | 13 | 58.06% |
| det_ctx7 | 6 | 30 | 29 | 1 | 96.67% |
| strong_ctx6 | 21 | 21 | 13 | 8 | 61.9% |
| det_ctx4 | 20 | 20 | 15 | 5 | 75.0% |
| syntax_same_indent_simple_assign | 14 | 14 | 8 | 6 | 57.14% |
| syntax_for_in | 11 | 11 | 11 | 0 | 100.0% |
| det_ctx5 | 6 | 9 | 7 | 2 | 77.78% |
| det_ctx6 | 3 | 9 | 9 | 0 | 100.0% |
| strong_ctx5 | 8 | 8 | 4 | 4 | 50.0% |
| syntax_name_eq | 8 | 8 | 8 | 0 | 100.0% |
| syntax_minus_one | 7 | 7 | 5 | 2 | 71.43% |
| syntax_return_terminal | 5 | 7 | 4 | 3 | 57.14% |
| syntax_block_colon | 3 | 6 | 6 | 0 | 100.0% |
| strong_ctx7 | 5 | 6 | 6 | 0 | 100.0% |
| syntax_dedent_after_terminal | 3 | 3 | 2 | 1 | 66.67% |

## Output Files
- Greedy/spec completions: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_adaptive_balanced\humaneval_outputs.jsonl`
- Full pass trace CSV: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_adaptive_balanced\humaneval_trace.csv`
- Tier summary CSV: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_adaptive_balanced\humaneval_tier_summary.csv`
- Rejections CSV: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_adaptive_balanced\humaneval_rejections.csv`
- Machine summary JSON: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_adaptive_balanced\humaneval_summary.json`
