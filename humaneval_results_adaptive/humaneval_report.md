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
| Greedy wall time | 72.691s |
| StructSpec wall time | 54.645s |
| Wall speedup | **1.330x** |
| Greedy passes | 2560 |
| StructSpec passes | 1953 |
| Pass speedup | **1.311x** |
| Greedy target eval tokens | 2540 |
| StructSpec target eval tokens | 2620 |
| Eval token ratio | 0.969x |
| Pattern fire rate | 378/1933 = 19.56% |
| Draft token acceptance | 615/695 = 88.49% |
| Reject events | 66 |
| No-rule passes | 1555 |
| Greedy-identical outputs | 14/20 |

## Phase Time Breakdown
| Phase | Greedy seconds | StructSpec seconds |
|---|---:|---:|
| Prompt prefill | 2.0267 | 1.7925 |
| Pattern prediction | 0.0000 | 0.1631 |
| Target decode | 72.5022 | 54.2066 |
| Python verify | 0.0000 | 0.1914 |
| Detokenize | 0.0041 | 0.0034 |

## Top Tiers
| Tier | Rows | Proposed | Accepted | Reject Events | Accept % |
|---|---:|---:|---:|---:|---:|
| det_ctx8 | 109 | 350 | 331 | 10 | 94.57% |
| strong_ctx8 | 53 | 89 | 84 | 5 | 94.38% |
| syntax_indent | 53 | 68 | 62 | 2 | 91.18% |
| strong_ctx4 | 31 | 31 | 18 | 13 | 58.06% |
| strong_ctx6 | 21 | 21 | 13 | 8 | 61.9% |
| det_ctx4 | 20 | 20 | 15 | 5 | 75.0% |
| syntax_same_indent_simple_assign | 14 | 14 | 8 | 6 | 57.14% |
| det_ctx7 | 6 | 14 | 13 | 1 | 92.86% |
| freq_ctx5 | 6 | 12 | 10 | 1 | 83.33% |
| syntax_for_in | 11 | 11 | 11 | 0 | 100.0% |
| strong_ctx5 | 8 | 8 | 4 | 4 | 50.0% |
| syntax_name_eq | 8 | 8 | 8 | 0 | 100.0% |
| syntax_minus_one | 7 | 7 | 5 | 2 | 71.43% |
| syntax_return_terminal | 5 | 7 | 4 | 3 | 57.14% |
| det_ctx5 | 6 | 6 | 4 | 2 | 66.67% |
| det_ctx6 | 3 | 6 | 6 | 0 | 100.0% |
| syntax_block_colon | 3 | 6 | 6 | 0 | 100.0% |
| strong_ctx7 | 5 | 6 | 6 | 0 | 100.0% |

## Output Files
- Greedy/spec completions: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_adaptive\humaneval_outputs.jsonl`
- Full pass trace CSV: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_adaptive\humaneval_trace.csv`
- Tier summary CSV: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_adaptive\humaneval_tier_summary.csv`
- Rejections CSV: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_adaptive\humaneval_rejections.csv`
- Machine summary JSON: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_adaptive\humaneval_summary.json`
