# HumanEval StructSpec Benchmark

Source dataset: OpenAI HumanEval (`HumanEval.jsonl.gz`) from https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz

## Config
- Tasks: 20
- Max new tokens per task: 128
- Draft length k: 12
- Reject mode: truncate
- Live mining: True
- Corpus: C:\Users\neera\OneDrive\Desktop\sep\engineering_dsa_tokens.json

## Aggregate Results
| Metric | Value |
|---|---:|
| Greedy wall time | 72.050s |
| StructSpec wall time | 52.689s |
| Wall speedup | **1.367x** |
| Greedy passes | 2560 |
| StructSpec passes | 1895 |
| Pass speedup | **1.351x** |
| Greedy target eval tokens | 2540 |
| StructSpec target eval tokens | 2751 |
| Eval token ratio | 0.923x |
| Pattern fire rate | 363/1875 = 19.36% |
| Draft token acceptance | 673/884 = 76.13% |
| Reject events | 85 |
| No-rule passes | 1512 |
| Greedy-identical outputs | 15/20 |

## Phase Time Breakdown
| Phase | Greedy seconds | StructSpec seconds |
|---|---:|---:|
| Prompt prefill | 2.1238 | 1.6633 |
| Pattern prediction | 0.0000 | 0.1736 |
| Target decode | 71.8548 | 52.2296 |
| Python verify | 0.0000 | 0.2036 |
| Detokenize | 0.0040 | 0.0030 |

## Top Tiers
| Tier | Rows | Proposed | Accepted | Reject Events | Accept % |
|---|---:|---:|---:|---:|---:|
| det_ctx8 | 93 | 331 | 307 | 9 | 92.75% |
| strong_ctx8 | 56 | 124 | 105 | 8 | 84.68% |
| syntax_pluseq_space | 6 | 72 | 3 | 6 | 4.17% |
| syntax_indent | 49 | 61 | 59 | 2 | 96.72% |
| det_ctx7 | 6 | 54 | 52 | 1 | 96.3% |
| det_ctx5 | 7 | 39 | 27 | 3 | 69.23% |
| strong_ctx4 | 29 | 32 | 22 | 10 | 68.75% |
| det_ctx4 | 20 | 30 | 19 | 5 | 63.33% |
| syntax_for_in | 11 | 27 | 12 | 8 | 44.44% |
| strong_ctx6 | 19 | 21 | 13 | 6 | 61.9% |
| syntax_range_paren | 3 | 14 | 0 | 3 | 0.0% |
| syntax_same_indent_simple_assign | 14 | 14 | 8 | 6 | 57.14% |
| strong_ctx5 | 8 | 10 | 5 | 4 | 50.0% |
| det_ctx6 | 3 | 9 | 9 | 0 | 100.0% |
| syntax_name_eq | 8 | 8 | 8 | 0 | 100.0% |
| syntax_minus_one | 7 | 7 | 5 | 2 | 71.43% |
| syntax_return_terminal | 5 | 7 | 4 | 3 | 57.14% |
| syntax_block_colon | 3 | 6 | 6 | 0 | 100.0% |

## Output Files
- Greedy/spec completions: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results\humaneval_outputs.jsonl`
- Full pass trace CSV: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results\humaneval_trace.csv`
- Tier summary CSV: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results\humaneval_tier_summary.csv`
- Rejections CSV: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results\humaneval_rejections.csv`
- Machine summary JSON: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results\humaneval_summary.json`
