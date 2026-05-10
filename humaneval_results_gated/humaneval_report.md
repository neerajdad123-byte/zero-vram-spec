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
| Greedy wall time | 64.279s |
| StructSpec wall time | 49.706s |
| Wall speedup | **1.293x** |
| Greedy passes | 2560 |
| StructSpec passes | 1897 |
| Pass speedup | **1.349x** |
| Greedy target eval tokens | 2540 |
| StructSpec target eval tokens | 2681 |
| Eval token ratio | 0.947x |
| Pattern fire rate | 358/1877 = 19.07% |
| Draft token acceptance | 671/812 = 82.64% |
| Reject events | 74 |
| No-rule passes | 1519 |
| Greedy-identical outputs | 15/20 |

## Phase Time Breakdown
| Phase | Greedy seconds | StructSpec seconds |
|---|---:|---:|
| Prompt prefill | 1.3184 | 1.1993 |
| Pattern prediction | 0.0000 | 0.1748 |
| Target decode | 64.0983 | 49.2549 |
| Python verify | 0.0000 | 0.1965 |
| Detokenize | 0.0039 | 0.0032 |

## Top Tiers
| Tier | Rows | Proposed | Accepted | Reject Events | Accept % |
|---|---:|---:|---:|---:|---:|
| det_ctx8 | 93 | 331 | 307 | 9 | 92.75% |
| strong_ctx8 | 56 | 124 | 105 | 8 | 84.68% |
| syntax_indent | 49 | 61 | 59 | 2 | 96.72% |
| det_ctx7 | 6 | 54 | 52 | 1 | 96.3% |
| det_ctx5 | 7 | 39 | 27 | 3 | 69.23% |
| strong_ctx4 | 29 | 32 | 22 | 10 | 68.75% |
| det_ctx4 | 20 | 30 | 19 | 5 | 63.33% |
| strong_ctx6 | 19 | 21 | 13 | 6 | 61.9% |
| syntax_for_in | 11 | 15 | 12 | 2 | 80.0% |
| syntax_range_paren | 3 | 14 | 0 | 3 | 0.0% |
| syntax_same_indent_simple_assign | 14 | 14 | 8 | 6 | 57.14% |
| syntax_pluseq_space | 1 | 12 | 1 | 1 | 8.33% |
| strong_ctx5 | 8 | 10 | 5 | 4 | 50.0% |
| det_ctx6 | 3 | 9 | 9 | 0 | 100.0% |
| syntax_name_eq | 8 | 8 | 8 | 0 | 100.0% |
| syntax_minus_one | 7 | 7 | 5 | 2 | 71.43% |
| syntax_return_terminal | 5 | 7 | 4 | 3 | 57.14% |
| syntax_block_colon | 3 | 6 | 6 | 0 | 100.0% |

## Output Files
- Greedy/spec completions: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_gated\humaneval_outputs.jsonl`
- Full pass trace CSV: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_gated\humaneval_trace.csv`
- Tier summary CSV: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_gated\humaneval_tier_summary.csv`
- Rejections CSV: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_gated\humaneval_rejections.csv`
- Machine summary JSON: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_gated\humaneval_summary.json`
