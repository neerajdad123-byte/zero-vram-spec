# HumanEval StructSpec Benchmark

Source dataset: OpenAI HumanEval (`HumanEval.jsonl.gz`) from https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz

## Config
- Tasks: 2
- Max new tokens per task: 32
- Draft length k: 12
- Reject mode: truncate
- Live mining: True
- Corpus: C:\Users\neera\OneDrive\Desktop\sep\engineering_dsa_tokens.json

## Aggregate Results
| Metric | Value |
|---|---:|
| Greedy wall time | 1.581s |
| StructSpec wall time | 1.542s |
| Wall speedup | **1.025x** |
| Greedy passes | 64 |
| StructSpec passes | 61 |
| Pass speedup | **1.049x** |
| Greedy target eval tokens | 62 |
| StructSpec target eval tokens | 68 |
| Eval token ratio | 0.912x |
| Pattern fire rate | 8/59 = 13.56% |
| Draft token acceptance | 3/9 = 33.33% |
| Reject events | 6 |
| No-rule passes | 51 |
| Greedy-identical outputs | 2/2 |

## Phase Time Breakdown
| Phase | Greedy seconds | StructSpec seconds |
|---|---:|---:|
| Prompt prefill | 0.2214 | 0.1278 |
| Pattern prediction | 0.0000 | 0.0054 |
| Target decode | 1.5770 | 1.5288 |
| Python verify | 0.0000 | 0.0057 |
| Detokenize | 0.0002 | 0.0002 |

## Top Tiers
| Tier | Rows | Proposed | Accepted | Reject Events | Accept % |
|---|---:|---:|---:|---:|---:|
| syntax_for_in | 2 | 3 | 2 | 1 | 66.67% |
| strong_ctx5 | 2 | 2 | 0 | 2 | 0.0% |
| syntax_minus_one | 2 | 2 | 0 | 2 | 0.0% |
| syntax_range_paren | 1 | 1 | 0 | 1 | 0.0% |
| syntax_indent | 1 | 1 | 1 | 0 | 100.0% |
| none | 51 | 0 | 0 | 0 | 0.0% |

## Output Files
- Greedy/spec completions: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_smoke\humaneval_outputs.jsonl`
- Full pass trace CSV: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_smoke\humaneval_trace.csv`
- Tier summary CSV: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_smoke\humaneval_tier_summary.csv`
- Rejections CSV: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_smoke\humaneval_rejections.csv`
- Machine summary JSON: `C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_smoke\humaneval_summary.json`
