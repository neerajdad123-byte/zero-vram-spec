# HumanEval StructSpec Comparison

Dataset: OpenAI HumanEval first 20 tasks, downloaded from https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz

## Headline
| Run | Wall Speedup | Pass Speedup | Accepted/Proposed | Accept % | Fire % | Rejects | No-rule | Identical |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw fast rules | 1.367x | 1.351x | 673/884 | 76.13% | 19.36% | 85 | 1512 | 15/20 |
| Gated HumanEval-safe rules | 1.293x | 1.349x | 671/812 | 82.64% | 19.07% | 74 | 1519 | 15/20 |

## Interpretation
- The raw run is faster: **1.367x wall** and **1.351x passes**.
- The gated run is safer: acceptance rises from **76.13%** to **82.64%**, and rejects drop from **85** to **74**.
- HumanEval is harder than the DSA benchmark because the DSA pattern bank only fires on about **19.07%** of passes here. That is why HumanEval speedup is lower than DSA.
- Pattern prediction time is tiny. In the gated run it used **0.174757s**, while target decode used **49.254918s**.

## Files
### Raw fast run
- Report: C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results\humaneval_report.md
- Outputs: C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results\humaneval_outputs.jsonl
- Trace CSV: C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results\humaneval_trace.csv
- Tier summary: C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results\humaneval_tier_summary.csv
- Rejections: C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results\humaneval_rejections.csv

### Gated safer run
- Report: C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_gated\humaneval_report.md
- Outputs: C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_gated\humaneval_outputs.jsonl
- Trace CSV: C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_gated\humaneval_trace.csv
- Tier summary: C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_gated\humaneval_tier_summary.csv
- Rejections: C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_gated\humaneval_rejections.csv
