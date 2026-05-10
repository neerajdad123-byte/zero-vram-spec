# HumanEval Adaptive K Test

Adaptive K was implemented in `cli.py` behind `--adaptive-k`. It caps draft chain length by rule trust: deep ctx8 gets longer drafts, shallow ctx4/5 gets short drafts, and HumanEval-toxic structural rules are stopped.

## Results
| Run | Wall Speedup | Pass Speedup | Accepted/Proposed | Accept % | Fire % | Rejects | No-rule | Identical | Report |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Raw fixed K=12 | 1.367x | 1.351x | 673/884 | 76.13% | 19.36% | 85 | 1512 | 15/20 | [humaneval_results](<C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results\humaneval_report.md>) |
| Gated fixed K=12 | 1.293x | 1.349x | 671/812 | 82.64% | 19.07% | 74 | 1519 | 15/20 | [humaneval_results_gated](<C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_gated\humaneval_report.md>) |
| Adaptive K conservative | 1.330x | 1.311x | 615/695 | 88.49% | 19.56% | 66 | 1555 | 14/20 | [humaneval_results_adaptive](<C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_adaptive\humaneval_report.md>) |
| Adaptive K balanced trial | 1.273x | 1.323x | 633/716 | 88.41% | 19.32% | 66 | 1545 | 14/20 | [humaneval_results_adaptive_balanced](<C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_adaptive_balanced\humaneval_report.md>) |

## What Happened
- Adaptive K did exactly what it was supposed to do on quality of drafts: acceptance increased from gated fixed-K **82.64%** to conservative adaptive **88.49%**.
- Reject events dropped from gated fixed-K **74** to conservative adaptive **66**.
- But it capped too many drafts, so accepted draft output tokens dropped from **671** to **615**.
- Because pass reduction mostly tracks accepted draft tokens, conservative adaptive pass speed became **1.311x**, lower than gated fixed-K **1.349x**.
- The fastest HumanEval result is still raw fixed K=12: **1.367x wall**, but it has lower acceptance and more rejects.

## Conclusion
Adaptive K is useful, but this heuristic is not yet better than raw fixed-K on HumanEval. The right next version should use online per-rule EV: keep long K only when a specific rule has positive accepted_tokens - rejected_tokens cost in the current run, instead of static tier caps.

## Files
- Adaptive outputs: C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_adaptive\humaneval_outputs.jsonl
- Adaptive trace: C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_adaptive\humaneval_trace.csv
- Adaptive accounting: C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_adaptive\humaneval_token_accounting_report.md
- Adaptive rejection diffs: C:\Users\neera\OneDrive\Desktop\Zero_vram_spec\humaneval_results_adaptive\humaneval_rejection_diffs.csv
