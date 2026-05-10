from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_int(value) -> int:
    if value in ("", None):
        return 0
    return int(value)


def as_float(value) -> float:
    if value in ("", None):
        return 0.0
    return float(value)


def split_ids(value: str) -> list[int]:
    if not value:
        return []
    return [int(x) for x in str(value).split() if x.strip()]


def first_mismatch(draft: list[int], model: list[int]) -> int | None:
    for idx, (want, got) in enumerate(zip(draft, model)):
        if want != got:
            return idx
    if len(draft) != len(model):
        return min(len(draft), len(model))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--tokens-per-task", type=int, default=128)
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    trace_path = result_dir / "humaneval_trace.csv"
    outputs_path = result_dir / "humaneval_outputs.jsonl"
    summary_path = result_dir / "humaneval_summary.json"
    out_report = result_dir / "humaneval_token_accounting_report.md"
    out_task_csv = result_dir / "humaneval_token_accounting_by_task.csv"
    out_reject_csv = result_dir / "humaneval_rejection_diffs.csv"

    trace = read_csv(trace_path)
    outputs = [json.loads(line) for line in outputs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tasks = len(outputs)
    total_output_tokens = tasks * args.tokens_per_task

    by_task: dict[str, list[dict]] = {}
    for row in trace:
        by_task.setdefault(row["task_id"], []).append(row)

    per_task_rows = []
    rejection_rows = []
    aggregate = Counter()
    for output in outputs:
        task_id = output["task_id"]
        rows = by_task.get(task_id, [])
        proposed = sum(as_int(r.get("draft")) for r in rows)
        accepted = sum(as_int(r.get("accepted_draft")) for r in rows)
        rejected_est = proposed - accepted
        reject_events = sum(1 for r in rows if r.get("why") == "draft_token_mismatch")
        no_rule_rows = sum(1 for r in rows if r.get("why") == "no_pattern_fired")
        draft_rows = sum(1 for r in rows if as_int(r.get("draft")) > 0)
        model_tokens = args.tokens_per_task - accepted

        per_task_rows.append({
            "task_id": task_id,
            "entry_point": output.get("entry_point", ""),
            "output_tokens": args.tokens_per_task,
            "proposed_draft_tokens": proposed,
            "accepted_draft_tokens": accepted,
            "rejected_draft_tokens_est": rejected_est,
            "target_model_tokens_est": model_tokens,
            "proposed_pct_of_output": round(100 * proposed / args.tokens_per_task, 2),
            "accepted_pct_of_output": round(100 * accepted / args.tokens_per_task, 2),
            "target_model_pct_of_output": round(100 * model_tokens / args.tokens_per_task, 2),
            "accept_rate_of_proposed": round(100 * accepted / proposed, 2) if proposed else 0.0,
            "draft_pass_rows": draft_rows,
            "no_rule_rows": no_rule_rows,
            "reject_events": reject_events,
            "identical_to_greedy": output.get("identical", False),
        })

        aggregate["proposed"] += proposed
        aggregate["accepted"] += accepted
        aggregate["rejected_est"] += rejected_est
        aggregate["reject_events"] += reject_events
        aggregate["no_rule_rows"] += no_rule_rows
        aggregate["draft_rows"] += draft_rows
        aggregate["rows"] += len(rows)

        for row in rows:
            if row.get("why") != "draft_token_mismatch":
                continue
            draft_ids = split_ids(row.get("draft_ids", ""))
            model_ids = split_ids(row.get("model_draft_ids", ""))
            idx = first_mismatch(draft_ids, model_ids)
            wanted_id = draft_ids[idx] if idx is not None and idx < len(draft_ids) else ""
            got_id = model_ids[idx] if idx is not None and idx < len(model_ids) else ""
            rejection_rows.append({
                "task_id": task_id,
                "pos": row.get("pos", ""),
                "tier": row.get("tier", ""),
                "rule_chain": row.get("rule_chain", ""),
                "mismatch_at": row.get("mismatch_at", idx if idx is not None else ""),
                "wanted_token_id": wanted_id,
                "target_token_id": got_id,
                "draft_text": row.get("draft_text", ""),
                "target_model_text": row.get("model_draft_text", ""),
                "ctx_tail_text": row.get("ctx_tail_text", ""),
            })

    write_csv(out_task_csv, per_task_rows)
    write_csv(out_reject_csv, rejection_rows)

    pattern_s = summary["phase_times_s"].get("spec_pattern", 0.0)
    decode_s = summary["phase_times_s"].get("spec_decode", 0.0)
    verify_s = summary["phase_times_s"].get("spec_verify", 0.0)
    prompt_s = summary["phase_times_s"].get("spec_prompt", 0.0)
    spec_time = summary["spec_time_s"]
    target_model_tokens = total_output_tokens - aggregate["accepted"]

    top_rejections = Counter(r["tier"] for r in rejection_rows).most_common(12)
    top_rej_md = "\n".join(f"| {tier} | {count} |" for tier, count in top_rejections)
    sample_rej_md = "\n".join(
        "| {task_id} | {pos} | {tier} | {mismatch_at} | `{draft_text}` | `{target_model_text}` |".format(**r)
        for r in rejection_rows[:18]
    )

    report = f"""# HumanEval Token Accounting Report

This report fixes the confusing denominator issue:

- **Proposed draft tokens** are attempts. Rejected proposed tokens are not output.
- **Accepted draft tokens** are the actual output tokens produced by StructSpec prediction and verified by Qwen.
- **Target/model tokens** are the remaining output tokens produced by the target model path, including bonus/pending/normal greedy tokens.

## Output Token Accounting
| Metric | Value |
|---|---:|
| Tasks | {tasks} |
| Output tokens per task | {args.tokens_per_task} |
| Total output tokens | {total_output_tokens} |
| Proposed draft token attempts | {aggregate['proposed']} ({100 * aggregate['proposed'] / total_output_tokens:.2f}% of output tokens) |
| Accepted draft output tokens | {aggregate['accepted']} ({100 * aggregate['accepted'] / total_output_tokens:.2f}% of output tokens) |
| Rejected draft token attempts | {aggregate['rejected_est']} |
| Draft acceptance rate | {100 * aggregate['accepted'] / max(1, aggregate['proposed']):.2f}% of proposed |
| Target/model-produced output tokens | {target_model_tokens} ({100 * target_model_tokens / total_output_tokens:.2f}% of output tokens) |
| Draft pass rows | {aggregate['draft_rows']} |
| No-rule rows | {aggregate['no_rule_rows']} |
| Reject events | {aggregate['reject_events']} |

## Speed
| Metric | Value |
|---|---:|
| Greedy wall time | {summary['greedy_time_s']:.3f}s |
| StructSpec wall time | {summary['spec_time_s']:.3f}s |
| Wall speedup | **{summary['wall_speedup']:.3f}x** |
| Greedy passes | {summary['greedy_passes']} |
| StructSpec passes | {summary['spec_passes']} |
| Pass speedup | **{summary['pass_speedup']:.3f}x** |
| Greedy target eval tokens | {summary['greedy_eval_tokens']} |
| StructSpec target eval tokens | {summary['spec_eval_tokens']} |
| Eval token ratio | {summary['eval_token_ratio']:.3f}x |

## Time Breakdown
| Phase | Seconds | Percent of StructSpec timed wall |
|---|---:|---:|
| Prompt prefill | {prompt_s:.4f} | outside loop |
| Pattern prediction | {pattern_s:.4f} | {100 * pattern_s / spec_time:.2f}% |
| Target decode | {decode_s:.4f} | {100 * decode_s / spec_time:.2f}% |
| Python verification | {verify_s:.4f} | {100 * verify_s / spec_time:.2f}% |

## Token Throughput View
| Token bucket | Tokens | Tokens/sec over StructSpec wall |
|---|---:|---:|
| All output tokens | {total_output_tokens} | {total_output_tokens / spec_time:.2f} |
| Accepted draft tokens | {aggregate['accepted']} | {aggregate['accepted'] / spec_time:.2f} |
| Target/model-produced tokens | {target_model_tokens} | {target_model_tokens / spec_time:.2f} |
| Rejected draft attempts | {aggregate['rejected_est']} | {aggregate['rejected_est'] / spec_time:.2f} |

## Top Rejection Tiers
| Tier | Reject events |
|---|---:|
{top_rej_md}

## Sample Rejection Diffs
These are the tokens StructSpec proposed versus what the target model wanted during verification.

| Task | Pos | Tier | Mismatch | Proposed draft text | Target model text |
|---|---:|---|---:|---|---|
{sample_rej_md}

## Saved Files
- Greedy/spec outputs: `{outputs_path}`
- Per-task token accounting CSV: `{out_task_csv}`
- Rejection diff CSV: `{out_reject_csv}`
- Original full pass trace: `{trace_path}`
"""
    out_report.write_text(report, encoding="utf-8")
    print(out_report)
    print(out_task_csv)
    print(out_reject_csv)


if __name__ == "__main__":
    main()
