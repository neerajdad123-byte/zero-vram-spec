from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
DEFAULT_MODEL = r"C:\Users\neera\.lmstudio\models\Qwen\Qwen2.5-7B-Instruct-GGUF\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
DEFAULT_CORPUS = r"C:\Users\neera\OneDrive\Desktop\sep\engineering_dsa_tokens.json"


def load_structspec():
    spec = importlib.util.spec_from_file_location("zero_vram_spec_cli", ROOT / "cli.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import cli.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def ensure_humaneval(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading HumanEval dataset from {HUMANEVAL_URL}")
        urllib.request.urlretrieve(HUMANEVAL_URL, path)
    return path


def load_humaneval(path: Path, limit: int) -> list[dict]:
    tasks = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
            if limit and len(tasks) >= limit:
                break
    return tasks


def build_prompt(task: dict) -> str:
    # HumanEval prompts already contain the function signature and docstring.
    # The extra line nudges instruct models to complete code only.
    return task["prompt"].rstrip() + "\n# Complete the function body only.\n"


def make_miner(mod, corpus_path: str, args):
    corpus = mod.QwenTokenCorpus(corpus_path)
    miner = mod.PatternMiner(
        corpus.token_text,
        max_ctx=args.max_ctx,
        min_support=args.min_support,
        min_conf=args.min_conf,
        det_conf=args.det_conf,
        min_rule_ctx=args.min_rule_ctx,
    ).fit(corpus.sequences)

    if args.extra_corpus:
        model_for_tokens = args.model_for_extra
        if model_for_tokens is None:
            return corpus, miner, list(corpus.sequences)
        extra = []
        for code in mod.EXTRA_CODE_EXAMPLES.values():
            extra.append(model_for_tokens.tokenize(code))
        sequences = list(corpus.sequences) + extra
        mod.refit_miner(miner, sequences)
        return corpus, miner, sequences
    return corpus, miner, list(corpus.sequences)


def row_int(row: dict, key: str) -> int:
    value = row.get(key, 0)
    return int(value or 0)


def aggregate_tiers(rows: list[dict]) -> list[dict]:
    by_tier: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_tier[str(row.get("tier", "none"))].append(row)
    out = []
    for tier, group in by_tier.items():
        proposed = sum(row_int(r, "draft") for r in group)
        accepted = sum(row_int(r, "accepted_draft") for r in group)
        rejects = sum(1 for r in group if r.get("why") == "draft_token_mismatch")
        pattern_s = sum(float(r.get("pattern_s", 0) or 0) for r in group)
        decode_s = sum(float(r.get("decode_s", 0) or 0) for r in group)
        verify_s = sum(float(r.get("verify_s", 0) or 0) for r in group)
        out.append({
            "tier": tier,
            "rows": len(group),
            "proposed_tokens": proposed,
            "accepted_tokens": accepted,
            "rejected_tokens_est": proposed - accepted,
            "reject_events": rejects,
            "accept_rate_pct": round(100 * accepted / proposed, 2) if proposed else 0.0,
            "pattern_s": round(pattern_s, 6),
            "decode_s": round(decode_s, 6),
            "verify_s": round(verify_s, 6),
        })
    return sorted(out, key=lambda r: r["proposed_tokens"], reverse=True)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--token-json", default=DEFAULT_CORPUS)
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--min-support", type=int, default=1)
    ap.add_argument("--min-conf", type=float, default=0.88)
    ap.add_argument("--det-conf", type=float, default=0.92)
    ap.add_argument("--min-rule-ctx", type=int, default=4)
    ap.add_argument("--max-ctx", type=int, default=8)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--reject-mode", choices=["truncate", "seq-bonus", "rebuild"], default="truncate")
    ap.add_argument("--strike-limit", type=int, default=2)
    ap.add_argument("--adaptive-k", action="store_true",
                    help="Cap draft chain length by rule trust to avoid verifying long weak drafts.")
    ap.add_argument("--live-mining", action="store_true", default=True)
    ap.add_argument("--no-live-mining", dest="live_mining", action="store_false")
    ap.add_argument("--extra-corpus", action="store_true", default=True)
    ap.add_argument("--no-extra-corpus", dest="extra_corpus", action="store_false")
    ap.add_argument("--dataset", default=str(ROOT / "HumanEval.jsonl.gz"))
    ap.add_argument("--out-dir", default=str(ROOT / "humaneval_results"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = ensure_humaneval(Path(args.dataset))
    tasks = load_humaneval(dataset_path, args.tasks)
    mod = load_structspec()

    print(f"HumanEval tasks: {len(tasks)}")
    print(f"Model: {args.model}")
    print(f"Max new tokens: {args.tokens}, k={args.k}, reject_mode={args.reject_mode}")

    model = mod.FastGreedyLlama(args.model, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers)
    args.model_for_extra = model
    corpus, miner, train_sequences = make_miner(mod, args.token_json, args)
    syntax = mod.PythonSyntaxProposer(model, mode="cluster")

    outputs_path = out_dir / "humaneval_outputs.jsonl"
    trace_path = out_dir / "humaneval_trace.csv"
    tier_path = out_dir / "humaneval_tier_summary.csv"
    reject_path = out_dir / "humaneval_rejections.csv"
    summary_path = out_dir / "humaneval_summary.json"
    report_path = out_dir / "humaneval_report.md"

    all_records: list[dict] = []
    output_rows: list[dict] = []
    per_task: list[dict] = []
    totals = Counter()
    phase_totals = Counter()
    match_count = 0
    t_suite = time.perf_counter()

    with outputs_path.open("w", encoding="utf-8") as out_f:
        for idx, task in enumerate(tasks, 1):
            prompt = build_prompt(task)
            prompt_ids = model.tokenize(prompt)
            greedy_ids, greedy_text, gm = mod.run_greedy(model, prompt, args.tokens)
            spec_ids, spec_text, sm = mod.run_speculative(
                model,
                miner,
                syntax,
                prompt,
                args.tokens,
                args.k,
                reject_mode=args.reject_mode,
                strike_limit=args.strike_limit,
                adaptive_k=args.adaptive_k,
            )
            if args.live_mining:
                train_sequences.append(spec_ids[len(prompt_ids):])
                mod.refit_miner(miner, train_sequences)

            identical = greedy_ids == spec_ids
            match_count += int(identical)
            recs = sm.get("records", [])
            task_records = [{"task_id": task["task_id"], "task_index": idx, **r} for r in recs]
            all_records.extend(task_records)

            proposed = sum(row_int(r, "draft") for r in recs)
            accepted = sum(row_int(r, "accepted_draft") for r in recs)
            rejects = sum(1 for r in recs if r.get("why") == "draft_token_mismatch")
            no_rule = sum(1 for r in recs if r.get("why") == "no_pattern_fired")
            fires = sum(1 for r in recs if row_int(r, "draft") > 0)
            task_summary = {
                "task_index": idx,
                "task_id": task["task_id"],
                "entry_point": task.get("entry_point", ""),
                "identical": identical,
                "greedy_time_s": gm["time"],
                "spec_time_s": sm["time"],
                "wall_speedup": gm["time"] / max(1e-9, sm["time"]),
                "greedy_passes": gm["passes"],
                "spec_passes": sm["passes"],
                "pass_speedup": gm["passes"] / max(1, sm["passes"]),
                "greedy_eval_tokens": gm["target_tokens_evaluated"],
                "spec_eval_tokens": sm["target_tokens_evaluated"],
                "proposed_tokens": proposed,
                "accepted_tokens": accepted,
                "accept_rate": accepted / proposed if proposed else 0.0,
                "reject_events": rejects,
                "no_rule_passes": no_rule,
                "pattern_fire_rate": fires / max(1, len(recs)),
            }
            per_task.append(task_summary)

            output_row = {
                "task_id": task["task_id"],
                "entry_point": task.get("entry_point", ""),
                "prompt": prompt,
                "greedy_completion": greedy_text,
                "spec_completion": spec_text,
                "identical": identical,
                "metrics": task_summary,
            }
            out_f.write(json.dumps(output_row, ensure_ascii=False) + "\n")

            totals["greedy_passes"] += gm["passes"]
            totals["spec_passes"] += sm["passes"]
            totals["greedy_eval_tokens"] += gm["target_tokens_evaluated"]
            totals["spec_eval_tokens"] += sm["target_tokens_evaluated"]
            totals["proposed_tokens"] += proposed
            totals["accepted_tokens"] += accepted
            totals["reject_events"] += rejects
            totals["no_rule_passes"] += no_rule
            totals["draft_rows"] += fires
            totals["rows"] += len(recs)
            phase_totals.update({f"greedy_{k}": v for k, v in gm["times"].items()})
            phase_totals.update({f"spec_{k}": v for k, v in sm["times"].items()})
            totals["greedy_time_ms"] += int(gm["time"] * 1000_000)
            totals["spec_time_ms"] += int(sm["time"] * 1000_000)

            print(
                f"[{idx:02d}/{len(tasks)}] {task['task_id']:<14} "
                f"wall={task_summary['wall_speedup']:.3f}x "
                f"pass={task_summary['pass_speedup']:.3f}x "
                f"acc={task_summary['accept_rate']*100:.1f}% "
                f"match={'YES' if identical else 'NO'}"
            )

    enriched = mod.enrich_trace_records(model, all_records)
    trace_fields = ["task_id", "task_index"] + mod.TRACE_FIELDS
    write_csv(trace_path, enriched, trace_fields)
    tier_rows = aggregate_tiers(all_records)
    write_csv(tier_path, tier_rows)
    rejection_rows = [
        r for r in enriched
        if r.get("why") in {"draft_token_mismatch", "pending_bonus_mismatch"}
    ]
    write_csv(reject_path, rejection_rows, trace_fields)

    greedy_time = totals["greedy_time_ms"] / 1000_000
    spec_time = totals["spec_time_ms"] / 1000_000
    summary = {
        "dataset": "OpenAI HumanEval",
        "dataset_url": HUMANEVAL_URL,
        "tasks": len(tasks),
        "tokens_per_task": args.tokens,
        "k": args.k,
        "reject_mode": args.reject_mode,
        "adaptive_k": args.adaptive_k,
        "identical_outputs": match_count,
        "greedy_time_s": greedy_time,
        "spec_time_s": spec_time,
        "wall_speedup": greedy_time / max(1e-9, spec_time),
        "greedy_passes": totals["greedy_passes"],
        "spec_passes": totals["spec_passes"],
        "pass_speedup": totals["greedy_passes"] / max(1, totals["spec_passes"]),
        "greedy_eval_tokens": totals["greedy_eval_tokens"],
        "spec_eval_tokens": totals["spec_eval_tokens"],
        "eval_token_ratio": totals["greedy_eval_tokens"] / max(1, totals["spec_eval_tokens"]),
        "pattern_fire_rate": totals["draft_rows"] / max(1, totals["rows"]),
        "accepted_tokens": totals["accepted_tokens"],
        "proposed_tokens": totals["proposed_tokens"],
        "accept_rate": totals["accepted_tokens"] / max(1, totals["proposed_tokens"]),
        "reject_events": totals["reject_events"],
        "no_rule_passes": totals["no_rule_passes"],
        "phase_times_s": {k: round(v, 6) for k, v in sorted(phase_totals.items())},
        "suite_wall_s": time.perf_counter() - t_suite,
        "files": {
            "outputs": str(outputs_path),
            "trace_csv": str(trace_path),
            "tier_summary_csv": str(tier_path),
            "rejections_csv": str(reject_path),
            "summary_json": str(summary_path),
            "report_md": str(report_path),
        },
        "per_task": per_task,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    top_tiers = "\n".join(
        f"| {r['tier']} | {r['rows']} | {r['proposed_tokens']} | {r['accepted_tokens']} | "
        f"{r['reject_events']} | {r['accept_rate_pct']}% |"
        for r in tier_rows[:18]
    )
    report = f"""# HumanEval StructSpec Benchmark

Source dataset: OpenAI HumanEval (`HumanEval.jsonl.gz`) from {HUMANEVAL_URL}

## Config
- Tasks: {len(tasks)}
- Max new tokens per task: {args.tokens}
- Draft length k: {args.k}
- Reject mode: {args.reject_mode}
- Adaptive K: {args.adaptive_k}
- Live mining: {args.live_mining}
- Corpus: {args.token_json}

## Aggregate Results
| Metric | Value |
|---|---:|
| Greedy wall time | {greedy_time:.3f}s |
| StructSpec wall time | {spec_time:.3f}s |
| Wall speedup | **{summary['wall_speedup']:.3f}x** |
| Greedy passes | {totals['greedy_passes']} |
| StructSpec passes | {totals['spec_passes']} |
| Pass speedup | **{summary['pass_speedup']:.3f}x** |
| Greedy target eval tokens | {totals['greedy_eval_tokens']} |
| StructSpec target eval tokens | {totals['spec_eval_tokens']} |
| Eval token ratio | {summary['eval_token_ratio']:.3f}x |
| Pattern fire rate | {totals['draft_rows']}/{totals['rows']} = {summary['pattern_fire_rate']*100:.2f}% |
| Draft token acceptance | {totals['accepted_tokens']}/{totals['proposed_tokens']} = {summary['accept_rate']*100:.2f}% |
| Reject events | {totals['reject_events']} |
| No-rule passes | {totals['no_rule_passes']} |
| Greedy-identical outputs | {match_count}/{len(tasks)} |

## Phase Time Breakdown
| Phase | Greedy seconds | StructSpec seconds |
|---|---:|---:|
| Prompt prefill | {phase_totals['greedy_prompt']:.4f} | {phase_totals['spec_prompt']:.4f} |
| Pattern prediction | 0.0000 | {phase_totals['spec_pattern']:.4f} |
| Target decode | {phase_totals['greedy_decode']:.4f} | {phase_totals['spec_decode']:.4f} |
| Python verify | 0.0000 | {phase_totals['spec_verify']:.4f} |
| Detokenize | {phase_totals['greedy_detok']:.4f} | {phase_totals['spec_detok']:.4f} |

## Top Tiers
| Tier | Rows | Proposed | Accepted | Reject Events | Accept % |
|---|---:|---:|---:|---:|---:|
{top_tiers}

## Output Files
- Greedy/spec completions: `{outputs_path}`
- Full pass trace CSV: `{trace_path}`
- Tier summary CSV: `{tier_path}`
- Rejections CSV: `{reject_path}`
- Machine summary JSON: `{summary_path}`
"""
    report_path.write_text(report, encoding="utf-8")
    print("\nAGGREGATE")
    print(f"  wall speedup : {summary['wall_speedup']:.3f}x")
    print(f"  pass speedup : {summary['pass_speedup']:.3f}x")
    print(f"  accept rate  : {summary['accept_rate']*100:.2f}%")
    print(f"  fire rate    : {summary['pattern_fire_rate']*100:.2f}%")
    print(f"  outputs      : {outputs_path}")
    print(f"  trace csv    : {trace_path}")
    print(f"  report       : {report_path}")


if __name__ == "__main__":
    main()
