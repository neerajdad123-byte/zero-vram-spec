"""stat_compare.py - Statistical side-by-side comparison of two trace CSVs."""
import csv, sys
from collections import Counter, defaultdict
from pathlib import Path

OLD = r"C:\Users\neera\OneDrive\Desktop\Structspec_Artifacts\humaneval_results\humaneval_trace.csv"
NEW = "comparison_new2.csv"

def read_trace(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def analyze(trace, label):
    pattern_t, decode_t, verify_t = [], [], []
    draft_total = accepted_total = draft_starts = total_steps = 0
    tier_acc = defaultdict(lambda: [0, 0])
    why_counts = Counter()

    for r in trace:
        try:
            d = int(r.get('draft', 0))
            a = int(r.get('accepted_draft', 0))
            total_steps += 1
            pattern_t.append(float(r.get('pattern_s', 0)))
            decode_t.append(float(r.get('decode_s', 0)))
            verify_t.append(float(r.get('verify_s', 0)))
            tier = r.get('tier', 'none')
            why = r.get('why', '')
            draft_total += d
            accepted_total += a
            why_counts[why] += 1
            if d > 0:
                draft_starts += 1
                tier_acc[tier][0] += d
                tier_acc[tier][1] += a
        except (ValueError, KeyError):
            continue

    pt_sorted = sorted(pattern_t)
    dt_sorted = sorted(decode_t)
    vt_sorted = sorted(verify_t)
    mid_pt = len(pt_sorted) // 2
    mid_dt = len(dt_sorted) // 2
    mid_vt = len(vt_sorted) // 2

    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    print(f"  Total steps            : {total_steps}")
    print(f"  Draft fire rate        : {draft_starts}/{total_steps} = {draft_starts/max(1,total_steps)*100:.1f}%")
    print(f"  Draft tokens proposed  : {draft_total}")
    print(f"  Draft tokens accepted  : {accepted_total}")
    print(f"  Draft tokens rejected  : {draft_total - accepted_total}")
    print(f"  ACCEPT RATE            : {accepted_total/max(1,draft_total)*100:.1f}%")
    print(f"  No-pattern steps       : {total_steps - draft_starts}")
    print(f"")
    print(f"  Pattern time TOTAL     : {sum(pattern_t):.4f}s")
    print(f"  Pattern time MEDIAN    : {pt_sorted[mid_pt]*1000 if pt_sorted else 0:.3f}ms")
    print(f"  Pattern time PER STEP  : {sum(pattern_t)/max(1,total_steps)*1000:.3f}ms")
    print(f"")
    print(f"  Model decode TOTAL     : {sum(decode_t):.3f}s")
    print(f"  Model decode MEDIAN    : {dt_sorted[mid_dt]*1000 if dt_sorted else 0:.2f}ms")
    print(f"  Model decode PER STEP  : {sum(decode_t)/max(1,total_steps)*1000:.1f}ms")
    print(f"")
    print(f"  Verification TOTAL     : {sum(verify_t):.4f}s")
    print(f"  Verification MEDIAN    : {vt_sorted[mid_vt]*1000 if vt_sorted else 0:.3f}ms")
    print(f"  Verification PER STEP  : {sum(verify_t)/max(1,total_steps)*1000:.3f}ms")
    print(f"")
    print(f"  Rejection reasons:")
    for why, cnt in why_counts.most_common():
        print(f"    {why:<25} {cnt:>5}")
    print(f"")
    print(f"  Tier acceptance breakdown:")
    for tier, (tot, acc) in sorted(tier_acc.items(), key=lambda x: -x[1][0]):
        rate = acc / max(1, tot) * 100
        bar = "#" * int(rate / 10) + "." * (10 - int(rate / 10))
        print(f"    {tier:<30} {acc:>3}/{tot:<3} = {rate:>5.1f}%  [{bar}]")

    return {
        'steps': total_steps, 'draft_total': draft_total, 'accepted': accepted_total,
        'accept_rate': accepted_total/max(1,draft_total)*100,
        'pattern_total': sum(pattern_t), 'decode_total': sum(decode_t),
        'verify_total': sum(verify_t), 'fire_rate': draft_starts/max(1,total_steps)*100,
        'no_pattern': total_steps - draft_starts, 'rejected': draft_total - accepted_total,
        'pattern_median': pt_sorted[mid_pt] * 1000 if pt_sorted else 0,
        'decode_median': dt_sorted[mid_dt] * 1000 if dt_sorted else 0,
        'verify_median': vt_sorted[mid_vt] * 1000 if vt_sorted else 0,
    }

def main():
    print("\n" + "="*80)
    print("  STRUCTSPEC — OLD vs NEW STATISTICAL COMPARISON")
    print("="*80)

    old = analyze(read_trace(OLD), "OLD TRACE (HumanEval 20 tasks, K=6, truncate)")
    new = analyze(read_trace(NEW), "NEW TRACE (DSA 5 prompts, K=6, truncate)")

    print(f"\n{'='*80}")
    print(f"  SIDE-BY-SIDE COMPARISON")
    print(f"{'='*80}")
    FMT = "  {:<35} {:>15} {:>15} {:>15}"
    print(FMT.format("METRIC", "OLD", "NEW", "DELTA"))
    print("  " + "-" * 77)

    metrics = [
        ("Total steps", "steps", "d"),
        ("Draft tokens proposed", "draft_total", "d"),
        ("Draft tokens accepted", "accepted", "d"),
        ("Draft tokens rejected", "rejected", "d"),
        ("Accept rate (%)", "accept_rate", ".1f"),
        ("Fire rate (%)", "fire_rate", ".1f"),
        ("No-pattern steps", "no_pattern", "d"),
        ("Pattern time total (s)", "pattern_total", ".4f"),
        ("Pattern time median (ms)", "pattern_median", ".3f"),
        ("Decode time total (s)", "decode_total", ".3f"),
        ("Decode time median (ms)", "decode_median", ".1f"),
        ("Verify time total (s)", "verify_total", ".4f"),
        ("Verify time median (ms)", "verify_median", ".3f"),
    ]

    for label, key, fmt in metrics:
        o = old.get(key, 0) or 0
        n = new.get(key, 0) or 0
        d = n - o
        d_str = f"{d:+{fmt}}" if fmt.endswith("f") else f"{d:+d}"
        print(FMT.format(label[:35], f"{o:{fmt}}", f"{n:{fmt}}", d_str))

    # Save CSV
    rows = []
    for label, key, fmt in metrics:
        o = old.get(key, 0) or 0
        n = new.get(key, 0) or 0
        d = n - o
        rows.append({"metric": label, "old": f"{o:{fmt}}", "new": f"{n:{fmt}}", "delta": f"{d:+{fmt}}" if fmt.endswith("f") else f"{d:+d}"})

    csv_path = "stat_compare_report.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "old", "new", "delta"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n  CSV report saved: {csv_path}")
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
