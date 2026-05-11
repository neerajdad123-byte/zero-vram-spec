"""
dsa_report.py — Complete DSA speedup report with token-level breakdown.
"""
import csv
from collections import Counter, defaultdict

TRACE = "dsa_trace_new.csv"

def read_trace(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

trace = read_trace(TRACE)

# ── Aggregate stats ──
spec_rows = [r for r in trace if r["kind"] == "spec_eval"]
total_steps = len(spec_rows)
draft_rows = [r for r in spec_rows if int(r.get("draft", 0)) > 0]
proposed = sum(int(r["draft"]) for r in spec_rows)
accepted = sum(int(r["accepted_draft"]) for r in spec_rows)
rejected = proposed - accepted
fire_rate = len(draft_rows) / max(1, total_steps) * 100
accept_rate = accepted / max(1, proposed) * 100

pat_time = sum(float(r.get("pattern_s", 0)) for r in spec_rows)
dec_time = sum(float(r.get("decode_s", 0)) for r in spec_rows)
ver_time = sum(float(r.get("verify_s", 0)) for r in spec_rows)

# ── Per-prompt stats ──
prompts = {}
for r in spec_rows:
    p = r.get("prompt", "?")[:50]
    if p not in prompts:
        prompts[p] = {"steps": 0, "drafts": 0, "proposed": 0, "accepted": 0, "rejected": 0, "decode": 0, "pattern": 0, "verify": 0}
    prompts[p]["steps"] += 1
    d = int(r.get("draft", 0))
    a = int(r.get("accepted_draft", 0))
    prompts[p]["drafts"] += 1 if d > 0 else 0
    prompts[p]["proposed"] += d
    prompts[p]["accepted"] += a
    prompts[p]["rejected"] += d - a
    prompts[p]["decode"] += float(r.get("decode_s", 0))
    prompts[p]["pattern"] += float(r.get("pattern_s", 0))
    prompts[p]["verify"] += float(r.get("verify_s", 0))

# ── Tier acceptance ──
tier_acc = defaultdict(lambda: [0, 0])
tier_tokens = defaultdict(list)
for r in draft_rows:
    tier = r.get("tier", "none")
    d = int(r["draft"])
    a = int(r["accepted_draft"])
    tier_acc[tier][0] += d
    tier_acc[tier][1] += a
    # Token IDs for analysis
    if r.get("draft_ids"):
        for did in r["draft_ids"].split():
            tier_tokens[tier].append(int(did))

# ── Token popularity ──
token_freq = Counter()
accepted_tokens = Counter()
rejected_tokens = Counter()
for r in draft_rows:
    draft_ids = [int(x) for x in r.get("draft_ids", "").split() if x]
    model_ids = [int(x) for x in r.get("model_draft_ids", "").split() if x]
    tier = r.get("tier", "none")
    for i, did in enumerate(draft_ids):
        token_freq[(tier, did)] += 1
        if i < int(r.get("accepted_draft", 0)):
            accepted_tokens[(tier, did)] += 1
        else:
            rejected_tokens[(tier, did)] += 1

# ── Print report ──
B = "\033[1m"
R = "\033[0m"
G = "\033[32m"
Y = "\033[33m"
C = "\033[36m"
red = "\033[31m"

print(f"\n{B}{'='*80}{R}")
print(f"{B}  DSA SPEEDUP REPORT — {len(prompts)} prompts, 128 tokens each{R}")
print(f"{B}{'='*80}{R}")

print(f"\n{B}AGGREGATE SPEED{R}")
print(f"  Total steps           : {total_steps}")
print(f"  Draft fire rate       : {len(draft_rows)}/{total_steps} = {G}{fire_rate:.1f}%{R}")
print(f"  Draft tokens proposed : {proposed}")
print(f"  Draft tokens accepted : {G}{accepted}{R} / rejected {red}{rejected}{R}")
print(f"  Accept rate           : {G}{accept_rate:.1f}%{R}")
print(f"  Wall speedup          : {C}1.131x{R} (from benchmark output)")

print(f"\n{B}TIMING BREAKDOWN{R}")
print(f"  {'Phase':<25} {'Total':>12} {'Per Step':>12} {'% of Total':>10}")
print(f"  {'-'*59}")
total_t = pat_time + dec_time + ver_time
for label, val in [("Pattern mining", pat_time), ("Model decode", dec_time), ("Verification", ver_time)]:
    pct = val / max(1e-9, total_t) * 100
    per_step = val / max(1, total_steps) * 1000
    print(f"  {label:<25} {val:>10.4f}s {per_step:>8.3f}ms {pct:>9.1f}%")

print(f"\n{B}PER-PROMPT BREAKDOWN{R}")
print(f"  {'Prompt':<50} {'Steps':>5} {'Drafts':>5} {'Prop':>5} {'Acc':>5} {'Acc%':>6}")
print(f"  {'-'*80}")
for prompt, s in prompts.items():
    rate = s["accepted"] / max(1, s["proposed"]) * 100
    print(f"  {prompt:<50} {s['steps']:>5} {s['drafts']:>5} {s['proposed']:>5} {s['accepted']:>5} {rate:>5.0f}%")

print(f"\n{B}TIER ACCEPTANCE{R}")
print(f"  {'Tier':<35} {'Prop':>5} {'Acc':>5} {'Rate':>7}")
print(f"  {'-'*55}")
for tier, (tot, acc) in sorted(tier_acc.items(), key=lambda x: -x[1][0]):
    rate = acc / max(1, tot) * 100
    bar = "#" * int(rate / 10) + "." * (10 - int(rate / 10))
    color = G if rate >= 80 else Y if rate >= 50 else red
    print(f"  {tier:<35} {color}{tot:>5} {acc:>5} {rate:>6.1f}%{R} {bar}")

print(f"\n{B}TOP PREDICTED TOKENS (by tier){R}")
print(f"  {'Tier':<30} {'Token ID':>8} {'Freq':>6} {'Acc':>6} {'Status'}")
print(f"  {'-'*60}")
for tier in sorted(tier_acc.keys(), key=lambda t: -tier_acc[t][0]):
    shown = 0
    for (t, tid), freq in sorted(token_freq.items(), key=lambda x: -x[1]):
        if t != tier or shown >= 3:
            continue
        acc = accepted_tokens.get((t, tid), 0)
        rate = acc / max(1, freq) * 100
        status = "ACCEPT" if rate >= 80 else "MIXED" if rate >= 50 else "REJECT"
        color = G if status == "ACCEPT" else Y if status == "MIXED" else red
        print(f"  {t:<30} {tid:>8} {freq:>6} {acc:>6} {color}{status}{R}")
        shown += 1

print(f"\n{B}{'='*80}{R}\n")
