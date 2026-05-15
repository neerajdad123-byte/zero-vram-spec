"""Speculative: Structspec pattern-based speculative decoding."""
import sys, os, time, json
sys.stderr = sys.stdout

os.environ["PATH"] = r"C:\Users\neera\.lmstudio\extensions\backends\vendor\win-llama-cuda12-vendor-v2" + os.pathsep + os.environ.get("PATH", "")

from structspec.engine import PatternMiner, PythonSyntaxProposer
from structspec.llama_backend import FastGreedyLlama, run_speculative

MODEL = r"C:\Users\neera\.lmstudio\models\lmstudio-community\Qwen2.5-Coder-7B-Instruct-GGUF\Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
TOKEN_JSON = r"C:\Users\neera\OneDrive\Desktop\1\zero-vram-spec\qwen_coder_dsa_tokens.json"
PROMPT = "write function to check given number prime or not in python only code no comments"
MAX_TOKENS = 128
K = 6

print("=" * 60)
print("STRUCTSPEC: Pattern-Based Speculative Decoding")
print("=" * 60)
print(f"\nPROMPT: {PROMPT}\n")
print("-" * 60)

# Load token corpus
with open(TOKEN_JSON, encoding="utf-8") as f:
    corpus_data = json.load(f)

token_text = {}
sequences = []
for item in corpus_data.values():
    ids = [int(t["id"]) for t in item["tokens"]]
    for t in item["tokens"]:
        token_text[int(t["id"])] = str(t["token"])
    sequences.append(ids)

print(f"Corpus: {len(sequences)} examples, {len(token_text)} unique tokens")

# Build pattern miner
miner = PatternMiner(
    token_text=token_text,
    max_ctx=8,
    min_support=2,
    min_conf=0.96,
    det_conf=0.96,
    min_rule_ctx=4,
)
miner.fit(sequences)
print(f"Rules mined: {len(miner.rules_by_ctx)}")

# Load model
fgl = FastGreedyLlama(MODEL, n_ctx=2048, n_gpu_layers=-1)
syntax = PythonSyntaxProposer(fgl, mode="cluster")
print("Model loaded.\n")

# Run speculative decoding
t0 = time.perf_counter()

gen_tokens, output_text, stats = run_speculative(
    fgl,
    miner,
    syntax,
    PROMPT,
    MAX_TOKENS,
    k=K,
    reject_mode="truncate",
)

t1 = time.perf_counter()

elapsed = t1 - t0
tokens_generated = len(gen_tokens)
tok_per_sec = tokens_generated / elapsed if elapsed > 0 else 0

print(output_text)
print()
print("-" * 60)
print(f"Tokens generated : {tokens_generated}")
print(f"Total time       : {elapsed:.3f}s")
print(f"Tokens/sec       : {tok_per_sec:.2f}")
print(f"Draft proposed   : {stats.get('proposed', 0)}")
print(f"Draft accepted   : {stats.get('accepted', 0)}")
accept_rate = stats.get('accepted', 0) / max(1, stats.get('proposed', 1))
print(f"Accept rate      : {accept_rate*100:.1f}%")
print(f"Model passes     : {stats.get('passes', 0)} (baseline would need {stats.get('target_tokens_evaluated', 0)})")
print(f"Pass reduction   : {stats.get('target_tokens_evaluated', 0) / max(1, stats.get('passes', 1)):.2f}x")
print("=" * 60)
