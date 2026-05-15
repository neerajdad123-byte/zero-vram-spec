"""Baseline: Standard greedy decoding (no speculative decoding)."""
import sys, os, time
sys.stderr = sys.stdout

os.environ["PATH"] = r"C:\Users\neera\.lmstudio\extensions\backends\vendor\win-llama-cuda12-vendor-v2" + os.pathsep + os.environ.get("PATH", "")

from structspec.llama_backend import FastGreedyLlama, run_greedy

MODEL = r"C:\Users\neera\.lmstudio\models\lmstudio-community\Qwen2.5-Coder-7B-Instruct-GGUF\Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
PROMPT = "write function to check given number prime or not in python only code no comments"
MAX_TOKENS = 128

print("=" * 60)
print("BASELINE: Standard Greedy Decoding (No Speculation)")
print("=" * 60)
print(f"\nPROMPT: {PROMPT}\n")
print("-" * 60)

# Load model
fgl = FastGreedyLlama(MODEL, n_ctx=2048, n_gpu_layers=-1)
print("Model loaded.\n")

# Run greedy decoding
t0 = time.perf_counter()

gen_tokens, output_text, stats = run_greedy(fgl, PROMPT, MAX_TOKENS)

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
print("=" * 60)
