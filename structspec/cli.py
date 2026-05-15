from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from .detect import detect_all, format_report
from .proxy import create_app, infer_target_base_url
from .status import print_status_loop, run_textual_status
from .vllm_integration import build_vllm_command, shell_join


def cmd_detect(args: argparse.Namespace) -> int:
    report = detect_all(args.model_path)
    if args.json:
        print(json.dumps(report, default=lambda obj: getattr(obj, "__dict__", str(obj)), indent=2))
    else:
        print(format_report(report))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    backend = args.backend
    if backend == "auto":
        report = detect_all(args.model_path)
        backend = report.selected_backend or "vllm"
    target = infer_target_base_url(backend, args.target_base_url)
    api_key = args.target_api_key or os.environ.get("STRUCTSPEC_TARGET_API_KEY")
    app = create_app(target_base_url=target, backend=backend, api_key=api_key, safety=args.safety)
    print(f"Structspec proxy listening on http://{args.host}:{args.port}/v1")
    print(f"Forwarding to {target} | backend={backend} | safety={args.safety}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    if args.tui and not args.once:
        run_textual_status(args.base_url, interval=args.interval)
    else:
        print_status_loop(args.base_url, once=args.once, interval=args.interval)
    return 0


def cmd_vllm_command(args: argparse.Namespace) -> int:
    cmd = build_vllm_command(
        model=args.model,
        host=args.host,
        port=args.port,
        method=args.method,
        num_speculative_tokens=args.num_speculative_tokens,
        prompt_lookup_min=args.prompt_lookup_min,
        prompt_lookup_max=args.prompt_lookup_max,
        draft_model=args.draft_model,
        extra_args=args.extra_args,
    )
    print(shell_join(cmd))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    try:
        from openai import OpenAI
    except ImportError:
        print("Install openai for `structspec run`: pip install openai", file=sys.stderr)
        return 2
    client = OpenAI(base_url=args.base_url.rstrip("/") + "/v1", api_key=args.api_key)
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        stream=False,
    )
    print(response.choices[0].message.content)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    script = "run_humaneval_benchmark.py" if args.dataset == "humaneval" else None
    if script is None or not os.path.exists(script):
        print("Benchmark runner not found for dataset:", args.dataset, file=sys.stderr)
        return 2
    cmd = [sys.executable, script]
    if args.extra_args:
        cmd.extend(args.extra_args)
    return subprocess.call(cmd)


def cmd_generate(args: argparse.Namespace) -> int:
    try:
        from .engine import PatternMiner, PythonSyntaxProposer, QwenTokenCorpus
        from .llama_backend import FastGreedyLlama, run_greedy, run_speculative
    except ImportError as exc:
        print(f"Missing dependency for generation: {exc}", file=sys.stderr)
        print("Install with: pip install structspec[research]", file=sys.stderr)
        return 2

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}", file=sys.stderr)
        return 2
    if not os.path.exists(args.token_json):
        print(f"Token JSON not found: {args.token_json}", file=sys.stderr)
        return 2

    corpus = QwenTokenCorpus(args.token_json)
    miner = PatternMiner(
        corpus.token_text,
        max_ctx=args.max_ctx,
        min_support=args.min_support,
        min_conf=args.min_conf,
        det_conf=args.det_conf,
        min_rule_ctx=args.min_rule_ctx,
    ).fit(corpus.sequences)

    print(f"Loading model: {args.model}")
    model = FastGreedyLlama(args.model, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers)

    syntax = None if args.no_syntax_patterns else PythonSyntaxProposer(model, mode=args.syntax_mode)

    if args.greedy:
        ids, text, stats = run_greedy(model, args.prompt, args.max_tokens)
    else:
        ids, text, stats = run_speculative(
            model, miner, syntax, args.prompt, args.max_tokens,
            k=args.k, reject_mode=args.reject_mode,
            strike_limit=args.strike_limit,
            live_viz=args.live_viz,
        )

    print(text)
    if args.verbose:
        print(f"\npasses={stats['passes']} time={stats['time']:.3f}s", file=sys.stderr)
    return 0


def cmd_config(_: argparse.Namespace) -> int:
    print("Structspec config wizard")
    print("1. Start your backend, for example vLLM on :8000.")
    print("2. Run: structspec serve --backend vllm --target-base-url http://localhost:8000/v1")
    print("3. Point tools at: OPENAI_BASE_URL=http://localhost:8080/v1")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="structspec")
    sub = parser.add_subparsers(dest="command", required=True)

    detect = sub.add_parser("detect", help="Scan local LLM environment.")
    detect.add_argument("--model-path", action="append")
    detect.add_argument("--json", action="store_true")
    detect.set_defaults(func=cmd_detect)

    serve = sub.add_parser("serve", help="Start OpenAI-compatible proxy.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--backend", default="auto", choices=["auto", "vllm", "llama-cpp", "lm-studio", "ollama"])
    serve.add_argument("--target-base-url")
    serve.add_argument("--target-api-key")
    serve.add_argument("--model-path", action="append")
    serve.add_argument("--safety", default="strict", choices=["strict", "fast", "audit"])
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=cmd_serve)

    run = sub.add_parser("run", help="One-shot generation through a Structspec-compatible endpoint.")
    run.add_argument("--base-url", default="http://localhost:8080")
    run.add_argument("--api-key", default="dummy")
    run.add_argument("--model", default="default")
    run.add_argument("--prompt", required=True)
    run.set_defaults(func=cmd_run)

    bench = sub.add_parser("benchmark", help="Run benchmark helper.")
    bench.add_argument("--dataset", default="humaneval", choices=["humaneval"])
    bench.add_argument("extra_args", nargs=argparse.REMAINDER)
    bench.set_defaults(func=cmd_benchmark)

    status = sub.add_parser("status", help="Attach to live status view.")
    status.add_argument("--base-url", default="http://localhost:8080")
    status.add_argument("--once", action="store_true")
    status.add_argument("--interval", type=float, default=1.0)
    status.add_argument("--tui", action="store_true")
    status.set_defaults(func=cmd_status)

    config = sub.add_parser("config", help="Print setup wizard.")
    config.set_defaults(func=cmd_config)

    gen = sub.add_parser("generate", help="Generate text with zero-VRAM speculative decoding (local model).")
    gen.add_argument("--model", required=True, help="Path to GGUF model.")
    gen.add_argument("--token-json", required=True, help="Path to token corpus JSON.")
    gen.add_argument("--prompt", required=True, help="Generation prompt.")
    gen.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate.")
    gen.add_argument("--k", type=int, default=6, help="Max draft length.")
    gen.add_argument("--max-ctx", type=int, default=8)
    gen.add_argument("--min-support", type=int, default=2)
    gen.add_argument("--min-conf", type=float, default=0.96)
    gen.add_argument("--det-conf", type=float, default=0.96)
    gen.add_argument("--min-rule-ctx", type=int, default=4)
    gen.add_argument("--n-ctx", type=int, default=2048)
    gen.add_argument("--n-gpu-layers", type=int, default=-1)
    gen.add_argument("--reject-mode", choices=["truncate", "seq-bonus", "rebuild"], default="truncate")
    gen.add_argument("--strike-limit", type=int, default=3)
    gen.add_argument("--no-syntax-patterns", action="store_true")
    gen.add_argument("--syntax-mode", choices=["off", "basic", "cluster"], default="cluster")
    gen.add_argument("--live-viz", action="store_true")
    gen.add_argument("--greedy", action="store_true", help="Run greedy baseline instead of speculative.")
    gen.add_argument("--verbose", action="store_true")
    gen.set_defaults(func=cmd_generate)

    vllm_cmd = sub.add_parser("vllm-command", help="Print a vLLM speculative decoding launch command.")
    vllm_cmd.add_argument("--model", required=True)
    vllm_cmd.add_argument("--host", default="0.0.0.0")
    vllm_cmd.add_argument("--port", type=int, default=8000)
    vllm_cmd.add_argument("--method", default="ngram", choices=["ngram", "suffix", "draft_model", "mtp"])
    vllm_cmd.add_argument("--num-speculative-tokens", type=int, default=4)
    vllm_cmd.add_argument("--prompt-lookup-min", type=int, default=2)
    vllm_cmd.add_argument("--prompt-lookup-max", type=int, default=5)
    vllm_cmd.add_argument("--draft-model")
    vllm_cmd.add_argument("extra_args", nargs=argparse.REMAINDER)
    vllm_cmd.set_defaults(func=cmd_vllm_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
