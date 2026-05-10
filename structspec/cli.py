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
