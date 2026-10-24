from __future__ import annotations

import argparse
import json
import os
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

    if args.passthrough:
        backend = args.backend
        if backend == "auto":
            report = detect_all(args.model_path)
            backend = report.selected_backend or "vllm"
        target = infer_target_base_url(backend, args.target_base_url)
        api_key = args.target_api_key or os.environ.get("STRUCTSPEC_TARGET_API_KEY")
        app = create_app(target_base_url=target, backend=backend, api_key=api_key, safety=args.safety)
        print(f"Structspec passthrough proxy on http://{args.host}:{args.port}/v1 -> {target} (no speculation)")
    else:
        if not args.model:
            print("serve needs --model PATH.gguf (or --passthrough to proxy another server)", file=sys.stderr)
            return 2
        from .proposer import default_proposer
        from .server import create_spec_app

        model = _load_model(args)
        proposer = default_proposer(model, args.index, syntax=args.syntax)
        name = os.path.splitext(os.path.basename(args.model))[0]
        app = create_spec_app(model, proposer, k=args.k, model_name=name, thinking=args.thinking)
        print(f"Structspec serving {name} on http://{args.host}:{args.port}/v1 (greedy, speculative k={args.k})")
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


def _load_model(args: argparse.Namespace, vocab_only: bool = False):
    from .llama_native import NativeLlama

    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found: {args.model}")
    if vocab_only:
        return NativeLlama(args.model, vocab_only=True)
    return NativeLlama(
        args.model, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers, cpu_moe=args.cpu_moe, n_threads=args.threads
    )


def cmd_generate(args: argparse.Namespace) -> int:
    from .llama_backend import run_greedy, run_speculative
    from .proposer import default_proposer

    model = _load_model(args)
    if args.greedy:
        ids, text, stats = run_greedy(model, args.prompt, args.max_tokens)
    else:
        proposer = default_proposer(model, args.index, syntax=args.syntax)
        ids, text, stats = run_speculative(
            model, proposer, args.prompt, args.max_tokens, k=args.k, reject_mode=args.reject_mode
        )
    print(text)
    if args.verbose:
        print(f"\npasses={stats['passes']} time={stats['time']:.3f}s", file=sys.stderr)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from .llama_backend import chat_prompt_ids
    from .proposer import default_proposer
    from .verify import DEFAULT_PROMPTS, verify_prompt

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        with open(args.prompts, encoding="utf-8") as f:
            prompts = [json.loads(line)["prompt"] for line in f if line.strip()]
    model = _load_model(args)
    proposer = default_proposer(model, args.index)
    results = []
    for i, prompt in enumerate(prompts):
        ids = chat_prompt_ids(model, [{"role": "user", "content": prompt}])
        r = verify_prompt(model, proposer, ids, args.max_tokens, k=args.k)
        results.append(r)
        print(
            f"[{i}] {r['classification']:<9} tokens={r['tokens']} accepted={r['accepted']}/{r['proposed']} "
            f"first_mismatch={r['first_mismatch']} greedy_margin={r['greedy_margin']}",
            flush=True,
        )
    same = sum(r["identical"] for r in results)
    print(f"identical: {same}/{len(results)}")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    return 0 if same == len(results) else 1


def cmd_index(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .proposer import build_index, default_index_path, save_index

    model = _load_model(args, vocab_only=True)
    index = build_index(model, [Path(p) for p in args.paths])
    out = Path(args.out) if args.out else default_index_path(index["tokenizer_hash"])
    save_index(index, out)
    print(f"Indexed {index['files']} files -> {len(index['rules'])} rules -> {out}")
    return 0


def cmd_config(_: argparse.Namespace) -> int:
    print("Structspec config wizard")
    print("1. Start your backend, for example vLLM on :8000.")
    print("2. Run: structspec serve --model path/to/model.gguf  (or --passthrough --target-base-url ...)")
    print("3. Point tools at: OPENAI_BASE_URL=http://localhost:8080/v1")
    return 0


def _model_args(p: argparse.ArgumentParser, required: bool = True) -> None:
    p.add_argument("--model", required=required, help="Path to GGUF model.")
    p.add_argument("--n-ctx", type=int, default=4096)
    p.add_argument("--n-gpu-layers", type=int, default=-1)
    p.add_argument("--cpu-moe", action="store_true", help="Keep MoE expert weights on the CPU.")
    p.add_argument("--threads", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="structspec")
    sub = parser.add_subparsers(dest="command", required=True)

    detect = sub.add_parser("detect", help="Scan local LLM environment.")
    detect.add_argument("--model-path", action="append")
    detect.add_argument("--json", action="store_true")
    detect.set_defaults(func=cmd_detect)

    serve = sub.add_parser("serve", help="OpenAI-compatible server with speculative decoding on a GGUF model.")
    _model_args(serve, required=False)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--k", type=int, default=4, help="Max draft length.")
    serve.add_argument("--index", help="N-gram index (default: cache file for this tokenizer, if present).")
    serve.add_argument("--syntax", action="store_true", help="Also use the Python syntax rules.")
    serve.add_argument("--thinking", action="store_true", help="Leave reasoning on for templates that support it.")
    serve.add_argument("--log-level", default="info")
    proxy = serve.add_argument_group("passthrough (old behaviour: forward to another server, no speculation)")
    proxy.add_argument("--passthrough", action="store_true")
    proxy.add_argument("--backend", default="auto", choices=["auto", "vllm", "llama-cpp", "lm-studio", "ollama"])
    proxy.add_argument("--target-base-url")
    proxy.add_argument("--target-api-key")
    proxy.add_argument("--model-path", action="append")
    proxy.add_argument("--safety", default="strict", choices=["strict", "fast", "audit"])
    serve.set_defaults(func=cmd_serve)

    run = sub.add_parser("run", help="One-shot generation through a Structspec-compatible endpoint.")
    run.add_argument("--base-url", default="http://localhost:8080")
    run.add_argument("--api-key", default="dummy")
    run.add_argument("--model", default="default")
    run.add_argument("--prompt", required=True)
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status", help="Attach to live status view.")
    status.add_argument("--base-url", default="http://localhost:8080")
    status.add_argument("--once", action="store_true")
    status.add_argument("--interval", type=float, default=1.0)
    status.add_argument("--tui", action="store_true")
    status.set_defaults(func=cmd_status)

    config = sub.add_parser("config", help="Print setup wizard.")
    config.set_defaults(func=cmd_config)

    gen = sub.add_parser("generate", help="Generate text with speculative decoding on a local GGUF model.")
    _model_args(gen)
    gen.add_argument("--prompt", required=True, help="Generation prompt.")
    gen.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate.")
    gen.add_argument("--k", type=int, default=4, help="Max draft length.")
    gen.add_argument("--index", help="N-gram index from `structspec index` (default: cache file for this tokenizer).")
    gen.add_argument("--syntax", action="store_true", help="Also use the Python syntax rules.")
    gen.add_argument("--reject-mode", choices=["truncate", "seq-bonus", "rebuild"], default="truncate")
    gen.add_argument("--greedy", action="store_true", help="Run greedy baseline instead of speculative.")
    gen.add_argument("--verbose", action="store_true")
    gen.set_defaults(func=cmd_generate)

    ver = sub.add_parser("verify", help="Check speculative output is identical to greedy on a local model.")
    _model_args(ver)
    ver.add_argument("--prompts", help="JSONL file with a 'prompt' field per line (default: built-in set).")
    ver.add_argument("--max-tokens", type=int, default=128)
    ver.add_argument("--k", type=int, default=4)
    ver.add_argument("--index")
    ver.add_argument("--json-out")
    ver.set_defaults(func=cmd_verify)

    index = sub.add_parser("index", help="Mine an n-gram draft index from source files with a model's tokenizer.")
    index.add_argument("paths", nargs="+", help="Files or directories to mine.")
    index.add_argument("--model", required=True, help="GGUF model whose tokenizer to use.")
    index.add_argument("--out", help="Output path (default: ~/.cache/structspec/index-<tokenizer>.json).")
    index.set_defaults(func=cmd_index)

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
