from __future__ import annotations

import argparse
import json
import shlex


def speculative_config(
    method: str = "ngram",
    num_speculative_tokens: int = 4,
    prompt_lookup_min: int = 2,
    prompt_lookup_max: int = 5,
    draft_model: str | None = None,
) -> dict:
    cfg: dict[str, object] = {
        "method": method,
        "num_speculative_tokens": num_speculative_tokens,
    }
    if method == "ngram":
        cfg["prompt_lookup_min"] = prompt_lookup_min
        cfg["prompt_lookup_max"] = prompt_lookup_max
    elif method == "suffix":
        cfg["suffix_decoding_max_tree_depth"] = max(8, num_speculative_tokens * 3)
        cfg["suffix_decoding_max_cached_requests"] = 10000
        cfg["suffix_decoding_max_spec_factor"] = 1.0
    elif method == "draft_model":
        if not draft_model:
            raise ValueError("--draft-model is required for method=draft_model")
        cfg["model"] = draft_model
    return cfg


def build_vllm_command(
    model: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    method: str = "ngram",
    num_speculative_tokens: int = 4,
    prompt_lookup_min: int = 2,
    prompt_lookup_max: int = 5,
    draft_model: str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    cfg = speculative_config(
        method=method,
        num_speculative_tokens=num_speculative_tokens,
        prompt_lookup_min=prompt_lookup_min,
        prompt_lookup_max=prompt_lookup_max,
        draft_model=draft_model,
    )
    cmd = [
        "vllm",
        "serve",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--speculative-config",
        json.dumps(cfg, separators=(",", ":")),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


class StructSpecDraftWorker:
    """Extension point for future in-process vLLM rule drafting.

    vLLM exposes stable user-facing speculative decoding through
    --speculative-config. A pure Python rule-draft worker needs a stable
    internal draft-runner hook from vLLM before it can be reliable across
    releases, so this class deliberately remains a narrow protocol boundary.
    """

    def propose(self, token_ids: list[int], max_tokens: int) -> list[int]:
        raise NotImplementedError("Use vLLM ngram/suffix/draft_model configs until the plugin hook is stable.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a vLLM command with speculative decoding enabled.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--method", default="ngram", choices=["ngram", "suffix", "draft_model", "mtp"])
    parser.add_argument("--num-speculative-tokens", type=int, default=4)
    parser.add_argument("--prompt-lookup-min", type=int, default=2)
    parser.add_argument("--prompt-lookup-max", type=int, default=5)
    parser.add_argument("--draft-model")
    args, extra = parser.parse_known_args(argv)
    cmd = build_vllm_command(
        model=args.model,
        host=args.host,
        port=args.port,
        method=args.method,
        num_speculative_tokens=args.num_speculative_tokens,
        prompt_lookup_min=args.prompt_lookup_min,
        prompt_lookup_max=args.prompt_lookup_max,
        draft_model=args.draft_model,
        extra_args=extra,
    )
    print(shell_join(cmd))


if __name__ == "__main__":
    main()
