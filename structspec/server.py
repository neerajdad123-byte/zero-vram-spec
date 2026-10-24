"""In-process OpenAI-compatible server over the speculative engine (greedy, lossless)."""

from __future__ import annotations

import codecs
import json
import threading
import time
import uuid
from typing import Iterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .llama_backend import SpecStats, chat_prompt_ids, speculate
from .metrics import MetricsStore

DEFAULT_MAX_TOKENS = 256


class BadRequest(ValueError):
    pass


def _error(msg: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": {"message": msg, "type": "invalid_request_error"}}, status_code=status)


def _stops(body: dict) -> list[str]:
    stop = body.get("stop") or []
    stops = [stop] if isinstance(stop, str) else stop
    if not isinstance(stops, list) or not all(isinstance(s, str) for s in stops):
        raise BadRequest("'stop' must be a string or a list of strings")
    return [s for s in stops if s]


def _held_back(text: str, stops: list[str]) -> int:
    """Length of the longest suffix of `text` that could still grow into a stop string."""
    best = 0
    for s in stops:
        for n in range(min(len(s) - 1, len(text)), best, -1):
            if text.endswith(s[:n]):
                best = n
                break
    return best


def create_spec_app(model, proposer, k: int = 4, model_name: str = "structspec", thinking: bool = False) -> FastAPI:
    app = FastAPI(title="Structspec", version="0.2.0")
    lock = threading.Lock()  # one llama context: requests run one at a time
    metrics = MetricsStore()
    metrics.set_backend("structspec-native", model_name)

    def prepare(body: dict, prompt_ids: list[int]) -> int:
        max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
        if max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise BadRequest("'max_tokens' must be a positive integer")
        room = model.n_ctx - len(prompt_ids)
        if room < 1:
            raise BadRequest(f"prompt is {len(prompt_ids)} tokens; the context window is {model.n_ctx}")
        return min(max_tokens, room)

    def generate(prompt_ids: list[int], max_tokens: int, stops: list[str], out: dict) -> Iterator[str]:
        with lock:
            stats = SpecStats()
            decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
            pending = ""
            finish = "length"
            metrics.inc("requests_total")
            metrics.inc("requests_active")
            try:
                for chunk in speculate(model, proposer, prompt_ids, max_tokens, k, stats=stats):
                    for tok in chunk:
                        if model.is_eog(tok):
                            finish = "stop"
                        else:
                            pending += decoder.decode(model.piece_bytes(tok))
                    hits = [pending.find(s) for s in stops if s in pending]
                    if hits:
                        pending, finish = pending[: min(hits)], "stop"
                        break
                    keep = _held_back(pending, stops)
                    if len(pending) > keep:
                        yield pending[: len(pending) - keep]
                        pending = pending[len(pending) - keep :]
                else:
                    pending += decoder.decode(b"", final=True)
                if pending:
                    yield pending
            finally:
                metrics.inc("requests_active", -1)
                metrics.inc("generated_tokens", stats.generated)
                metrics.inc("target_passes", stats.passes)
                metrics.inc("proposed_draft_tokens", stats.proposed)
                metrics.inc("accepted_draft_tokens", stats.accepted)
                metrics.inc("rejected_draft_tokens", stats.proposed - stats.accepted)
                out.update(finish_reason=finish, prompt_tokens=len(prompt_ids), completion_tokens=stats.generated)

    def usage(out: dict) -> dict:
        return {
            "prompt_tokens": out["prompt_tokens"],
            "completion_tokens": out["completion_tokens"],
            "total_tokens": out["prompt_tokens"] + out["completion_tokens"],
        }

    def respond(body: dict, prompt_ids: list[int], chat: bool) -> Response:
        stops = _stops(body)
        max_tokens = prepare(body, prompt_ids)
        out: dict = {}
        rid = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex}"
        created = int(time.time())
        obj = "chat.completion" if chat else "text_completion"

        if not body.get("stream"):
            text = "".join(generate(prompt_ids, max_tokens, stops, out))
            choice = {"index": 0, "finish_reason": out["finish_reason"]}
            if chat:
                choice["message"] = {"role": "assistant", "content": text}
            else:
                choice.update(text=text, logprobs=None)
            return JSONResponse(
                {"id": rid, "object": obj, "created": created, "model": model_name,
                 "choices": [choice], "usage": usage(out)}
            )

        def event(delta: dict | str | None, finish: str | None) -> str:
            if chat:
                choice = {"index": 0, "delta": delta or {}, "finish_reason": finish}
            else:
                choice = {"index": 0, "text": delta or "", "logprobs": None, "finish_reason": finish}
            chunk = {"id": rid, "object": obj + (".chunk" if chat else ""), "created": created,
                     "model": model_name, "choices": [choice]}
            if finish:
                chunk["usage"] = usage(out)
            return f"data: {json.dumps(chunk)}\n\n"

        def sse() -> Iterator[str]:
            if chat:
                yield event({"role": "assistant"}, None)
            for piece in generate(prompt_ids, max_tokens, stops, out):
                yield event({"content": piece} if chat else piece, None)
            yield event(None, out["finish_reason"])
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    async def handle(request: Request, chat: bool) -> Response:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise BadRequest("request body must be a JSON object")
            if chat:
                messages = body.get("messages")
                if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
                    raise BadRequest("'messages' must be a non-empty list of objects")
                prompt_ids = chat_prompt_ids(model, messages, thinking=thinking)
            else:
                prompt = body.get("prompt")
                if isinstance(prompt, list) and len(prompt) == 1:
                    prompt = prompt[0]
                if not isinstance(prompt, str) or not prompt:
                    raise BadRequest("'prompt' must be a non-empty string")
                prompt_ids = model.encode(prompt, special=True)
            if body.get("n", 1) != 1:
                raise BadRequest("only n=1 is supported")
            return respond(body, prompt_ids, chat)
        except (BadRequest, json.JSONDecodeError) as exc:
            return _error(str(exc))

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await handle(request, chat=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await handle(request, chat=True)

    @app.get("/v1/models")
    async def models() -> dict:
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "structspec"}]}

    @app.get("/health")
    async def health() -> dict:
        snap = metrics.snapshot()
        snap.update(status="ok", mode="speculative", decoding="greedy", backend="structspec-native",
                    model=model_name, proposer=getattr(proposer, "name", "?"), k=k)
        return snap

    @app.get("/v1/structspec/metrics")
    async def prometheus() -> Response:
        return Response(metrics.prometheus(), media_type="text/plain")

    @app.get("/")
    async def root() -> dict:
        return {"name": "structspec", "health": "/health", "metrics": "/v1/structspec/metrics"}

    return app
