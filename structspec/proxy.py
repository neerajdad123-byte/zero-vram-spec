from __future__ import annotations

import json
import os
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .domains import detect_domain, rules_for_domain
from .metrics import metrics

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


def _headers(request: Request, api_key: str | None) -> dict[str, str]:
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    return headers


def _count_output_tokens_rough(body: bytes) -> int:
    try:
        data = json.loads(body)
    except Exception:
        return 0
    text = ""
    for choice in data.get("choices", []):
        if "message" in choice:
            text += str(choice.get("message", {}).get("content", ""))
        else:
            text += str(choice.get("text", ""))
    return max(0, len(text.split()))


def _request_prompt_text(body: bytes) -> str:
    try:
        data = json.loads(body)
    except Exception:
        return ""
    if "prompt" in data:
        return str(data.get("prompt") or "")
    messages = data.get("messages") or []
    return "\n".join(str(msg.get("content", "")) for msg in messages if isinstance(msg, dict))


def create_app(
    target_base_url: str,
    backend: str = "auto",
    api_key: str | None = None,
    safety: str = "strict",
) -> FastAPI:
    app = FastAPI(title="Structspec", version="0.1.0")
    target = target_base_url.rstrip("/")
    metrics.set_backend(backend, "proxy")

    async def forward(request: Request, path: str) -> Response:
        body = await request.body()
        metrics.inc("requests_total")
        metrics.inc("requests_active")
        metrics.inc("bytes_in", len(body))
        headers = _headers(request, api_key)
        url = f"{target}{path}"
        try:
            is_stream = bool(json.loads(body).get("stream", False))
        except (json.JSONDecodeError, UnicodeDecodeError):
            is_stream = False
        domain = detect_domain(_request_prompt_text(body))
        request.state.structspec_domain = domain
        metrics.inc_domain(domain)

        if is_stream:
            client = httpx.AsyncClient(timeout=None)
            try:
                req = client.build_request(request.method, url, headers=headers, content=body)
                upstream = await client.send(req, stream=True)
            except Exception as exc:
                metrics.inc("requests_failed")
                metrics.inc("requests_active", -1)
                await client.aclose()
                return JSONResponse({"error": {"message": str(exc), "type": "structspec_proxy_error"}}, status_code=502)

            async def chunks() -> AsyncIterator[bytes]:
                try:
                    async for chunk in upstream.aiter_bytes():
                        metrics.inc("streamed_chunks")
                        metrics.inc("bytes_out", len(chunk))
                        yield chunk
                finally:
                    await upstream.aclose()
                    await client.aclose()
                    metrics.inc("requests_active", -1)

            return StreamingResponse(
                chunks(),
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "text/event-stream"),
            )

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                upstream = await client.request(request.method, url, headers=headers, content=body)
                metrics.inc("bytes_out", len(upstream.content))
                metrics.inc("target_tokens", _count_output_tokens_rough(upstream.content))
                metrics.inc("requests_active", -1)
                return Response(
                    content=upstream.content,
                    status_code=upstream.status_code,
                    media_type=upstream.headers.get("content-type", "application/json"),
                )
            except Exception as exc:
                metrics.inc("requests_failed")
                metrics.inc("requests_active", -1)
                return JSONResponse({"error": {"message": str(exc), "type": "structspec_proxy_error"}}, status_code=502)

    @app.get("/health")
    async def health() -> dict:
        snap = metrics.snapshot()
        snap["status"] = "ok"
        snap["backend"] = backend
        snap["target_base_url"] = target
        snap["safety"] = safety
        snap["domains"] = {
            name: rules_for_domain(name)
            for name in ("python", "json", "html", "sql", "go", "generic")
        }
        return snap

    @app.get("/v1/structspec/metrics")
    async def prometheus() -> Response:
        return Response(metrics.prometheus(), media_type="text/plain")

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        return await forward(request, "/models")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await forward(request, "/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await forward(request, "/completions")

    @app.get("/")
    async def root() -> dict:
        return {"name": "structspec", "health": "/health", "metrics": "/v1/structspec/metrics"}

    return app


def infer_target_base_url(backend: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit.rstrip("/")
    env = os.environ.get("STRUCTSPEC_TARGET_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if env and "localhost:8080" not in env:
        return env.rstrip("/")
    defaults = {
        "vllm": "http://localhost:8000/v1",
        "lm-studio": "http://localhost:1234/v1",
        "ollama": "http://localhost:11434/v1",
        "llama-cpp": "http://localhost:8000/v1",
        "auto": "http://localhost:8000/v1",
    }
    return defaults.get(backend, "http://localhost:8000/v1")
