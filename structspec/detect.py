from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

KEY_NAMES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN", "VLLM_API_KEY")
MODEL_SUFFIXES = (".gguf", ".safetensors", ".bin")


@dataclass
class DetectionReport:
    env_files: list[str] = field(default_factory=list)
    keys: dict[str, str] = field(default_factory=dict)
    llama_cpp: dict = field(default_factory=dict)
    vllm: dict = field(default_factory=dict)
    lm_studio: dict = field(default_factory=dict)
    ollama: dict = field(default_factory=dict)
    models: list[dict] = field(default_factory=list)
    gpus: list[str] = field(default_factory=list)
    selected_backend: str | None = None


def _run(cmd: list[str], timeout: int = 5) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout or proc.stderr).strip()
    except Exception as exc:
        return 1, str(exc)


def _http_json(url: str, timeout: float = 0.8) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


def _port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def find_env_files(start: Path | None = None) -> list[Path]:
    cur = (start or Path.cwd()).resolve()
    out = []
    for parent in [cur, *cur.parents]:
        for name in (".env", ".env.local", ".structspec.env"):
            p = parent / name
            if p.exists():
                out.append(p)
    return out


def load_env_keys(paths: list[Path]) -> dict[str, str]:
    keys = {name: "environment" for name in KEY_NAMES if os.environ.get(name)}
    for path in paths:
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if "=" not in line or line.lstrip().startswith("#"):
                    continue
                name, value = line.split("=", 1)
                name = name.strip()
                if name in KEY_NAMES and value.strip():
                    keys.setdefault(name, str(path))
        except OSError:
            continue
    return keys


def detect_gpus() -> list[str]:
    code, out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], timeout=4)
    if code == 0 and out:
        return [line.strip() for line in out.splitlines() if line.strip()]
    return []


def detect_vllm() -> dict:
    code, out = _run([sys.executable, "-c", "import vllm; print(vllm.__version__)"], timeout=8)
    exe = shutil.which("vllm")
    return {
        "installed": code == 0,
        "version": out if code == 0 else None,
        "executable": exe,
    }


def detect_llama_cpp() -> dict:
    candidates = ["llama-server", "llama-cli", "main"]
    found = []
    for name in candidates:
        path = shutil.which(name)
        if path and name == "main" and Path(path).suffix.lower() not in ("", ".exe"):
            path = None
        if path:
            code, out = _run([path, "--version"], timeout=4)
            found.append({"name": name, "path": path, "version": out if code == 0 else None})
    return {"found": bool(found), "binaries": found}


def detect_lm_studio() -> dict:
    hits = []
    for port in (1234, 8080, 5000):
        data = _http_json(f"http://localhost:{port}/v1/models")
        if data is not None:
            hits.append({"base_url": f"http://localhost:{port}/v1", "models": data.get("data", [])})
    return {"running": bool(hits), "servers": hits}


def detect_ollama() -> dict:
    exe = shutil.which("ollama")
    models = []
    code, out = _run(["ollama", "list"], timeout=5) if exe else (1, "")
    if code == 0:
        models = [line for line in out.splitlines()[1:] if line.strip()]
    return {
        "installed": bool(exe),
        "executable": exe,
        "running": _port_open("localhost", 11434),
        "models": models,
    }


def scan_models(paths: list[str] | None = None) -> list[dict]:
    roots = [Path("~/models").expanduser(), Path.cwd() / "models"]
    if paths:
        roots.extend(Path(p).expanduser() for p in paths)
    out = []
    seen = set()
    for root in roots:
        if not root.exists() or root in seen:
            continue
        seen.add(root)
        for suffix in MODEL_SUFFIXES:
            for path in root.rglob(f"*{suffix}"):
                out.append({"path": str(path), "format": suffix.lstrip("."), "name": path.name})
    return out[:200]


def select_backend(report: DetectionReport) -> str | None:
    if report.vllm.get("installed") and report.gpus:
        return "vllm"
    if report.llama_cpp.get("found"):
        return "llama-cpp"
    if report.ollama.get("running"):
        return "ollama"
    if report.lm_studio.get("running"):
        return "lm-studio"
    return None


def detect_all(model_paths: list[str] | None = None) -> DetectionReport:
    env_files = find_env_files()
    report = DetectionReport()
    report.env_files = [str(p) for p in env_files]
    report.keys = load_env_keys(env_files)
    report.gpus = detect_gpus()
    report.vllm = detect_vllm()
    report.llama_cpp = detect_llama_cpp()
    report.lm_studio = detect_lm_studio()
    report.ollama = detect_ollama()
    report.models = scan_models(model_paths)
    report.selected_backend = select_backend(report)
    return report


def format_report(report: DetectionReport) -> str:
    lines = ["Structspec detection report", ""]
    lines.append(f"Selected backend: {report.selected_backend or 'none'}")
    lines.append(f"GPU(s): {', '.join(report.gpus) if report.gpus else 'none detected'}")
    lines.append(f"Env files: {', '.join(report.env_files) if report.env_files else 'none'}")
    lines.append(f"Keys: {', '.join(report.keys) if report.keys else 'none'}")
    lines.append(f"vLLM: {'yes ' + str(report.vllm.get('version')) if report.vllm.get('installed') else 'not found'}")
    lines.append(f"llama.cpp: {'yes' if report.llama_cpp.get('found') else 'not found'}")
    lines.append(f"LM Studio: {'running' if report.lm_studio.get('running') else 'not running'}")
    lines.append(f"Ollama: {'running' if report.ollama.get('running') else 'not running'}")
    lines.append(f"Models found: {len(report.models)}")
    for model in report.models[:12]:
        lines.append(f"  - {model['name']} ({model['format']})")
    return "\n".join(lines)
