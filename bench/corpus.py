"""Tokenizer artifacts for offline replay: vocab pieces (from the GGUF) and the
tokenized mining corpus (via a running llama-server's /tokenize). Cached in bench/data/."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).parent / "data"
MODEL = r"C:\models\gguf\Qwen3.6-35B-A3B-Q4_K_M.gguf"


def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode table."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


@lru_cache(maxsize=1)
def vocab_pieces(model: str = MODEL) -> list[str]:
    """id -> decoded text piece (partial UTF-8 bytes become U+FFFD)."""
    cache = DATA / "vocab_pieces.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    import gguf

    r = gguf.GGUFReader(model)
    f = r.fields["tokenizer.ggml.tokens"]
    types = r.fields["tokenizer.ggml.token_type"]
    dec = _byte_decoder()
    pieces = []
    for idx, i in enumerate(f.data):
        raw = bytes(f.parts[i]).decode("utf-8")
        if int(types.parts[types.data[idx]][0]) != 1:  # control/special tokens keep their literal text
            pieces.append(raw)
            continue
        pieces.append(bytes(dec.get(ch, 63) for ch in raw).decode("utf-8", errors="replace"))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(pieces), encoding="utf-8")
    return pieces


def mining_tokens(url: str = "http://127.0.0.1:8090") -> list[dict]:
    """[{"lang", "ids"}] for every mining-corpus file, tokenized by the model."""
    cache = DATA / "mining_tokens.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import httpx

    out = []
    with httpx.Client(base_url=url, timeout=120) as c:
        for line in (Path(__file__).parent / "mining_manifest.jsonl").read_text().splitlines():
            m = json.loads(line)
            text = (DATA / m["local"]).read_text(encoding="utf-8", errors="replace")
            out.append({"lang": m["lang"], "ids": c.post("/tokenize", json={"content": text}).json()["tokens"]})
    cache.write_text(json.dumps(out))
    return out
