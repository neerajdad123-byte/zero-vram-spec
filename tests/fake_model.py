"""Deterministic stand-in for NativeLlama: no GPU, same interface.

The next token is a hash of the model's own KV contents, so a stale or wrongly
rolled-back cache changes the output and the lossless tests catch it.
"""

from __future__ import annotations

import hashlib

import numpy as np

EOG = 0
# id -> bytes; ids 29/30 are the two halves of "é" so streaming must handle split UTF-8.
PIECES: list[bytes] = [b""] + [bytes([c]) for c in b"abcdefghijklmnopqrstuvwxyz"] + [b" ", b"\n", b"\xc3", b"\xa9"]
PIECES += [bytes([c]) for c in b"0123456789(){}:=+-.,"][: 50 - len(PIECES)]
_BY_CHAR = {p.decode(): i for i, p in enumerate(PIECES) if len(p) == 1 and p[0] < 128}


def next_token(prefix: list[int], vocab: int = len(PIECES)) -> int:
    h = 0
    for t in prefix:
        h = (h * 1_000_003 + t + 1) % 2_147_483_647
    return (h * 2_654_435_761 >> 7) % vocab


class FakeModel:
    def __init__(self, n_ctx: int = 4096, batch_noise=None):
        self.n_vocab = len(PIECES)
        self.eos = EOG
        self.n_ctx = n_ctx
        self.kv: list[int] = []
        self.calls = 0
        self.batch_noise = batch_noise  # optional fn(row_index, batch_len, row) -> row, to simulate numerics

    # tokenizer
    def encode(self, text: str, special: bool = False) -> list[int]:
        out = []
        for ch in text:
            if ch == "é":
                out += [29, 30]
            else:
                out.append(_BY_CHAR.get(ch, _BY_CHAR[" "]))
        return out

    def piece_bytes(self, tok: int) -> bytes:
        return PIECES[int(tok)]

    def piece(self, tok: int) -> str:
        return self.piece_bytes(tok).decode("utf-8", errors="ignore")

    def decode(self, toks: list[int]) -> str:
        return b"".join(PIECES[int(t)] for t in toks).decode("utf-8", errors="ignore")

    def is_eog(self, tok: int) -> bool:
        return int(tok) == EOG

    def tokenizer_hash(self) -> str:
        return hashlib.sha256(b"\x00".join(PIECES)).hexdigest()

    def apply_chat_template(self, messages: list[dict]) -> str:
        return "".join(f"{m['role']}: {m['content']}\n" for m in messages) + "assistant: "

    # decoding
    def reset(self) -> None:
        self.kv = []

    def truncate_kv(self, n_tokens: int) -> None:
        self.kv = self.kv[:n_tokens]

    def decode_logits(self, tokens: list[int], logits_all: bool) -> np.ndarray:
        if not tokens:
            raise ValueError("decode_logits requires at least one token")
        if len(self.kv) + len(tokens) > self.n_ctx:
            raise ValueError("context overflow")
        self.calls += 1
        rows = []
        for i, tok in enumerate(tokens):
            self.kv.append(int(tok))
            row = np.zeros(self.n_vocab, dtype=np.float32)
            row[next_token(self.kv, self.n_vocab)] = 1.0
            if self.batch_noise is not None:
                row = self.batch_noise(i, len(tokens), row)
            rows.append(row)
        return np.stack(rows if logits_all else rows[-1:])

    @staticmethod
    def argmax(row: np.ndarray) -> int:
        return int(np.argmax(row))


def greedy_continuation(prefix: list[int], n: int) -> list[int]:
    out, cur = [], list(prefix)
    for _ in range(n):
        t = next_token(cur)
        out.append(t)
        cur.append(t)
    return out
