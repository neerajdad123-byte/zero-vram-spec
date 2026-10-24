"""Minimal ctypes binding to the official llama.cpp shared library (pinned to b11756).

Struct layouts are copied from include/llama.h at tag b11756. The library is loaded
lazily, so importing this module never needs the DLL.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    LIB_NAME, GGML_NAME, GGML_BASE_NAME = "llama.dll", "ggml.dll", "ggml-base.dll"
elif sys.platform == "darwin":
    LIB_NAME, GGML_NAME, GGML_BASE_NAME = "libllama.dylib", "libggml.dylib", "libggml-base.dylib"
else:
    LIB_NAME, GGML_NAME, GGML_BASE_NAME = "libllama.so", "libggml.so", "libggml-base.so"

PINNED_BUILD = "b11756"
SUPPORTED_VERSIONS = {"0.7.0-dev"}
# Symbols that only exist in recent builds; their absence means an older, ABI-incompatible llama.cpp.
REQUIRED_SYMBOLS = ("llama_n_rs_seq", "llama_model_is_hybrid", "llama_memory_seq_rm", "llama_get_memory")


class LlamaLoadError(RuntimeError):
    pass


def find_llama_dir() -> Path:
    searched: list[str] = []
    env = os.environ.get("STRUCTSPEC_LLAMA_DIR")
    if env:
        if (Path(env) / LIB_NAME).exists():
            return Path(env)
        searched.append(f"STRUCTSPEC_LLAMA_DIR={env}")
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry and (Path(entry) / LIB_NAME).exists():
            return Path(entry)
    searched.append("PATH")
    tools = sorted((Path.home() / "tools").glob("llama.cpp-*/bin"), reverse=True)
    for cand in tools:
        if (cand / LIB_NAME).exists():
            return cand
    searched.append(str(Path.home() / "tools" / "llama.cpp-*" / "bin"))
    raise LlamaLoadError(
        f"Could not find {LIB_NAME} (searched: {', '.join(searched)}). Download the official llama.cpp "
        f"{PINNED_BUILD} release and set STRUCTSPEC_LLAMA_DIR to its bin directory."
    )


def check_version(lib) -> None:
    missing = [s for s in REQUIRED_SYMBOLS if not hasattr(lib, s)]
    lib.llama_version.restype = C.c_char_p
    version = (lib.llama_version() or b"").decode()
    if version not in SUPPORTED_VERSIONS or missing:
        detail = f"missing symbols {', '.join(missing)}" if missing else f"version {version!r}"
        raise LlamaLoadError(
            f"llama.cpp library is incompatible ({detail}). Structspec is built against llama.cpp "
            f"{PINNED_BUILD} (llama_version {sorted(SUPPORTED_VERSIONS)}); point STRUCTSPEC_LLAMA_DIR at that build."
        )


# ---- structs (llama.h @ b11756) ------------------------------------------------------------

class _BuftOverride(C.Structure):
    _fields_ = [("pattern", C.c_char_p), ("buft", C.c_void_p)]


class ModelParams(C.Structure):
    _fields_ = [
        ("devices", C.c_void_p),
        ("tensor_buft_overrides", C.c_void_p),
        ("n_gpu_layers", C.c_int32),
        ("split_mode", C.c_int),
        ("load_mode", C.c_int),
        ("lazy_mode", C.c_int),
        ("main_gpu", C.c_int32),
        ("tensor_split", C.c_void_p),
        ("progress_callback", C.c_void_p),
        ("progress_callback_user_data", C.c_void_p),
        ("kv_overrides", C.c_void_p),
        ("vocab_only", C.c_bool),
        ("check_tensors", C.c_bool),
        ("use_extra_bufts", C.c_bool),
        ("no_host", C.c_bool),
        ("no_alloc", C.c_bool),
        ("load_mtp", C.c_bool),
    ]


class ContextParams(C.Structure):
    _fields_ = [
        ("n_ctx", C.c_uint32),
        ("n_batch", C.c_uint32),
        ("n_ubatch", C.c_uint32),
        ("n_seq_max", C.c_uint32),
        ("n_rs_seq", C.c_uint32),
        ("n_outputs_max", C.c_uint32),
        ("n_outputs_max_per_seq", C.c_uint32),
        ("n_threads", C.c_int32),
        ("n_threads_batch", C.c_int32),
        ("ctx_type", C.c_int),
        ("rope_scaling_type", C.c_int),
        ("pooling_type", C.c_int),
        ("attention_type", C.c_int),
        ("flash_attn_type", C.c_int),
        ("rope_freq_base", C.c_float),
        ("rope_freq_scale", C.c_float),
        ("yarn_ext_factor", C.c_float),
        ("yarn_attn_factor", C.c_float),
        ("yarn_beta_fast", C.c_float),
        ("yarn_beta_slow", C.c_float),
        ("yarn_orig_ctx", C.c_uint32),
        ("defrag_thold", C.c_float),
        ("cb_eval", C.c_void_p),
        ("cb_eval_user_data", C.c_void_p),
        ("type_k", C.c_int),
        ("type_v", C.c_int),
        ("moe_cache_size", C.c_size_t),
        ("abort_callback", C.c_void_p),
        ("abort_callback_data", C.c_void_p),
        ("embeddings", C.c_bool),
        ("offload_kqv", C.c_bool),
        ("no_perf", C.c_bool),
        ("op_offload", C.c_bool),
        ("swa_full", C.c_bool),
        ("kv_unified", C.c_bool),
        ("samplers", C.c_void_p),
        ("n_samplers", C.c_size_t),
        ("ctx_other", C.c_void_p),
    ]


class Batch(C.Structure):
    _fields_ = [
        ("n_tokens", C.c_int32),
        ("token", C.POINTER(C.c_int32)),
        ("embd", C.POINTER(C.c_float)),
        ("pos", C.POINTER(C.c_int32)),
        ("n_seq_id", C.POINTER(C.c_int32)),
        ("seq_id", C.POINTER(C.POINTER(C.c_int32))),
        ("logits", C.POINTER(C.c_int8)),
    ]


class ChatMessage(C.Structure):
    _fields_ = [("role", C.c_char_p), ("content", C.c_char_p)]


_LOG_CB = C.CFUNCTYPE(None, C.c_int, C.c_char_p, C.c_void_p)
_GGML_LOG_LEVEL_ERROR = 4


@_LOG_CB
def _quiet_log(level, text, _user):
    if level == _GGML_LOG_LEVEL_ERROR and text:
        sys.stderr.write(text.decode("utf-8", "replace"))


_LIB = None
_CPU_BUFT = None


def _sig(lib, name, restype, *argtypes):
    fn = getattr(lib, name)
    fn.restype = restype
    fn.argtypes = list(argtypes)


def load_library(verbose: bool = False):
    """Load llama.cpp once per process: ggml backends, version check, signatures."""
    global _LIB, _CPU_BUFT
    if _LIB is not None:
        return _LIB
    d = find_llama_dir()
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(d))
    try:
        ggml = C.CDLL(str(d / GGML_NAME))
        ggml_base = C.CDLL(str(d / GGML_BASE_NAME))
        lib = C.CDLL(str(d / LIB_NAME))
    except OSError as exc:
        raise LlamaLoadError(f"Failed to load llama.cpp from {d}: {exc}") from exc
    check_version(lib)

    v, p, i32, u32, b = C.c_void_p, C.c_char_p, C.c_int32, C.c_uint32, C.c_bool
    _sig(ggml, "ggml_backend_load_all_from_path", None, p)
    _sig(ggml_base, "ggml_backend_cpu_buffer_type", v)
    _sig(lib, "llama_log_set", None, _LOG_CB, v)
    _sig(lib, "llama_backend_init", None)
    _sig(lib, "llama_model_default_params", ModelParams)
    _sig(lib, "llama_context_default_params", ContextParams)
    _sig(lib, "llama_model_load_from_file", v, p, ModelParams)
    _sig(lib, "llama_model_free", None, v)
    _sig(lib, "llama_init_from_model", v, v, ContextParams)
    _sig(lib, "llama_free", None, v)
    _sig(lib, "llama_model_get_vocab", v, v)
    _sig(lib, "llama_model_is_hybrid", b, v)
    _sig(lib, "llama_model_is_recurrent", b, v)
    _sig(lib, "llama_model_chat_template", p, v, p)
    _sig(lib, "llama_vocab_n_tokens", i32, v)
    _sig(lib, "llama_vocab_eos", i32, v)
    _sig(lib, "llama_vocab_is_eog", b, v, i32)
    _sig(lib, "llama_tokenize", i32, v, p, i32, C.POINTER(i32), i32, b, b)
    _sig(lib, "llama_token_to_piece", i32, v, i32, p, i32, i32, b)
    _sig(lib, "llama_detokenize", i32, v, C.POINTER(i32), i32, p, i32, b, b)
    _sig(lib, "llama_chat_apply_template", i32, p, C.POINTER(ChatMessage), C.c_size_t, b, p, i32)
    _sig(lib, "llama_n_ctx", u32, v)
    _sig(lib, "llama_get_memory", v, v)
    _sig(lib, "llama_memory_clear", None, v, b)
    _sig(lib, "llama_memory_seq_rm", b, v, i32, i32, i32)
    _sig(lib, "llama_batch_init", Batch, i32, i32, i32)
    _sig(lib, "llama_batch_free", None, Batch)
    _sig(lib, "llama_decode", i32, v, Batch)
    _sig(lib, "llama_get_logits", C.POINTER(C.c_float), v)

    if not verbose:
        lib.llama_log_set(_quiet_log, None)
    ggml.ggml_backend_load_all_from_path(str(d).encode())
    lib.llama_backend_init()
    _check_layout(lib)
    _CPU_BUFT = ggml_base.ggml_backend_cpu_buffer_type()
    _LIB = lib
    return lib


def _check_layout(lib) -> None:
    # Known b11756 defaults spread across both structs; a mismatch means our layout drifted.
    cp = lib.llama_context_default_params()
    mp = lib.llama_model_default_params()
    ok = (
        cp.n_batch == 2048 and cp.n_ubatch == 512 and cp.n_seq_max == 1
        and cp.yarn_beta_fast == -1.0 and cp.defrag_thold == -1.0 and cp.type_k == 1 and cp.type_v == 1
        and cp.offload_kqv and cp.swa_full and not cp.kv_unified and cp.n_samplers == 0
        and mp.n_gpu_layers == -1 and mp.load_mode == -1 and mp.use_extra_bufts and not mp.vocab_only
    )
    if not ok:
        raise LlamaLoadError(f"llama.cpp struct layout does not match {PINNED_BUILD}; wrong build?")


class NativeLlama:
    """Greedy-decoding model handle with the interface the speculative loop needs."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_batch: int = 512,
        n_gpu_layers: int = -1,
        cpu_moe: bool = False,
        max_rollback: int = 16,
        n_threads: int | None = None,
        vocab_only: bool = False,
        verbose: bool = False,
    ):
        lib = load_library(verbose)
        self._lib = lib
        self._ctx = self._model = None
        self._batch = None
        mp = lib.llama_model_default_params()
        mp.n_gpu_layers = n_gpu_layers
        mp.vocab_only = vocab_only
        if cpu_moe:
            self._overrides = (_BuftOverride * 2)(_BuftOverride(rb"\.ffn_(up|down|gate)_exps", _CPU_BUFT))
            mp.tensor_buft_overrides = C.cast(self._overrides, C.c_void_p)
        if not Path(model_path).exists():
            raise FileNotFoundError(model_path)
        self._model = lib.llama_model_load_from_file(str(model_path).encode(), mp)
        if not self._model:
            raise LlamaLoadError(f"llama.cpp failed to load model {model_path}")
        self._vocab = lib.llama_model_get_vocab(self._model)
        self.n_vocab = lib.llama_vocab_n_tokens(self._vocab)
        self.eos = lib.llama_vocab_eos(self._vocab)
        self.is_hybrid = bool(lib.llama_model_is_hybrid(self._model) or lib.llama_model_is_recurrent(self._model))
        self.max_rollback = max_rollback
        self._piece_cache: dict[int, bytes] = {}
        self.n_past = 0
        self.n_ctx = 0
        if vocab_only:
            return

        cp = lib.llama_context_default_params()
        cp.n_ctx = n_ctx
        cp.n_batch = cp.n_ubatch = n_batch
        if self.is_hybrid:
            # Recurrent state cannot be partially removed without snapshots to roll back to.
            cp.n_rs_seq = max_rollback
        if n_threads:
            cp.n_threads = cp.n_threads_batch = n_threads
        self._ctx = lib.llama_init_from_model(self._model, cp)
        if not self._ctx:
            raise LlamaLoadError("llama.cpp failed to create a context (out of memory?)")
        self.n_ctx = lib.llama_n_ctx(self._ctx)
        self.n_batch = n_batch
        self._mem = lib.llama_get_memory(self._ctx)
        self._batch = lib.llama_batch_init(n_batch, 0, 1)

    # ---- tokenizer -------------------------------------------------------------------------

    def encode(self, text: str, special: bool = False) -> list[int]:
        data = text.encode("utf-8")
        n_max = len(data) + 8
        buf = (C.c_int32 * n_max)()
        n = self._lib.llama_tokenize(self._vocab, data, len(data), buf, n_max, False, special)
        if n < 0:
            n_max = -n
            buf = (C.c_int32 * n_max)()
            n = self._lib.llama_tokenize(self._vocab, data, len(data), buf, n_max, False, special)
        return list(buf[:n])

    def decode(self, toks: list[int]) -> str:
        if not toks:
            return ""
        arr = (C.c_int32 * len(toks))(*toks)
        size = len(toks) * 16 + 16
        buf = C.create_string_buffer(size)
        n = self._lib.llama_detokenize(self._vocab, arr, len(toks), buf, size, False, False)
        if n < 0:
            size = -n
            buf = C.create_string_buffer(size)
            n = self._lib.llama_detokenize(self._vocab, arr, len(toks), buf, size, False, False)
        return buf.raw[:n].decode("utf-8", errors="ignore")

    def _token_bytes(self, tok: int, special: bool) -> bytes:
        buf = C.create_string_buffer(64)
        n = self._lib.llama_token_to_piece(self._vocab, tok, buf, 64, 0, special)
        if n < 0:
            buf = C.create_string_buffer(-n)
            n = self._lib.llama_token_to_piece(self._vocab, tok, buf, -n, 0, special)
        return buf.raw[:n]

    def piece_bytes(self, tok: int) -> bytes:
        tok = int(tok)
        piece = self._piece_cache.get(tok)
        if piece is None:
            piece = self._piece_cache[tok] = self._token_bytes(tok, False)
        return piece

    def piece(self, tok: int) -> str:
        return self.piece_bytes(tok).decode("utf-8", errors="ignore")

    def is_eog(self, tok: int) -> bool:
        return bool(self._lib.llama_vocab_is_eog(self._vocab, int(tok)))

    def tokenizer_hash(self) -> str:
        h = hashlib.sha256(str(self.n_vocab).encode())
        for tok in range(self.n_vocab):
            h.update(self._token_bytes(tok, True))
            h.update(b"\x00")
        return h.hexdigest()

    def chat_template(self) -> str:
        tmpl = self._lib.llama_model_chat_template(self._model, None)
        return tmpl.decode("utf-8", errors="replace") if tmpl else ""

    def apply_chat_template(self, messages: list[dict]) -> str:
        tmpl = self._lib.llama_model_chat_template(self._model, None)
        enc = [(str(m.get("role", "user")).encode(), str(m.get("content") or "").encode()) for m in messages]
        chat = (ChatMessage * len(enc))(*[ChatMessage(r, c) for r, c in enc])
        size = 2 * sum(len(c) for _, c in enc) + 256
        buf = C.create_string_buffer(size)
        n = self._lib.llama_chat_apply_template(tmpl, chat, len(enc), True, buf, size)
        if n > size:
            size = n
            buf = C.create_string_buffer(size)
            n = self._lib.llama_chat_apply_template(tmpl, chat, len(enc), True, buf, size)
        if n < 0:
            raise ValueError("model chat template is not supported by llama_chat_apply_template")
        return buf.raw[:n].decode("utf-8")

    # ---- decoding --------------------------------------------------------------------------

    def reset(self) -> None:
        self._lib.llama_memory_clear(self._mem, True)
        self.n_past = 0

    def truncate_kv(self, n_tokens: int) -> None:
        if n_tokens >= self.n_past:
            return
        if not self._lib.llama_memory_seq_rm(self._mem, 0, n_tokens, -1):
            raise RuntimeError(
                f"llama.cpp could not roll the cache back from {self.n_past} to {n_tokens} tokens "
                f"(hybrid={self.is_hybrid}, max_rollback={self.max_rollback})"
            )
        self.n_past = n_tokens

    def decode_logits(self, tokens: list[int], logits_all: bool) -> np.ndarray:
        if not tokens:
            raise ValueError("decode_logits requires at least one token")
        if self.n_past + len(tokens) > self.n_ctx:
            raise ValueError(f"context overflow: {self.n_past} + {len(tokens)} > n_ctx {self.n_ctx}")
        b = self._batch
        out = None
        for start in range(0, len(tokens), self.n_batch):
            chunk = tokens[start : start + self.n_batch]
            last_chunk = start + self.n_batch >= len(tokens)
            for i, tok in enumerate(chunk):
                b.token[i] = tok
                b.pos[i] = self.n_past + i
                b.n_seq_id[i] = 1
                b.seq_id[i][0] = 0
                b.logits[i] = 1 if (logits_all or (last_chunk and i == len(chunk) - 1)) else 0
            b.n_tokens = len(chunk)
            rc = self._lib.llama_decode(self._ctx, b)
            if rc != 0:
                raise RuntimeError(f"llama_decode failed with code {rc}")
            self.n_past += len(chunk)
            rows = len(chunk) if logits_all else (1 if last_chunk else 0)
            if rows:
                ptr = self._lib.llama_get_logits(self._ctx)
                block = np.ctypeslib.as_array(ptr, shape=(rows * self.n_vocab,)).reshape(rows, self.n_vocab).copy()
                out = block if out is None or not logits_all else np.vstack([out, block])
        assert out is not None
        return out

    @staticmethod
    def argmax(row: np.ndarray) -> int:
        return int(np.argmax(row))

    def close(self) -> None:
        if self._batch is not None:
            self._lib.llama_batch_free(self._batch)
            self._batch = None
        if self._ctx:
            self._lib.llama_free(self._ctx)
            self._ctx = None
        if self._model:
            self._lib.llama_model_free(self._model)
            self._model = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
