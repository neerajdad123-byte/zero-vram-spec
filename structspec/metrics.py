from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from threading import Lock


@dataclass
class Metrics:
    started_at: float = field(default_factory=time.time)
    requests_total: int = 0
    requests_active: int = 0
    requests_failed: int = 0
    streamed_chunks: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    target_tokens: int = 0
    accepted_draft_tokens: int = 0
    proposed_draft_tokens: int = 0
    rejected_draft_tokens: int = 0
    kv_repair_ops: int = 0
    last_backend: str = "unknown"
    last_model: str = "unknown"

    @property
    def uptime_s(self) -> float:
        return max(0.0, time.time() - self.started_at)

    @property
    def acceptance_rate(self) -> float:
        if self.proposed_draft_tokens <= 0:
            return 0.0
        return self.accepted_draft_tokens / self.proposed_draft_tokens

    @property
    def estimated_multiplier(self) -> float:
        total = self.target_tokens + self.accepted_draft_tokens
        if total <= 0 or self.target_tokens <= 0:
            return 1.0
        return total / self.target_tokens


class MetricsStore:
    def __init__(self) -> None:
        self._metrics = Metrics()
        self._lock = Lock()

    def snapshot(self) -> dict:
        with self._lock:
            data = asdict(self._metrics)
            data["uptime_s"] = self._metrics.uptime_s
            data["acceptance_rate"] = self._metrics.acceptance_rate
            data["estimated_multiplier"] = self._metrics.estimated_multiplier
            return data

    def inc(self, field: str, amount: int = 1) -> None:
        with self._lock:
            setattr(self._metrics, field, getattr(self._metrics, field) + amount)

    def set_backend(self, backend: str, model: str = "unknown") -> None:
        with self._lock:
            self._metrics.last_backend = backend
            self._metrics.last_model = model

    def prometheus(self) -> str:
        snap = self.snapshot()
        lines = []
        for key, value in snap.items():
            if isinstance(value, (int, float)):
                lines.append(f"structspec_{key} {value}")
        return "\n".join(lines) + "\n"


metrics = MetricsStore()
