"""In-memory metrics for reranker fallback observability."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Any, Dict
from urllib.parse import urlparse


class RerankerMetrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self._requests_total = 0
        self._success_total = 0
        self._fallback_total = 0
        self._fallback_reason_counts = defaultdict(int)
        self._by_provider_model = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    def _provider(self, model: str, base_url: str) -> str:
        m = (model or "").strip()
        if "/" in m:
            return m.split("/", 1)[0]
        try:
            host = (urlparse(base_url or "").hostname or "").strip()
            return host or "unknown"
        except Exception:
            return "unknown"

    def _model(self, model: str) -> str:
        return (model or "unknown").strip()

    def record_request(self) -> None:
        with self._lock:
            self._requests_total += 1

    def record_success(self) -> None:
        with self._lock:
            self._success_total += 1

    def record_fallback(self, reason: str, model: str, base_url: str) -> None:
        rsn = (reason or "unknown").strip().lower()
        provider = self._provider(model, base_url)
        model_name = self._model(model)
        with self._lock:
            self._fallback_total += 1
            self._fallback_reason_counts[rsn] += 1
            self._by_provider_model[rsn][provider][model_name] += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            nested = {
                reason: {
                    provider: dict(model_counts)
                    for provider, model_counts in providers.items()
                }
                for reason, providers in self._by_provider_model.items()
            }
            return {
                "requests_total": self._requests_total,
                "success_total": self._success_total,
                "fallback_total": self._fallback_total,
                "fallback_reason_counts": dict(self._fallback_reason_counts),
                "fallback_by_provider_model": nested,
            }


metrics = RerankerMetrics()

