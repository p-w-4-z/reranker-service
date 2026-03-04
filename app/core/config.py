"""
Runtime configuration management for reranker-service.

Goals:
- Load defaults from environment variables at startup
- Optionally override with a persisted runtime JSON file
- Allow safe, validated updates at runtime via API
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError


class LLMRuntimeConfig(BaseModel):
    base_url: str = Field(default="http://os-v2-inference-proxy:8000/v1", description="OpenAI-compatible inference proxy base URL")
    api_key: Optional[str] = Field(default=None, description="API key for inference proxy (if required)")
    model: str = Field(default="openai/gpt-oss-20b", description="Default model name for reranking")
    timeout: int = Field(default=30, ge=1, le=600, description="Default request timeout in seconds")
    max_tokens: int = Field(default=1024, ge=1, le=16384, description="Max tokens for reranking output")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature")
    reasoning: Optional[Dict[str, Any]] = Field(default=None, description="OpenRouter reasoning config, e.g. {\"effort\": \"high\"}. null = disabled.")


class RerankerRuntimeConfig(BaseModel):
    default_top_n: int = Field(default=5, ge=1, le=50, description="Default number of items to return if not specified")


class LoggingRuntimeConfig(BaseModel):
    level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)")


class ServiceRuntimeConfig(BaseModel):
    llm: LLMRuntimeConfig = Field(default_factory=LLMRuntimeConfig)
    reranker: RerankerRuntimeConfig = Field(default_factory=RerankerRuntimeConfig)
    logging: LoggingRuntimeConfig = Field(default_factory=LoggingRuntimeConfig)


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge patch into base, returning a new dict."""
    merged: Dict[str, Any] = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)  # type: ignore[arg-type]
        else:
            merged[k] = v
    return merged


class RuntimeConfigManager:
    """
    Thread-safe runtime config manager backed by a JSON file.
    """

    def __init__(self, runtime_path: str, defaults: ServiceRuntimeConfig):
        self._runtime_path = Path(runtime_path)
        self._lock = threading.RLock()
        self._defaults = defaults
        self._current = defaults
        self._load()

    @property
    def runtime_path(self) -> str:
        return str(self._runtime_path)

    def defaults_dict(self) -> Dict[str, Any]:
        return self._defaults.model_dump()

    def current_dict(self) -> Dict[str, Any]:
        with self._lock:
            return self._current.model_dump()

    def _load(self) -> None:
        with self._lock:
            if not self._runtime_path.exists():
                self._current = self._defaults
                return

            try:
                raw = json.loads(self._runtime_path.read_text(encoding="utf-8"))
                merged = _deep_merge(self._defaults.model_dump(), raw if isinstance(raw, dict) else {})
                self._current = ServiceRuntimeConfig.model_validate(merged)
            except Exception:
                # If runtime file is invalid/corrupt, fall back to defaults.
                self._current = self._defaults

    def reload(self) -> None:
        self._load()

    def update(self, patch: Dict[str, Any]) -> ServiceRuntimeConfig:
        if not isinstance(patch, dict):
            raise ValueError("Config patch must be a JSON object")

        with self._lock:
            merged = _deep_merge(self._current.model_dump(), patch)
            try:
                validated = ServiceRuntimeConfig.model_validate(merged)
            except ValidationError as e:
                raise ValueError(e.errors()) from e

            self._runtime_path.write_text(json.dumps(validated.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
            self._current = validated
            return validated

    def reset(self) -> None:
        with self._lock:
            try:
                if self._runtime_path.exists():
                    self._runtime_path.unlink()
            finally:
                self._current = self._defaults

    def schema(self) -> Dict[str, Any]:
        return ServiceRuntimeConfig.model_json_schema()


def build_defaults_from_env() -> ServiceRuntimeConfig:
    return ServiceRuntimeConfig(
        llm=LLMRuntimeConfig(
            base_url=os.getenv("LLM_BASE_URL", "http://os-v2-inference-proxy:8000/v1"),
            api_key=os.getenv("LLM_API_KEY") or None,
            model=os.getenv("DEFAULT_MODEL", "openai/gpt-oss-20b"),
            timeout=int(os.getenv("LLM_TIMEOUT", "30")),
        ),
        logging=LoggingRuntimeConfig(level=os.getenv("LOG_LEVEL", "INFO")),
    )
