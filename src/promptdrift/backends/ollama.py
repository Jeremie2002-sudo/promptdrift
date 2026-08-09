"""Ollama backend -- talks to a locally running Ollama server.

Local-first is the default for a reason: an eval suite gets run on every prompt
change, and a suite that costs money on every run is a suite people stop
running. Cost is a correctness feature here, not a nicety.
"""

from __future__ import annotations

import time

import httpx

from promptdrift.backends.base import BackendError, Completion
from promptdrift.spec import BackendSpec

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaBackend:
    name = "ollama"

    def __init__(self, spec: BackendSpec, client: httpx.AsyncClient | None = None) -> None:
        self.spec = spec
        self._base_url = str(spec.options.get("base_url", DEFAULT_BASE_URL)).rstrip("/")
        timeout = float(spec.options.get("timeout_s", 120.0))
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def complete(self, prompt: str, system: str | None = None) -> Completion:
        payload: dict[str, object] = {
            "model": self.spec.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.spec.temperature},
        }
        if system:
            payload["system"] = system
        if self.spec.max_tokens:
            payload["options"] = {
                **payload["options"],  # type: ignore[dict-item]
                "num_predict": self.spec.max_tokens,
            }

        started = time.perf_counter()
        try:
            resp = await self._client.post(f"{self._base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:200]
            raise BackendError(
                f"ollama returned {exc.response.status_code}: {detail}. "
                f"Is model {self.spec.model!r} pulled? Try: ollama pull {self.spec.model}"
            ) from exc
        except httpx.HTTPError as exc:
            raise BackendError(
                f"could not reach ollama at {self._base_url}: {exc}. Is `ollama serve` running?"
            ) from exc

        latency_ms = (time.perf_counter() - started) * 1000.0
        return Completion(
            text=data.get("response", ""),
            latency_ms=latency_ms,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
