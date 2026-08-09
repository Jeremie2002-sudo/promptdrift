"""Groq backend -- OpenAI-compatible chat completions.

Optional. Reads its key from an environment variable (default GROQ_API_KEY) and
never from the suite file, so suites stay safe to commit.
"""

from __future__ import annotations

import os
import time

import httpx

from promptdrift.backends.base import BackendError, Completion
from promptdrift.spec import BackendSpec

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"


class GroqBackend:
    name = "groq"

    def __init__(self, spec: BackendSpec, client: httpx.AsyncClient | None = None) -> None:
        self.spec = spec
        self._base_url = str(spec.options.get("base_url", DEFAULT_BASE_URL)).rstrip("/")
        self._key_env = str(spec.options.get("api_key_env", "GROQ_API_KEY"))
        timeout = float(spec.options.get("timeout_s", 60.0))
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    def _api_key(self) -> str:
        key = os.environ.get(self._key_env)
        if not key:
            raise BackendError(
                f"{self._key_env} is not set. Export it, or switch the suite to "
                f"`provider: ollama` to run locally for free."
            )
        return key

    async def complete(self, prompt: str, system: str | None = None) -> Completion:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, object] = {
            "model": self.spec.model,
            "messages": messages,
            "temperature": self.spec.temperature,
        }
        if self.spec.max_tokens:
            payload["max_tokens"] = self.spec.max_tokens

        started = time.perf_counter()
        try:
            resp = await self._client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key()}"},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise BackendError(
                f"groq returned {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise BackendError(f"could not reach groq: {exc}") from exc

        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = data.get("usage") or {}
        return Completion(
            text=data["choices"][0]["message"]["content"],
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
