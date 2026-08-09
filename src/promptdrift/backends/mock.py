"""A deterministic backend that never touches the network.

This exists so the test suite and CI can exercise the *entire* pipeline --
runner, concurrency, assertions, baseline diffing, exit codes -- without an API
key, without cost, and without flakes. A test tool whose own tests are
non-deterministic has no business telling you your prompts are unstable.

Two modes:

  responses:   an explicit map of prompt-substring -> canned reply, for tests
               that need a specific output.
  (default)    a hash of the prompt, so the same prompt always yields the same
               reply and different prompts yield different ones.

Set ``failure_rate`` to make it fail deterministically-but-pseudorandomly, for
exercising retry and error paths.
"""

from __future__ import annotations

import asyncio
import hashlib

from promptdrift.backends.base import BackendError, Completion
from promptdrift.spec import BackendSpec


class MockBackend:
    name = "mock"

    def __init__(self, spec: BackendSpec) -> None:
        self.spec = spec
        opts = spec.options
        self._responses: dict[str, str] = dict(opts.get("responses") or {})
        self._default: str | None = opts.get("default")
        self._latency_ms: float = float(opts.get("latency_ms", 0.0))
        self._failure_rate: float = float(opts.get("failure_rate", 0.0))
        # Non-zero makes repeated samples of the same prompt differ, which is how
        # we test pass_threshold / flakiness detection deterministically.
        self._jitter: int = int(opts.get("variants", 0))
        self._calls = 0

    async def complete(self, prompt: str, system: str | None = None) -> Completion:
        self._calls += 1
        if self._latency_ms:
            await asyncio.sleep(self._latency_ms / 1000.0)

        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        if self._failure_rate > 0:
            bucket = int(digest[:8], 16) % 100
            if bucket < self._failure_rate * 100:
                raise BackendError(f"mock: simulated failure for prompt hash {digest[:8]}")

        text = self._lookup(prompt, digest)
        return Completion(
            text=text,
            latency_ms=self._latency_ms,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
        )

    def _lookup(self, prompt: str, digest: str) -> str:
        for needle, reply in self._responses.items():
            if needle in prompt:
                if self._jitter:
                    # Cycle through "<reply>", "<reply> (v2)", ... across samples
                    # so a case can be made to pass only some of the time.
                    n = self._calls % (self._jitter + 1)
                    return reply if n == 0 else f"{reply} (v{n + 1})"
                return reply
        if self._default is not None:
            return self._default
        return f"mock-response-{digest[:12]}"

    async def aclose(self) -> None:
        return None
