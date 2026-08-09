"""Backend protocol and the shared Completion result type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from promptdrift.spec import BackendSpec


class BackendError(RuntimeError):
    """A backend failed to produce a completion (network, auth, bad model...)."""


@dataclass(frozen=True)
class Completion:
    text: str
    latency_ms: float
    # None when the provider does not report usage.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@runtime_checkable
class Backend(Protocol):
    """Anything that can turn a prompt into text.

    Kept this narrow on purpose: the runner should not know or care whether it
    is talking to a local model, a hosted API, or a canned fixture.
    """

    name: str

    async def complete(self, prompt: str, system: str | None = None) -> Completion:
        ...

    async def aclose(self) -> None:
        ...


def build_backend(spec: BackendSpec) -> Backend:
    """Construct a backend from its spec. Imports lazily so that a missing
    optional dependency for one provider never breaks the others."""
    provider = spec.provider.lower()

    if provider == "mock":
        from promptdrift.backends.mock import MockBackend

        return MockBackend(spec)
    if provider == "ollama":
        from promptdrift.backends.ollama import OllamaBackend

        return OllamaBackend(spec)
    if provider == "groq":
        from promptdrift.backends.groq import GroqBackend

        return GroqBackend(spec)

    raise BackendError(
        f"unknown provider {spec.provider!r}. Available: mock, ollama, groq"
    )
