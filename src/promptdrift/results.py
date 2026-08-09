"""Result types, and their serialised form.

These are what gets written to a baseline and diffed on the next run, so the
shape here is effectively the tool's on-disk format. Kept flat and boring on
purpose -- a baseline that is hard to read in a PR diff defeats the point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from promptdrift.assertions import AssertionResult


@dataclass
class SampleResult:
    index: int
    output: str
    latency_ms: float
    assertions: list[AssertionResult] = field(default_factory=list)
    error: str | None = None

    @property
    def passed(self) -> bool:
        if self.error is not None:
            return False
        return all(a.passed for a in self.assertions)

    def failure_summary(self) -> str:
        if self.error:
            return self.error
        return "; ".join(a.detail for a in self.assertions if not a.passed)


@dataclass
class CaseResult:
    name: str
    samples: list[SampleResult]
    threshold: float

    @property
    def pass_rate(self) -> float:
        if not self.samples:
            return 0.0
        return sum(1 for s in self.samples if s.passed) / len(self.samples)

    @property
    def passed(self) -> bool:
        return self.pass_rate >= self.threshold

    @property
    def flaky(self) -> bool:
        """Passed some samples but not all. Worth surfacing separately: a case
        that passes 4 times in 5 is not a pass, it is an unstable prompt."""
        return 0.0 < self.pass_rate < 1.0

    @property
    def mean_latency_ms(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.latency_ms for s in self.samples) / len(self.samples)

    def first_failure(self) -> str:
        for s in self.samples:
            if not s.passed:
                return s.failure_summary()
        return ""

    def to_dict(self) -> dict[str, Any]:
        """Baseline form. Deliberately excludes latency and raw outputs.

        Latency is machine-dependent and would make every baseline diff noisy.
        Raw outputs are excluded because for a non-deterministic model they
        change on every run without indicating a regression -- what we snapshot
        is *whether the case passed and how reliably*, which is the thing a
        prompt change can actually break.
        """
        return {
            "name": self.name,
            "pass_rate": round(self.pass_rate, 4),
            "passed": self.passed,
            "threshold": self.threshold,
            "samples": len(self.samples),
        }


@dataclass
class SuiteResult:
    suite_name: str
    backend: str
    model: str
    cases: list[CaseResult]
    duration_s: float = 0.0

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.cases)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def n_failed(self) -> int:
        return len(self.cases) - self.n_passed

    @property
    def flaky_cases(self) -> list[CaseResult]:
        return [c for c in self.cases if c.flaky]

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite_name,
            "backend": self.backend,
            "model": self.model,
            "cases": [c.to_dict() for c in self.cases],
        }
