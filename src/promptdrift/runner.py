"""Suite execution.

Every (case, sample) pair is an independent unit of work, so they all go into
one flat task list bounded by a semaphore rather than being run case-by-case.
With 20 cases x 3 samples against a local model, the difference between
sequential and concurrency=8 is minutes versus seconds, and a slow suite is a
suite that stops getting run.

Backend failures are captured per-sample rather than aborting the run. One
model timeout should not throw away the other 59 results.
"""

from __future__ import annotations

import asyncio
import time

from promptdrift.assertions import (
    FREE,
    AssertionError_,
    AssertionResult,
    run_free,
    run_judge,
    tier_of,
)
from promptdrift.backends.base import Backend, BackendError, Completion, build_backend
from promptdrift.results import CaseResult, SampleResult, SuiteResult
from promptdrift.spec import Case, Suite

DEFAULT_CONCURRENCY = 8


def validate_assertions(suite: Suite) -> None:
    """Fail fast on unknown assertion types, before spending a single call."""
    for case in suite.cases:
        for a in case.assertions:
            tier_of(a.type)  # raises AssertionError_ if unknown
            if a.type == "judge" and not a.rubric:
                raise AssertionError_(
                    f"case {case.name!r}: assertion 'judge' requires a 'rubric'"
                )


async def run_suite(
    suite: Suite,
    backend: Backend | None = None,
    judge: Backend | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    progress: object | None = None,
) -> SuiteResult:
    """Execute every case in the suite and return the collected results.

    ``judge`` defaults to ``backend``: grading with the same model that produced
    the output is cheap and usually fine for regression testing, where what
    matters is the delta between runs rather than absolute quality. Pass a
    stronger judge when you care about the absolute number.
    """
    validate_assertions(suite)

    owns_backend = backend is None
    backend = backend or build_backend(suite.backend)
    judge = judge or backend

    semaphore = asyncio.Semaphore(max(1, concurrency))
    started = time.perf_counter()

    # Flat task list: one entry per (case, sample).
    units: list[tuple[int, int]] = [
        (ci, si)
        for ci, case in enumerate(suite.cases)
        for si in range(suite.samples_for(case))
    ]

    async def run_unit(ci: int, si: int) -> tuple[int, SampleResult]:
        async with semaphore:
            result = await _run_sample(suite, suite.cases[ci], si, backend, judge)
        if progress is not None and hasattr(progress, "advance"):
            progress.advance(1)  # type: ignore[attr-defined]
        return ci, result

    try:
        gathered = await asyncio.gather(*(run_unit(ci, si) for ci, si in units))
    finally:
        if owns_backend:
            await backend.aclose()

    by_case: dict[int, list[SampleResult]] = {i: [] for i in range(len(suite.cases))}
    for ci, sample in gathered:
        by_case[ci].append(sample)

    cases = [
        CaseResult(
            name=case.name,
            samples=sorted(by_case[ci], key=lambda s: s.index),
            threshold=suite.threshold_for(case),
        )
        for ci, case in enumerate(suite.cases)
    ]

    return SuiteResult(
        suite_name=suite.name,
        backend=suite.backend.provider,
        model=suite.backend.model,
        cases=cases,
        duration_s=time.perf_counter() - started,
    )


async def _run_sample(
    suite: Suite, case: Case, index: int, backend: Backend, judge: Backend
) -> SampleResult:
    prompt = suite.render(case)

    try:
        completion = await backend.complete(prompt, system=suite.system)
    except BackendError as exc:
        return SampleResult(index=index, output="", latency_ms=0.0, error=str(exc))

    assertions = await _evaluate(case, completion, judge)
    return SampleResult(
        index=index,
        output=completion.text,
        latency_ms=completion.latency_ms,
        assertions=assertions,
    )


async def _evaluate(
    case: Case, completion: Completion, judge: Backend
) -> list[AssertionResult]:
    """Deterministic checks first; skip the judge if one already failed."""
    results: list[AssertionResult] = []

    free = [a for a in case.assertions if tier_of(a.type) == FREE]
    graded = [a for a in case.assertions if tier_of(a.type) != FREE]

    for a in free:
        results.append(run_free(a, completion))

    if not graded:
        return results

    if any(not r.passed for r in results):
        results.extend(
            AssertionResult(
                type=a.type,
                passed=False,
                detail="skipped -- a deterministic assertion already failed",
            )
            for a in graded
        )
        return results

    graded_results = await asyncio.gather(
        *(run_judge(a, completion, judge) for a in graded),
        return_exceptions=True,
    )
    for a, r in zip(graded, graded_results, strict=True):
        if isinstance(r, BaseException):
            results.append(
                AssertionResult(type=a.type, passed=False, detail=f"judge failed: {r}")
            )
        else:
            results.append(r)

    return results
