"""Assertions: the checks run against a single model output.

Two tiers, and the distinction is load-bearing:

  FREE    deterministic string/JSON/latency checks. No model call, no cost,
          no flakiness.
  GRADED  model-graded checks (LLM-as-judge). Costs a call, and is itself
          fallible.

The runner evaluates FREE assertions first and skips GRADED ones if a FREE one
already failed. Most regressions are caught by a cheap check -- there is no
reason to pay a judge to confirm that output which was supposed to be the single
word "BILLING" and is instead three paragraphs is bad.

Judge prompts are pinned in this file rather than being user-supplied templates,
because a judge whose prompt drifts is a ruler whose length changes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from promptdrift.spec import Assertion

if TYPE_CHECKING:
    from promptdrift.backends.base import Backend, Completion


class AssertionError_(ValueError):
    """Raised when an assertion is misconfigured (not when it fails)."""


@dataclass(frozen=True)
class AssertionResult:
    type: str
    passed: bool
    detail: str
    # 0.0-1.0 for graded assertions; None for boolean ones.
    score: float | None = None


FREE = "free"
GRADED = "graded"

# type name -> (tier, checker)
_REGISTRY: dict[str, tuple[str, Callable[..., AssertionResult]]] = {}


def _register(name: str, tier: str):
    def wrap(fn):
        _REGISTRY[name] = (tier, fn)
        return fn

    return wrap


def tier_of(assertion_type: str) -> str:
    entry = _REGISTRY.get(assertion_type)
    if entry is None:
        raise AssertionError_(
            f"unknown assertion type {assertion_type!r}. Available: {sorted(_REGISTRY)}"
        )
    return entry[0]


def known_types() -> list[str]:
    return sorted(_REGISTRY)


def _ok(t: str, detail: str) -> AssertionResult:
    return AssertionResult(type=t, passed=True, detail=detail)


def _fail(t: str, detail: str) -> AssertionResult:
    return AssertionResult(type=t, passed=False, detail=detail)


def _truncate(text: str, limit: int = 120) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "..."


def _need_value(a: Assertion) -> Any:
    if a.value is None:
        raise AssertionError_(f"assertion {a.type!r} requires a 'value'")
    return a.value


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


# --------------------------------------------------------------------------
# Deterministic assertions
# --------------------------------------------------------------------------


@_register("equals", FREE)
def _equals(a: Assertion, c: Completion) -> AssertionResult:
    want = str(_need_value(a))
    got = c.text.strip()
    if got == want:
        return _ok("equals", f"output == {want!r}")
    return _fail("equals", f"expected {want!r}, got {_truncate(got)!r}")


@_register("iequals", FREE)
def _iequals(a: Assertion, c: Completion) -> AssertionResult:
    want = str(_need_value(a)).strip().lower()
    got = c.text.strip().lower()
    if got == want:
        return _ok("iequals", f"output ~= {want!r} (case-insensitive)")
    return _fail("iequals", f"expected {want!r} (any case), got {_truncate(got)!r}")


@_register("contains", FREE)
def _contains(a: Assertion, c: Completion) -> AssertionResult:
    needle = str(_need_value(a))
    if needle in c.text:
        return _ok("contains", f"found {needle!r}")
    return _fail("contains", f"{needle!r} not present in {_truncate(c.text)!r}")


@_register("not_contains", FREE)
def _not_contains(a: Assertion, c: Completion) -> AssertionResult:
    needle = str(_need_value(a))
    if needle not in c.text:
        return _ok("not_contains", f"{needle!r} correctly absent")
    return _fail("not_contains", f"forbidden string {needle!r} appeared in output")


@_register("contains_any", FREE)
def _contains_any(a: Assertion, c: Completion) -> AssertionResult:
    needles = [str(v) for v in _as_list(_need_value(a))]
    hits = [n for n in needles if n in c.text]
    if hits:
        return _ok("contains_any", f"found {hits[0]!r}")
    return _fail("contains_any", f"none of {needles} present in {_truncate(c.text)!r}")


@_register("contains_all", FREE)
def _contains_all(a: Assertion, c: Completion) -> AssertionResult:
    needles = [str(v) for v in _as_list(_need_value(a))]
    missing = [n for n in needles if n not in c.text]
    if not missing:
        return _ok("contains_all", f"all {len(needles)} substrings present")
    return _fail("contains_all", f"missing {missing}")


@_register("one_of", FREE)
def _one_of(a: Assertion, c: Completion) -> AssertionResult:
    options = [str(v).strip() for v in _as_list(_need_value(a))]
    got = c.text.strip()
    if got in options:
        return _ok("one_of", f"output {got!r} is an allowed value")
    return _fail("one_of", f"expected one of {options}, got {_truncate(got)!r}")


@_register("regex", FREE)
def _regex(a: Assertion, c: Completion) -> AssertionResult:
    pattern = str(_need_value(a))
    try:
        compiled = re.compile(pattern, re.DOTALL)
    except re.error as exc:
        raise AssertionError_(f"invalid regex {pattern!r}: {exc}") from exc
    if compiled.search(c.text):
        return _ok("regex", f"matched /{pattern}/")
    return _fail("regex", f"/{pattern}/ did not match {_truncate(c.text)!r}")


@_register("json_valid", FREE)
def _json_valid(a: Assertion, c: Completion) -> AssertionResult:
    try:
        json.loads(c.text)
    except json.JSONDecodeError as exc:
        return _fail("json_valid", f"not valid JSON: {exc.msg} at pos {exc.pos}")
    return _ok("json_valid", "output parsed as JSON")


@_register("json_has_keys", FREE)
def _json_has_keys(a: Assertion, c: Completion) -> AssertionResult:
    keys = [str(k) for k in _as_list(_need_value(a))]
    try:
        parsed = json.loads(c.text)
    except json.JSONDecodeError as exc:
        return _fail("json_has_keys", f"not valid JSON: {exc.msg}")
    if not isinstance(parsed, dict):
        return _fail("json_has_keys", f"expected a JSON object, got {type(parsed).__name__}")
    if missing := [k for k in keys if k not in parsed]:
        return _fail("json_has_keys", f"missing key(s) {missing}")
    return _ok("json_has_keys", f"all {len(keys)} key(s) present")


@_register("max_latency_ms", FREE)
def _max_latency(a: Assertion, c: Completion) -> AssertionResult:
    limit = float(_need_value(a))
    if c.latency_ms <= limit:
        return _ok("max_latency_ms", f"{c.latency_ms:.0f}ms <= {limit:.0f}ms")
    return _fail("max_latency_ms", f"{c.latency_ms:.0f}ms exceeded budget of {limit:.0f}ms")


@_register("max_words", FREE)
def _max_words(a: Assertion, c: Completion) -> AssertionResult:
    limit = int(_need_value(a))
    count = len(c.text.split())
    if count <= limit:
        return _ok("max_words", f"{count} words <= {limit}")
    return _fail("max_words", f"{count} words exceeded limit of {limit}")


# --------------------------------------------------------------------------
# Model-graded assertions
# --------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are grading whether a model's output satisfies a specific criterion.

CRITERION:
{rubric}

OUTPUT TO GRADE:
{output}

Reply with ONLY a JSON object, no prose and no code fence:
{{"score": <number between 0.0 and 1.0>, "reason": "<one short sentence>"}}

A score of 1.0 means the output fully satisfies the criterion. A score of 0.0
means it does not satisfy it at all."""

_SCORE_RE = re.compile(r'"score"\s*:\s*([0-9.]+)')
_REASON_RE = re.compile(r'"reason"\s*:\s*"([^"]*)"')


async def run_judge(
    assertion: Assertion, completion: Completion, judge: Backend
) -> AssertionResult:
    """LLM-as-judge. Returns a graded result with a 0.0-1.0 score."""
    if not assertion.rubric:
        raise AssertionError_("assertion 'judge' requires a 'rubric'")
    threshold = assertion.threshold if assertion.threshold is not None else 0.7

    prompt = _JUDGE_PROMPT.format(rubric=assertion.rubric, output=completion.text)
    graded = await judge.complete(prompt)
    score, reason = _parse_judge_reply(graded.text)

    if score is None:
        return AssertionResult(
            type="judge",
            passed=False,
            detail=f"judge returned unparseable output: {_truncate(graded.text)!r}",
            score=None,
        )

    passed = score >= threshold
    verb = "met" if passed else "missed"
    return AssertionResult(
        type="judge",
        passed=passed,
        detail=f"score {score:.2f} {verb} threshold {threshold:.2f}: {reason}",
        score=score,
    )


def _parse_judge_reply(text: str) -> tuple[float | None, str]:
    """Extract score and reason, tolerating code fences and stray prose.

    Judges are themselves LLMs and will occasionally wrap JSON in a fence or add
    a preamble no matter how firmly the prompt says not to. Falling back to a
    regex is not elegant, but treating a formatting slip as a test failure would
    produce false alarms, which is the fastest way to get a suite ignored.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "score" in parsed:
            return _clamp(float(parsed["score"])), str(parsed.get("reason", ""))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    if m := _SCORE_RE.search(text):
        reason_match = _REASON_RE.search(text)
        try:
            return _clamp(float(m.group(1))), reason_match.group(1) if reason_match else ""
        except ValueError:
            return None, ""
    return None, ""


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def run_free(assertion: Assertion, completion: Completion) -> AssertionResult:
    """Run a deterministic assertion."""
    entry = _REGISTRY.get(assertion.type)
    if entry is None:
        raise AssertionError_(
            f"unknown assertion type {assertion.type!r}. Available: {known_types()}"
        )
    tier, fn = entry
    if tier != FREE or fn is None:
        raise AssertionError_(
            f"{assertion.type!r} is not a deterministic assertion -- "
            f"it is model-graded and must be dispatched through run_judge()"
        )
    return fn(assertion, completion)


# 'judge' is registered so that validation and tier_of() know about it, but it
# is dispatched through run_judge() because it needs a backend.
_REGISTRY["judge"] = (GRADED, None)  # type: ignore[assignment]
