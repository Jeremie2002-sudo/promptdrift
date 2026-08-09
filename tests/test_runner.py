from __future__ import annotations

import asyncio

import pytest

from promptdrift.assertions import AssertionError_
from promptdrift.backends.base import BackendError, Completion, build_backend
from promptdrift.backends.mock import MockBackend
from promptdrift.runner import run_suite, validate_assertions
from promptdrift.spec import BackendSpec, parse_suite
from tests.conftest import suite_dict


def mock_suite(responses: dict[str, str] | None = None, **overrides):
    data = suite_dict(**overrides)
    data["backend"] = {
        "provider": "mock",
        "model": "test",
        "options": {"responses": responses or {}},
    }
    return parse_suite(data)


class TestMockBackend:
    async def test_is_deterministic(self, mock_spec):
        b = MockBackend(mock_spec())
        a = await b.complete("same prompt")
        c = await b.complete("same prompt")
        assert a.text == c.text

    async def test_different_prompts_differ(self, mock_spec):
        b = MockBackend(mock_spec())
        a = await b.complete("prompt one")
        c = await b.complete("prompt two")
        assert a.text != c.text

    async def test_canned_response_by_substring(self, mock_spec):
        b = MockBackend(mock_spec(responses={"charged twice": "BILLING"}))
        assert (await b.complete("Classify: charged twice")).text == "BILLING"

    async def test_default_response(self, mock_spec):
        b = MockBackend(mock_spec(default="OTHER"))
        assert (await b.complete("anything at all")).text == "OTHER"

    async def test_simulated_failure_raises(self, mock_spec):
        b = MockBackend(mock_spec(failure_rate=1.0))
        with pytest.raises(BackendError, match="simulated failure"):
            await b.complete("x")

    async def test_reports_token_counts(self, mock_spec):
        b = MockBackend(mock_spec(default="one two three"))
        c = await b.complete("a b")
        assert c.prompt_tokens == 2
        assert c.completion_tokens == 3


class TestBackendRegistry:
    def test_builds_known_providers(self):
        assert build_backend(BackendSpec("mock", "m")).name == "mock"

    def test_unknown_provider_raises(self):
        with pytest.raises(BackendError, match="unknown provider"):
            build_backend(BackendSpec("gpt5000", "m"))


class TestRunSuite:
    async def test_passing_case(self):
        suite = mock_suite({"charged twice": "BILLING"})
        result = await run_suite(suite)
        assert result.passed
        assert result.n_passed == 1
        assert result.cases[0].pass_rate == 1.0

    async def test_failing_case(self):
        suite = mock_suite({"charged twice": "TECHNICAL"})
        result = await run_suite(suite)
        assert not result.passed
        assert result.n_failed == 1
        assert "TECHNICAL" in result.cases[0].first_failure()

    async def test_records_backend_and_model(self):
        result = await run_suite(mock_suite())
        assert result.backend == "mock"
        assert result.model == "test"

    async def test_samples_are_all_run(self):
        suite = mock_suite({"charged twice": "BILLING"}, samples=5)
        result = await run_suite(suite)
        assert len(result.cases[0].samples) == 5

    async def test_sample_indices_are_ordered(self):
        suite = mock_suite({"charged twice": "BILLING"}, samples=4)
        result = await run_suite(suite)
        assert [s.index for s in result.cases[0].samples] == [0, 1, 2, 3]

    async def test_backend_error_is_captured_not_raised(self):
        data = suite_dict()
        data["backend"] = {
            "provider": "mock", "model": "test", "options": {"failure_rate": 1.0}
        }
        result = await run_suite(parse_suite(data))
        assert not result.passed
        assert "simulated failure" in result.cases[0].first_failure()

    async def test_one_failing_case_does_not_stop_the_others(self):
        data = suite_dict(cases=[
            {"name": "good", "vars": {"ticket": "charged twice"},
             "assert": [{"type": "equals", "value": "BILLING"}]},
            {"name": "bad", "vars": {"ticket": "something else"},
             "assert": [{"type": "equals", "value": "NEVER_MATCHES"}]},
        ])
        data["backend"] = {"provider": "mock", "model": "t",
                           "options": {"responses": {"charged twice": "BILLING"}}}
        result = await run_suite(parse_suite(data))
        assert len(result.cases) == 2
        assert result.n_passed == 1
        assert result.n_failed == 1

    async def test_duration_is_recorded(self):
        result = await run_suite(mock_suite())
        assert result.duration_s >= 0.0


class TestPassThreshold:
    async def test_partial_pass_below_threshold_fails(self):
        """variants=1 makes every other sample differ, so pass rate lands near 50%."""
        data = suite_dict(samples=4, pass_threshold=1.0)
        data["backend"] = {
            "provider": "mock", "model": "t",
            "options": {"responses": {"charged twice": "BILLING"}, "variants": 1},
        }
        result = await run_suite(parse_suite(data), concurrency=1)
        assert not result.cases[0].passed
        assert 0.0 < result.cases[0].pass_rate < 1.0

    async def test_partial_pass_above_threshold_passes(self):
        data = suite_dict(samples=4, pass_threshold=0.25)
        data["backend"] = {
            "provider": "mock", "model": "t",
            "options": {"responses": {"charged twice": "BILLING"}, "variants": 1},
        }
        result = await run_suite(parse_suite(data), concurrency=1)
        assert result.cases[0].passed

    async def test_flaky_is_flagged(self):
        data = suite_dict(samples=4, pass_threshold=0.25)
        data["backend"] = {
            "provider": "mock", "model": "t",
            "options": {"responses": {"charged twice": "BILLING"}, "variants": 1},
        }
        result = await run_suite(parse_suite(data), concurrency=1)
        assert result.cases[0].flaky
        assert result.flaky_cases

    async def test_full_pass_is_not_flaky(self):
        result = await run_suite(mock_suite({"charged twice": "BILLING"}, samples=3))
        assert not result.cases[0].flaky


class TestConcurrency:
    async def test_runs_concurrently(self):
        """20 units at 50ms each: sequential is ~1s, concurrency=10 is ~0.1s."""
        data = suite_dict(samples=20)
        data["backend"] = {
            "provider": "mock", "model": "t",
            "options": {"responses": {"charged twice": "BILLING"}, "latency_ms": 50},
        }
        suite = parse_suite(data)

        started = asyncio.get_running_loop().time()
        await run_suite(suite, concurrency=10)
        elapsed = asyncio.get_running_loop().time() - started

        assert elapsed < 0.6, f"expected concurrent execution, took {elapsed:.2f}s"

    async def test_concurrency_of_zero_is_clamped_not_deadlocked(self):
        result = await run_suite(mock_suite({"charged twice": "BILLING"}), concurrency=0)
        assert result.passed


class ShortCircuitJudge:
    name = "counting-judge"

    def __init__(self):
        self.calls = 0

    async def complete(self, prompt: str, system: str | None = None) -> Completion:
        self.calls += 1
        return Completion(text='{"score": 1.0}', latency_ms=1.0)

    async def aclose(self) -> None:
        return None


class TestJudgeShortCircuit:
    async def test_judge_skipped_when_free_assertion_fails(self):
        data = suite_dict()
        data["cases"][0]["assert"] = [
            {"type": "equals", "value": "WILL_NOT_MATCH"},
            {"type": "judge", "rubric": "is it good?"},
        ]
        data["backend"] = {"provider": "mock", "model": "t", "options": {"default": "NOPE"}}
        judge = ShortCircuitJudge()

        result = await run_suite(parse_suite(data), judge=judge)

        assert judge.calls == 0, "judge should not be paid for an already-failed case"
        assert not result.passed
        details = [a.detail for a in result.cases[0].samples[0].assertions]
        assert any("skipped" in d for d in details)

    async def test_judge_runs_when_free_assertions_pass(self):
        data = suite_dict()
        data["cases"][0]["assert"] = [
            {"type": "equals", "value": "BILLING"},
            {"type": "judge", "rubric": "is it good?"},
        ]
        data["backend"] = {
            "provider": "mock", "model": "t",
            "options": {"responses": {"charged twice": "BILLING"}},
        }
        judge = ShortCircuitJudge()

        result = await run_suite(parse_suite(data), judge=judge)

        assert judge.calls == 1
        assert result.passed


class TestValidateAssertions:
    def test_rejects_unknown_type_before_any_call(self):
        data = suite_dict()
        data["cases"][0]["assert"] = [{"type": "vibes", "value": "good"}]
        with pytest.raises(AssertionError_, match="unknown assertion type"):
            validate_assertions(parse_suite(data))

    def test_rejects_judge_without_rubric(self):
        data = suite_dict()
        data["cases"][0]["assert"] = [{"type": "judge"}]
        with pytest.raises(AssertionError_, match="requires a 'rubric'"):
            validate_assertions(parse_suite(data))

    def test_accepts_a_valid_suite(self, suite):
        validate_assertions(suite)
