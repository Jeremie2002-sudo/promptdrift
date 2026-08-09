from __future__ import annotations

import json

import pytest

from promptdrift.assertions import AssertionResult
from promptdrift.baseline import (
    DEGRADED,
    IMPROVED,
    NEW,
    REGRESSED,
    REMOVED,
    UNCHANGED,
    BaselineError,
    baseline_path,
    diff_against,
    load_baseline,
    save_baseline,
)
from promptdrift.results import CaseResult, SampleResult, SuiteResult


def make_case(name: str, pass_rate: float, samples: int = 4, threshold: float = 1.0):
    """Build a CaseResult with the requested pass rate."""
    n_pass = round(pass_rate * samples)
    results = []
    for i in range(samples):
        ok = i < n_pass
        results.append(
            SampleResult(
                index=i,
                output="out" if ok else "wrong",
                latency_ms=10.0,
                assertions=[
                    AssertionResult(
                        type="equals",
                        passed=ok,
                        detail="matched" if ok else "expected 'out', got 'wrong'",
                    )
                ],
            )
        )
    return CaseResult(name=name, samples=results, threshold=threshold)


def make_result(cases, model: str = "test-model"):
    return SuiteResult(
        suite_name="demo", backend="mock", model=model, cases=cases, duration_s=1.0
    )


class TestResultProperties:
    def test_pass_rate(self):
        assert make_case("a", 0.5).pass_rate == 0.5

    def test_passed_respects_threshold(self):
        assert make_case("a", 0.5, threshold=0.5).passed
        assert not make_case("a", 0.5, threshold=1.0).passed

    def test_flaky_is_strictly_between(self):
        assert make_case("a", 0.5).flaky
        assert not make_case("a", 1.0).flaky
        assert not make_case("a", 0.0).flaky

    def test_first_failure_is_reported(self):
        assert "expected" in make_case("a", 0.5).first_failure()

    def test_no_failure_when_all_pass(self):
        assert make_case("a", 1.0).first_failure() == ""

    def test_sample_with_error_does_not_pass(self):
        s = SampleResult(index=0, output="", latency_ms=0.0, error="boom")
        assert not s.passed
        assert s.failure_summary() == "boom"

    def test_mean_latency(self):
        assert make_case("a", 1.0).mean_latency_ms == 10.0


class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        result = make_result([make_case("a", 1.0), make_case("b", 0.5)])
        path = save_baseline(result, tmp_path / "b.json")
        loaded = load_baseline(path)
        assert loaded["suite"] == "demo"
        assert len(loaded["cases"]) == 2

    def test_creates_parent_directory(self, tmp_path):
        path = save_baseline(make_result([make_case("a", 1.0)]), tmp_path / "x" / "y.json")
        assert path.exists()

    def test_baseline_excludes_outputs_and_latency(self, tmp_path):
        """Storing raw outputs would make every diff noise for a stochastic model."""
        path = save_baseline(make_result([make_case("a", 1.0)]), tmp_path / "b.json")
        case = json.loads(path.read_text())["cases"][0]
        assert set(case) == {"name", "pass_rate", "passed", "threshold", "samples"}

    def test_written_sorted_for_clean_diffs(self, tmp_path):
        path = save_baseline(make_result([make_case("a", 1.0)]), tmp_path / "b.json")
        text = path.read_text()
        assert text.endswith("\n")
        assert text.index('"backend"') < text.index('"cases"') < text.index('"created"')

    def test_missing_baseline_raises_with_guidance(self, tmp_path):
        with pytest.raises(BaselineError, match="promptdrift record"):
            load_baseline(tmp_path / "absent.json")

    def test_corrupt_baseline_raises(self, tmp_path):
        p = tmp_path / "b.json"
        p.write_text("{not json", encoding="utf-8")
        with pytest.raises(BaselineError, match="not valid JSON"):
            load_baseline(p)

    def test_version_mismatch_raises_with_guidance(self, tmp_path):
        p = tmp_path / "b.json"
        p.write_text(json.dumps({"version": 99, "cases": []}), encoding="utf-8")
        with pytest.raises(BaselineError, match="Re-record it"):
            load_baseline(p)

    def test_path_sanitises_suite_name(self):
        p = baseline_path("my suite/v2", ".promptdrift")
        assert "/" not in p.name and " " not in p.name
        assert p.name.endswith(".baseline.json")


class TestDiff:
    def _baseline_of(self, cases, tmp_path, model="test-model"):
        path = save_baseline(make_result(cases, model=model), tmp_path / "b.json")
        return load_baseline(path)

    def test_regression_detected(self, tmp_path):
        base = self._baseline_of([make_case("a", 1.0)], tmp_path)
        diff = diff_against(make_result([make_case("a", 0.0)]), base)
        assert diff.has_regressions
        assert diff.deltas[0].status == REGRESSED
        assert diff.deltas[0].before == 1.0
        assert diff.deltas[0].after == 0.0

    def test_regression_detail_carries_the_failure(self, tmp_path):
        base = self._baseline_of([make_case("a", 1.0)], tmp_path)
        diff = diff_against(make_result([make_case("a", 0.0)]), base)
        assert "expected" in diff.regressions[0].detail

    def test_improvement_detected(self, tmp_path):
        base = self._baseline_of([make_case("a", 0.0)], tmp_path)
        diff = diff_against(make_result([make_case("a", 1.0)]), base)
        assert diff.of(IMPROVED)
        assert not diff.has_regressions

    def test_unchanged_when_identical(self, tmp_path):
        base = self._baseline_of([make_case("a", 1.0)], tmp_path)
        diff = diff_against(make_result([make_case("a", 1.0)]), base)
        assert diff.of(UNCHANGED)
        assert not diff.has_regressions

    def test_degraded_when_still_passing_but_worse(self, tmp_path):
        """1.0 -> 0.5 while threshold is 0.5: green, but sliding."""
        base = self._baseline_of([make_case("a", 1.0, threshold=0.5)], tmp_path)
        diff = diff_against(
            make_result([make_case("a", 0.5, threshold=0.5)]), base, tolerance=0.2
        )
        assert diff.of(DEGRADED)
        assert not diff.has_regressions

    def test_small_drop_within_tolerance_is_unchanged(self, tmp_path):
        base = self._baseline_of([make_case("a", 1.0, threshold=0.25)], tmp_path)
        diff = diff_against(
            make_result([make_case("a", 0.75, threshold=0.25)]), base, tolerance=0.34
        )
        assert diff.of(UNCHANGED)

    def test_new_case_detected(self, tmp_path):
        base = self._baseline_of([make_case("a", 1.0)], tmp_path)
        diff = diff_against(make_result([make_case("a", 1.0), make_case("b", 1.0)]), base)
        assert [d.name for d in diff.of(NEW)] == ["b"]

    def test_removed_case_detected(self, tmp_path):
        base = self._baseline_of([make_case("a", 1.0), make_case("b", 1.0)], tmp_path)
        diff = diff_against(make_result([make_case("a", 1.0)]), base)
        assert [d.name for d in diff.of(REMOVED)] == ["b"]

    def test_new_failing_case_is_new_not_regressed(self, tmp_path):
        """It never passed, so it cannot have regressed."""
        base = self._baseline_of([make_case("a", 1.0)], tmp_path)
        diff = diff_against(make_result([make_case("a", 1.0), make_case("b", 0.0)]), base)
        assert not diff.has_regressions
        assert diff.of(NEW)[0].name == "b"

    def test_still_failing_is_not_a_regression(self, tmp_path):
        base = self._baseline_of([make_case("a", 0.0)], tmp_path)
        diff = diff_against(make_result([make_case("a", 0.0)]), base)
        assert not diff.has_regressions

    def test_model_change_is_flagged(self, tmp_path):
        base = self._baseline_of([make_case("a", 1.0)], tmp_path, model="llama3.2")
        diff = diff_against(make_result([make_case("a", 1.0)], model="mistral"), base)
        assert diff.model_changed

    def test_same_model_not_flagged(self, tmp_path):
        base = self._baseline_of([make_case("a", 1.0)], tmp_path, model="llama3.2")
        diff = diff_against(make_result([make_case("a", 1.0)], model="llama3.2"), base)
        assert not diff.model_changed
