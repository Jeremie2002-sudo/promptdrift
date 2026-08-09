from __future__ import annotations

import json

from promptdrift.baseline import save_baseline
from promptdrift.cli import EXIT_FAILED, EXIT_OK, EXIT_REGRESSED, EXIT_USAGE, main
from promptdrift.report import render_markdown
from promptdrift.results import SuiteResult
from tests.conftest import suite_dict
from tests.test_baseline import make_case, make_result


def mock_backend(response: str = "BILLING"):
    return {
        "provider": "mock",
        "model": "test",
        "options": {"responses": {"charged twice": response}},
    }


class TestExitCodes:
    def test_pass_exits_zero(self, write_suite, tmp_path):
        path = write_suite(suite_dict(backend=mock_backend("BILLING")))
        code = main(["run", str(path), "--no-baseline",
                     "--baseline-dir", str(tmp_path / "bl")])
        assert code == EXIT_OK

    def test_failure_exits_one(self, write_suite, tmp_path):
        path = write_suite(suite_dict(backend=mock_backend("TECHNICAL")))
        code = main(["run", str(path), "--no-baseline",
                     "--baseline-dir", str(tmp_path / "bl")])
        assert code == EXIT_FAILED

    def test_hard_failure_beats_regression(self, write_suite, tmp_path):
        """Record green, then flip the answer. The case both regressed *and*
        fails outright, and the more urgent code wins."""
        bl = str(tmp_path / "bl")
        good = write_suite(suite_dict(backend=mock_backend("BILLING")), "good.yaml")
        assert main(["record", str(good), "--baseline-dir", bl]) == EXIT_OK

        bad = write_suite(suite_dict(backend=mock_backend("TECHNICAL")), "bad.yaml")
        assert main(["run", str(bad), "--baseline-dir", bl]) == EXIT_FAILED

    def test_degraded_but_still_passing_exits_two(self, write_suite, tmp_path):
        """The case exit 2 exists for: green, but measurably worse than before.

        Baseline records 100%. The run uses `variants` to make half the samples
        differ, landing at 50% -- still above a 0.25 threshold, so the suite
        passes, but the drop exceeds tolerance.
        """
        bl = tmp_path / "bl"
        bl.mkdir()
        save_baseline(
            make_result([make_case("billing", 1.0, threshold=0.25)]),
            bl / "demo.baseline.json",
        )

        data = suite_dict(samples=4, pass_threshold=0.25)
        data["cases"][0]["name"] = "billing"
        data["backend"] = {
            "provider": "mock", "model": "test-model",
            "options": {"responses": {"charged twice": "BILLING"}, "variants": 1},
        }
        path = write_suite(data)

        code = main(["run", str(path), "--baseline-dir", str(bl),
                     "--concurrency", "1", "--tolerance", "0.2"])
        assert code == EXIT_REGRESSED

    def test_removed_case_is_not_drift(self, write_suite, tmp_path):
        """A case dropped from the suite is reported, but does not fail anything."""
        bl = tmp_path / "bl"
        bl.mkdir()
        save_baseline(
            make_result([make_case("billing", 1.0), make_case("extra", 1.0)]),
            bl / "demo.baseline.json",
        )
        data = suite_dict(backend=mock_backend("BILLING"))
        data["cases"][0]["name"] = "billing"
        path = write_suite(data)
        assert main(["run", str(path), "--baseline-dir", str(bl)]) == EXIT_OK

    def test_malformed_suite_exits_three(self, write_suite, tmp_path):
        path = write_suite(suite_dict(nonsense_key=1))
        assert main(["run", str(path), "--no-baseline"]) == EXIT_USAGE

    def test_unknown_assertion_exits_three(self, write_suite, tmp_path):
        data = suite_dict(backend=mock_backend())
        data["cases"][0]["assert"] = [{"type": "vibes", "value": "good"}]
        path = write_suite(data)
        assert main(["run", str(path), "--no-baseline"]) == EXIT_USAGE

    def test_unknown_provider_exits_three(self, write_suite):
        path = write_suite(suite_dict(backend={"provider": "nope", "model": "m"}))
        assert main(["run", str(path), "--no-baseline"]) == EXIT_USAGE


class TestRecord:
    def test_writes_a_baseline(self, write_suite, tmp_path):
        bl = tmp_path / "bl"
        path = write_suite(suite_dict(backend=mock_backend("BILLING")))
        main(["record", str(path), "--baseline-dir", str(bl)])
        written = bl / "demo.baseline.json"
        assert written.exists()
        assert json.loads(written.read_text())["cases"][0]["passed"] is True

    def test_recording_a_failing_suite_exits_one(self, write_suite, tmp_path):
        path = write_suite(suite_dict(backend=mock_backend("WRONG")))
        code = main(["record", str(path), "--baseline-dir", str(tmp_path / "bl")])
        assert code == EXIT_FAILED

    def test_missing_baseline_does_not_crash_run(self, write_suite, tmp_path, capsys):
        path = write_suite(suite_dict(backend=mock_backend("BILLING")))
        code = main(["run", str(path), "--baseline-dir", str(tmp_path / "absent")])
        assert code == EXIT_OK
        assert "No baseline comparison" in capsys.readouterr().out


class TestOverrides:
    def test_samples_override(self, write_suite, tmp_path, capsys):
        path = write_suite(suite_dict(samples=1, backend=mock_backend("BILLING")))
        main(["run", str(path), "--no-baseline", "--samples", "5", "-v",
              "--baseline-dir", str(tmp_path)])
        assert "1 cases" in capsys.readouterr().out

    def test_model_override_is_reported(self, write_suite, tmp_path, capsys):
        path = write_suite(suite_dict(backend=mock_backend("BILLING")))
        main(["run", str(path), "--no-baseline", "--model", "other-model",
              "--baseline-dir", str(tmp_path)])
        assert "other-model" in capsys.readouterr().out


class TestOutputFiles:
    def test_writes_markdown(self, write_suite, tmp_path):
        out = tmp_path / "summary.md"
        path = write_suite(suite_dict(backend=mock_backend("BILLING")))
        main(["run", str(path), "--no-baseline", "--markdown", str(out),
              "--baseline-dir", str(tmp_path)])
        assert "PromptDrift" in out.read_text(encoding="utf-8")

    def test_writes_json(self, write_suite, tmp_path):
        out = tmp_path / "results.json"
        path = write_suite(suite_dict(backend=mock_backend("BILLING")))
        main(["run", str(path), "--no-baseline", "--json", str(out),
              "--baseline-dir", str(tmp_path)])
        assert json.loads(out.read_text())["suite"] == "demo"

    def test_appends_to_github_step_summary(self, write_suite, tmp_path, monkeypatch):
        summary = tmp_path / "step_summary.md"
        summary.write_text("", encoding="utf-8")
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
        path = write_suite(suite_dict(backend=mock_backend("BILLING")))
        main(["run", str(path), "--no-baseline", "--baseline-dir", str(tmp_path)])
        assert "PromptDrift" in summary.read_text(encoding="utf-8")


class TestAssertionsCommand:
    def test_lists_types(self, capsys):
        assert main(["assertions"]) == EXIT_OK
        out = capsys.readouterr().out
        assert "equals" in out and "judge" in out


class TestMarkdownReport:
    def test_pass_summary(self):
        md = render_markdown(make_result([make_case("a", 1.0)]), None)
        assert "PASS" in md and "1/1" in md

    def test_failure_table_lists_the_case(self):
        md = render_markdown(make_result([make_case("a", 0.0)]), None)
        assert "FAIL" in md
        assert "`a`" in md
        assert "Failing cases" in md

    def test_pipes_in_detail_are_escaped(self):
        result = make_result([make_case("a", 0.0)])
        result.cases[0].samples[0].assertions[0] = type(
            result.cases[0].samples[0].assertions[0]
        )(type="equals", passed=False, detail="got a|b|c")
        md = render_markdown(result, None)
        assert r"a\|b\|c" in md

    def test_flaky_note_included(self):
        md = render_markdown(make_result([make_case("a", 0.5, threshold=0.25)]), None)
        assert "Flaky" in md

    def test_empty_suite_renders(self):
        md = render_markdown(SuiteResult("s", "mock", "m", [], 0.0), None)
        assert "PASS" in md
