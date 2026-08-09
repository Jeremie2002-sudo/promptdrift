from __future__ import annotations

import pytest

from promptdrift.spec import SpecError, load_suite, parse_suite
from tests.conftest import suite_dict


class TestParsing:
    def test_parses_a_minimal_suite(self, suite):
        assert suite.name == "demo"
        assert len(suite.cases) == 1
        assert suite.cases[0].assertions[0].type == "equals"

    def test_defaults(self, suite):
        assert suite.samples == 1
        assert suite.pass_threshold == 1.0
        assert suite.backend.temperature == 0.0

    def test_render_substitutes_vars(self, suite):
        assert suite.render(suite.cases[0]) == "Classify: charged twice"

    def test_render_tolerates_internal_whitespace(self):
        s = parse_suite(suite_dict(prompt="Hi {{ ticket }} bye"))
        assert s.render(s.cases[0]) == "Hi charged twice bye"

    def test_template_vars_discovered(self, suite):
        assert suite.template_vars() == {"ticket"}


class TestValidation:
    def test_rejects_unknown_top_level_key(self):
        with pytest.raises(SpecError, match="unknown key"):
            parse_suite(suite_dict(temperture=0.5))

    def test_rejects_unknown_case_key(self):
        data = suite_dict()
        data["cases"][0]["asserts"] = []
        with pytest.raises(SpecError, match="unknown key"):
            parse_suite(data)

    def test_rejects_missing_prompt(self):
        data = suite_dict()
        del data["prompt"]
        with pytest.raises(SpecError, match="missing required key 'prompt'"):
            parse_suite(data)

    def test_rejects_empty_cases(self):
        with pytest.raises(SpecError, match="non-empty list"):
            parse_suite(suite_dict(cases=[]))

    def test_rejects_case_without_assertions(self):
        data = suite_dict()
        data["cases"][0]["assert"] = []
        with pytest.raises(SpecError, match="non-empty list"):
            parse_suite(data)

    def test_rejects_duplicate_case_names(self):
        data = suite_dict()
        data["cases"].append(dict(data["cases"][0]))
        with pytest.raises(SpecError, match="duplicate case name"):
            parse_suite(data)

    def test_rejects_missing_template_var(self):
        data = suite_dict(prompt="Classify {{ticket}} for {{customer}}")
        with pytest.raises(SpecError, match=r"references \['customer'\]"):
            parse_suite(data)

    def test_rejects_unused_case_var(self):
        """The subtle one: prompt edited, cases not. Case silently tests nothing."""
        data = suite_dict()
        data["cases"][0]["vars"]["stale"] = "left over"
        with pytest.raises(SpecError, match=r"supplies \['stale'\]"):
            parse_suite(data)

    @pytest.mark.parametrize("bad", [0, -1, 1.5, "three", True])
    def test_rejects_bad_samples(self, bad):
        with pytest.raises(SpecError, match="positive integer"):
            parse_suite(suite_dict(samples=bad))

    @pytest.mark.parametrize("bad", [0.0, 1.5, -0.2])
    def test_rejects_bad_pass_threshold(self, bad):
        with pytest.raises(SpecError, match="pass_threshold"):
            parse_suite(suite_dict(pass_threshold=bad))

    def test_rejects_non_numeric_temperature(self):
        with pytest.raises(SpecError, match="expected a number"):
            parse_suite(suite_dict(backend={"provider": "mock", "model": "m",
                                            "temperature": "hot"}))

    def test_rejects_out_of_range_assertion_threshold(self):
        data = suite_dict()
        data["cases"][0]["assert"] = [{"type": "judge", "rubric": "x", "threshold": 2.0}]
        with pytest.raises(SpecError, match="between 0.0 and 1.0"):
            parse_suite(data)

    def test_error_message_includes_node_path(self):
        data = suite_dict()
        data["cases"][0]["assert"][0]["typo"] = 1
        with pytest.raises(SpecError, match=r"cases\[0\]\.assert\[0\]"):
            parse_suite(data)


class TestOverrides:
    def test_case_level_samples_wins(self):
        data = suite_dict(samples=5)
        data["cases"][0]["samples"] = 2
        s = parse_suite(data)
        assert s.samples_for(s.cases[0]) == 2

    def test_falls_back_to_suite_samples(self):
        s = parse_suite(suite_dict(samples=5))
        assert s.samples_for(s.cases[0]) == 5

    def test_case_level_threshold_wins(self):
        data = suite_dict(pass_threshold=1.0)
        data["cases"][0]["pass_threshold"] = 0.5
        s = parse_suite(data)
        assert s.threshold_for(s.cases[0]) == 0.5


class TestLoading:
    def test_loads_from_file(self, write_suite):
        s = load_suite(write_suite(suite_dict()))
        assert s.name == "demo"

    def test_missing_file_raises_spec_error(self, tmp_path):
        with pytest.raises(SpecError, match="could not read"):
            load_suite(tmp_path / "nope.yaml")

    def test_invalid_yaml_raises_spec_error(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("name: [unclosed\n", encoding="utf-8")
        with pytest.raises(SpecError, match="invalid YAML"):
            load_suite(p)
