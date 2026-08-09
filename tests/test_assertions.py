from __future__ import annotations

import pytest

from promptdrift.assertions import (
    FREE,
    GRADED,
    AssertionError_,
    _parse_judge_reply,
    known_types,
    run_free,
    run_judge,
    tier_of,
)
from promptdrift.backends.base import Completion
from promptdrift.spec import Assertion


def completion(text: str, latency_ms: float = 10.0) -> Completion:
    return Completion(text=text, latency_ms=latency_ms)


def check(type_: str, value=None, output: str = "", **kw) -> bool:
    return run_free(Assertion(type=type_, value=value, **kw), completion(output)).passed


class TestDeterministicAssertions:
    @pytest.mark.parametrize(
        ("type_", "value", "output", "expected"),
        [
            ("equals", "BILLING", "BILLING", True),
            ("equals", "BILLING", "  BILLING  ", True),   # output is stripped
            ("equals", "BILLING", "billing", False),      # but not case-folded
            ("equals", "BILLING", "BILLING now", False),
            ("iequals", "BILLING", "billing", True),
            ("iequals", "BILLING", " BiLLiNg ", True),
            ("contains", "BILL", "the BILL is due", True),
            ("contains", "BILL", "nothing here", False),
            ("not_contains", "sorry", "here is your answer", True),
            ("not_contains", "sorry", "sorry, I cannot", False),
            ("contains_any", ["A", "B"], "contains B", True),
            ("contains_any", ["A", "B"], "contains C", False),
            ("contains_all", ["A", "B"], "A and B", True),
            ("contains_all", ["A", "B"], "only A", False),
            ("one_of", ["YES", "NO"], "YES", True),
            ("one_of", ["YES", "NO"], "MAYBE", False),
            ("regex", r"^\d{3}-\d{4}$", "555-1234", True),
            ("regex", r"^\d{3}-\d{4}$", "not a number", False),
            ("max_words", 3, "one two three", True),
            ("max_words", 3, "one two three four", False),
            ("json_valid", None, '{"a": 1}', True),
            ("json_valid", None, "not json", False),
            ("json_has_keys", ["a", "b"], '{"a":1,"b":2}', True),
            ("json_has_keys", ["a", "z"], '{"a":1,"b":2}', False),
        ],
    )
    def test_cases(self, type_, value, output, expected):
        assert check(type_, value, output) is expected

    def test_contains_any_accepts_a_bare_string(self):
        assert check("contains_any", "solo", "a solo value") is True

    def test_json_has_keys_rejects_a_json_array(self):
        r = run_free(Assertion(type="json_has_keys", value=["a"]), completion("[1,2]"))
        assert not r.passed
        assert "expected a JSON object" in r.detail

    def test_max_latency_within_budget(self):
        r = run_free(Assertion(type="max_latency_ms", value=500), completion("x", 120))
        assert r.passed

    def test_max_latency_exceeded(self):
        r = run_free(Assertion(type="max_latency_ms", value=100), completion("x", 450))
        assert not r.passed
        assert "450ms" in r.detail

    def test_failure_detail_mentions_the_actual_output(self):
        r = run_free(Assertion(type="equals", value="A"), completion("B"))
        assert "'B'" in r.detail

    def test_long_output_is_truncated_in_detail(self):
        r = run_free(Assertion(type="equals", value="A"), completion("z" * 500))
        assert len(r.detail) < 200
        assert "..." in r.detail


class TestMisconfiguration:
    def test_unknown_type_raises(self):
        with pytest.raises(AssertionError_, match="unknown assertion type"):
            run_free(Assertion(type="definitely_not_real"), completion("x"))

    def test_missing_value_raises(self):
        with pytest.raises(AssertionError_, match="requires a 'value'"):
            run_free(Assertion(type="equals"), completion("x"))

    def test_invalid_regex_raises(self):
        with pytest.raises(AssertionError_, match="invalid regex"):
            run_free(Assertion(type="regex", value="([unclosed"), completion("x"))

    def test_graded_type_rejected_by_run_free(self):
        with pytest.raises(AssertionError_, match="not a deterministic assertion"):
            run_free(Assertion(type="judge", rubric="x"), completion("y"))


class TestTiers:
    def test_free_types_are_free(self):
        for t in ["equals", "contains", "regex", "json_valid", "max_words"]:
            assert tier_of(t) == FREE

    def test_judge_is_graded(self):
        assert tier_of("judge") == GRADED

    def test_unknown_tier_raises(self):
        with pytest.raises(AssertionError_):
            tier_of("nope")

    def test_judge_is_listed(self):
        assert "judge" in known_types()


class TestJudgeReplyParsing:
    @pytest.mark.parametrize(
        ("reply", "score"),
        [
            ('{"score": 0.9, "reason": "good"}', 0.9),
            ('```json\n{"score": 0.5, "reason": "ok"}\n```', 0.5),
            ('Sure!\n{"score": 1.0, "reason": "yes"}', 1.0),
            ('{"score": 1, "reason": "int"}', 1.0),
        ],
    )
    def test_extracts_score(self, reply, score):
        got, _ = _parse_judge_reply(reply)
        assert got == pytest.approx(score)

    def test_clamps_out_of_range_scores(self):
        assert _parse_judge_reply('{"score": 4.2}')[0] == 1.0
        assert _parse_judge_reply('{"score": -3}')[0] == 0.0

    def test_extracts_reason(self):
        _, reason = _parse_judge_reply('{"score": 0.8, "reason": "close enough"}')
        assert reason == "close enough"

    def test_unparseable_yields_none(self):
        assert _parse_judge_reply("I think it was fine, honestly")[0] is None


class FakeJudge:
    """A backend that returns a fixed judge reply."""

    name = "fake-judge"

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    async def complete(self, prompt: str, system: str | None = None) -> Completion:
        self.calls += 1
        self.last_prompt = prompt
        return Completion(text=self.reply, latency_ms=1.0)

    async def aclose(self) -> None:
        return None


class TestJudge:
    async def test_passes_above_threshold(self):
        judge = FakeJudge('{"score": 0.9, "reason": "accurate"}')
        r = await run_judge(
            Assertion(type="judge", rubric="Is it accurate?", threshold=0.7),
            completion("some output"),
            judge,
        )
        assert r.passed
        assert r.score == pytest.approx(0.9)

    async def test_fails_below_threshold(self):
        judge = FakeJudge('{"score": 0.4, "reason": "vague"}')
        r = await run_judge(
            Assertion(type="judge", rubric="Is it accurate?", threshold=0.7),
            completion("some output"),
            judge,
        )
        assert not r.passed
        assert "vague" in r.detail

    async def test_default_threshold_is_point_seven(self):
        judge = FakeJudge('{"score": 0.7}')
        r = await run_judge(
            Assertion(type="judge", rubric="x"), completion("out"), judge
        )
        assert r.passed

    async def test_rubric_and_output_reach_the_judge(self):
        judge = FakeJudge('{"score": 1.0}')
        await run_judge(
            Assertion(type="judge", rubric="MY RUBRIC"), completion("MY OUTPUT"), judge
        )
        assert "MY RUBRIC" in judge.last_prompt
        assert "MY OUTPUT" in judge.last_prompt

    async def test_unparseable_judge_reply_fails_loudly(self):
        judge = FakeJudge("no idea mate")
        r = await run_judge(Assertion(type="judge", rubric="x"), completion("y"), judge)
        assert not r.passed
        assert "unparseable" in r.detail

    async def test_missing_rubric_raises(self):
        with pytest.raises(AssertionError_, match="requires a 'rubric'"):
            await run_judge(Assertion(type="judge"), completion("y"), FakeJudge("{}"))
