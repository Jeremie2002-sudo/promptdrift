"""Suite specification: parsing and validation of the YAML eval file.

Validation is deliberately strict and eager. A typo in an assertion type or a
case variable that the prompt never uses is a silent no-op at runtime, which is
the worst possible failure mode for a test tool -- the suite goes green and
tells you nothing. So every unknown key is an error, raised with the path to
the offending node.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Matches {{var}} / {{ var }} in prompt templates.
_TEMPLATE_VAR = re.compile(r"\{\{\s*(\w+)\s*\}\}")


class SpecError(ValueError):
    """Raised when a suite file is malformed. Message includes the node path."""


@dataclass(frozen=True)
class BackendSpec:
    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int | None = None
    # Provider-specific extras (base_url, api_key_env, ...).
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Assertion:
    type: str
    value: Any = None
    # Only used by model-graded assertions.
    rubric: str | None = None
    threshold: float | None = None


@dataclass(frozen=True)
class Case:
    name: str
    vars: dict[str, Any]
    assertions: tuple[Assertion, ...]
    # Per-case overrides; fall back to suite-level when None.
    samples: int | None = None
    pass_threshold: float | None = None


@dataclass(frozen=True)
class Suite:
    name: str
    prompt: str
    backend: BackendSpec
    cases: tuple[Case, ...]
    description: str = ""
    samples: int = 1
    pass_threshold: float = 1.0
    system: str | None = None

    def template_vars(self) -> set[str]:
        """Variable names referenced by the prompt template."""
        return set(_TEMPLATE_VAR.findall(self.prompt))

    def render(self, case: Case) -> str:
        """Substitute case vars into the prompt template."""
        return _TEMPLATE_VAR.sub(lambda m: str(case.vars[m.group(1)]), self.prompt)

    def samples_for(self, case: Case) -> int:
        return case.samples if case.samples is not None else self.samples

    def threshold_for(self, case: Case) -> float:
        return case.pass_threshold if case.pass_threshold is not None else self.pass_threshold


_SUITE_KEYS = {
    "name", "description", "prompt", "system", "backend",
    "samples", "pass_threshold", "cases",
}
_CASE_KEYS = {"name", "vars", "assert", "samples", "pass_threshold"}
_BACKEND_KEYS = {"provider", "model", "temperature", "max_tokens", "options"}
_ASSERT_KEYS = {"type", "value", "rubric", "threshold"}


def _reject_unknown(node: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(node) - allowed
    if unknown:
        raise SpecError(
            f"{path}: unknown key(s) {sorted(unknown)}. Allowed: {sorted(allowed)}"
        )


def _require(node: dict[str, Any], key: str, path: str) -> Any:
    if key not in node:
        raise SpecError(f"{path}: missing required key '{key}'")
    return node[key]


def _parse_backend(node: Any, path: str) -> BackendSpec:
    if not isinstance(node, dict):
        raise SpecError(f"{path}: expected a mapping, got {type(node).__name__}")
    _reject_unknown(node, _BACKEND_KEYS, path)
    temperature = node.get("temperature", 0.0)
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise SpecError(f"{path}.temperature: expected a number, got {temperature!r}")
    return BackendSpec(
        provider=str(_require(node, "provider", path)),
        model=str(_require(node, "model", path)),
        temperature=float(temperature),
        max_tokens=node.get("max_tokens"),
        options=dict(node.get("options") or {}),
    )


def _parse_assertion(node: Any, path: str) -> Assertion:
    if not isinstance(node, dict):
        raise SpecError(f"{path}: expected a mapping, got {type(node).__name__}")
    _reject_unknown(node, _ASSERT_KEYS, path)
    threshold = node.get("threshold")
    if threshold is not None and not 0.0 <= float(threshold) <= 1.0:
        raise SpecError(f"{path}.threshold: must be between 0.0 and 1.0, got {threshold}")
    return Assertion(
        type=str(_require(node, "type", path)),
        value=node.get("value"),
        rubric=node.get("rubric"),
        threshold=None if threshold is None else float(threshold),
    )


def _parse_case(node: Any, path: str) -> Case:
    if not isinstance(node, dict):
        raise SpecError(f"{path}: expected a mapping, got {type(node).__name__}")
    _reject_unknown(node, _CASE_KEYS, path)

    raw_assertions = _require(node, "assert", path)
    if not isinstance(raw_assertions, list) or not raw_assertions:
        raise SpecError(f"{path}.assert: expected a non-empty list")

    return Case(
        name=str(_require(node, "name", path)),
        vars=dict(node.get("vars") or {}),
        assertions=tuple(
            _parse_assertion(a, f"{path}.assert[{i}]") for i, a in enumerate(raw_assertions)
        ),
        samples=node.get("samples"),
        pass_threshold=node.get("pass_threshold"),
    )


def parse_suite(raw: dict[str, Any], source: str = "<suite>") -> Suite:
    """Build a validated Suite from an already-loaded mapping."""
    if not isinstance(raw, dict):
        raise SpecError(f"{source}: expected a mapping at the top level")
    _reject_unknown(raw, _SUITE_KEYS, source)

    raw_cases = _require(raw, "cases", source)
    if not isinstance(raw_cases, list) or not raw_cases:
        raise SpecError(f"{source}.cases: expected a non-empty list")

    samples = raw.get("samples", 1)
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
        raise SpecError(f"{source}.samples: expected a positive integer, got {samples!r}")

    threshold = raw.get("pass_threshold", 1.0)
    if not 0.0 < float(threshold) <= 1.0:
        raise SpecError(
            f"{source}.pass_threshold: expected a value in (0.0, 1.0], got {threshold!r}"
        )

    suite = Suite(
        name=str(_require(raw, "name", source)),
        description=str(raw.get("description", "")),
        prompt=str(_require(raw, "prompt", source)),
        system=raw.get("system"),
        backend=_parse_backend(_require(raw, "backend", source), f"{source}.backend"),
        samples=samples,
        pass_threshold=float(threshold),
        cases=tuple(_parse_case(c, f"{source}.cases[{i}]") for i, c in enumerate(raw_cases)),
    )

    _validate_case_vars(suite, source)
    _validate_unique_names(suite, source)
    return suite


def _validate_case_vars(suite: Suite, source: str) -> None:
    """Every template var must be supplied, and every supplied var must be used.

    The second half matters as much as the first: an unused var usually means
    the prompt was edited and the cases were not, so the case is no longer
    testing what its author thinks it tests.
    """
    required = suite.template_vars()
    for i, case in enumerate(suite.cases):
        supplied = set(case.vars)
        if missing := required - supplied:
            raise SpecError(
                f"{source}.cases[{i}] ({case.name}): prompt references {sorted(missing)} "
                f"but the case does not supply {'it' if len(missing) == 1 else 'them'}"
            )
        if unused := supplied - required:
            raise SpecError(
                f"{source}.cases[{i}] ({case.name}): supplies {sorted(unused)} "
                f"but the prompt never references {'it' if len(unused) == 1 else 'them'}"
            )


def _validate_unique_names(suite: Suite, source: str) -> None:
    seen: set[str] = set()
    for i, case in enumerate(suite.cases):
        if case.name in seen:
            raise SpecError(f"{source}.cases[{i}]: duplicate case name {case.name!r}")
        seen.add(case.name)


def load_suite(path: str | Path) -> Suite:
    """Load and validate a suite from a YAML file."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"could not read suite file {p}: {exc}") from exc

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SpecError(f"{p}: invalid YAML: {exc}") from exc

    return parse_suite(raw, source=str(p))
