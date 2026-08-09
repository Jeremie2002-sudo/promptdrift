from __future__ import annotations

from typing import Any

import pytest
import yaml

from promptdrift.spec import BackendSpec, parse_suite

MINIMAL_SUITE: dict[str, Any] = {
    "name": "demo",
    "prompt": "Classify: {{ticket}}",
    "backend": {"provider": "mock", "model": "test"},
    "cases": [
        {
            "name": "billing",
            "vars": {"ticket": "charged twice"},
            "assert": [{"type": "equals", "value": "BILLING"}],
        }
    ],
}


def suite_dict(**overrides: Any) -> dict[str, Any]:
    """A valid suite mapping, with overrides merged at the top level."""
    import copy

    base = copy.deepcopy(MINIMAL_SUITE)
    base.update(copy.deepcopy(overrides))
    return base


@pytest.fixture
def suite():
    return parse_suite(suite_dict())


@pytest.fixture
def mock_spec():
    def _make(**options: Any) -> BackendSpec:
        return BackendSpec(provider="mock", model="test", options=options)

    return _make


@pytest.fixture
def write_suite(tmp_path):
    """Write a suite mapping to a YAML file and return its path."""

    def _write(data: dict[str, Any], name: str = "suite.yaml"):
        path = tmp_path / name
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    return _write
