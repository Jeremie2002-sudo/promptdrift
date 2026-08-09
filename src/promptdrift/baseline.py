"""Baselines and regression diffing -- the core of the tool.

The question this answers is not "is my prompt good?" (unanswerable in the
abstract) but "is my prompt *worse than it was yesterday?*" (answerable, and the
one that actually blocks a merge).

A baseline is a committed snapshot of how each case scored. The next run diffs
against it and classifies every case into one of five buckets. Only REGRESSED
fails the build.

DEGRADED deserves its own bucket rather than being folded into "passing": a case
whose pass rate slid from 1.0 to 0.7 while staying above its threshold is still
green, but it is on its way to red and you want to see it in the PR *before* it
starts failing intermittently on someone else's branch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from promptdrift.results import SuiteResult

BASELINE_VERSION = 1
DEFAULT_BASELINE_DIR = ".promptdrift"

# A case's pass rate may drift by this much before it counts as DEGRADED.
# Sampling noise on a small n is real: at samples=3, one unlucky sample moves the
# rate by 0.33, so a tighter tolerance would cry wolf constantly.
DEFAULT_TOLERANCE = 0.34


class BaselineError(RuntimeError):
    pass


REGRESSED = "regressed"
DEGRADED = "degraded"
IMPROVED = "improved"
UNCHANGED = "unchanged"
NEW = "new"
REMOVED = "removed"


@dataclass(frozen=True)
class CaseDelta:
    name: str
    status: str
    before: float | None
    after: float | None
    detail: str = ""


@dataclass
class Diff:
    deltas: list[CaseDelta] = field(default_factory=list)
    baseline_created: str | None = None
    baseline_model: str | None = None
    current_model: str | None = None

    def of(self, status: str) -> list[CaseDelta]:
        return [d for d in self.deltas if d.status == status]

    @property
    def regressions(self) -> list[CaseDelta]:
        """Hard regressions: was passing, now failing."""
        return self.of(REGRESSED)

    @property
    def degradations(self) -> list[CaseDelta]:
        """Still passing its threshold, but measurably worse than the baseline."""
        return self.of(DEGRADED)

    @property
    def has_regressions(self) -> bool:
        return bool(self.regressions)

    @property
    def drift(self) -> list[CaseDelta]:
        """Everything that got worse, hard failure or not.

        A hard REGRESSED case also fails the suite outright, so on its own it
        would only ever surface as exit 1. Folding DEGRADED in here is what
        makes the "still green, but sliding" exit code reachable -- and that is
        the case worth having a distinct signal for, because it is the one a
        human would otherwise never notice.
        """
        return self.regressions + self.degradations

    @property
    def has_drift(self) -> bool:
        return bool(self.drift)

    @property
    def model_changed(self) -> bool:
        return (
            self.baseline_model is not None
            and self.current_model is not None
            and self.baseline_model != self.current_model
        )


def baseline_path(suite_name: str, directory: str | Path = DEFAULT_BASELINE_DIR) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in suite_name)
    return Path(directory) / f"{safe}.baseline.json"


def save_baseline(result: SuiteResult, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": BASELINE_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **result.to_dict(),
    }
    # sort_keys + trailing newline so baselines diff cleanly in a PR.
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def load_baseline(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise BaselineError(
            f"no baseline at {p}. Create one with:  promptdrift record <suite.yaml>"
        )
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BaselineError(f"{p} is not valid JSON: {exc}") from exc

    version = data.get("version")
    if version != BASELINE_VERSION:
        raise BaselineError(
            f"{p} was written by baseline format v{version}, this build expects "
            f"v{BASELINE_VERSION}. Re-record it with:  promptdrift record <suite.yaml>"
        )
    return data


def diff_against(
    result: SuiteResult, baseline: dict[str, Any], tolerance: float = DEFAULT_TOLERANCE
) -> Diff:
    """Classify each case by comparing the current run to the baseline."""
    before = {c["name"]: c for c in baseline.get("cases", [])}
    deltas: list[CaseDelta] = []

    for case in result.cases:
        prior = before.pop(case.name, None)
        now = case.pass_rate

        if prior is None:
            deltas.append(
                CaseDelta(
                    name=case.name,
                    status=NEW,
                    before=None,
                    after=now,
                    detail="not in baseline" + ("" if case.passed else " -- and failing"),
                )
            )
            continue

        was_passing = bool(prior.get("passed", False))
        was_rate = float(prior.get("pass_rate", 0.0))

        if was_passing and not case.passed:
            deltas.append(
                CaseDelta(
                    name=case.name,
                    status=REGRESSED,
                    before=was_rate,
                    after=now,
                    detail=case.first_failure(),
                )
            )
        elif not was_passing and case.passed:
            deltas.append(
                CaseDelta(name=case.name, status=IMPROVED, before=was_rate, after=now)
            )
        elif case.passed and (was_rate - now) > tolerance:
            deltas.append(
                CaseDelta(
                    name=case.name,
                    status=DEGRADED,
                    before=was_rate,
                    after=now,
                    detail=f"pass rate fell {was_rate:.0%} -> {now:.0%}, still above threshold",
                )
            )
        else:
            deltas.append(
                CaseDelta(name=case.name, status=UNCHANGED, before=was_rate, after=now)
            )

    # Anything left in `before` was in the baseline but not in this run.
    for name, prior in before.items():
        deltas.append(
            CaseDelta(
                name=name,
                status=REMOVED,
                before=float(prior.get("pass_rate", 0.0)),
                after=None,
                detail="in baseline but not in this suite",
            )
        )

    return Diff(
        deltas=deltas,
        baseline_created=baseline.get("created"),
        baseline_model=baseline.get("model"),
        current_model=result.model,
    )
