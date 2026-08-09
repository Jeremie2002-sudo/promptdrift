"""Command line interface.

Exit codes are part of the contract -- CI reads them:

  0  everything passed, nothing worse than baseline
  1  one or more cases failed outright
  2  every case still passes, but one got measurably worse than the baseline
  3  bad usage: malformed suite, unknown assertion, missing baseline

2 is separate from 1 on purpose. "Everything is green but this case slid from
100% to 60%" is not a build-breaker, but it is the early warning you want to see
in the PR -- before it starts failing intermittently on someone else's branch.
A workflow can treat 2 as a warning and 1 as fatal.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from rich.console import Console

from promptdrift import __version__
from promptdrift.assertions import AssertionError_, known_types
from promptdrift.backends.base import BackendError, build_backend
from promptdrift.baseline import (
    DEFAULT_BASELINE_DIR,
    DEFAULT_TOLERANCE,
    BaselineError,
    baseline_path,
    diff_against,
    load_baseline,
    save_baseline,
)
from promptdrift.report import render_diff, render_markdown, render_run
from promptdrift.runner import DEFAULT_CONCURRENCY, run_suite
from promptdrift.spec import SpecError, load_suite

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REGRESSED = 2
EXIT_USAGE = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promptdrift",
        description="Regression testing for LLM prompts.",
    )
    parser.add_argument("--version", action="version", version=f"promptdrift {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("suite", help="path to the suite YAML file")
        p.add_argument(
            "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
            help=f"parallel model calls (default: {DEFAULT_CONCURRENCY})",
        )
        p.add_argument("--samples", type=int, help="override samples per case")
        p.add_argument("--model", help="override the suite's model")
        p.add_argument("--provider", help="override the suite's provider")
        p.add_argument(
            "--baseline-dir", default=DEFAULT_BASELINE_DIR,
            help=f"where baselines live (default: {DEFAULT_BASELINE_DIR})",
        )
        p.add_argument("-v", "--verbose", action="store_true", help="show every case")

    run = sub.add_parser("run", help="run a suite and compare against its baseline")
    add_common(run)
    run.add_argument(
        "--no-baseline", action="store_true", help="skip baseline comparison entirely"
    )
    run.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE,
        help=f"pass-rate drop tolerated before flagging (default: {DEFAULT_TOLERANCE})",
    )
    run.add_argument("--markdown", help="write a Markdown summary to this path")
    run.add_argument("--json", dest="json_out", help="write raw results as JSON to this path")

    record = sub.add_parser("record", help="run a suite and save the result as the baseline")
    add_common(record)

    sub.add_parser("assertions", help="list available assertion types")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console(stderr=False)

    if args.command == "assertions":
        for name in known_types():
            console.print(f"  {name}")
        return EXIT_OK

    try:
        suite = load_suite(args.suite)
    except SpecError as exc:
        console.print(f"[bold red]Invalid suite:[/bold red] {exc}")
        return EXIT_USAGE

    if args.samples:
        suite = _replace(suite, samples=args.samples)
    if args.model or args.provider:
        backend_spec = suite.backend
        suite = _replace(
            suite,
            backend=type(backend_spec)(
                provider=args.provider or backend_spec.provider,
                model=args.model or backend_spec.model,
                temperature=backend_spec.temperature,
                max_tokens=backend_spec.max_tokens,
                options=backend_spec.options,
            ),
        )

    try:
        backend = build_backend(suite.backend)
    except BackendError as exc:
        console.print(f"[bold red]Backend error:[/bold red] {exc}")
        return EXIT_USAGE

    try:
        result = asyncio.run(
            run_suite(suite, backend=backend, concurrency=args.concurrency)
        )
    except AssertionError_ as exc:
        console.print(f"[bold red]Invalid assertion:[/bold red] {exc}")
        return EXIT_USAGE
    except BackendError as exc:
        console.print(f"[bold red]Backend error:[/bold red] {exc}")
        return EXIT_USAGE

    render_run(result, console, verbose=args.verbose)

    path = baseline_path(suite.name, args.baseline_dir)

    if args.command == "record":
        save_baseline(result, path)
        console.print(f"\n[green]Baseline saved:[/green] {path}")
        console.print("[dim]Commit it so future runs have something to compare against.[/dim]")
        return EXIT_OK if result.passed else EXIT_FAILED

    diff = None
    if not args.no_baseline:
        try:
            diff = diff_against(result, load_baseline(path), tolerance=args.tolerance)
            render_diff(diff, console)
        except BaselineError as exc:
            console.print(f"\n[yellow]No baseline comparison:[/yellow] {exc}")

    if args.markdown:
        Path(args.markdown).write_text(render_markdown(result, diff), encoding="utf-8")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(render_markdown(result, diff) + "\n")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if not result.passed:
        return EXIT_FAILED
    if diff and diff.has_drift:
        console.print(
            f"\n[bold yellow]{len(diff.drift)} case(s) worse than baseline[/bold yellow] "
            f"(still above threshold)."
        )
        return EXIT_REGRESSED
    return EXIT_OK


def _replace(obj, **changes):
    """dataclasses.replace, imported lazily to keep the module header short."""
    import dataclasses

    return dataclasses.replace(obj, **changes)


if __name__ == "__main__":
    sys.exit(main())
