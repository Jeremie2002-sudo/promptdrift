"""Rendering: terminal output for humans, Markdown for CI.

Both renderers lead with what failed. A test tool that makes you scroll past 40
green rows to find the one red one is a test tool people stop reading.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.text import Text

from promptdrift.baseline import (
    DEGRADED,
    IMPROVED,
    NEW,
    REGRESSED,
    REMOVED,
    UNCHANGED,
    Diff,
)
from promptdrift.results import SuiteResult

_STATUS_STYLE = {
    REGRESSED: ("REGRESSED", "bold red"),
    DEGRADED: ("degraded", "yellow"),
    IMPROVED: ("improved", "green"),
    NEW: ("new", "cyan"),
    REMOVED: ("removed", "dim"),
    UNCHANGED: ("unchanged", "dim"),
}


def render_run(result: SuiteResult, console: Console, verbose: bool = False) -> None:
    """Print the outcome of a single run."""
    console.print()
    console.print(
        f"[bold]{result.suite_name}[/bold] "
        f"[dim]({result.backend}/{result.model}, {len(result.cases)} cases, "
        f"{result.duration_s:.1f}s)[/dim]"
    )

    failed = [c for c in result.cases if not c.passed]
    if failed:
        table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        table.add_column("", width=2)
        table.add_column("case")
        table.add_column("rate", justify="right")
        table.add_column("why")
        for case in failed:
            table.add_row(
                Text("x", style="bold red"),
                case.name,
                f"{case.pass_rate:.0%}",
                Text(_clip(case.first_failure()), style="red"),
            )
        console.print()
        console.print(table)

    if flaky := [c for c in result.flaky_cases if c.passed]:
        console.print()
        console.print("[yellow]Flaky[/yellow] [dim](passed, but not on every sample)[/dim]")
        for case in flaky:
            console.print(f"  [yellow]~[/yellow] {case.name} [dim]{case.pass_rate:.0%}[/dim]")

    if verbose:
        console.print()
        for case in result.cases:
            mark = "[green]v[/green]" if case.passed else "[red]x[/red]"
            console.print(
                f"  {mark} {case.name} [dim]{case.pass_rate:.0%} "
                f"{case.mean_latency_ms:.0f}ms[/dim]"
            )

    console.print()
    if result.passed:
        console.print(f"[bold green]PASS[/bold green]  {result.n_passed}/{len(result.cases)}")
    else:
        console.print(
            f"[bold red]FAIL[/bold red]  {result.n_passed}/{len(result.cases)} passed, "
            f"{result.n_failed} failed"
        )


def render_diff(diff: Diff, console: Console) -> None:
    """Print the comparison against the baseline."""
    if diff.model_changed:
        console.print()
        console.print(
            f"[yellow]Note:[/yellow] baseline was recorded against "
            f"[bold]{diff.baseline_model}[/bold], this run used "
            f"[bold]{diff.current_model}[/bold]. Differences may be the model, not your prompt."
        )

    ordered = (
        diff.of(REGRESSED)
        + diff.of(DEGRADED)
        + diff.of(NEW)
        + diff.of(REMOVED)
        + diff.of(IMPROVED)
    )
    if not ordered:
        console.print(f"[dim]No changes vs baseline ({len(diff.of(UNCHANGED))} cases).[/dim]")
        return

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("status")
    table.add_column("case")
    table.add_column("before", justify="right")
    table.add_column("after", justify="right")
    table.add_column("detail")

    for d in ordered:
        label, style = _STATUS_STYLE[d.status]
        table.add_row(
            Text(label, style=style),
            d.name,
            "-" if d.before is None else f"{d.before:.0%}",
            "-" if d.after is None else f"{d.after:.0%}",
            Text(_clip(d.detail), style="dim"),
        )

    console.print()
    console.print(table)


def render_markdown(result: SuiteResult, diff: Diff | None) -> str:
    """Markdown summary, for GITHUB_STEP_SUMMARY or a PR comment."""
    icon = "PASS" if result.passed else "FAIL"
    lines = [
        f"## PromptDrift: {icon} - `{result.suite_name}`",
        "",
        f"`{result.backend}/{result.model}` - {result.n_passed}/{len(result.cases)} cases "
        f"passed in {result.duration_s:.1f}s",
        "",
    ]

    if failed := [c for c in result.cases if not c.passed]:
        lines += ["### Failing cases", "", "| case | rate | why |", "|---|---|---|"]
        lines += [
            f"| `{c.name}` | {c.pass_rate:.0%} | {_md_escape(_clip(c.first_failure()))} |"
            for c in failed
        ]
        lines.append("")

    if diff and (changed := diff.of(REGRESSED) + diff.of(DEGRADED) + diff.of(IMPROVED)):
        lines += [
            "### Changes vs baseline",
            "",
            "| status | case | before | after |",
            "|---|---|---|---|",
        ]
        for d in changed:
            label = _STATUS_STYLE[d.status][0]
            before = "-" if d.before is None else f"{d.before:.0%}"
            after = "-" if d.after is None else f"{d.after:.0%}"
            lines.append(f"| {label} | `{d.name}` | {before} | {after} |")
        lines.append("")

    if flaky := [c for c in result.flaky_cases if c.passed]:
        names = ", ".join(f"`{c.name}`" for c in flaky)
        lines += [f"> Flaky (passed, but not on every sample): {names}", ""]

    return "\n".join(lines)


def _clip(text: str, limit: int = 90) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "..."


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|")
