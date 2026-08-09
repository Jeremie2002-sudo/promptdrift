# PromptDrift

**Regression testing for LLM prompts.** Catch the prompt change that quietly made things worse, before it ships.

[![CI](https://github.com/Jeremie2002-sudo/promptdrift/actions/workflows/ci.yml/badge.svg)](https://github.com/Jeremie2002-sudo/promptdrift/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## The problem

You tweak a prompt to fix one bad case. It works. You ship it.

Three other cases broke and you won't find out until a user does.

This is the normal state of affairs for LLM features, and it isn't a discipline problem — it's a tooling gap. Unit tests assume deterministic output. LLMs don't give you that. So teams fall back to eyeballing a handful of examples in a notebook, which doesn't scale past about five cases and leaves no record of what used to work.

PromptDrift treats prompts the way you'd treat any other code: **define cases, snapshot the current behaviour, and fail the build when a change makes it worse.**

## The distinction that matters

PromptDrift does not try to tell you whether your prompt is *good*. That question is unanswerable in the abstract and every tool that claims to answer it is really just asking a second LLM to guess.

It answers a narrower question that's actually decidable, and is the one that blocks a merge:

> **Is this prompt worse than it was yesterday?**

That reframing is what makes the whole thing tractable. You don't need absolute ground truth — you need a committed baseline and a diff.

## Quickstart

```bash
pip install git+https://github.com/Jeremie2002-sudo/promptdrift
```

Not on PyPI — install from the repo. Python 3.10+.

See it work immediately, with no model and no network:

```bash
promptdrift run examples/offline_demo/suite.yaml --no-baseline -v
```

For real use it runs against a **local model via [Ollama](https://ollama.com) — no API key, no cost.** PromptDrift's own test suite uses a deterministic mock backend, so CI needs no model at all.

```bash
ollama pull llama3.2
```

Write a suite:

```yaml
# examples/support_triage/suite.yaml
name: support-triage
description: Route inbound support tickets to the right queue

backend:
  provider: ollama
  model: llama3.2
  temperature: 0

prompt: |
  Classify the support ticket into exactly one of:
  BILLING, TECHNICAL, ACCOUNT, OTHER.
  Reply with only the label, nothing else.

  Ticket: {{ticket}}

samples: 3            # run each case 3x -- LLMs are not deterministic
pass_threshold: 1.0   # all 3 must pass

cases:
  - name: double-charge
    vars:
      ticket: "I was charged twice for my subscription this month."
    assert:
      - type: equals
        value: BILLING

  - name: password-reset
    vars:
      ticket: "The reset link in my email keeps expiring before I can use it."
    assert:
      - type: one_of
        value: [ACCOUNT, TECHNICAL]
      - type: max_words
        value: 3
```

Record a baseline, then use it:

```bash
promptdrift record examples/support_triage/suite.yaml
```

```bash
promptdrift run examples/support_triage/suite.yaml
```

Now change the prompt and run it again. Cases that used to pass and now don't are reported as `REGRESSED`, and the command exits non-zero.

## What a run looks like

```
support-triage (ollama/llama3.2, 6 cases, 4.2s)

  x  vague-complaint     33%   expected one of ['OTHER'], got 'ACCOUNT'

Flaky (passed, but not on every sample)
  ~  refund-request      67%

FAIL  5/6 passed, 1 failed

status       case              before  after  detail
REGRESSED    vague-complaint     100%    33%  expected one of ['OTHER'], got 'ACCOUNT'
degraded     refund-request      100%    67%  pass rate fell 100% -> 67%, still above threshold
```

## Design decisions

The interesting parts of this project are the judgement calls, so they're documented rather than buried.

### Sampling, because a single run tells you nothing

An LLM at `temperature: 0` is still not deterministic — batching, hardware, and quantisation all introduce variance. A case that runs once and passes has told you almost nothing.

So every case runs `samples: N` times and passes only if at least `pass_threshold` of those samples pass. A prompt that works 70% of the time is a *different thing* from one that works always, and the tool should be able to tell them apart.

`flaky` is reported as its own state: a case at 67% is still green against a 0.5 threshold, but you want to see it before it starts failing intermittently on someone else's branch.

### Deterministic assertions run first, and can short-circuit the judge

Assertions come in two tiers:

| tier | examples | cost |
|---|---|---|
| `FREE` | `equals`, `contains`, `regex`, `json_valid`, `max_words`, `max_latency_ms` | none |
| `GRADED` | `judge` (LLM-as-judge with a rubric) | one model call |

Free assertions are evaluated first, and if any fails, the graded ones are skipped. There's no reason to pay a judge to confirm that output which was supposed to be the single word `BILLING`, and is instead three paragraphs, is bad.

The corollary is a recommendation: **reach for a judge last.** Most regressions are caught by a cheap string check, and every judge call is a place where a second fallible model gets a vote.

### The baseline stores pass rates, not outputs

A baseline snapshots *whether each case passed and how reliably*. It deliberately does not store raw model outputs or latencies.

Storing outputs seems obviously useful and is actually a trap: for a non-deterministic model they change on every run without indicating anything, so the baseline diff becomes pure noise and people stop reading it. Latency is machine-dependent and would do the same.

Baselines are JSON, sorted, one case per line — meant to be committed and reviewed in a PR diff.

### Failures are captured per-sample, not per-run

One model timeout shouldn't discard the other 59 results. Backend errors are recorded against the sample that hit them and the run continues.

### Silent degradation gets its own exit code

| code | meaning |
|---|---|
| `0` | passed, nothing worse than baseline |
| `1` | one or more cases failed outright |
| `2` | every case still passes, but one got measurably worse than the baseline |
| `3` | bad usage — malformed suite, unknown assertion, missing baseline |

Code `2` is the interesting one. A case that slid from 100% to 60% while its threshold sits at 50% is still green — nothing is broken, no build should fail — but it is on a trajectory, and nobody is going to notice by reading a wall of passing tests. Giving it a distinct exit code lets a workflow surface it as a warning without blocking the merge.

This is worth calling out because the obvious design is wrong: if you define "regression" only as *was passing, now failing*, then the regression case always fails the suite too, exit `1` always wins, and exit `2` is unreachable code. Degradation is the thing that actually needs its own signal.

### The suite parser rejects unused variables

If a case supplies `{{customer_name}}` but the prompt no longer references it, that's an error — not a warning.

It usually means the prompt was edited and the cases weren't, so the case has silently stopped testing what its author thinks it tests. A test tool going green while testing nothing is the worst failure mode available, so the parser is strict about every unknown key and every unused variable.

### A mock backend is a first-class citizen

`provider: mock` is deterministic and never touches the network. It's what PromptDrift's own test suite runs against, which means CI exercises the entire pipeline — runner, concurrency, assertions, baseline diffing, exit codes — with no API key, no cost, and no flakes.

A test tool whose own tests are flaky has no business telling you your prompts are unstable.

## Assertion types

```bash
promptdrift assertions
```

| type | checks |
|---|---|
| `equals` / `iequals` | exact match (optionally case-insensitive) |
| `contains` / `not_contains` | substring present / absent |
| `contains_any` / `contains_all` | any / all of a list |
| `one_of` | output is one of an allowed set |
| `regex` | matches a pattern |
| `json_valid` | parses as JSON |
| `json_has_keys` | JSON object containing given keys |
| `max_words` | output length ceiling |
| `max_latency_ms` | per-call latency budget |
| `judge` | LLM-as-judge against a `rubric`, passing at `threshold` |

## Use in CI

```yaml
- name: Prompt regression tests
  run: promptdrift run prompts/triage.yaml --markdown $GITHUB_STEP_SUMMARY
```

PromptDrift writes its summary to `GITHUB_STEP_SUMMARY` automatically when that variable is set, so the result renders on the workflow page.

## Backends

| provider | notes |
|---|---|
| `ollama` | local, free, the default |
| `groq` | hosted; reads `GROQ_API_KEY` from the environment, never from the suite file |
| `mock` | deterministic, offline, for tests |

Compare two models against the same suite without touching the YAML:

```bash
promptdrift run suite.yaml --provider groq --model llama-3.3-70b-versatile
```

## Development

```bash
git clone https://github.com/Jeremie2002-sudo/promptdrift && cd promptdrift && pip install -e ".[dev]"
```

```bash
pytest
```

```bash
ruff check .
```

157 tests, all offline — the mock backend means the suite needs no model, no API key and no network.

## License

MIT — see [LICENSE](LICENSE).
