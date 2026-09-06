# <img src="https://raw.githubusercontent.com/kecsap/opencode-tokenstats/master/assets/logo.svg" alt="opencode-tokenstats" width="40" valign="middle" /> opencode-tokenstats

<p align="center">
  <strong>See where your OpenCode tokens and cost actually go.</strong><br />
  Local-first CLI for session analytics, model cost breakdowns, tool usage, and exportable reports.
</p>

<p align="center">
  <a href="https://github.com/kecsap/opencode-tokenstats"><img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" /></a>
  <a href="https://github.com/kecsap/opencode-tokenstats/stargazers"><img src="https://img.shields.io/github/stars/kecsap/opencode-tokenstats?style=flat-square" alt="GitHub stars" /></a>
  <a href="https://github.com/kecsap/opencode-tokenstats"><img src="https://img.shields.io/badge/OpenCode-local--first-7c3aed" alt="OpenCode local-first" /></a>
</p>

opencode-tokenstats reads OpenCode session data and turns it into useful summaries:

- **token composition**: input, output, reasoning, cache read/write
- **costs by model**: API cost when present, estimates when needed
- **tool usage**: top tools, MCP servers, component families, OpenCode core usage
- **period reports**: daily, weekly, monthly, custom range, lifetime
- **machine output**: JSON or Markdown for downstream automation

It is built for one job: **make OpenCode usage legible without Grafana, proxies, or extra infra**.

---

## Why this exists

OpenCode already has the raw data. This project makes it readable.

- No dashboard stack to deploy
- No data leaving your machine in local mode
- No wrapper around OpenCode
- Fast enough for daily use
- Useful for both humans and scripts

---

## Install

### From PyPI

```bash
pip install opencode-tokenstats
octoken --help
```

### From source

```bash
git clone git@github.com:kecsap/opencode-tokenstats.git
cd opencode-tokenstats
pip install -e .[dev]
octoken --help
```

### Run without installing

```bash
PYTHONPATH=src python -m opencode_tokenstats.cli --help
PYTHONPATH=src python -m opencode_tokenstats.cli --mode local health
PYTHONPATH=src python -m opencode_tokenstats.cli --mode api health
```

---

## Quick start

```bash
# local mode is the default
octoken status
octoken daily
octoken weekly

# inspect one session
octoken session --session-id <session-id>

# export machine-readable output
octoken json --period daily --format json
```

### Common flows

```bash
# API-backed checks
octoken --mode api health
octoken --mode api weekly

# explicit time window
octoken range --from-date 2026-05-01 --to-date 2026-05-07

# warm tokenizer caches up front
octoken tokenizer-warmup --pair local:qwen3.6-27b --pair openai:gpt-5.3-codex
```

---

## What you get

The report focuses on the stuff that matters when AI usage gets expensive or noisy:

- **Period Summary** for totals and high-level usage
- **Token Composition** to separate productive output from context overhead
- **Model Costs** to see where spend concentrates
- **Top Tools** to spot tool-heavy sessions
- **Component Contribution** to understand MCP/skill/subagent families
- **OpenCode Contribution** to measure built-in tool overhead
- **MCP Servers** to isolate external tool usage
- **By Activity / Top Sessions** for period dashboards

If you know [CodeBurn](https://github.com/getagentseal/codeburn), the goal is similar: make AI usage visible. This project is narrower and lazier on purpose: **OpenCode-first, simple CLI, no extra stack**.

---

## Commands

| Command | What it does |
| --- | --- |
| `health` | Check data source health, tokenizer, and compatibility |
| `status` | Current mode, session count, latest session |
| `session` | Canonical summary for one session |
| `daily` | Aggregate last 24 hours |
| `weekly` | Aggregate last 7 days |
| `month [month]` | Aggregate last 30 days or a named month |
| `range --from-date --to-date` | Aggregate an explicit date window |
| `lifetime` | Aggregate all sessions |
| `json --period ... --format json|md` | Structured export |
| `tokenizer-warmup` | Preload tokenizer caches |

### Global options

- `--mode [local|api]`
- `--timeout <seconds>`
- `--retries <count>`
- `--no-warmup`
- `--model-alias-file <path>`
- `--session-filter <root1,root2,...>`

---

## Local mode vs API mode

### Local mode

- reads OpenCode SQLite data directly
- default mode
- fastest path for local analytics

### API mode

- talks to the OpenCode API
- useful when API telemetry is the source of truth
- good for compatibility and remote checks

Default base URL: `http://127.0.0.1:4096`

---

## Model aliases

Use `models.conf` to merge model variants under one label and mark local models as zero-cost:

```ini
gpt-unified = azure/gpt-5.4 openai/gpt-5.4
claude-pro = anthropic/claude-sonnet-4

@local myollama/* myllamacpp/*
@local *qwen36*
```

Load order:

1. `--model-alias-file`
2. `OPTOKEN_MODEL_ALIAS_FILE`
3. `./models.conf`

---

## Development

```bash
make help
make run
make run-api
make health
make health-api
make warmup
make test
make build
make install-wheel
```

Notes:

- `make run*` and `make health*` run from source with `PYTHONPATH=src`
- `make build` creates wheel and sdist in `dist/`

---

## Release to PyPI

```bash
python -m pip install -U build twine
rm -rf dist build *.egg-info
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

Recommended once, before first release:

1. create the `opencode-tokenstats` project on PyPI
2. switch to **trusted publishing** in PyPI instead of API tokens
3. add a license file
4. add screenshots to `assets/` and reference them with absolute GitHub URLs so they render on PyPI too

---

## README polish ideas

Want it to feel more like CodeBurn or Magic Context?

1. add a real dashboard screenshot: `assets/dashboard.png`
2. add a second image for session drill-down: `assets/session.png`
3. add a tiny animated GIF of `octoken weekly`
4. add a short "Why local-first?" section near the top
5. add a PyPI badge after first release
6. add a "sample report" block copied from real output

The shortest path: **logo + one good screenshot + tighter intro**. That gets you 80% of the fancy.
