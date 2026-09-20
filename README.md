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

# compare read-only local query strategies
octoken local-query-benchmark --period lifetime

# override local period-query workers for reports
octoken --local-query-workers 4 weekly
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

Model Costs shows actual API charges separately from local estimates. `*` marks
rows made entirely from generic fallback rates; `†` marks later hosted-market
rates. Hosted-market and generic estimates are counterfactual pricing, not money
spent. Mixed-basis rows have no `*` marker.
- **Top Tools** to spot tool-heavy sessions
- **Component Contribution** to understand MCP/skill/subagent families
- **OpenCode Contribution** to measure built-in tool overhead
- **MCP Servers** to isolate external tool usage
- **By Activity / Top Sessions** for period dashboards
- **Period Trends** for correlated token usage, lines changed, and tokens per changed LOC

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
| `local-query-benchmark --period ...` | Compare read-only 1/2/4/8 local query strategies |
| `tokenizer-warmup` | Preload tokenizer caches |
| `pricing status` | Inspect ledger records, Fast aliases, and coverage gaps |
| `pricing import <source> --target <path>` | Merge reviewed dated records into the ledger |
| `pricing refresh --target <path>` | Propose dated records from the official OpenAI pricing page |
| `pricing snapshot --target <path>` | Preview or write the current all-provider models.dev snapshot |
| `pricing backfill <manifest> --target <path>` | Propose dated records from pinned models.dev Git revisions |

### Global options

- `--mode [local|api]`
- `--timeout <seconds>`
- `--retries <count>`
- `--no-warmup`
- `--model-alias-file <path>`
- `--session-filter <root1,root2,...>`
- `--loc-scope code|all` (default: `code`)
- `--loc-exclude <pattern1,pattern2,...>`
- `--local-query-workers auto|1|N` (default: `auto`)

Local period and lifetime reports keep one query for small collections. With the
fixed `auto` policy, collections of 1,800 or more session IDs use four concurrent
session-ID buckets. Every bucket contains at most 900 IDs, and workers never exceed
eight. Use `--local-query-workers 1` to force sequential bounded buckets for large
collections (`<=900` IDs still use one query), or
`N` to choose another bounded worker count. `local-query-benchmark` reads the local
database, compares 1/2/4/8 workers, prints a recommendation, and never writes
benchmark results, configuration, or database state.

Period reports show three fixed-width aggregate trend charts when selected sessions map
to Git repositories and have timestamped token telemetry. Git churn counts added plus
deleted lines in `HEAD` history; the report also shows signed net LOC. Sessions without
both sources are excluded from the charts, while the normal report totals are unchanged.

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

# Map local model IDs to hosted catalog IDs when pricing is available.
@market-model qwen3.8-27b* = openai/qwen3.8-27b
@cloud-equivalent qwen3.8-27b = openai/qwen3.8-27b
```

`@market-model` rules are authoritative for local-model pricing estimates.
Exact sources win; otherwise the most-specific wildcard wins, then the later
rule. Matching ignores case and separators in the final model-ID segment. If
an explicit market target is unavailable in the pricing ledger, its
`@cloud-equivalent` rule is tried, then the generic fallback is used; automatic
matching is not retried. These rules do not change aliases, `@local`
classification, actual costs, or report aggregation.

Load order:

1. `--model-alias-file`
2. `OPTOKEN_MODEL_ALIAS_FILE`
3. `./models.conf`

---

## Pricing ledger

Report costs come from a tracked historical ledger (`src/opencode_tokenstats/data/pricing-history.json`), not from live price lookups. Each record pins a rate to an effective period and the source where it was observed, so historical rates stay reviewable and reproducible. Ledger records can come from official provider pricing or models.dev catalog observations; official-provider records take precedence when both cover the same model and period.

### Maintaining rates

```bash
# review what is active, retired, and missing
octoken pricing status

# import reviewed dated records from a ledger-format JSON file
octoken pricing import reviewed-rates.json --target my-ledger.json --yes

# fetch the official OpenAI pricing page and propose dated records
octoken pricing refresh --effective-from 2026-09-18 --target my-ledger.json --yes

# collect the current all-provider models.dev catalog (preview unless --yes)
octoken pricing snapshot --target my-ledger.json --yes

# reconstruct historical records from reviewed, pinned models.dev revisions
octoken pricing backfill models-dev-history.json --target my-ledger.json --yes
```

- `pricing status` lists every record with its status (active/retired), effective period, source URL, retrieval time, and confidence, plus Fast aliases and coverage gaps. `--ledger` points it at any ledger file.
- `import`, `refresh`, and `backfill` are preview-only unless you pass `--yes`; writes always go to the explicit `--target` path, never to the bundled ledger. `snapshot` also previews unless `--yes` is passed.
- `snapshot` fetches the current models.dev catalog, records its URL, retrieval time, catalog revision, and `source.kind: "models.dev"`, then validates the complete schema before writing. It describes current catalog observations, not historical prices.
- A failed fetch or validation leaves the ledger unchanged.
- `refresh` only accepts official OpenAI https URLs (`openai.com` or `*.openai.com`); no third-party scrapers are supported. `--effective-from` (default: today UTC) stamps the proposed records.
- `backfill` requires a JSON list (or `{"revisions": [...]}`) of `{ "revision": "<40-char SHA>", "effective_from": "YYYY-MM-DD" }` entries. It is preview-only without `--yes`, and each catalog is fetched at its pinned revision.

### Merge rules

- A record's identity is `provider` + `model` + `service_profile`; a model ID ending in `-fast` gets its own `fast` record, all others are `standard`.
- Updates are append-only: a new rate closes the open record for that identity at the new `effective_from` and adds a new record. Existing history is never rewritten or deleted.
- A proposed record supersedes an open one only when their rates differ; identical rates are reported as `unchanged`.
- Models missing from a proposal are never retired.
- A proposed `effective_from` must start after the open record's start, or the command fails without writing.
- Official-provider records take precedence over `models.dev` catalog observations for pricing lookup. A catalog observation does not replace an official record solely because it is newer.

### Ledger format

```json
{
  "schema_version": 2,
  "unit": "USD per 1M tokens",
  "records": [
    {
      "provider": "openai",
      "model": "gpt-5.6-terra",
      "aliases": ["gpt-5.6-terra", "openai/gpt-5.6-terra"],
      "service_profile": "standard",
      "context": "short",
      "effective_from": "2026-09-18T00:00:00Z",
      "effective_to": null,
      "status": "active",
      "confidence": "observed/inferred",
      "billing_channel": "direct_api",
      "source_kind": "official",
      "source_revision": "",
      "observed_at": "2026-09-18T00:00:00Z",
      "source": {"url": "https://openai.com/api/pricing/", "retrieved_at": "2026-09-18T00:00:00Z", "kind": "official", "revision": ""},
      "rates": {"input": 2.0, "output": 0.2, "cacheRead": 2.5, "cacheWrite": 12.0}
    }
  ]
}
```

- Rates are USD per 1M tokens. `effective_to: null` means the period is open (active).
- `aliases` are the exact model spellings a call can match (with and without the `provider/` prefix).
- `source.url` and `source.retrieved_at` record where and when the rate was observed.
- Optional rate fields: `webSearch`, `fastMultiplier`, `tiers`, `contextOver200k`.
- Intervals for one identity must not overlap or run backwards.

### How reports price calls

Each API call in a report shows an `API` cost (what the OpenCode telemetry actually reported) and an `Est.` cost (computed from the rate active at the call's timestamp):

- A reported positive `API` cost is authoritative: it takes precedence over the estimate, and the model row's `Est.` cost is zeroed. Estimates apply only to calls without a reported API cost.
- For known models, the historical ledger record whose effective period contains the call's timestamp wins.
- A call before the first known period uses the earliest later rate and is counted as `future-fallback`, with a warning.
- Calls without a matching historical record use a flat rate from `OPENCODE_MODEL_PRICING_FILE` or a local `models.json` when available; otherwise they use legacy default rates for estimation. These fallback estimates do not add unpriced warning noise.
- The session report prints `Pricing coverage: NN% (n priced, n unpriced, n future-fallback)`. Estimated totals remain estimates, not bills.

Current official rates do not prove what you were billed historically: ledger records are observations tied to effective dates, and anything before the first observed record is an estimate. To correct historical rates, add dated records via `pricing import`.

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
