# opencode-tokenstats

Local-first OpenCode TokenStats CLI.

## Install

```bash
pip install -e .[dev]
```

## Run Locally (No Installation)

```bash
PYTHONPATH=src python -m opencode_tokenstats.cli --help
python src/opencode_tokenstats/cli.py --help
PYTHONPATH=src python -m opencode_tokenstats.cli --mode local health
PYTHONPATH=src python -m opencode_tokenstats.cli --mode api health
```

This runs the CLI directly from source code without installing the package.

## CLI

```bash
octoken health
```

## Command Reference

- `health`: checks data source health; optional tokenizer and compatibility diagnostics
- `session`: show canonical summary for one session
- `status`: source mode + session count + latest session id
- `daily`: aggregate last 24h
- `weekly`: aggregate last 7d
- `month [month]`: aggregate last 30d, or stats for a specific month (e.g. `month may`, `month 05`)
- `lifetime`: aggregate all available sessions
- `range --from-date YYYY-MM-DD --to-date YYYY-MM-DD`: explicit window aggregate (e.g. `--from-date 2026-05-01 --to-date 2026-05-07`)
- `json --period daily|weekly|month|lifetime --format json|md`: canonical report schema output
- `tokenizer-warmup`: preload tokenizer caches

### Global Options

- `--model-alias-file <path>`: path to models.conf alias file (overrides default search)

## Makefile Shortcuts

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
- `make run*` and `make health*` run from source (`PYTHONPATH=src`), no install required.
- `make build` creates wheel/sdist in `dist/`.

## Common Workflows

```bash
# Local default status + quick session aggregate
octoken status
octoken daily
octoken lifetime

# API mode
octoken --mode api status
octoken --mode api weekly

# One specific session
octoken session --session-id <session-id>

# Canonical machine output
octoken json --period month --format json
octoken json --period daily --format md
```

## Tokenizer Check Examples

```bash
octoken health --check-tokenizer
octoken health --check-tokenizer --model-id qwen3.6-27b
octoken --mode api health --check-tokenizer --provider-id openai --model-id gpt-5.3-codex
```

## Compatibility Mode Examples

```bash
# Conservative local-only compatibility signals
octoken health --compat-mode strict_local

# API-only strict compatibility signals
octoken --mode api health --compat-mode strict_api --compat-source api

# TokenScope-like heuristic schema estimates from observed tool calls
octoken health --compat-mode tokenscope_compat

# Run compatibility check for an explicit session id
octoken health --compat-mode tokenscope_compat --compat-session-id <session-id>
```

## Warmup Behavior

- Tokenizer warmup is enabled by default for normal commands.
- Warmup runs in parallel by default (up to 4 workers) for faster cache loading.
- Disable with `--no-warmup` when needed.

```bash
# Default (auto warmup on)
octoken daily

# Disable warmup for this run
octoken --no-warmup daily

# Explicit preload command
octoken tokenizer-warmup --pair local:qwen3.6-27b --pair openai:gpt-5.3-codex
```

## Model Costs

The Model Costs table shows both API costs (from telemetry) and estimated costs (from pricing lookup):

- **Cost (API)**: Actual cost reported by the API provider
- **Cost (Est)**: Estimated cost based on token counts and pricing data
- Primary cost uses API cost when available, falls back to estimated cost otherwise

## Component Contribution

The Component Contribution table groups tools by component family (e.g., `lean-ctx`, `jcodemunch`, `codegraph2`). Each row aggregates all tools within the same family, showing combined token usage and call counts. Skills and subagents are also included as separate families.

## Model Aliases (models.conf)

Merge multiple model IDs under a single alias via `models.conf`:

```
# models.conf (in current working directory)
gpt-unified = azure/gpt-5.4 openai/gpt-5.4
claude-pro = anthropic/claude-sonnet-4

# Mark models as local (no API cost) using wildcard patterns
@local myollama/* myllamacpp/*
@local *qwen36*
```

**Configuration:**
- File format: `alias_name = model1 model2 model3`
- Local models: `@local pattern1 pattern2` (supports `*` wildcard)
- Comments: lines starting with `#`
- Blank lines are skipped

**Load locations (priority order):**
1. `--model-alias-file` CLI option
2. `OPTOKEN_MODEL_ALIAS_FILE` environment variable
3. Current working directory: `models.conf`

Aliases are applied when aggregating model costs across sessions. Local models have API cost set to 0.
