# opencode-tokenstats

Local-first OpenCode TokenStats CLI.

## Install

```bash
pip install -e .[dev]
```

## Run Locally (No Installation)

```bash
PYTHONPATH=src python -m opencode_tokenstats.cli --help
PYTHONPATH=src python -m opencode_tokenstats.cli --mode local doctor
PYTHONPATH=src python -m opencode_tokenstats.cli --mode api doctor
```

This runs the CLI directly from source code without installing the package.

## CLI

```bash
octoken doctor
```

## Command Reference

- `doctor`: checks data source health; optional tokenizer and compatibility diagnostics
- `session`: show canonical summary for one session
- `status`: source mode + session count + latest session id
- `daily`: aggregate last 24h
- `weekly`: aggregate last 7d
- `month`: aggregate last 30d
- `range --from-date YYYY-MM-DD --to-date YYYY-MM-DD`: explicit window aggregate
- `json --period daily|weekly|month --format json|md`: canonical report schema output
- `tokenizer-warmup`: preload tokenizer caches

## Makefile Shortcuts

```bash
make help
make run
make run-api
make doctor
make doctor-api
make warmup
make test
make build
make install-wheel
```

Notes:
- `make run*` and `make doctor*` run from source (`PYTHONPATH=src`), no install required.
- `make build` creates wheel/sdist in `dist/`.

## Common Workflows

```bash
# Local default status + quick session aggregate
octoken status
octoken daily

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
octoken doctor --check-tokenizer
octoken doctor --check-tokenizer --model-id qwen3.6-27b
octoken --mode api doctor --check-tokenizer --provider-id openai --model-id gpt-5.3-codex
```

## Compatibility Mode Examples

```bash
# Conservative local-only compatibility signals
octoken doctor --compat-mode strict_local

# API-only strict compatibility signals
octoken --mode api doctor --compat-mode strict_api --compat-source api

# TokenScope-like heuristic schema estimates from observed tool calls
octoken doctor --compat-mode tokenscope_compat

# Run compatibility check for an explicit session id
octoken doctor --compat-mode tokenscope_compat --compat-session-id <session-id>
```

## Warmup Behavior

- Tokenizer warmup is enabled by default for normal commands.
- Disable with `--no-warmup` when needed.

```bash
# Default (auto warmup on)
octoken daily

# Disable warmup for this run
octoken --no-warmup daily

# Explicit preload command
octoken tokenizer-warmup --pair local:qwen3.6-27b --pair openai:gpt-5.3-codex
```
