# opencode-tokenstats

Local-first OpenCode TokenStats CLI.

## Install

```bash
pip install -e .[dev]
```

## CLI

```bash
octoken doctor
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
