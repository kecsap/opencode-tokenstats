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
