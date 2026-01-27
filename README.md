# LLM Provider Feature Catalog (SA Playbook)

**Purpose:** A provider-neutral LLM API feature catalog + migration guide for Cerebras SAs.

This repo helps SAs answer:
- Does provider X support feature Y (logprobs, structured outputs, tool calling, streaming)?
- How does it behave operationally (limits, burst queueing, priority, caching)?
- What rework is required to migrate to a different provider API?

## Python Environment (uv)

### Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Sync dependencies
```bash
uv sync
```

## Running examples (multi-provider)

Most Python examples in `examples/python/` accept `--provider` and `--model` arguments and will
look up the correct API key environment variable for that provider.

### Common env vars
Set the one(s) you need for the provider you’re testing:

- `CEREBRAS_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GROQ_API_KEY`
- `FIREWORKS_API_KEY`
- `TOGETHER_API_KEY`
- AWS Bedrock: standard AWS credentials + `AWS_REGION` (or `AWS_DEFAULT_REGION`)

### Streaming
```bash
uv run --env-file .env python examples/python/streaming.py --provider cerebras --model zai-glm-4.7
```

### Chat completions (non-streaming)
```bash
uv run --env-file .env python examples/python/chat_completions.py --provider groq --model llama-3.1-70b-versatile
```

### Structured outputs
```bash
uv run --env-file .env python examples/python/structured_output.py --provider cerebras --model zai-glm-4.7 --mode both
```

### Dev dependencies
```bash
uv sync --extra dev
```

## What’s in here
- docs/ — canonical feature definitions + SA guidance
- data/ — machine-readable feature matrix (CSV/JSON)
- examples/ — parsing/streaming/tools/logprobs snippets

_Last updated: 2026-01-27_
