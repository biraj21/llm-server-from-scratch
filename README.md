# LLM Serving from Scratch

FastAPI server for Google's Gemma 3 270M model with streaming support.

> [!NOTE]
> This project is a learning experiment and is not intended for production use.

## Setup

Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv install
source .venv/bin/activate
echo "HF_TOKEN=your_token" > .env
make dev
```

## Usage

```bash
# Non-streaming
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"text": "What is AI?", "max_output_tokens": 100}'

# Streaming
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Write a story", "stream": true}'
```

## Commands

- `make dev` - Start FastAPI dev server
- `make format` - Format code using ruff
- `make type-check` - Type check using pyright

## Current Status

- ✅ Basic inference
- ✅ Token streaming
- ✅ Request batching
- ✅ Concurrent inference
- ❌ Continuous batching
- ❌ KV caching optimization
