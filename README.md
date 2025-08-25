# Model Serving from Scratch

FastAPI server for serving language and speech models with batched inference and streaming support.

> [!NOTE]
> This project is a learning experiment and is not intended for production use.

## Models

- google/gemma-3-270m-it
- openai/whisper-large-v3-turbo

## Features

- **Text Generation**: Google Gemma 3 270M with token streaming via Server-Sent Events
- **Speech-to-Text**: OpenAI Whisper Large v3 Turbo (non-streaming only)
- **Batched Inference**: Efficient processing of multiple requests
- **Independent Completion**: Requests finish as soon as they're done (no batch stragglers)

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
- ✅ Whisper for STT
- ❌ Continuous batching
- ❌ KV caching optimization
