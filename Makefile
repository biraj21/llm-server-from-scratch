format:
	ruff format && ruff check

type-check:
	uv run pyright .

dev:
	uv run fastapi dev main.py
