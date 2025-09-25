src_dir = src
server = $(src_dir)/server.py

format:
	ruff format $(src_dir) && ruff check $(src_dir)

type-check:
	uv run pyright $(src_dir)

dev:
	uv run fastapi dev $(server)
