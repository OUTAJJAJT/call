
.PHONY: install run debug clean lint

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m src --model Qwen/Qwen3-0.6B

clean:
	rm -rf __pycache__ src/__pycache__ .mypy_cache .pytest_cache
	rm -rf data/output/functions_results.json

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
