SHELL := /bin/bash

.PHONY: setup play arena test zip gate candidate

setup:
	uv sync

play:
	uv run python -m harness.play --white . --black baselines/greedy $(if $(FEN),--fen "$(FEN)")

arena:
	uv run python -m harness.arena --opponent baselines/greedy --games 20

test:
	uv run ruff check .
	uv run mypy
	uv run python -m unittest discover -s tests -v

zip:
	uv run python -m harness.package

gate:
	$(MAKE) test
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000

candidate: test
	uv run python -m harness.arena --opponent baselines/greedy --games 8 --base-ms 3000
	uv run python -m harness.arena --opponent baselines/minimax --games 8 --base-ms 3000
	uv run python -m harness.arena --opponent baselines/numba --games 8 --base-ms 3000
	$(MAKE) zip
