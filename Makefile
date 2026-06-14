.PHONY: sync test check smoke

sync:
	uv sync --all-groups

test:
	uv run pytest

check:
	uv run ruff check .
	uv run ruff format --check .

smoke: test check
	uv run python -c "import inventory_replenishment_rl; print(inventory_replenishment_rl.__version__)"
