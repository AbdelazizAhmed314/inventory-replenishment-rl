.PHONY: sync test check validate-design simulator-evidence baseline-evidence smoke

sync:
	uv sync --all-groups

test:
	uv run pytest

check:
	uv run ruff check .
	uv run ruff format --check .

validate-design:
	uv run python scripts/validate_mdp_design.py

simulator-evidence:
	uv run python scripts/generate_simulator_evidence.py

baseline-evidence:
	uv run python scripts/evaluate_baselines.py

smoke: test check validate-design simulator-evidence baseline-evidence
	uv run python -c "import inventory_replenishment_rl; print(inventory_replenishment_rl.__version__)"
