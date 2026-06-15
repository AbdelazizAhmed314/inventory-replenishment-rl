.PHONY: sync test check validate-design simulator-evidence baseline-evidence reward-audit train verify-training smoke

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

reward-audit:
	uv run python scripts/audit_reward.py

train:
	uv run python scripts/train_q_learning.py

verify-training:
	uv run python scripts/verify_training_artifacts.py

smoke: test check validate-design simulator-evidence baseline-evidence reward-audit
	uv run python -c "import inventory_replenishment_rl; print(inventory_replenishment_rl.__version__)"
