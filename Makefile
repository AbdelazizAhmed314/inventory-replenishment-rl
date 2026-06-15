.PHONY: sync test check validate-design simulator-evidence baseline-evidence reward-audit train evaluate verify-training verify-evaluation demo run verify final-audit smoke

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

evaluate:
	uv run python scripts/evaluate_learned_policy.py

verify-training:
	uv run python scripts/verify_training_artifacts.py

verify-evaluation:
	uv run python scripts/verify_evaluation_artifacts.py

demo:
	uv run python scripts/demo_policy.py

run: train evaluate demo

verify: test check validate-design verify-training verify-evaluation
	uv run python scripts/verify_artifacts.py

final-audit: verify
	uv run python scripts/final_submission_audit.py

smoke: test check validate-design simulator-evidence baseline-evidence reward-audit
	uv run python -c "import inventory_replenishment_rl; print(inventory_replenishment_rl.__version__)"
