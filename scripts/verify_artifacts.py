"""Verify the complete generated artifact contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_verifier(script_name: str) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script_name)],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    required_documents = [
        "README.md",
        "TRAINING_REPORT.md",
        "REWARD_HACKING_AUDIT.md",
        "EVALUATION_REPORT.md",
        "FAILURE_AND_SAFETY_ANALYSIS.md",
        "BUSINESS_MEMO.md",
        "SUBMISSION_CHECKLIST.md",
    ]
    missing = [name for name in required_documents if not (ROOT / name).exists()]
    if missing:
        raise SystemExit(f"Missing required documents: {missing}")

    run_verifier("verify_training_artifacts.py")
    run_verifier("verify_evaluation_artifacts.py")
    print(f"Complete artifact contract valid: {len(required_documents)} documents verified")


if __name__ == "__main__":
    main()
