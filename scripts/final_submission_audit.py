"""Audit the final tracked repository and locally generated evidence."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def main() -> None:
    tracked = set(git_output("ls-files").splitlines())
    required_tracked = {
        "README.md",
        "BUSINESS_MEMO.md",
        "EVALUATION_REPORT.md",
        "FAILURE_AND_SAFETY_ANALYSIS.md",
        "REWARD_HACKING_AUDIT.md",
        "SUBMISSION_CHECKLIST.md",
        "TRAINING_REPORT.md",
        "Makefile",
    }
    missing_tracked = sorted(required_tracked - tracked)
    if missing_tracked:
        raise SystemExit(f"Missing tracked deliverables: {missing_tracked}")

    private_documents = {
        "Instructions.md",
        "IMPLEMENTATION_ROADMAP.md",
        "MDP_SPEC.md",
        "PROJECT_BRIEF.md",
    }
    unignored_private = [
        name
        for name in private_documents
        if subprocess.run(
            ["git", "check-ignore", "-q", name],
            cwd=ROOT,
            check=False,
        ).returncode
        != 0
    ]
    if unignored_private:
        raise SystemExit(f"Private planning documents are not ignored: {unignored_private}")

    tracked_generated = sorted(
        path
        for path in tracked
        if (path.startswith("artifacts/") or path.startswith("evidence/"))
        and not path.endswith(".gitkeep")
    )
    if tracked_generated:
        raise SystemExit(f"Generated outputs must remain untracked: {tracked_generated}")

    fragments = (
        "cour" + "se",
        "assign" + "ment",
        "lec" + "ture",
        "note" + "booklm",
        "mc" + "gill",
        "show" + "case",
    )
    reference_hits: list[str] = []
    for relative_path in sorted(tracked):
        path = ROOT / relative_path
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if any(fragment in text for fragment in fragments):
            reference_hits.append(relative_path)
    if reference_hits:
        raise SystemExit(f"Tracked files contain private-source references: {reference_hits}")

    evaluation = json.loads(
        (ROOT / "artifacts" / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    if evaluation["capacity_violations"] != 0:
        raise SystemExit("Evaluation contains capacity violations.")
    if evaluation["unseen_seed_start"] < 150000:
        raise SystemExit("Evaluation seed range overlaps training.")

    print(
        "Final repository audit passed: required deliverables tracked, generated outputs "
        "ignored, private planning documents ignored, and public references clean"
    )


if __name__ == "__main__":
    main()
