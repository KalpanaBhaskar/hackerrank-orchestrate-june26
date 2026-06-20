"""
Phase 9 — Submission Packaging Script
======================================
Creates code.zip for submission.

Includes:
  code/           (all Python modules, README, requirements.txt)
  evaluation/     (evaluation_report.md, eval_*.csv)
  .env.local      (template only — not the real key file)

Excludes:
  __pycache__/, .cache/, telemetry.jsonl, dataset/, .git/,
  venv/, node_modules/, *.pyc, .env (real secrets file)

Usage:
    python scripts/package.py [--output code.zip]
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent
CODE_DIR   = REPO_ROOT / "code"
EVAL_DIR   = CODE_DIR / "evaluation"
OUTPUT_CSV = REPO_ROOT / "dataset" / "output.csv"

# Paths to include in the zip
INCLUDE_ROOTS = [
    CODE_DIR,
    EVAL_DIR,
]

# Relative paths from repo root that should always be included
ALWAYS_INCLUDE = [
    ".env.local",           # template (no real key)
    "problem_statement.md",
    "README.md",
    # NOTE: output.csv is uploaded separately to HackerRank — NOT in the zip
]

# Exclude patterns (matched against any path component)
EXCLUDE_DIRS = {
    "__pycache__", ".cache", ".git", "venv", ".venv",
    "node_modules", "dataset", "images",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".env"}
EXCLUDE_FILES = {"telemetry.jsonl", ".env"}


def should_exclude(path: Path) -> bool:
    """Return True if this path should be omitted from the zip."""
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
    if path.name in EXCLUDE_FILES:
        return True
    if path.suffix in EXCLUDE_SUFFIXES:
        return True
    return False


def build_zip(output_path: Path) -> None:
    print(f"\n[Package] Building {output_path} ...")

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        included = []
        skipped  = []

        # Walk code/ and evaluation/
        for root_dir in [CODE_DIR]:
            for filepath in sorted(root_dir.rglob("*")):
                if not filepath.is_file():
                    continue
                if should_exclude(filepath):
                    skipped.append(filepath)
                    continue
                arcname = filepath.relative_to(REPO_ROOT)
                zf.write(filepath, arcname)
                included.append(arcname)

        # Always-include files from repo root
        for rel in ALWAYS_INCLUDE:
            abs_path = REPO_ROOT / rel
            if abs_path.exists():
                if not should_exclude(abs_path):
                    zf.write(abs_path, Path(rel))
                    included.append(Path(rel))

    print(f"\n[Package] Included {len(included)} files:")
    for f in included:
        print(f"  {f}")

    if skipped:
        print(f"\n[Package] Excluded {len(skipped)} files (pycache, cache, secrets, etc.)")

    size_kb = output_path.stat().st_size / 1024
    print(f"\n[Package] Done: {output_path}  ({size_kb:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Package submission zip")
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "code.zip",
        help="Output zip path (default: code.zip in repo root)",
    )
    args = parser.parse_args()

    # Pre-flight checks
    ok = True
    if not OUTPUT_CSV.exists():
        print(f"[WARN] output.csv not found at {OUTPUT_CSV}. Run main.py first.")
        ok = False
    else:
        print(f"[CHECK] output.csv found.")

    eval_report = EVAL_DIR / "evaluation_report.md"
    if not eval_report.exists():
        print(f"[WARN] evaluation_report.md not found. Run evaluation/main.py first.")
        ok = False
    else:
        print(f"[CHECK] evaluation_report.md found.")

    if not ok:
        print("\n[WARN] Packaging anyway — fix warnings before submitting.")

    build_zip(args.output)
    print("\n[Package] Submission checklist:")
    print("  [x] code.zip built")
    print("  [ ] output.csv row count verified == claims.csv row count")
    print("  [ ] evaluation_report.md has 2 strategies + 3+ disagreement examples")
    print("  [ ] log.txt from %USERPROFILE%\\hackerrank_orchestrate\\log.txt included as chat transcript")
    print("  [ ] No hardcoded answers in any file")


if __name__ == "__main__":
    main()
