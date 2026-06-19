"""
Phase 0-2 Verification Script
Checks: images exist, CSV files valid, lookups working — no API calls.
Run: python verify_phases_0_2.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))

from ingestion import (
    DATASET_ROOT,
    get_evidence_requirements,
    get_user_history,
    load_claims,
    load_evidence_requirements,
    load_user_history,
)

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

def check(label, condition, detail=""):
    if condition:
        print(f"{PASS}  {label}" + (f" - {detail}" if detail else ""))
    else:
        print(f"{FAIL}  {label}" + (f" - {detail}" if detail else ""))
    return condition

def main():
    ok = True
    print("\n" + "="*60)
    print(" PHASE 0 — Dataset & Image Verification")
    print("="*60)

    # Phase 0: image directories exist
    sample_dir = DATASET_ROOT / "images" / "sample"
    test_dir = DATASET_ROOT / "images" / "test"
    ok &= check("dataset/images/sample/ exists", sample_dir.is_dir())
    ok &= check("dataset/images/test/ exists", test_dir.is_dir())

    sample_imgs = list(sample_dir.rglob("*.jpg")) + list(sample_dir.rglob("*.png"))
    test_imgs = list(test_dir.rglob("*.jpg")) + list(test_dir.rglob("*.png"))
    ok &= check(f"sample images found", len(sample_imgs) > 0, f"{len(sample_imgs)} files")
    ok &= check(f"test images found", len(test_imgs) > 0, f"{len(test_imgs)} files")

    hist_path = DATASET_ROOT / "user_history.csv"
    ev_path = DATASET_ROOT / "evidence_requirements.csv"
    ok &= check("user_history.csv exists and non-empty", hist_path.exists() and hist_path.stat().st_size > 0)
    ok &= check("evidence_requirements.csv exists and non-empty", ev_path.exists() and ev_path.stat().st_size > 0)

    print("\n" + "="*60)
    print(" PHASE 1 — Data Loaders & Path Resolution")
    print("="*60)

    # Load sample_claims.csv
    sample_csv = DATASET_ROOT / "sample_claims.csv"
    sample_records = load_claims(sample_csv, mode="evaluate")
    ok &= check(f"sample_claims.csv loaded", len(sample_records) > 0, f"{len(sample_records)} rows")

    # Verify input record has exactly 4 input fields + image helpers
    r = sample_records[0]
    input_fields = {"user_id", "image_paths", "user_claim", "claim_object"}
    has_4 = input_fields.issubset(r.keys())
    ok &= check("Input record has 4 required fields", has_4)

    # Verify expected_output is separate (not in input)
    has_expected = "expected_output" in r and isinstance(r["expected_output"], dict)
    ok &= check("Expected outputs stored separately (not in prompt)", has_expected)
    if has_expected:
        print(f"      Expected output keys: {list(r['expected_output'].keys())}")

    # Load claims.csv (predict mode — only 4 input fields)
    claims_csv = DATASET_ROOT / "claims.csv"
    test_records = load_claims(claims_csv, mode="predict")
    ok &= check(f"claims.csv loaded", len(test_records) > 0, f"{len(test_records)} rows")
    ok &= check("claims.csv has empty expected_output", test_records[0]["expected_output"] == {})

    # Check every image resolves
    all_invalid = []
    for rec in sample_records + test_records:
        all_invalid.extend(rec["invalid_images"])

    if all_invalid:
        print(f"{WARN}  Some images did not resolve ({len(all_invalid)} total):")
        for p in all_invalid[:5]:
            print(f"      {p}")
    else:
        print(f"{PASS}  All images resolved for both CSVs - {len(sample_records)+len(test_records)} records checked")

    print("\n" + "="*60)
    print(" PHASE 2 — Reference Data Joins")
    print("="*60)

    history = load_user_history()
    ev_reqs = load_evidence_requirements()
    ok &= check(f"User history loaded", len(history) > 0, f"{len(history)} users")
    ok &= check(f"Evidence requirements loaded", len(ev_reqs) > 0, f"{len(ev_reqs)} rules")

    # Every distinct claim_object resolves to at least one requirement
    all_objects = {r["claim_object"] for r in sample_records + test_records}
    for obj in sorted(all_objects):
        reqs = get_evidence_requirements(ev_reqs, obj)
        ok &= check(f"claim_object '{obj}' -> evidence requirements", len(reqs) > 0, f"{len(reqs)} rules")

    # Every user_id in both CSVs looked up
    missing_users = []
    for rec in sample_records + test_records:
        uid = rec["user_id"]
        hist_row = get_user_history(history, uid)
        if hist_row is None:
            missing_users.append(uid)

    if missing_users:
        print(f"{WARN}  Users with no history row: {sorted(set(missing_users))}")
    else:
        print(f"{PASS}  All user IDs found in user_history.csv")

    print("\n" + "="*60)
    result_label = "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"
    print(f" RESULT: {result_label}")
    print("="*60 + "\n")

    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
