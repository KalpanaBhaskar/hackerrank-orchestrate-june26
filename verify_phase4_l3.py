"""
Phase 4 Verification — L3 Visual Evidence Reasoning
Tests vision assessment on 3 sample cases.
Run: python verify_phase4_l3.py
Requires GEMINI_API_KEY in .env or .env.local
"""

import sys
import json
from pathlib import Path

_root = Path(__file__).resolve().parent
for env_file in (".env", ".env.local"):
    ef = _root / env_file
    if ef.exists():
        from dotenv import load_dotenv
        load_dotenv(ef)
        break

sys.path.insert(0, str(_root / "code"))

from ingestion import (
    DATASET_ROOT, load_claims, load_evidence_requirements, load_user_history,
    get_evidence_requirements, get_user_history, format_history_risk_summary,
    ALLOWED, get_object_part_enum,
)
from claim_extraction import extract_claim
from vision_reasoning import assess_visual_evidence
from validation import validate_and_repair

SAMPLE_CSV = DATASET_ROOT / "sample_claims.csv"

def main():
    print("\n" + "="*60)
    print(" PHASE 4 - L3 Visual Evidence Reasoning Verification")
    print("="*60)

    import os
    if not os.environ.get("GEMINI_API_KEY"):
        print("[FAIL] GEMINI_API_KEY not set. Set it in .env.local first.")
        sys.exit(1)

    records = load_claims(SAMPLE_CSV, mode="evaluate")
    history = load_user_history()
    ev_reqs = load_evidence_requirements()

    # Test first 3 records
    test_records = records[:3]
    all_ok = True

    for i, rec in enumerate(test_records):
        claim_id = f"{rec['user_id']}_verify_{i}"
        print(f"\n--- Case {i+1}: {rec['user_id']} ({rec['claim_object']}) ---")
        print(f"  Images: {rec['image_id_list']}")

        # L2
        l2 = extract_claim(rec["user_claim"], rec["claim_object"], claim_id)
        print(f"  L2 part: {l2['primary_object_part']}, issue: {l2['primary_issue_family']}")

        # Reference data
        hist = get_user_history(history, rec["user_id"])
        history_risk = format_history_risk_summary(hist)
        ev = get_evidence_requirements(ev_reqs, rec["claim_object"], l2.get("primary_issue_family"))

        # L3
        verdict = assess_visual_evidence(
            claim_object=rec["claim_object"],
            image_paths=rec["resolved_images"],
            image_ids=rec["image_id_list"],
            l2_result=l2,
            evidence_requirements=ev,
            history_risk_summary=history_risk,
            claim_id=claim_id,
        )

        print(f"  L3 raw verdict:")
        for k, v in verdict.items():
            print(f"    {k}: {v}")

        # L4 Validate
        repaired = validate_and_repair(verdict, rec["claim_object"], rec["image_id_list"])

        # Check dynamic enum: object_part correct for claim_object
        part_enum = get_object_part_enum(rec["claim_object"])
        part_ok = repaired.get("object_part", "") in part_enum
        print(f"  object_part '{repaired.get('object_part')}' in {rec['claim_object']} enum: {'[PASS]' if part_ok else '[FAIL]'}")
        if not part_ok:
            all_ok = False

        # Check supporting_image_ids are all real IDs
        sup_ids_raw = repaired.get("supporting_image_ids", "none")
        if sup_ids_raw != "none":
            sup_ids = set(sup_ids_raw.split(";"))
            real_ids = set(rec["image_id_list"])
            ids_ok = sup_ids.issubset(real_ids)
            print(f"  supporting_image_ids subset of real IDs: {'[PASS]' if ids_ok else '[FAIL]'}")
            if not ids_ok:
                all_ok = False
        else:
            print(f"  supporting_image_ids: none (acceptable)")

        # Check all enum values
        enum_ok = True
        for field, allowed_set in [
            ("claim_status", ALLOWED["claim_status"]),
            ("issue_type", ALLOWED["issue_type"]),
            ("severity", ALLOWED["severity"]),
            ("evidence_standard_met", ALLOWED["evidence_standard_met"]),
            ("valid_image", ALLOWED["valid_image"]),
        ]:
            val = repaired.get(field, "")
            if val not in allowed_set:
                print(f"  [FAIL] {field}='{val}' not in allowed set")
                enum_ok = False
                all_ok = False
        if enum_ok:
            print(f"  [PASS] All enum values valid")

        # Compare with expected
        expected = rec.get("expected_output", {})
        if expected:
            print(f"  Expected claim_status: {expected.get('claim_status', '?')}")
            print(f"  Predicted claim_status: {repaired.get('claim_status', '?')}")

    print("\n" + "="*60)
    result_label = "ALL L3 CHECKS PASSED" if all_ok else "SOME L3 CHECKS FAILED"
    print(f" RESULT: {result_label}")
    print("="*60 + "\n")

    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
