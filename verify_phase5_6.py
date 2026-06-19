"""
Phase 5 & 6 Verification — L4 Validation + L5 Assembly (pure code, no API calls)
==================================================================================
Phase 5 — Validates that:
  [5a] Every enum field in a verdict is in its allowed list (hard-repair check)
  [5b] supporting_image_ids are always a subset of real image IDs
  [5c] Logical consistency rules fire correctly
  [5d] The retry_fn path is exercised (intentionally broken response -> repair confirmed)
  [5e] Safe fallback is returned when retry also fails

Phase 6 — Validates that:
  [6a] Header row matches the required 14-column order exactly
  [6b] A full mock run produces N rows for N input records
  [6c] Pass-through fields (user_id, image_paths, user_claim, claim_object)
       are byte-identical to the input

Run: python verify_phase5_6.py
No GEMINI_API_KEY required.
"""

import csv
import io
import sys
import tempfile
from pathlib import Path

# ── path bootstrap ──────────────────────────────────────────────────────────
_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root / "code"))

from ingestion import ALLOWED, OUTPUT_COLUMNS, get_object_part_enum
from validation import validate_and_repair
from assembly import assemble_row, write_output_csv

PASS = "[PASS]"
FAIL = "[FAIL]"


def check(label: str, condition: bool, detail: str = "") -> bool:
    tag = PASS if condition else FAIL
    print(f"  {tag}  {label}" + (f" -- {detail}" if detail else ""))
    return condition


# ════════════════════════════════════════════════════════════════════════════
# Phase 5 helpers
# ════════════════════════════════════════════════════════════════════════════

_GOOD_VERDICT = {
    "evidence_standard_met": "true",
    "evidence_standard_met_reason": "Image shows clear damage.",
    "valid_image": "true",
    "risk_flags": "none",
    "issue_type": "dent",
    "object_part": "rear_bumper",
    "claim_status": "supported",
    "claim_status_justification": "The image clearly shows a dent on the rear bumper.",
    "supporting_image_ids": "img_1",
    "severity": "medium",
}

_REAL_IDS = ["img_1", "img_2"]


def _verdict(**overrides) -> dict:
    v = dict(_GOOD_VERDICT)
    v.update(overrides)
    return v


def test_5a_enum_repair():
    """5a: Bad enum values are corrected to nearest allowed value."""
    all_ok = True
    print("\n  [5a] Enum repair checks...")

    # Bad claim_status
    v = validate_and_repair(
        _verdict(claim_status="INVALID_VALUE"), "car", _REAL_IDS
    )
    all_ok &= check("bad claim_status repaired to not_enough_information",
                    v["claim_status"] == "not_enough_information")

    # Bad issue_type
    v = validate_and_repair(
        _verdict(issue_type="destroyed"), "car", _REAL_IDS
    )
    all_ok &= check("bad issue_type repaired (closest allowed or unknown)",
                    v["issue_type"] in ALLOWED["issue_type"])

    # Bad object_part for car
    v = validate_and_repair(
        _verdict(object_part="steering_wheel"), "car", _REAL_IDS
    )
    car_parts = get_object_part_enum("car")
    all_ok &= check("bad car object_part repaired to allowed car part",
                    v["object_part"] in car_parts)

    # Bad severity
    v = validate_and_repair(
        _verdict(severity="severe"), "car", _REAL_IDS
    )
    all_ok &= check("bad severity repaired to unknown",
                    v["severity"] in ALLOWED["severity"])

    # Wrong object_part for laptop (car part in laptop claim)
    v = validate_and_repair(
        _verdict(object_part="rear_bumper"), "laptop", _REAL_IDS
    )
    laptop_parts = get_object_part_enum("laptop")
    all_ok &= check("car part repaired for laptop claim",
                    v["object_part"] in laptop_parts)

    # Boolean coercion
    v = validate_and_repair(
        _verdict(evidence_standard_met=True, valid_image=1), "car", _REAL_IDS
    )
    all_ok &= check("bool True coerced to string 'true'",
                    v["evidence_standard_met"] == "true" and v["valid_image"] == "true")

    return all_ok


def test_5b_image_ids():
    """5b: supporting_image_ids must be a subset of real image IDs."""
    all_ok = True
    print("\n  [5b] Image ID subset checks...")

    # All valid
    v = validate_and_repair(
        _verdict(supporting_image_ids="img_1;img_2"), "car", ["img_1", "img_2"]
    )
    all_ok &= check("valid image IDs pass through",
                    v["supporting_image_ids"] == "img_1;img_2")

    # One invalid ID → only valid IDs remain
    v = validate_and_repair(
        _verdict(supporting_image_ids="img_1;img_99"), "car", ["img_1", "img_2"]
    )
    ids = set(v["supporting_image_ids"].split(";"))
    all_ok &= check("invalid img_99 removed, img_1 kept",
                    "img_99" not in ids and "img_1" in ids)

    # All invalid → becomes 'none'
    v = validate_and_repair(
        _verdict(supporting_image_ids="img_99;img_100"), "car", ["img_1", "img_2"]
    )
    all_ok &= check("all-invalid IDs -> 'none'",
                    v["supporting_image_ids"] == "none")

    # Already 'none' → stays 'none'
    v = validate_and_repair(
        _verdict(supporting_image_ids="none"), "car", ["img_1"]
    )
    all_ok &= check("'none' supporting IDs unchanged",
                    v["supporting_image_ids"] == "none")

    return all_ok


def test_5c_consistency():
    """5c: Logical consistency rules."""
    all_ok = True
    print("\n  [5c] Logical consistency rules...")

    # Rule A: valid_image=false + claim_status=supported -> not_enough_information
    v = validate_and_repair(
        _verdict(valid_image="false", claim_status="supported"), "car", _REAL_IDS
    )
    all_ok &= check("valid_image=false overrides claim_status=supported",
                    v["claim_status"] == "not_enough_information")

    # Rule B: evidence_standard_met=false + claim_status=supported -> not_enough_information
    v = validate_and_repair(
        _verdict(evidence_standard_met="false", claim_status="supported"), "car", _REAL_IDS
    )
    all_ok &= check("evidence_standard_met=false overrides claim_status=supported",
                    v["claim_status"] == "not_enough_information")

    # Rule C: issue_type=none -> severity=none
    v = validate_and_repair(
        _verdict(issue_type="none", severity="high"), "car", _REAL_IDS
    )
    all_ok &= check("issue_type=none forces severity=none",
                    v["severity"] == "none")

    # Rule: risk_flags cleaned
    v = validate_and_repair(
        _verdict(risk_flags="blurry_image;INVALID_FLAG;low_light_or_glare"), "car", _REAL_IDS
    )
    flags = set(v["risk_flags"].split(";"))
    all_ok &= check("invalid risk flags removed, valid ones kept",
                    "invalid_flag" not in flags and "blurry_image" in flags)

    return all_ok


def test_5d_retry_path():
    """5d: Retry path is exercised (intentionally broken response -> retry fn called)."""
    print("\n  [5d] Retry path exercise...")

    retry_called = {"count": 0}

    def mock_retry(violation_desc: str) -> dict:
        retry_called["count"] += 1
        # Return a VALID verdict on retry
        return dict(_GOOD_VERDICT)

    broken = _verdict(claim_status="TOTALLY_BROKEN_VALUE")
    v = validate_and_repair(
        verdict=broken,
        claim_object="car",
        real_image_ids=_REAL_IDS,
        retry_fn=mock_retry,
    )
    ok1 = check("retry_fn was called once on violation",
                retry_called["count"] == 1)
    ok2 = check("repaired verdict is valid after retry",
                v["claim_status"] in ALLOWED["claim_status"])
    return ok1 and ok2


def test_5e_fallback():
    """5e: When retry also returns invalid output, hard-repair is applied (no crash)."""
    print("\n  [5e] Fallback after failed retry...")

    def bad_retry(violation_desc: str) -> dict:
        # Retry also returns invalid → should be hard-repaired, not crash
        return {"claim_status": "STILL_BROKEN", "issue_type": "fake",
                "object_part": "??", "severity": "extreme",
                "evidence_standard_met": "yes_kinda", "valid_image": "maybe",
                "risk_flags": "INVALID", "supporting_image_ids": "img_999",
                "evidence_standard_met_reason": "x", "claim_status_justification": "x"}

    v = validate_and_repair(
        verdict=_verdict(claim_status="BAD"),
        claim_object="car",
        real_image_ids=["img_1"],
        retry_fn=bad_retry,
    )
    ok = check("no crash after double-invalid path; all enums valid",
               v["claim_status"] in ALLOWED["claim_status"]
               and v["issue_type"] in ALLOWED["issue_type"]
               and v["severity"] in ALLOWED["severity"])
    return ok


# ════════════════════════════════════════════════════════════════════════════
# Phase 6
# ════════════════════════════════════════════════════════════════════════════

def test_6a_header_order():
    """6a: CSV header matches 14-column required order exactly."""
    print("\n  [6a] Output CSV header check...")

    expected = [
        "user_id", "image_paths", "user_claim", "claim_object",
        "evidence_standard_met", "evidence_standard_met_reason",
        "risk_flags", "issue_type", "object_part",
        "claim_status", "claim_status_justification",
        "supporting_image_ids", "valid_image", "severity",
    ]
    ok1 = check("OUTPUT_COLUMNS has exactly 14 fields",
                len(OUTPUT_COLUMNS) == 14, str(len(OUTPUT_COLUMNS)))
    ok2 = check("OUTPUT_COLUMNS order matches required spec",
                OUTPUT_COLUMNS == expected)
    return ok1 and ok2


def test_6b_row_count():
    """6b: write_output_csv produces exactly N rows for N input records."""
    print("\n  [6b] Row count integrity check...")

    mock_input = [
        {"user_id": f"u{i}", "image_paths": f"images/test/case_{i}/img_1.jpg",
         "user_claim": f"claim {i}", "claim_object": "car",
         "image_id_list": ["img_1"], "resolved_images": [],
         "invalid_images": [], "expected_output": {}}
        for i in range(5)
    ]

    rows = [assemble_row(rec, _GOOD_VERDICT) for rec in mock_input]

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        tmp_path = Path(f.name)

    write_output_csv(rows, tmp_path)

    with open(tmp_path, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    tmp_path.unlink()

    ok = check("CSV row count == input record count",
               len(reader) == 5, f"{len(reader)} rows for 5 inputs")
    return ok


def test_6c_passthrough():
    """6c: Pass-through fields are byte-identical to input."""
    print("\n  [6c] Pass-through field identity check...")

    input_rec = {
        "user_id":      "user_007",
        "image_paths":  "images/test/case_005/img_1.jpg;images/test/case_005/img_2.jpg",
        "user_claim":   "Customer: Hi, my side mirror is broken. | Support: We will review.",
        "claim_object": "car",
        "image_id_list": ["img_1", "img_2"],
        "resolved_images": [],
        "invalid_images": [],
        "expected_output": {},
    }

    row = assemble_row(input_rec, _GOOD_VERDICT)

    all_ok = True
    for field in ("user_id", "image_paths", "user_claim", "claim_object"):
        all_ok &= check(f"'{field}' is byte-identical",
                        row[field] == input_rec[field])
    return all_ok


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("\n" + "="*60)
    print(" PHASE 5 -- L4 Validation & Repair (code only)")
    print("="*60)

    p5_ok = True
    p5_ok &= test_5a_enum_repair()
    p5_ok &= test_5b_image_ids()
    p5_ok &= test_5c_consistency()
    p5_ok &= test_5d_retry_path()
    p5_ok &= test_5e_fallback()

    print("\n" + "="*60)
    print(f" PHASE 5 RESULT: {'ALL CHECKS PASSED' if p5_ok else 'SOME CHECKS FAILED'}")
    print("="*60)

    print("\n" + "="*60)
    print(" PHASE 6 -- L5 Output Assembly (code only)")
    print("="*60)

    p6_ok = True
    p6_ok &= test_6a_header_order()
    p6_ok &= test_6b_row_count()
    p6_ok &= test_6c_passthrough()

    print("\n" + "="*60)
    print(f" PHASE 6 RESULT: {'ALL CHECKS PASSED' if p6_ok else 'SOME CHECKS FAILED'}")
    print("="*60)

    overall = p5_ok and p6_ok
    print(f"\n OVERALL: {'ALL CHECKS PASSED' if overall else 'SOME CHECKS FAILED'}\n")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
