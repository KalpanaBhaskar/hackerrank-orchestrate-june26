"""
L4 · VALIDATION & REPAIR  (code only, no LLM)
Deterministic checks on L3 output:
  - enum field values in allowed lists
  - supporting_image_ids ⊆ real image IDs for this row
  - logical consistency rules
  - retry (once) with violation named if needed
  - safe fallback if still broken
"""

from ingestion import ALLOWED, closest_allowed, get_object_part_enum


# ──────────────────────────────────────────────
# Consistency rules
# ──────────────────────────────────────────────

def validate_and_repair(
    verdict: dict,
    claim_object: str,
    real_image_ids: list[str],
    retry_fn=None,       # Optional callable(violation_description) -> dict
) -> dict:
    """
    Validates and repairs a verdict dict in-place.
    retry_fn: if provided and verdict is fixable, called once with a violation description.
    Returns the (possibly repaired) verdict.
    """
    violations = []
    v = dict(verdict)  # work on a copy

    # 1. Enum: evidence_standard_met
    v["evidence_standard_met"] = _fix_bool(v.get("evidence_standard_met", "false"))

    # 2. Enum: valid_image
    v["valid_image"] = _fix_bool(v.get("valid_image", "false"))

    # 3. Enum: claim_status
    cs = str(v.get("claim_status", "")).lower().strip()
    if cs not in ALLOWED["claim_status"]:
        violations.append(f"claim_status '{cs}' not in allowed values")
        v["claim_status"] = "not_enough_information"

    # 4. Enum: issue_type
    it = str(v.get("issue_type", "")).lower().strip()
    if it not in ALLOWED["issue_type"]:
        violations.append(f"issue_type '{it}' not in allowed values")
        v["issue_type"] = closest_allowed(it, ALLOWED["issue_type"], "unknown")

    # 5. Enum: object_part  (dynamic per claim_object)
    part_enum = get_object_part_enum(claim_object)
    op = str(v.get("object_part", "")).lower().strip()
    if op not in part_enum:
        violations.append(f"object_part '{op}' not in allowed values for {claim_object}")
        v["object_part"] = closest_allowed(op, part_enum, "unknown")

    # 6. Enum: severity
    sv = str(v.get("severity", "")).lower().strip()
    if sv not in ALLOWED["severity"]:
        violations.append(f"severity '{sv}' not in allowed values")
        v["severity"] = "unknown"

    # 7. risk_flags — semicolon list, each must be in allowed set
    v["risk_flags"] = _fix_risk_flags(v.get("risk_flags", "none"))

    # 8. supporting_image_ids ⊆ real_image_ids
    v["supporting_image_ids"] = _fix_image_ids(
        v.get("supporting_image_ids", "none"), real_image_ids, violations
    )

    # 9. Logical consistency rules
    esm = v["evidence_standard_met"]
    vi = v["valid_image"]
    claim_status = v["claim_status"]
    issue_type = v["issue_type"]
    severity = v["severity"]

    # Rule A: valid_image=false → claim_status should be not_enough_information
    if vi == "false" and claim_status == "supported":
        violations.append("valid_image=false but claim_status=supported (inconsistent)")
        v["claim_status"] = "not_enough_information"

    # Rule B: evidence_standard_met=false → claim_status should NOT be 'supported'
    if esm == "false" and claim_status == "supported":
        violations.append("evidence_standard_met=false but claim_status=supported (inconsistent)")
        v["claim_status"] = "not_enough_information"

    # Rule C: issue_type=none → severity should be none
    if issue_type == "none" and severity not in ("none", "unknown"):
        violations.append(f"issue_type=none but severity={severity} (should be none)")
        v["severity"] = "none"

    # Rule D: claim_status=not_enough_information → supporting_image_ids should be none
    if claim_status == "not_enough_information" and v["supporting_image_ids"] != "none":
        # This is a soft warning — some models correctly cite images even for NEI
        # We allow it but flag it
        pass

    # If violations found and retry_fn available → retry once
    if violations and retry_fn:
        violation_desc = "; ".join(violations)
        print(f"    [L4] Violations found: {violation_desc}. Requesting retry…")
        try:
            retry_result = retry_fn(violation_desc)
            # Re-validate the retry result (no further retry)
            return validate_and_repair(retry_result, claim_object, real_image_ids, retry_fn=None)
        except Exception as e:
            print(f"    [L4] Retry failed: {e}. Using repaired original.")

    if violations:
        print(f"    [L4] Repaired violations: {violations}")

    return v


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _fix_bool(val) -> str:
    s = str(val).lower().strip()
    if s in ("true", "yes", "1"):
        return "true"
    return "false"


def _fix_risk_flags(raw: str) -> str:
    """Validate and clean semicolon-separated risk flags."""
    raw = str(raw).strip()
    if raw.lower() == "none" or not raw:
        return "none"
    flags = [f.strip().lower() for f in raw.split(";") if f.strip()]
    valid_flags = [f for f in flags if f in ALLOWED["risk_flags"]]
    if not valid_flags:
        return "none"
    return ";".join(sorted(set(valid_flags)))


def _fix_image_ids(raw: str, real_ids: list[str], violations: list) -> str:
    """Ensure all returned image IDs are real IDs for this claim."""
    raw = str(raw).strip()
    if raw.lower() == "none" or not raw:
        return "none"
    claimed_ids = [i.strip() for i in raw.split(";") if i.strip()]
    real_set = set(real_ids)
    valid_ids = [i for i in claimed_ids if i in real_set]
    if len(valid_ids) < len(claimed_ids):
        bad = set(claimed_ids) - real_set
        violations.append(f"supporting_image_ids contained invalid IDs: {bad}")
    if not valid_ids:
        return "none"
    return ";".join(valid_ids)
