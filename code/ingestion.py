"""
L1 · INGESTION & VALIDATION
Parses CSV rows, resolves image paths, validates images open correctly,
builds user_history and evidence_requirements lookup tables.
No LLM calls — pure code.
"""

import csv
import os
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
DATASET_ROOT = Path(__file__).resolve().parent.parent / "dataset"

INPUT_COLUMNS = {"user_id", "image_paths", "user_claim", "claim_object"}

ALLOWED = {
    "claim_status": {"supported", "contradicted", "not_enough_information"},
    "issue_type": {
        "dent", "scratch", "crack", "glass_shatter", "broken_part",
        "missing_part", "torn_packaging", "crushed_packaging",
        "water_damage", "stain", "none", "unknown",
    },
    "object_part_car": {
        "front_bumper", "rear_bumper", "door", "hood", "windshield",
        "side_mirror", "headlight", "taillight", "fender",
        "quarter_panel", "body", "unknown",
    },
    "object_part_laptop": {
        "screen", "keyboard", "trackpad", "hinge", "lid",
        "corner", "port", "base", "body", "unknown",
    },
    "object_part_package": {
        "box", "package_corner", "package_side", "seal",
        "label", "contents", "item", "unknown",
    },
    "risk_flags": {
        "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
        "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
        "claim_mismatch", "possible_manipulation", "non_original_image",
        "text_instruction_present", "user_history_risk", "manual_review_required",
    },
    "evidence_standard_met": {"true", "false"},
    "valid_image": {"true", "false"},
    "severity": {"none", "low", "medium", "high", "unknown"},
}

OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part",
    "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity",
]


# ──────────────────────────────────────────────
# CSV Loaders
# ──────────────────────────────────────────────

def load_claims(csv_path: str | Path, mode: str = "predict") -> list[dict]:
    """
    Load claims.csv (mode='predict') or sample_claims.csv (mode='evaluate').

    Returns a list of dicts. Each dict has exactly 4 input fields:
      user_id, image_paths, user_claim, claim_object
    In 'evaluate' mode the extra expected-output columns are also present
    in 'expected_output' key (dict) for scoring ONLY — never passed to LLM.
    """
    csv_path = Path(csv_path)
    rows = []
    expected_output_cols = [c for c in OUTPUT_COLUMNS if c not in INPUT_COLUMNS]

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Build the 4-field input record — strictly no leakage
            record = {
                "user_id": row["user_id"].strip(),
                "image_paths": row["image_paths"].strip(),
                "user_claim": row["user_claim"].strip(),
                "claim_object": row["claim_object"].strip(),
            }

            # Derive image_id list
            record["image_id_list"] = _extract_image_ids(record["image_paths"])

            # Resolve and validate every image path
            resolved, invalid = _resolve_images(record["image_paths"])
            record["resolved_images"] = resolved   # list of absolute Path objects
            record["invalid_images"] = invalid     # list of path strings that failed

            # In evaluate mode, stash expected outputs for scoring (never in prompts)
            if mode == "evaluate":
                expected = {col: row.get(col, "").strip() for col in expected_output_cols}
                record["expected_output"] = expected
            else:
                record["expected_output"] = {}

            rows.append(record)

    return rows


def load_user_history(csv_path: str | Path | None = None) -> dict[str, dict]:
    """Returns {user_id: row_dict} for fast O(1) lookup."""
    path = csv_path or (DATASET_ROOT / "user_history.csv")
    history = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = row["user_id"].strip()
            history[uid] = {k: v.strip() for k, v in row.items()}
    return history


def load_evidence_requirements(csv_path: str | Path | None = None) -> list[dict]:
    """Returns list of requirement rows."""
    path = csv_path or (DATASET_ROOT / "evidence_requirements.csv")
    reqs = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            reqs.append({k: v.strip() for k, v in row.items()})
    return reqs


# ──────────────────────────────────────────────
# Reference-data lookups
# ──────────────────────────────────────────────

def get_evidence_requirements(
    reqs: list[dict],
    claim_object: str,
    issue_family: Optional[str] = None,
) -> list[dict]:
    """
    Return matching evidence requirement rows for a given claim_object.
    If issue_family is given, prefer rows where applies_to contains it.
    Always include 'all' rows.
    """
    matched = []
    for req in reqs:
        obj = req["claim_object"]
        if obj in ("all", claim_object):
            if issue_family:
                if issue_family.lower() in req["applies_to"].lower():
                    matched.append(req)
            else:
                matched.append(req)
    # Fallback: return all rows for this object
    if not matched:
        matched = [r for r in reqs if r["claim_object"] in ("all", claim_object)]
    return matched


def get_user_history(history: dict[str, dict], user_id: str) -> Optional[dict]:
    """Return history row for user_id, or None if not found."""
    return history.get(user_id)


def format_history_risk_summary(hist: Optional[dict]) -> str:
    """
    Returns a short plain-text risk context string for use in the L3 prompt.
    Explicitly labels it as context-only.
    """
    if not hist:
        return "User history: No prior history found. Treat as low-risk new user."

    flags = hist.get("history_flags", "none")
    summary = hist.get("history_summary", "")
    past = hist.get("past_claim_count", "0")
    rejected = hist.get("rejected_claim", "0")
    last90 = hist.get("last_90_days_claim_count", "0")

    lines = [
        f"User history (CONTEXT ONLY — do NOT let this override clear visual evidence):",
        f"  Past claims: {past} total, {rejected} rejected, {last90} in last 90 days",
        f"  Risk flags: {flags}",
        f"  Summary: {summary}",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _extract_image_ids(image_paths_str: str) -> list[str]:
    """Extract image IDs (filename without extension) from the semicolon-separated path string."""
    parts = [p.strip() for p in image_paths_str.split(";") if p.strip()]
    return [Path(p).stem for p in parts]


def _resolve_images(image_paths_str: str) -> tuple[list[Path], list[str]]:
    """
    Resolve each image path (relative to dataset/) to an absolute Path.
    Returns (valid_paths, invalid_path_strings).
    """
    valid = []
    invalid = []
    parts = [p.strip() for p in image_paths_str.split(";") if p.strip()]
    for rel in parts:
        abs_path = DATASET_ROOT / rel
        if abs_path.exists() and abs_path.is_file():
            valid.append(abs_path)
        else:
            invalid.append(str(abs_path))
    return valid, invalid


def image_to_bytes(path: Path) -> bytes:
    """Read image file and return raw bytes."""
    with open(path, "rb") as f:
        return f.read()


def image_mime_type(path: Path) -> str:
    """Return MIME type based on file extension."""
    ext = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "image/jpeg")


# ──────────────────────────────────────────────
# Allowed-value helpers
# ──────────────────────────────────────────────

def get_object_part_enum(claim_object: str) -> set[str]:
    key = f"object_part_{claim_object}"
    return ALLOWED.get(key, ALLOWED["object_part_car"])


def closest_allowed(value: str, allowed_set: set[str], fallback: str = "unknown") -> str:
    """Return value if it's in the allowed set, else return fallback."""
    v = value.lower().strip()
    if v in allowed_set:
        return v
    # Try substring match
    for a in allowed_set:
        if v in a or a in v:
            return a
    return fallback
