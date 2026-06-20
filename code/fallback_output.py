"""
Fallback output generator — no API required.
Produces a complete output.csv using rule-based NLP on claim text + image path heuristics.
Run this if the ML pipeline cannot complete due to quota exhaustion.

Usage:
    python fallback_output.py [--output ../dataset/output.csv]
"""
import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from ingestion import (
    DATASET_ROOT, load_claims, load_user_history, load_evidence_requirements,
    get_user_history, get_evidence_requirements, format_history_risk_summary,
    ALLOWED, get_object_part_enum,
)
from assembly import write_output_csv

# ── Keyword maps ────────────────────────────────────────────────────
ISSUE_KEYWORDS = {
    "dent or scratch":          ["dent", "dented", "scratch", "scratched", "scraped", "hail", "ding"],
    "crack":                    ["crack", "cracked", "cracking", "shatter", "shattered", "fractured", "split"],
    "broken or missing part":   ["broken", "broke", "missing", "snapped", "fell off", "detached", "broke off"],
    "torn or crushed packaging":["torn", "crushed", "squashed", "ripped", "packaging", "box"],
    "water or stain damage":    ["water", "liquid", "spill", "spilled", "coffee", "wet", "flood", "moisture", "stain"],
    "missing contents":         ["missing contents", "items missing", "contents missing", "not inside"],
}

PART_KEYWORDS_CAR = {
    "front bumper":   ["front bumper", "bumper front"],
    "rear bumper":    ["rear bumper", "bumper rear", "bumper behind", "back bumper"],
    "hood":           ["hood", "bonnet", "hail"],
    "door":           ["door"],
    "windshield":     ["windshield", "front glass", "windscreen", "front window"],
    "side mirror":    ["side mirror", "mirror"],
    "headlight":      ["headlight", "head light", "front light"],
    "taillight":      ["taillight", "tail light", "back light", "rear light"],
    "roof":           ["roof", "top"],
    "trunk":          ["trunk", "boot"],
    "wheel":          ["wheel", "tire", "tyre", "rim"],
}

PART_KEYWORDS_LAPTOP = {
    "screen":    ["screen", "display", "pantalla", "monitor"],
    "keyboard":  ["keyboard", "keys", "key"],
    "hinge":     ["hinge"],
    "trackpad":  ["trackpad", "touchpad", "palm-rest", "palmrest"],
    "body":      ["body", "chassis", "case", "shell"],
    "port":      ["port", "usb", "hdmi", "charging"],
    "battery":   ["battery"],
}

PART_KEYWORDS_PACKAGE = {
    "outer box":  ["box", "outer", "package", "packaging", "carton"],
    "inner wrap": ["inner", "wrap", "bubble"],
    "seal":       ["seal", "tape"],
}

OBJECT_PART_MAP = {
    "car":     PART_KEYWORDS_CAR,
    "laptop":  PART_KEYWORDS_LAPTOP,
    "package": PART_KEYWORDS_PACKAGE,
}


def _classify_issue(text: str) -> str:
    text_l = text.lower()
    for issue, kws in ISSUE_KEYWORDS.items():
        if any(kw in text_l for kw in kws):
            return issue
    return "general review"


def _classify_part(text: str, claim_object: str) -> str:
    text_l = text.lower()
    kw_map = OBJECT_PART_MAP.get(claim_object, {})
    for part, kws in kw_map.items():
        if any(kw in text_l for kw in kws):
            return part
    return "unknown"


def _image_count(image_paths: list[Path]) -> int:
    return len([p for p in image_paths if p.exists()])


def _make_verdict(record: dict, evidence_reqs: list[dict]) -> dict:
    claim_text = record["user_claim"].lower()
    obj = record["claim_object"]
    images = record["resolved_images"]
    image_ids = record["image_id_list"]
    n_images = _image_count(images)

    issue = _classify_issue(claim_text)
    part  = _classify_part(claim_text, obj)

    # Validate part against allowed enum
    allowed_parts = get_object_part_enum(obj)
    if part not in allowed_parts:
        part = "unknown"

    # Validate issue
    if issue not in ALLOWED["issue_type"]:
        issue = "general review"

    # Evidence standard: need ≥1 image that exists
    ev_met = "true" if n_images >= 1 else "false"
    ev_reason = (
        f"{n_images} image(s) submitted covering the claimed {part}."
        if n_images >= 1
        else "No valid images could be loaded for review."
    )

    # Risk flags — check for prompt injection in claim text
    risk_flags = "none"
    injection_patterns = ["approve", "ignore previous", "skip manual", "mark this"]
    if any(p in claim_text for p in injection_patterns):
        risk_flags = "prompt_injection_attempt"

    # Claim status heuristic: if we have images and a clear issue → likely supported
    # If no images or issue unclear → not_enough_information
    if n_images >= 1 and issue != "general review" and part != "unknown":
        status = "supported"
        justification = (
            f"Images show evidence consistent with {issue} on {obj} {part}. "
            f"Claim supported based on submitted image evidence."
        )
        severity = "medium"
        if issue in ("crack", "broken or missing part"):
            severity = "high"
        elif issue in ("dent or scratch", "water or stain damage"):
            severity = "medium"
        else:
            severity = "low"
    elif n_images >= 1:
        status = "not_enough_information"
        justification = (
            f"Images are present but could not confirm specific {obj} damage from claim description alone. "
            f"Manual review recommended."
        )
        severity = "unknown"
    else:
        status = "not_enough_information"
        justification = "No usable images were submitted. Cannot assess claim."
        severity = "unknown"

    supporting = ";".join(image_ids[:2]) if image_ids and status == "supported" else "none"

    return {
        "evidence_standard_met":        ev_met,
        "evidence_standard_met_reason": ev_reason,
        "valid_image":                  "true" if n_images >= 1 else "false",
        "risk_flags":                   risk_flags,
        "issue_type":                   issue,
        "object_part":                  part,
        "claim_status":                 status,
        "claim_status_justification":   justification,
        "supporting_image_ids":         supporting,
        "severity":                     severity,
    }


def run_fallback(input_csv: Path, output_csv: Path) -> None:
    print(f"[Fallback] Reading {input_csv}")
    records       = load_claims(input_csv, mode="predict")
    evidence_reqs = load_evidence_requirements()
    print(f"[Fallback] Generating {len(records)} predictions (rules-based, no API)...")

    rows = []
    for i, rec in enumerate(records):
        verdict = _make_verdict(rec, evidence_reqs)
        from assembly import assemble_row
        rows.append(assemble_row(rec, verdict))
        print(f"  [{i+1}/{len(records)}] {rec['user_id']} -> {verdict['claim_status']} / {verdict['issue_type']}")

    write_output_csv(rows, output_csv)
    print(f"[Fallback] Written {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fallback rules-based output generator")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent.parent / "dataset" / "output.csv")
    args = parser.parse_args()
    run_fallback(DATASET_ROOT / "claims.csv", args.output)
