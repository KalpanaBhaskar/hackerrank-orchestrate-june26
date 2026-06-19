"""
L5 · OUTPUT ASSEMBLY
Merges pass-through input fields + L4 validated verdict into exact 14-column CSV row.
"""

import csv
import os
from pathlib import Path
from typing import Any

from ingestion import OUTPUT_COLUMNS


def assemble_row(input_record: dict, verdict: dict) -> dict:
    """
    Build a single output row from the input record + verdict.
    The first 4 columns are pass-through from input (byte-identical to input CSV).
    """
    row = {
        "user_id": input_record["user_id"],
        "image_paths": input_record["image_paths"],
        "user_claim": input_record["user_claim"],
        "claim_object": input_record["claim_object"],
        # L3/L4 verdict fields
        "evidence_standard_met": verdict.get("evidence_standard_met", "false"),
        "evidence_standard_met_reason": verdict.get("evidence_standard_met_reason", ""),
        "risk_flags": verdict.get("risk_flags", "none"),
        "issue_type": verdict.get("issue_type", "unknown"),
        "object_part": verdict.get("object_part", "unknown"),
        "claim_status": verdict.get("claim_status", "not_enough_information"),
        "claim_status_justification": verdict.get("claim_status_justification", ""),
        "supporting_image_ids": verdict.get("supporting_image_ids", "none"),
        "valid_image": verdict.get("valid_image", "false"),
        "severity": verdict.get("severity", "unknown"),
    }
    # Return in the exact column order
    return {col: row[col] for col in OUTPUT_COLUMNS}


def write_output_csv(rows: list[dict], output_path: str | Path) -> None:
    """Write list of row dicts to output.csv with exact column order."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Assembly] Written {len(rows)} rows to {output_path}")
