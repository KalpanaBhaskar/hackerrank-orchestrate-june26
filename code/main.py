"""
Main pipeline entry point — Mode B (Final/Predict).
Reads dataset/claims.csv → produces output.csv.

Usage:
    python main.py [--input PATH] [--output PATH] [--workers N]
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from functools import partial

# Seconds to sleep between claims in sequential mode (workers=1)
# Free tier: 15 RPM for gemini-2.0-flash; each claim makes 2 calls → 4s gap is safe
CLAIM_SLEEP_S = float(os.environ.get("CLAIM_SLEEP_S", "4"))

# Ensure code/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

# Load .env or .env.local from project root
_root = Path(__file__).resolve().parent.parent
for env_file in (".env", ".env.local"):
    _ef = _root / env_file
    if _ef.exists():
        load_dotenv(_ef)
        break

from assembly import assemble_row, write_output_csv
from claim_extraction import extract_claim
from ingestion import (
    DATASET_ROOT,
    format_history_risk_summary,
    get_evidence_requirements,
    get_user_history,
    load_claims,
    load_evidence_requirements,
    load_user_history,
)
from validation import validate_and_repair
from vision_reasoning import assess_visual_evidence

# ──────────────────────────────────────────────
# Safe fallback row
# ──────────────────────────────────────────────

def _error_verdict(reason: str) -> dict:
    return {
        "evidence_standard_met": "false",
        "evidence_standard_met_reason": reason,
        "valid_image": "false",
        "risk_flags": "manual_review_required",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": "An error occurred during processing.",
        "supporting_image_ids": "none",
        "severity": "unknown",
    }


# ──────────────────────────────────────────────
# Per-claim pipeline
# ──────────────────────────────────────────────

def process_claim(
    record: dict,
    history_lookup: dict,
    evidence_reqs: list[dict],
    claim_idx: int,
    total: int,
) -> dict:
    """
    Full L1→L5 pipeline for one claim record.
    L4 retry_fn is wired: if L3 output fails validation, L3 is called once more
    with the violation description appended to the prompt context.
    Returns assembled output row dict.
    """
    claim_id = f"{record['user_id']}_idx{claim_idx}"
    print(f"[{claim_idx}/{total}] Processing {claim_id} ({record['claim_object']})...")

    # ── L1: Image resolution already done in load_claims ──
    image_paths = record["resolved_images"]
    image_ids   = record["image_id_list"]

    if not image_paths:
        print(f"  [WARN] No valid images for {claim_id}; using fallback.")
        return assemble_row(record, _error_verdict("No valid images could be loaded."))

    # ── L2: Claim extraction (text-only) ──
    l2 = extract_claim(
        user_claim=record["user_claim"],
        claim_object=record["claim_object"],
        claim_id=claim_id,
    )
    print(f"  [L2] part={l2['primary_object_part']} | issue={l2['primary_issue_family']}")

    # ── Reference data ──
    hist = get_user_history(history_lookup, record["user_id"])
    if hist is None:
        print(f"  [WARN] No history row for {record['user_id']}")
    history_risk = format_history_risk_summary(hist)

    ev_reqs = get_evidence_requirements(
        evidence_reqs,
        record["claim_object"],
        l2.get("primary_issue_family"),
    )

    # ── L3: Visual evidence reasoning ──
    verdict = assess_visual_evidence(
        claim_object=record["claim_object"],
        image_paths=image_paths,
        image_ids=image_ids,
        l2_result=l2,
        evidence_requirements=ev_reqs,
        history_risk_summary=history_risk,
        claim_id=claim_id,
    )
    print(f"  [L3] status={verdict.get('claim_status')} | severity={verdict.get('severity')}")

    # ── L4: Validation & repair — retry_fn wired to L3 ──
    # The retry_fn builds a new L3 prompt that names the specific violation,
    # giving the model one correction opportunity before we hard-repair.
    def _l4_retry_fn(violation_desc: str) -> dict:
        print(f"  [L4→L3 retry] Violation: {violation_desc}")
        # Inject violation context into the L2 result so L3 prompt carries it
        l2_with_correction = dict(l2)
        l2_with_correction["_correction_note"] = (
            f"CORRECTION REQUIRED: Your previous response had the following schema violation: "
            f"{violation_desc}. Please fix only those fields and re-output the full JSON."
        )
        return assess_visual_evidence(
            claim_object=record["claim_object"],
            image_paths=image_paths,
            image_ids=image_ids,
            l2_result=l2_with_correction,
            evidence_requirements=ev_reqs,
            history_risk_summary=history_risk,
            claim_id=f"{claim_id}_retry",
        )

    verdict = validate_and_repair(
        verdict=verdict,
        claim_object=record["claim_object"],
        real_image_ids=image_ids,
        retry_fn=_l4_retry_fn,
    )

    # ── L5: Assemble output row ──
    return assemble_row(record, verdict)


# ──────────────────────────────────────────────
# Pipeline runner (shared by main.py and evaluation)
# ──────────────────────────────────────────────

def run_pipeline(
    input_csv: Path,
    output_csv: Path,
    mode: str = "predict",
    max_workers: int = 4,
) -> list[dict]:
    """
    Run the full L1→L5 pipeline on input_csv.
    mode='predict'  → dataset/claims.csv (4 input cols only)
    mode='evaluate' → dataset/sample_claims.csv (14 cols, expected outputs stashed separately)
    """
    print(f"\n{'='*60}")
    print(f"Mode     : {mode.upper()}")
    print(f"Input    : {input_csv}")
    print(f"Output   : {output_csv}")
    print(f"Workers  : {max_workers}")
    print(f"{'='*60}\n")

    records       = load_claims(input_csv, mode=mode)
    history_lookup = load_user_history()
    evidence_reqs  = load_evidence_requirements()

    print(f"Loaded {len(records)} claims.")
    print(f"Loaded {len(history_lookup)} history entries.")
    print(f"Loaded {len(evidence_reqs)} evidence requirements.\n")

    for i, r in enumerate(records):
        if r["invalid_images"]:
            print(f"[WARN] Claim {i+1} ({r['user_id']}): unresolved images: {r['invalid_images']}")

    total       = len(records)
    output_rows = [None] * total
    t_start     = time.perf_counter()

    if max_workers == 1:
        # Sequential mode: respect free-tier RPM by sleeping between claims
        for i, rec in enumerate(records):
            try:
                output_rows[i] = process_claim(rec, history_lookup, evidence_reqs, i + 1, total)
            except Exception as exc:
                print(f"[ERROR] Claim {i+1} failed: {exc}")
                output_rows[i] = assemble_row(rec, _error_verdict(f"Pipeline error: {exc}"))
            if i < total - 1:
                print(f"  [rate-limit] sleeping {CLAIM_SLEEP_S}s before next claim...")
                time.sleep(CLAIM_SLEEP_S)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_claim, rec, history_lookup, evidence_reqs, i + 1, total
                ): i
                for i, rec in enumerate(records)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    output_rows[idx] = future.result()
                except Exception as exc:
                    print(f"[ERROR] Claim {idx+1} failed: {exc}")
                    output_rows[idx] = assemble_row(
                        records[idx], _error_verdict(f"Pipeline error: {exc}")
                    )

    elapsed = time.perf_counter() - t_start
    print(f"\n[Done] {total} claims in {elapsed:.1f}s")

    write_output_csv(output_rows, output_csv)
    return output_rows


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HackerRank Orchestrate — Damage Claim Verifier (Mode B)"
    )
    parser.add_argument(
        "--input", type=Path,
        default=DATASET_ROOT / "claims.csv",
        help="Input CSV (default: dataset/claims.csv)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent.parent / "output.csv",
        help="Output CSV path (default: output.csv in repo root)",
    )
    parser.add_argument(
        "--workers", type=int,
        default=int(os.environ.get("MAX_WORKERS", 4)),
        help="Concurrent workers (default: 4)",
    )
    args = parser.parse_args()

    run_pipeline(
        input_csv=args.input,
        output_csv=args.output,
        mode="predict",
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
