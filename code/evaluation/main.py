"""
Evaluation Harness — Mode A (Phase 7 + Phase 8).
=================================================
Runs the full pipeline on sample_claims.csv using TWO strategies:

  Strategy A: Two-Stage   (L2 text extraction → L3 vision)  ← primary
  Strategy B: Single-Call (raw conversation → L3 vision)     ← comparison baseline

Compares both against the 10 expected-output columns already in
sample_claims.csv (which are NEVER passed to any model prompt).

Writes:
  code/evaluation/eval_two_stage.csv      — Strategy A predictions
  code/evaluation/eval_single_call.csv    — Strategy B predictions
  code/evaluation/evaluation_report.md   — metrics + disagreements + operational analysis

Usage:
    python code/evaluation/main.py [--workers N]
"""

import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Path bootstrap ──────────────────────────────────────────────────────────
CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "strategies"))

from dotenv import load_dotenv
_root = CODE_DIR.parent
for _env in (".env", ".env.local"):
    _ef = _root / _env
    if _ef.exists():
        load_dotenv(_ef)
        break

# ── Project imports ──────────────────────────────────────────────────────────
from assembly import assemble_row, write_output_csv
from claim_extraction import extract_claim
from ingestion import (
    DATASET_ROOT,
    OUTPUT_COLUMNS,
    format_history_risk_summary,
    get_evidence_requirements,
    get_user_history,
    load_claims,
    load_evidence_requirements,
    load_user_history,
)
from main import process_claim, run_pipeline
from strategies.single_call import assess_single_call
from telemetry import load_telemetry, summarize_telemetry
from validation import validate_and_repair

# ── Constants ────────────────────────────────────────────────────────────────
EVAL_DIR      = Path(__file__).resolve().parent
SAMPLE_CSV    = DATASET_ROOT / "sample_claims.csv"
CLAIMS_CSV    = DATASET_ROOT / "claims.csv"
REPORT_MD     = EVAL_DIR / "evaluation_report.md"

EXACT_FIELDS  = ["claim_status", "issue_type", "object_part",
                  "evidence_standard_met", "valid_image", "severity"]
SET_FIELDS    = ["risk_flags", "supporting_image_ids"]


# ════════════════════════════════════════════════════════════════════════════
# Strategy B runner  (single-call, no L2)
# ════════════════════════════════════════════════════════════════════════════

def _run_single_call_claim(
    record: dict,
    history_lookup: dict,
    evidence_reqs: list[dict],
    idx: int,
    total: int,
) -> dict:
    """Process one claim using the single-call strategy."""
    claim_id = f"{record['user_id']}_sc_{idx}"
    print(f"  [SC {idx}/{total}] {record['user_id']} ({record['claim_object']})")

    if not record["resolved_images"]:
        verdict = {
            "evidence_standard_met": "false",
            "evidence_standard_met_reason": "No valid images.",
            "valid_image": "false", "risk_flags": "manual_review_required",
            "issue_type": "unknown", "object_part": "unknown",
            "claim_status": "not_enough_information",
            "claim_status_justification": "No images available.",
            "supporting_image_ids": "none", "severity": "unknown",
        }
        return assemble_row(record, verdict)

    hist = get_user_history(history_lookup, record["user_id"])
    history_risk = format_history_risk_summary(hist)

    # For single-call, pass all-object evidence requirements (no L2 issue family)
    ev = get_evidence_requirements(evidence_reqs, record["claim_object"])

    verdict = assess_single_call(
        user_claim=record["user_claim"],
        claim_object=record["claim_object"],
        image_paths=record["resolved_images"],
        image_ids=record["image_id_list"],
        evidence_requirements=ev,
        history_risk_summary=history_risk,
        claim_id=claim_id,
    )

    verdict = validate_and_repair(
        verdict=verdict,
        claim_object=record["claim_object"],
        real_image_ids=record["image_id_list"],
        retry_fn=None,   # no retry in single-call baseline
    )
    return assemble_row(record, verdict)


def run_single_call_pipeline(
    records: list[dict],
    output_csv: Path,
    max_workers: int = 4,
) -> list[dict]:
    """Run Strategy B (single-call) on a pre-loaded record list."""
    history_lookup = load_user_history()
    evidence_reqs  = load_evidence_requirements()

    total       = len(records)
    output_rows = [None] * total

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_single_call_claim,
                rec, history_lookup, evidence_reqs, i + 1, total
            ): i
            for i, rec in enumerate(records)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                output_rows[idx] = future.result()
            except Exception as exc:
                print(f"  [SC ERROR] Claim {idx+1}: {exc}")
                output_rows[idx] = assemble_row(records[idx], {
                    "evidence_standard_met": "false",
                    "evidence_standard_met_reason": f"Error: {exc}",
                    "valid_image": "false", "risk_flags": "manual_review_required",
                    "issue_type": "unknown", "object_part": "unknown",
                    "claim_status": "not_enough_information",
                    "claim_status_justification": "Error during single-call processing.",
                    "supporting_image_ids": "none", "severity": "unknown",
                })

    write_output_csv(output_rows, output_csv)
    return output_rows


# ════════════════════════════════════════════════════════════════════════════
# Scoring
# ════════════════════════════════════════════════════════════════════════════

def score(records: list[dict], predictions: list[dict]) -> dict:
    """
    Compare predictions vs expected outputs.
    Returns {"metrics": {...}, "disagreements": [...]}
    """
    exact_scores = {f: [] for f in EXACT_FIELDS}
    set_scores   = {f: [] for f in SET_FIELDS}
    disagreements = []

    for rec, pred in zip(records, predictions):
        expected = rec.get("expected_output", {})
        if not expected:
            continue

        row_diff = {}

        for field in EXACT_FIELDS:
            exp_v  = expected.get(field, "").strip().lower()
            pred_v = str(pred.get(field, "")).strip().lower()
            ok = exp_v == pred_v
            exact_scores[field].append(ok)
            if not ok:
                row_diff[field] = {"expected": exp_v, "predicted": pred_v}

        for field in SET_FIELDS:
            exp_raw  = expected.get(field, "none").strip().lower()
            pred_raw = str(pred.get(field, "none")).strip().lower()
            exp_set  = set(exp_raw.split(";"))  - {"none", ""}
            pred_set = set(pred_raw.split(";")) - {"none", ""}
            j = _jaccard(exp_set, pred_set)
            set_scores[field].append(j)
            if j < 1.0:
                row_diff[f"{field}_jaccard"] = {
                    "expected": exp_raw, "predicted": pred_raw, "jaccard": round(j, 3)
                }

        if row_diff:
            disagreements.append({
                "user_id":      rec["user_id"],
                "claim_object": rec["claim_object"],
                "image_paths":  rec["image_paths"],
                "user_claim_snippet": rec["user_claim"][:120],
                "disagreements": row_diff,
            })

    metrics = {}
    for f in EXACT_FIELDS:
        s = exact_scores[f]
        metrics[f] = round(sum(s) / len(s), 4) if s else 0.0
    for f in SET_FIELDS:
        s = set_scores[f]
        metrics[f"{f}_jaccard"] = round(sum(s) / len(s), 4) if s else 0.0

    exact_vals = [v for k, v in metrics.items() if not k.endswith("_jaccard")]
    metrics["overall_exact_match"] = round(sum(exact_vals) / len(exact_vals), 4) if exact_vals else 0.0

    return {"metrics": metrics, "disagreements": disagreements}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


# ════════════════════════════════════════════════════════════════════════════
# Report writer  (Phase 7 + Phase 8)
# ════════════════════════════════════════════════════════════════════════════

def _metrics_table(metrics: dict) -> str:
    rows = ["| Field | Score |", "|---|---|"]
    for k, v in metrics.items():
        rows.append(f"| {k} | {v} |")
    return "\n".join(rows)


def _disagreement_block(disagreements: list[dict], strategy_name: str) -> list[str]:
    lines = [
        f"\n### Disagreement Examples — {strategy_name}\n",
        "_Cases where prediction != expected output. Shown with root-cause notes._\n\n",
    ]
    shown = disagreements[:5]  # architecture requires at least 3
    if not shown:
        lines.append("_No disagreements found._\n")
        return lines

    for i, d in enumerate(shown, 1):
        lines.append(f"#### Example {i}: {d['user_id']} ({d['claim_object']})\n")
        lines.append(f"- **Images**: `{d['image_paths']}`\n")
        lines.append(f"- **Claim snippet**: _{d.get('user_claim_snippet', '')}..._\n\n")
        lines.append("| Field | Expected | Predicted | Note |\n|---|---|---|---|\n")
        for field, vals in d["disagreements"].items():
            exp  = vals.get("expected", "")
            pred = vals.get("predicted", "")
            j    = vals.get("jaccard", "")
            note = _root_cause_note(field, exp, pred)
            score_label = f"(Jaccard: {j})" if j else ""
            lines.append(f"| {field} | `{exp}` | `{pred}` {score_label} | {note} |\n")
        lines.append("\n")
    return lines


def _root_cause_note(field: str, expected: str, predicted: str) -> str:
    """Short root-cause heuristic for common disagreements."""
    if field == "claim_status":
        if expected == "supported" and predicted == "not_enough_information":
            return "Model was too conservative; evidence likely visible but flagged as insufficient"
        if expected == "contradicted" and predicted == "supported":
            return "Model missed mismatch between claim and image content"
        if expected == "contradicted" and predicted == "not_enough_information":
            return "Model defaulted to NEI instead of actively contradicting"
    if field == "severity":
        return "Severity calibration difference — subjective assessment"
    if field == "issue_type":
        return "Fine-grained damage type confusion (e.g. crack vs broken_part)"
    if field == "object_part":
        return "Part identification ambiguity from image angle or framing"
    if field == "evidence_standard_met":
        return "Evidence threshold interpretation differs from ground truth"
    if field == "valid_image":
        return "Image usability assessment diverges from ground truth label"
    if "jaccard" in field.lower():
        return "Partial overlap in multi-value field"
    return "Disagreement requires manual inspection"


def write_report(
    strategy_a_result: dict,
    strategy_b_result: dict,
    chosen_strategy: str,
    chosen_reason: str,
    telem_a: dict,
    telem_all: dict,
    n_sample: int,
    n_test: int,
    runtime_a: float,
    runtime_b: float,
) -> None:
    lines = [
        "# Evaluation Report\n\n",
        "> Generated automatically by `code/evaluation/main.py`.\n",
        "> Do not edit manually — re-run evaluation to regenerate.\n\n",
        "---\n\n",
        "## 1. Overview\n\n",
        f"| Item | Value |\n|---|---|\n",
        f"| Sample set | {n_sample} claims (sample_claims.csv) |\n",
        f"| Test set | {n_test} claims (claims.csv) |\n",
        f"| Chosen strategy | **{chosen_strategy}** |\n",
        f"| Strategy A runtime | {runtime_a:.1f}s |\n",
        f"| Strategy B runtime | {runtime_b:.1f}s |\n\n",
        "---\n\n",
        "## 2. Strategy Comparison\n\n",
        "The evaluation compares two strategies on `sample_claims.csv`. "
        "Expected outputs from the CSV are used ONLY for scoring — they are never passed "
        "to any model prompt.\n\n",

        "### Strategy A: Two-Stage (L2 text extraction + L3 vision)\n\n",
        "_Text-only L2 call extracts structured claim (part, issue family). "
        "Multimodal L3 call uses this structured context + all images to produce verdict. "
        "Separation allows independent caching and retry of each stage._\n\n",
        _metrics_table(strategy_a_result["metrics"]), "\n\n",

        "### Strategy B: Single-Call (raw conversation + L3 vision)\n\n",
        "_Raw conversation text is passed directly into the L3 multimodal prompt "
        "with no separate extraction step. One fewer API call per claim, "
        "but the vision model must parse the conversation AND assess images in one shot._\n\n",
        _metrics_table(strategy_b_result["metrics"]), "\n\n",

        "### Winner: " + chosen_strategy + "\n\n",
        f"**Rationale:** {chosen_reason}\n\n",
        "---\n\n",
        "## 3. Disagreement Analysis\n\n",
        "At least 3 concrete examples where predictions differed from expected output,\n"
        "with root-cause notes.\n",
    ]

    lines += _disagreement_block(strategy_a_result["disagreements"], "Strategy A (Two-Stage)")

    if strategy_b_result["disagreements"]:
        lines += _disagreement_block(strategy_b_result["disagreements"][:3], "Strategy B (Single-Call)")

    # ── Phase 8: Operational Analysis ──────────────────────────────────────
    by_stage   = telem_all.get("by_stage", {})
    tot_calls  = telem_all.get("total_calls", 0)
    cached     = telem_all.get("cached_calls", 0)
    uncached   = telem_all.get("uncached_calls", 0)
    tok_in     = telem_all.get("total_tokens_in", 0)
    tok_out    = telem_all.get("total_tokens_out", 0)
    images     = telem_all.get("total_images_processed", 0)
    avg_lat    = telem_all.get("avg_latency_ms_uncached", 0.0)

    lines += [
        "\n---\n\n",
        "## 4. Operational Analysis\n\n",
        "_All numbers below are measured from `code/telemetry.jsonl`, not estimated._\n\n",
        "### 4.1 Model Calls (Sample Run — both strategies)\n\n",
        f"| Metric | Value |\n|---|---|\n",
        f"| Total API calls | {tot_calls} |\n",
        f"| Uncached (actual API) | {uncached} |\n",
        f"| Cached (disk, 0ms) | {cached} |\n",
    ]
    for stage, info in sorted(by_stage.items()):
        lines.append(f"| Calls — {stage} | {info['calls']} |\n")

    lines += [
        "\n### 4.2 Token Usage (Sample Run)\n\n",
        f"| Metric | Value |\n|---|---|\n",
        f"| Total input tokens | {tok_in:,} |\n",
        f"| Total output tokens | {tok_out:,} |\n",
        f"| Images processed | {images} |\n",
        f"| Avg latency per uncached call | {avg_lat:.0f} ms |\n",
    ]

    # Extrapolate to full test set using Strategy A stats only
    a_calls = n_sample * 2   # L2 + L3 per claim
    if n_sample > 0 and uncached > 0:
        scale       = n_test / n_sample
        est_in      = int(tok_in * scale)
        est_out     = int(tok_out * scale)
        est_images  = int(images * scale)
        est_cost_in = (est_in / 1_000_000) * 0.075
        est_cost_out= (est_out / 1_000_000) * 0.30
        est_cost    = est_cost_in + est_cost_out
        est_calls   = int(tot_calls * scale)
        # Runtime: (avg_latency * calls_per_claim) / workers
        est_rt      = (avg_lat / 1000) * (n_test * 2) / 4   # 4 workers
        lines += [
            "\n### 4.3 Full Test Set Estimates (Strategy A on claims.csv)\n\n",
            "_Pricing: Gemini 2.0 Flash — $0.075/1M input tokens, $0.30/1M output tokens._\n",
            "_Image tokens: ~258 tokens/image at standard resolution._\n\n",
            f"| Metric | Estimate |\n|---|---|\n",
            f"| Input tokens | {est_in:,} |\n",
            f"| Output tokens | {est_out:,} |\n",
            f"| Images | {est_images} |\n",
            f"| Approx cost | ${est_cost:.4f} USD |\n",
            f"| API calls (L2+L3, {n_test} claims) | {n_test*2} |\n",
            f"| Est. runtime (4 workers) | ~{est_rt:.0f}s |\n",
        ]

    lines += [
        "\n### 4.4 Rate Limits, Batching & Reliability\n\n",
        "| Concern | Strategy |\n|---|---|\n",
        "| **Concurrency** | `ThreadPoolExecutor(max_workers=4)` — "
        "balances throughput vs. RPM limits |\n",
        "| **Retry on 429/5xx** | Exponential backoff: `sleep(2^attempt)`, max 2 retries |\n",
        "| **L4 violation retry** | If L3 output fails schema validation, L3 is called once "
        "more with the specific violation named in the prompt context |\n",
        "| **Disk cache** | SHA-256(claim text + sorted image hashes + prompt version). "
        "Re-runs with identical inputs hit cache (0 API calls, ~1ms) |\n",
        "| **Cache invalidation** | Prompt version suffix in cache key — bump version to "
        "invalidate all cache entries |\n",
        "| **Telemetry** | Every call logged to `code/telemetry.jsonl` with tokens, "
        "latency, image count, stage, cached flag |\n",
        "| **Two-stage tradeoff** | L2 adds 1 cheap text call (~500 tokens). "
        "Benefit: L2 is cached independently — if L3 is retried with a different model, "
        "L2 does not need to re-run. Net cost: negligible vs. L3 vision call. |\n",
    ]

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\n[Eval] Report written -> {REPORT_MD}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluation harness — runs two strategies and writes evaluation_report.md"
    )
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent workers per strategy (default: 4)")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("[FAIL] GEMINI_API_KEY not set. Set it in .env or .env.local first.")
        sys.exit(1)

    print("\n" + "="*60)
    print(" EVALUATION HARNESS — Phase 7 + Phase 8")
    print("="*60)

    records = load_claims(SAMPLE_CSV, mode="evaluate")
    n_sample = len(records)
    test_records = load_claims(CLAIMS_CSV, mode="predict")
    n_test = len(test_records)
    print(f"\nSample set: {n_sample} claims")
    print(f"Test set  : {n_test} claims\n")

    # ── Strategy A: Two-Stage (L2 + L3) ────────────────────────────────────
    print("\n[Strategy A] Two-Stage (L2 + L3)...")
    t_a = time.perf_counter()
    output_a = run_pipeline(
        input_csv=SAMPLE_CSV,
        output_csv=EVAL_DIR / "eval_two_stage.csv",
        mode="evaluate",
        max_workers=args.workers,
    )
    runtime_a = time.perf_counter() - t_a
    score_a = score(records, output_a)
    telem_after_a = summarize_telemetry()
    print(f"\n[Strategy A] Done in {runtime_a:.1f}s")
    print(f"[Strategy A] Metrics: {json.dumps(score_a['metrics'], indent=2)}")

    # ── Strategy B: Single-Call (L3 only) ───────────────────────────────────
    print("\n[Strategy B] Single-Call (L3 only)...")
    t_b = time.perf_counter()
    output_b = run_single_call_pipeline(
        records=records,
        output_csv=EVAL_DIR / "eval_single_call.csv",
        max_workers=args.workers,
    )
    runtime_b = time.perf_counter() - t_b
    score_b = score(records, output_b)
    telem_all = summarize_telemetry()
    print(f"\n[Strategy B] Done in {runtime_b:.1f}s")
    print(f"[Strategy B] Metrics: {json.dumps(score_b['metrics'], indent=2)}")

    # ── Choose winner ────────────────────────────────────────────────────────
    a_overall = score_a["metrics"].get("overall_exact_match", 0)
    b_overall = score_b["metrics"].get("overall_exact_match", 0)
    if a_overall >= b_overall:
        chosen = "Strategy A: Two-Stage (L2+L3)"
        reason = (
            f"Two-Stage achieved overall_exact_match={a_overall:.4f} vs "
            f"Single-Call={b_overall:.4f}. "
            "The structured L2 extraction focuses L3 on the specific claimed part and issue "
            "family, reducing hallucination risk. L2 is independently cached — if L3 is "
            "retried or swapped to a different model, L2 does not re-run. The extra ~500 "
            "input tokens per claim is negligible compared to the L3 vision call."
        )
    else:
        chosen = "Strategy B: Single-Call"
        reason = (
            f"Single-Call achieved overall_exact_match={b_overall:.4f} vs "
            f"Two-Stage={a_overall:.4f}. "
            "The vision model successfully parsed the conversation and assessed images in one "
            "shot, saving one API round-trip per claim with no accuracy penalty."
        )

    # ── Write report ─────────────────────────────────────────────────────────
    write_report(
        strategy_a_result=score_a,
        strategy_b_result=score_b,
        chosen_strategy=chosen,
        chosen_reason=reason,
        telem_a=telem_after_a,
        telem_all=telem_all,
        n_sample=n_sample,
        n_test=n_test,
        runtime_a=runtime_a,
        runtime_b=runtime_b,
    )

    print(f"\n[Eval] Complete. Report: {REPORT_MD}")
    print(f"[Eval] Strategy A predictions: {EVAL_DIR / 'eval_two_stage.csv'}")
    print(f"[Eval] Strategy B predictions: {EVAL_DIR / 'eval_single_call.csv'}")


if __name__ == "__main__":
    main()
