# Damage Claim Verifier — Code README

Multi-modal evidence review system for the HackerRank Orchestrate hackathon.
Verifies damage claims (car / laptop / package) using images, claim conversations,
user history, and evidence requirements — powered by Gemini 2.0 Flash.

---

## Architecture

```
dataset/claims.csv
       |
   [L1] ingestion.py          — parse CSV, resolve images, load reference data
       |
   [L2] claim_extraction.py   — text-only Gemini call: extract structured claim
       |                         (handles Hinglish/Spanish/Chinese)
   [L3] vision_reasoning.py   — multimodal Gemini call: image + claim → verdict
       |                         (dynamic object_part enum per claim_object)
   [L4] validation.py         — deterministic enum checks + consistency rules + retry
       |
   [L5] assembly.py           — exact 14-column output row → output.csv

   [CACHE] cache.py           — SHA-256 disk cache (zero API calls on re-run)
   [TELEMETRY] telemetry.py   — per-call token/latency log → JSONL
```

---

## Quick Start

### 1. Install dependencies

```bash
cd code/
pip install -r requirements.txt
```

### 2. Set your API key

Copy `.env.local` (in the repo root) and fill in your key:

```
GEMINI_API_KEY=your_key_here
```

### 3. Run verification (no API key needed for Phases 0-2)

```bash
# From repo root:
python verify_phases_0_2.py
```

### 4. Run the full pipeline (Mode B — produces output.csv)

```bash
# From repo root:
python code/main.py
# or with options:
python code/main.py --input dataset/claims.csv --output output.csv --workers 4
```

### 5. Run evaluation (Mode A — sample_claims.csv with metrics)

```bash
python code/evaluation/main.py
```

---

## File Structure

```
code/
├── main.py                  # Mode B entry point
├── ingestion.py             # L1: CSV loaders, image resolver, allowed values
├── claim_extraction.py      # L2: text-only claim extraction (Gemini)
├── vision_reasoning.py      # L3: multimodal visual reasoning (Gemini)
├── validation.py            # L4: deterministic validation & repair
├── assembly.py              # L5: output row assembly + CSV writer
├── cache.py                 # Disk cache (SHA-256 keyed)
├── telemetry.py             # API call telemetry (JSONL)
├── requirements.txt         # google-genai, python-dotenv
├── .cache/                  # Auto-created: cached API responses
├── telemetry.jsonl          # Auto-created: API call log
└── evaluation/
    ├── main.py              # Mode A evaluation harness
    └── evaluation_report.md # Auto-generated after eval run
```

---

## Key Design Decisions

### Two-stage LLM calls (L2 + L3)
- **L2** (text-only): extracts structured claim (part, issue family) from conversation.
  Cheap (~500 tokens), deterministic, separately cacheable.
- **L3** (multimodal): uses L2 output as structured context + all images → verdict.
  Separation allows L2 to be reused if L3 is retried or swapped.

### Dynamic object_part enum
The L3 prompt's `object_part` field is constrained to the *correct enum for the
claim_object* (car / laptop / package). This eliminates the most common schema
error without needing a correction call.

### Images as primary truth
The L3 system prompt explicitly states:
> "The images are the PRIMARY source of truth. User history is CONTEXT ONLY —
>  it may add risk flags but MUST NEVER flip a decision that images clearly
>  support or contradict."

### Prompt injection resistance
The system prompt explicitly instructs the model to ignore any claim-language
or image-embedded text that says "approve this", "ignore previous instructions", etc.

### Caching
Every L2 and L3 response is cached to `.cache/` by SHA-256 of (claim text + image
file hashes + prompt version). Re-running with identical inputs is instant.

### Concurrency
`ThreadPoolExecutor(max_workers=4)` by default. Tune with `--workers N` or
`MAX_WORKERS` env var. Exponential backoff on 429/5xx errors.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | (required) | Gemini API key |
| `GEMINI_MODEL_TEXT` | `gemini-2.0-flash` | Model for L2 text calls |
| `GEMINI_MODEL_VISION` | `gemini-2.0-flash` | Model for L3 vision calls |
| `MAX_WORKERS` | `4` | Concurrent pipeline workers |
| `CACHE_DIR` | `code/.cache` | Disk cache directory |

---

## Output Schema

14 columns, exact order:

```
user_id, image_paths, user_claim, claim_object,
evidence_standard_met, evidence_standard_met_reason,
risk_flags, issue_type, object_part,
claim_status, claim_status_justification,
supporting_image_ids, valid_image, severity
```
