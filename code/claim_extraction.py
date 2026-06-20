"""
L2 · CLAIM UNDERSTANDING  (text-only LLM call)
Extracts structured claim info from the user_claim conversation.
Uses Gemini 2.0 Flash with JSON mode (forced schema).
Handles Hinglish / mixed-language conversations.

Uses the new google.genai SDK (v2.x).
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

from cache import DiskCache
from telemetry import log_call

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
TEXT_MODEL = os.environ.get("GEMINI_MODEL_TEXT", "gemini-2.0-flash")
L2_PROMPT_VERSION = "l2_v3"

_cache = DiskCache()
_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")
        _client = genai.Client(api_key=api_key)
    return _client


# ──────────────────────────────────────────────
# L2 prompts and schema
# ──────────────────────────────────────────────

L2_SCHEMA_DESCRIPTION = """
{
  "primary_issue_family": "<string: one of: dent or scratch | crack | broken or missing part | torn or crushed packaging | water or stain damage | missing contents | general review>",
  "primary_object_part": "<string: the specific part the user claims is damaged, in plain English, e.g. 'rear bumper', 'screen', 'keyboard', 'package corner'>",
  "secondary_parts": ["<optional list of additional parts mentioned>"],
  "exclusions": ["<parts explicitly excluded from the claim, e.g. 'windshield', 'headlight'>"],
  "contains_language_other_than_english": "<true|false>",
  "claim_summary": "<1-2 sentence plain-English summary of what the user is claiming was damaged and how>"
}
"""

L2_SYSTEM_PROMPT = """You are a claim extraction assistant. Your job is to read a customer support
conversation and extract the structured damage claim. The conversation may be in English,
Hinglish (Hindi+English), Spanish, Chinese, or other languages — handle them correctly.

Extract ONLY what the user is actually claiming. Focus on:
1. What physical part is damaged (primary_object_part)
2. What kind of damage it is (primary_issue_family)
3. Any secondary parts mentioned
4. Any parts the user explicitly says should NOT be reviewed (exclusions)

Rules:
- If the user changes their mind mid-conversation, use their FINAL stated claim.
- Never invent damage not mentioned by the user.
- Keep claim_summary to 1-2 sentences max.
- Always respond with valid JSON only. No markdown, no explanation outside the JSON.
"""

L2_USER_TEMPLATE = """Claim conversation:
\"\"\"
{user_claim}
\"\"\"

Claim object type: {claim_object}

Extract the structured claim and respond with JSON matching this schema exactly:
{schema}"""


# ──────────────────────────────────────────────
# Public function
# ──────────────────────────────────────────────

def extract_claim(
    user_claim: str,
    claim_object: str,
    claim_id: str = "unknown",
    max_retries: int = 5,
) -> dict:
    """
    Runs L2 claim extraction.
    Returns a dict with keys: primary_issue_family, primary_object_part,
    secondary_parts, exclusions, contains_language_other_than_english, claim_summary.
    On failure returns a safe fallback dict.
    """
    cache_key = _cache.make_key(L2_PROMPT_VERSION, user_claim, claim_object)
    cached = _cache.get(cache_key)
    if cached:
        log_call("L2", claim_id, 0, 0, 0, 0.0, cached=True, model=TEXT_MODEL)
        return cached

    prompt = L2_USER_TEMPLATE.format(
        user_claim=user_claim,
        claim_object=claim_object,
        schema=L2_SCHEMA_DESCRIPTION,
    )
    full_prompt = L2_SYSTEM_PROMPT + "\n\n" + prompt

    client = _get_client()
    last_exc = None

    for attempt in range(max_retries):
        try:
            t0 = time.perf_counter()
            response = client.models.generate_content(
                model=TEXT_MODEL,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            raw_text = response.text.strip()
            result = _parse_l2_json(raw_text)

            # Telemetry
            usage = response.usage_metadata
            tokens_in = getattr(usage, "prompt_token_count", 0) or 0
            tokens_out = getattr(usage, "candidates_token_count", 0) or 0
            log_call("L2", claim_id, tokens_in, tokens_out, 0, latency_ms, model=TEXT_MODEL)

            _cache.set(cache_key, result)
            return result

        except Exception as exc:
            last_exc = exc
            err_str = str(exc)
            # For 429 rate limit, wait longer
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = min(30, 5 * (2 ** attempt))  # 5s, 10s, 20s, 30s...
            else:
                wait = 2 ** attempt
            print(f"  [L2] attempt {attempt + 1} failed for {claim_id}: {exc}. Retrying in {wait}s…")
            time.sleep(wait)

    print(f"  [L2] All retries exhausted for {claim_id}: {last_exc}. Using fallback.")
    return _fallback_l2(claim_object)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _parse_l2_json(raw: str) -> dict:
    """Parse and validate the L2 JSON response."""
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    data = json.loads(raw)

    defaults = {
        "primary_issue_family": "general review",
        "primary_object_part": "unknown",
        "secondary_parts": [],
        "exclusions": [],
        "contains_language_other_than_english": "false",
        "claim_summary": "",
    }
    for k, v in defaults.items():
        data.setdefault(k, v)

    for k in ("secondary_parts", "exclusions"):
        if isinstance(data[k], str):
            data[k] = [data[k]] if data[k] else []

    return data


def _fallback_l2(claim_object: str) -> dict:
    return {
        "primary_issue_family": "general review",
        "primary_object_part": "unknown",
        "secondary_parts": [],
        "exclusions": [],
        "contains_language_other_than_english": "false",
        "claim_summary": "Could not parse claim; treating as general review.",
    }
