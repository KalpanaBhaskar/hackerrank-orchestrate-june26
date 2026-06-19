"""
Phase 3 Verification — L2 Claim Extraction
Tests extraction on 5 sample conversations (including Hinglish) using Gemini.
Run: python verify_phase3_l2.py
Requires GEMINI_API_KEY in .env or .env.local
"""

import sys
import os
from pathlib import Path

# Load env
_root = Path(__file__).resolve().parent
for env_file in (".env", ".env.local"):
    ef = _root / env_file
    if ef.exists():
        from dotenv import load_dotenv
        load_dotenv(ef)
        print(f"[ENV] Loaded {env_file}")
        break

sys.path.insert(0, str(_root / "code"))

from claim_extraction import extract_claim

TEST_CASES = [
    {
        "id": "case_001_eng",
        "user_claim": "Customer: Hi, I found new damage on my car after it was parked outside overnight. | Support: Sorry to hear that. Can you describe what changed? | Customer: The back of the car has a dent now. It was not there before. | Support: Did anything else break or is it mostly body damage? | Customer: Mostly the rear bumper area. I attached the photo I took this morning.",
        "claim_object": "car",
        "expected_part": "rear bumper",
        "expected_issue": "dent",
    },
    {
        "id": "case_002_hinglish",
        "user_claim": "Customer: Parking lot mein meri car ko scrape lag gaya. | Support: Aap kis type ka damage report karna chahte hain? | Customer: Front side par mark aa gaya hai, bumper ke upar. | Support: Light damage hai ya body par scratch? | Customer: Light theek hai, front bumper par scratch hai. Photos upload kar diye hain.",
        "claim_object": "car",
        "expected_part": "front bumper",
        "expected_issue": "scratch",
    },
    {
        "id": "case_009_laptop",
        "user_claim": "Customer: My laptop fell from the table yesterday. | Support: Is it turning on? | Customer: It turns on, but the display glass has a crack now. | Support: Are you reporting the screen only or the whole laptop? | Customer: The screen is the issue. I attached a photo of it open.",
        "claim_object": "laptop",
        "expected_part": "screen",
        "expected_issue": "crack",
    },
    {
        "id": "case_016_hinglish_package",
        "user_claim": "Customer: Package receive hua toh opened jaisa lag raha tha. | Support: Tape broken tha ya box crush hua tha? | Customer: Seal wali side phati hui thi, jaise parcel khola gaya ho. | Support: Kya andar ka item missing hai? | Customer: Abhi item missing claim nahi kar raha, sirf torn packaging review karwana hai.",
        "claim_object": "package",
        "expected_part": "seal",
        "expected_issue": "torn",
    },
    {
        "id": "case_006_confused_user",
        "user_claim": "Customer: Hi, I am not fully sure how to explain this because I noticed it only after reaching home. | Support: No problem, please tell me what happened. | Customer: There was a small bump earlier, nothing major, and I first thought everything was fine. | Support: Are you reporting a general vehicle issue or a specific damaged part? | Customer: I was confused at first because I checked the side and the front. | Support: That is okay. What part do you want reviewed? | Customer: After checking again, I think the issue is with the headlight. It looks like the headlight may be cracked, so please review that part.",
        "claim_object": "car",
        "expected_part": "headlight",
        "expected_issue": "crack",
    },
]

def main():
    print("\n" + "="*60)
    print(" PHASE 3 - L2 Claim Extraction Verification")
    print("="*60)

    if not os.environ.get("GEMINI_API_KEY"):
        print("[FAIL] GEMINI_API_KEY not set. Set it in .env.local first.")
        sys.exit(1)

    all_ok = True
    for tc in TEST_CASES:
        print(f"\n--- {tc['id']} ({tc['claim_object']}) ---")
        result = extract_claim(
            user_claim=tc["user_claim"],
            claim_object=tc["claim_object"],
            claim_id=tc["id"],
        )

        print(f"  primary_object_part:  {result.get('primary_object_part', '?')}")
        print(f"  primary_issue_family: {result.get('primary_issue_family', '?')}")
        print(f"  claim_summary:        {result.get('claim_summary', '?')}")
        print(f"  secondary_parts:      {result.get('secondary_parts', [])}")
        print(f"  exclusions:           {result.get('exclusions', [])}")
        print(f"  multilingual:         {result.get('contains_language_other_than_english', '?')}")

        # Check required JSON keys
        required_keys = ["primary_issue_family", "primary_object_part", "secondary_parts",
                         "exclusions", "contains_language_other_than_english", "claim_summary"]
        has_all_keys = all(k in result for k in required_keys)

        if has_all_keys:
            print(f"  [PASS] All required JSON keys present")
        else:
            missing = [k for k in required_keys if k not in result]
            print(f"  [FAIL] Missing keys: {missing}")
            all_ok = False

        # Manual check guidance
        expected_part = tc["expected_part"].lower()
        extracted_part = result.get("primary_object_part", "").lower()
        part_match = expected_part in extracted_part or extracted_part in expected_part
        print(f"  Part match: expected '{expected_part}' got '{extracted_part}' -> {'[PASS]' if part_match else '[CHECK MANUALLY]'}")

    print("\n" + "="*60)
    print(f" RESULT: {'ALL JSON STRUCTURES VALID' if all_ok else 'SOME ISSUES FOUND'}")
    print("="*60 + "\n")
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
