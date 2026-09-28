"""Builds the vector index: one Gemini embedding per policy, saved to data/policy_index.json.

Run once, and again only if company_policies.csv changes:  py build_index.py
"""
import json
from datetime import date

from google import genai

import policy_engine as pe


def main():
    client = genai.Client(api_key=pe.get_secret("GEMINI_API_KEY"))
    policies = pe.load_policies()
    vectors = pe.embed_texts(client, [pe.index_text(p) for p in policies], "RETRIEVAL_DOCUMENT")
    index = {
        "model": pe.EMBED_MODEL,
        "dim": pe.EMBED_DIM,
        "built": date.today().strftime("%d.%m.%Y"),
        "titles": [p["title"] for p in policies],
        "vectors": [[round(float(x), 6) for x in v] for v in vectors],
    }
    with open(pe.INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f)
    print(f"Saved {len(vectors)} vectors of length {vectors.shape[1]} to {pe.INDEX_FILE}")


if __name__ == "__main__":
    main()
