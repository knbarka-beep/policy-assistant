"""Shared logic for the three policy methods.

Used by app.py (website), evaluate.py (comparison) and slack_bot.py (Slack).
Method 1: rules-based keyword search, no model call.
Method 2: LLM with no vector index. The whole policy file goes into the prompt.
Method 3: LLM with a vector index. Only the closest policies go into the prompt.
"""
import csv
import json
import math
import os
import re
import time
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
POLICY_CSV = ROOT / "data" / "company_policies.csv"
INDEX_FILE = ROOT / "data" / "policy_index.json"
SECRETS_FILE = ROOT / ".streamlit" / "secrets.toml"

# Same fallback list as the P-card app. If one model is overloaded (503) or closed (404), the next is tried.
GEN_MODELS = ["gemini-3.1-flash-lite-preview", "gemini-flash-latest", "gemini-3.5-flash-lite"]
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
TOP_K = 3

# "full_context": method 2 gets all 98 policies in the prompt (default).
# "no_context": method 2 gets no policies at all and answers from the model's own knowledge.
NO_INDEX_MODE = "full_context"

RULES_MIN_SCORE = 6.0  # below this the rules-based search declines
NOT_FOUND = "No policy in the database covers this question."

METHOD_NAMES = {
    "rules": "Rules-based search",
    "no_index": "LLM, no vector index",
    "index": "LLM with vector index",
    "baseline": "LLM, no policy data",
}


# ---------- secrets and data ----------

def get_secret(name):
    """Environment variable first, then the local .streamlit/secrets.toml (never committed)."""
    if os.environ.get(name):
        return os.environ[name]
    if SECRETS_FILE.exists():
        with open(SECRETS_FILE, "rb") as f:
            value = tomllib.load(f).get(name)
        if value:
            return value
    raise RuntimeError(f"{name} not found. Put it in .streamlit/secrets.toml or set it as an environment variable.")


def load_policies():
    with open(POLICY_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        {
            "id": i + 1,
            "title": r["title"].strip(),
            "department": r["department"].strip(),
            "category": r["category"].strip(),
            "text": r["policy_text"].strip(),
        }
        for i, r in enumerate(rows)
    ]


def policy_block(policies):
    return "\n".join(f"[{p['id']}] {p['title']} | {p['category']} | {p['text']}" for p in policies)


# ---------- method 1: rules-based search ----------

STOPWORDS = set("""
a an the i im me my we our us you your he she it its they them their is are was were be been being am
do does did doing have has had can could may might must shall should will would what whats when where which
who whom why how to of in on at for from by with about as into over under than then so if or and but not no
yes there this that these those any some all each every much many more most get got need needs want wants
please tell know policy policies company employee employees staff rule rules just also per
""".split())

# Phrases in the question are rewritten to the words the policy texts actually use.
SYNONYMS = [
    (r"\bholiday (days|allowance)\b", "vacation days"),
    (r"\b(pto|annual leave|days off|day off)\b", "vacation"),
    (r"\b(from home|wfh)\b", "remote"),
    (r"\b(overseas|another country|other country|spain|germany|france|italy|norway|sweden|denmark|usa)\b", "abroad"),
    (r"\b(ill|illness|unwell)\b", "sick"),
    (r"\b(baby|maternity|paternity|newborn)\b", "parental"),
    (r"\b(funeral|died|passed away|death)\b", "bereavement"),
    (r"\b(hacked|virus|phishing|breach|malware)\b", "security incident"),
    (r"\b(journalists?|reporters?|newspapers?)\b", "media"),
    (r"\b(anonymous|without giving my name)\b", "anonymously"),
    (r"\b(wear|clothes|clothing|outfit)\b", "dress attire"),
    (r"\b(business trip|trip|travelling|traveling)\b", "travel"),
    (r"\b(refund|pay back|paid back|reimbursed)\b", "reimburse"),
    (r"\b(broken|not working|repair|fix|fixed)\b", "issue helpdesk"),
    (r"\b(shop online|online shopping|browse|browsing)\b", "internet"),
    (r"\b(computer|pc)\b", "laptop"),
    (r"\bcar\b", "vehicle"),
    (r"\b(course|degree|education|university)\b", "tuition training"),
    (r"\b(buy|bought)\b", "purchase"),
    (r"\b(spend|spending|cost)\b", "expense"),
]

SUFFIXES = ["ments", "ment", "ings", "ing", "ies", "ied", "ly", "es", "ed", "al", "s", "e"]


def _stem(word):
    """Crude suffix stripping so 'reimbursed', 'reimburse' and 'reimbursement' all become 'reimburs'."""
    for _ in range(2):
        for suf in SUFFIXES:
            if word.endswith(suf) and len(word) - len(suf) >= 3:
                word = word[: -len(suf)]
                break
    return word


def _tokens(text, expand=False):
    text = text.lower()
    if expand:
        for pattern, repl in SYNONYMS:
            text = re.sub(pattern, repl, text)
    return {_stem(w) for w in re.findall(r"[a-z]+", text) if w not in STOPWORDS}


class RulesIndex:
    """Keyword index. Words that appear in few policies (high IDF) count more than common words."""

    def __init__(self, policies):
        self.policies = policies
        self.title_tokens = [_tokens(p["title"]) for p in policies]
        self.text_tokens = [_tokens(p["text"]) for p in policies]
        n = len(policies)
        df = {}
        for t_set, x_set in zip(self.title_tokens, self.text_tokens):
            for tok in t_set | x_set:
                df[tok] = df.get(tok, 0) + 1
        self.idf = {tok: math.log(n / c) for tok, c in df.items()}

    def score(self, question):
        q = _tokens(question, expand=True)
        scores = []
        for i, p in enumerate(self.policies):
            s = 0.0
            for tok in q:
                w = self.idf.get(tok, 0.0)
                if tok in self.title_tokens[i]:
                    s += 1.5 * w
                if tok in self.text_tokens[i]:
                    s += 1.0 * w
            scores.append((s, p))
        scores.sort(key=lambda x: -x[0])
        return scores


def rules_based(question, rules_index):
    start = time.perf_counter()
    ranked = rules_index.score(question)
    best_score, best = ranked[0]
    declined = best_score < RULES_MIN_SCORE
    result = {
        "method": "rules",
        "answer": NOT_FOUND if declined else best["text"],
        "policies": [] if declined else [best],
        "declined": declined,
        "tokens": {"input": 0, "output": 0, "total": 0},
        "model": None,
        "retrieved": [(p, round(s, 2)) for s, p in ranked[:TOP_K]],
    }
    result["seconds"] = time.perf_counter() - start
    result["support"] = support_check(result)
    return result


# ---------- LLM helpers ----------

SYSTEM_WITH_POLICIES = f"""You answer employee questions using only the company policies provided.
Rules:
1. Use only facts written in the policies. Never add numbers, limits, amounts or conditions that are not stated there.
2. If no policy covers the question, set "supported" to false, set "policy_ids" to [] and answer exactly: "{NOT_FOUND}"
3. If a policy covers the topic but does not state the detail asked for, say that the policy does not specify it.
4. Keep the answer to one to three sentences. Do not write policy id numbers in the answer text.
Return JSON only: {{"answer": string, "policy_ids": [integers], "supported": boolean}}"""

SYSTEM_NO_DATA = """You are a company policy assistant. Answer the employee's question and name the company policy that applies.
Keep the answer to one to three sentences.
Return JSON only: {"answer": string, "policy_titles": [strings], "supported": boolean}"""


def _generate(client, system, contents):
    from google.genai import types

    last_err = None
    for model in GEN_MODELS:
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
            u = resp.usage_metadata
            tokens = {
                "input": u.prompt_token_count or 0,
                "output": (u.candidates_token_count or 0) + (getattr(u, "thoughts_token_count", 0) or 0),
                "total": u.total_token_count or 0,
            }
            return resp.text, tokens, model
        except Exception as e:  # overloaded, closed or rate-limited model: try the next one
            last_err = e
    raise RuntimeError(f"All Gemini models failed. Last error: {last_err}")


def _parse_json(text):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", text or "", re.S)
        return json.loads(m.group(0)) if m else {"answer": text or "", "policy_ids": [], "supported": False}


def _finish(result, data, policies_by_id, start):
    answer = str(data.get("answer", "")).strip()
    ids = [int(i) for i in data.get("policy_ids", []) if str(i).isdigit() and int(i) in policies_by_id]
    result["answer"] = answer
    result["policies"] = [policies_by_id[i] for i in ids]
    result["cited_missing"] = []
    result["declined"] = (not data.get("supported", False)) or answer.startswith(NOT_FOUND)
    result["seconds"] = time.perf_counter() - start
    result["support"] = support_check(result)
    return result


# ---------- method 2: LLM, no vector index ----------

def llm_no_index(client, question, policies, mode=None):
    mode = mode or NO_INDEX_MODE
    start = time.perf_counter()
    by_id = {p["id"]: p for p in policies}
    if mode == "no_context":
        text, tokens, model = _generate(client, SYSTEM_NO_DATA, f"Question: {question}")
        data = _parse_json(text)
        by_title = {p["title"].lower(): p for p in policies}
        titles = [str(t) for t in data.get("policy_titles", [])]
        result = {"method": "baseline", "tokens": tokens, "model": model, "retrieved": []}
        result["answer"] = str(data.get("answer", "")).strip()
        result["policies"] = [by_title[t.lower()] for t in titles if t.lower() in by_title]
        result["cited_missing"] = [t for t in titles if t.lower() not in by_title]
        result["declined"] = not data.get("supported", True)
        result["seconds"] = time.perf_counter() - start
        result["support"] = support_check(result)
        return result

    contents = f"Company policies:\n{policy_block(policies)}\n\nQuestion: {question}"
    text, tokens, model = _generate(client, SYSTEM_WITH_POLICIES, contents)
    result = {"method": "no_index", "tokens": tokens, "model": model, "retrieved": []}
    return _finish(result, _parse_json(text), by_id, start)


# ---------- method 3: LLM with vector index ----------

def _normalise(m):
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


def embed_texts(client, texts, task_type):
    from google.genai import types

    vectors = []
    for i in range(0, len(texts), 50):
        resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts[i : i + 50],
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBED_DIM),
        )
        vectors.extend(e.values for e in resp.embeddings)
    return _normalise(np.array(vectors, dtype=np.float32))


def index_text(p):
    return f"{p['title']}. Category: {p['category']}. {p['text']}"


def load_index(policies):
    if not INDEX_FILE.exists():
        raise RuntimeError("data/policy_index.json is missing. Run: py build_index.py")
    with open(INDEX_FILE, encoding="utf-8") as f:
        idx = json.load(f)
    if idx["titles"] != [p["title"] for p in policies]:
        raise RuntimeError("The vector index does not match company_policies.csv. Run: py build_index.py")
    return {"vectors": _normalise(np.array(idx["vectors"], dtype=np.float32)), "model": idx["model"]}


def llm_with_index(client, question, policies, index):
    start = time.perf_counter()
    q_vec = embed_texts(client, [question], "RETRIEVAL_QUERY")[0]
    sims = index["vectors"] @ q_vec  # cosine similarity, vectors are normalised
    top = np.argsort(-sims)[:TOP_K]
    retrieved = [(policies[i], round(float(sims[i]), 3)) for i in top]
    contents = f"Company policies:\n{policy_block([p for p, _ in retrieved])}\n\nQuestion: {question}"
    text, tokens, model = _generate(client, SYSTEM_WITH_POLICIES, contents)
    result = {"method": "index", "tokens": tokens, "model": model, "retrieved": retrieved, "embedding_calls": 1}
    return _finish(result, _parse_json(text), {p["id"]: p for p, _ in retrieved}, start)


# ---------- support check ----------

NUM_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "fifteen": "15",
    "twenty": "20", "thirty": "30", "ninety": "90",
}


def _numbers(text):
    text = re.sub(r"\[\d+\]", "", text.lower()).replace(",", "")
    for word, digit in NUM_WORDS.items():
        text = re.sub(rf"\b{word}\b", digit, text)
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def support_check(result):
    """Deterministic check of whether an answer is backed by the policy database.

    Flags answers that cite no policy that exists, and answers that contain numbers
    (days, amounts, limits) that do not appear in the policies they cite.
    """
    if result["declined"]:
        return {"label": "Declined, no supporting policy", "invented": False, "unsupported": False}
    if not result["policies"]:
        return {"label": "Unsupported: cites no policy that exists in the database", "invented": True, "unsupported": True}
    if result.get("cited_missing"):
        missing = ", ".join(result["cited_missing"])
        return {"label": f"Unsupported: cites a policy not in the database ({missing})", "invented": True, "unsupported": True}
    source_numbers = set().union(*(_numbers(p["text"]) for p in result["policies"]))
    extra = sorted(_numbers(result["answer"]) - source_numbers)
    if extra:
        return {
            "label": f"Possibly unsupported: {', '.join(extra)} not in the cited policy",
            "invented": True,
            "unsupported": True,
        }
    return {"label": "Consistent with the cited policy", "invented": False, "unsupported": False}


# ---------- formatting ----------

def fmt_seconds(s):
    return f"{s:.2f}".replace(".", ",")


def fmt_int(n):
    return f"{n:,}".replace(",", ".")


def format_for_slack(r):
    lines = [r["answer"]]
    for p in r["policies"]:
        lines.append(f"\n*Relevant policy:* {p['title']} ({p['category']})\n> {p['text']}")
    lines.append(
        f"\n_{METHOD_NAMES[r['method']]}, {fmt_seconds(r['seconds'])} s, {fmt_int(r['tokens']['total'])} tokens_"
    )
    return "\n".join(lines)
