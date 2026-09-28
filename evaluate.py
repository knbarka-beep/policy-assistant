"""Runs every test question through all three methods and saves the comparison.

    py evaluate.py              three methods
    py evaluate.py --baseline   also an LLM that gets no policy data at all
    py evaluate.py --rescore    redo the support check on saved answers without calling Gemini

Writes results/eval_results.csv (one row per question and method),
results/summary.csv and results/per_question.csv (the two tables on the website).
"""
import argparse
import csv
import time
from pathlib import Path

import policy_engine as pe

RESULTS = Path(__file__).parent / "results"


def expected_titles(value):
    return set() if value == "NONE" else {t.strip() for t in value.split("|")}


def score(row, r):
    expected = expected_titles(row["expected_policy"])
    cited = {p["title"] for p in r["policies"]}
    answerable = bool(expected)
    return {
        "question": row["question"],
        "type": row["type"],
        "expected": row["expected_policy"],
        "method": pe.METHOD_NAMES[r["method"]],
        "answer": r["answer"],
        "cited_policy": "; ".join(sorted(cited)) or "",
        "declined": r["declined"],
        "right_policy": answerable and bool(cited & expected),
        "declined_correctly": (not answerable) and r["declined"],
        "invented_details": r["support"]["invented"],
        "unsupported": r["support"]["unsupported"] or ((not answerable) and not r["declined"]),
        "support_check": r["support"]["label"],
        "seconds": round(r["seconds"], 3),
        "tokens_total": r["tokens"]["total"],
        "model": r["model"] or "",
        "error": "",
    }


def run_with_retry(fn, tries=3, wait=20):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            if attempt == tries - 1:
                raise
            print(f"   retrying after error: {str(e)[:120]}")
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="also run an LLM with no policy data")
    parser.add_argument("--pause", type=float, default=4.0, help="seconds between questions (free-tier rate limit)")
    parser.add_argument("--rescore", action="store_true", help="redo the support check on saved answers, no API calls")
    args = parser.parse_args()
    if args.rescore:
        rescore()
        return

    from google import genai  # imported here so --rescore works without the SDK

    client = genai.Client(api_key=pe.get_secret("GEMINI_API_KEY"))
    policies = pe.load_policies()
    rules = pe.RulesIndex(policies)
    index = pe.load_index(policies)
    with open(pe.ROOT / "data" / "eval_questions.csv", newline="", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    methods = [
        ("rules", lambda q: pe.rules_based(q, rules)),
        ("no_index", lambda q: pe.llm_no_index(client, q, policies)),
        ("index", lambda q: pe.llm_with_index(client, q, policies, index)),
    ]
    if args.baseline:
        methods.append(("baseline", lambda q: pe.llm_no_index(client, q, policies, mode="no_context")))

    rows = []
    for n, row in enumerate(questions, 1):
        print(f"{n}/{len(questions)} {row['question']}")
        for key, fn in methods:
            try:
                r = run_with_retry(lambda: fn(row["question"]))
                rows.append(score(row, r))
            except Exception as e:
                rows.append({"question": row["question"], "type": row["type"], "expected": row["expected_policy"],
                             "method": pe.METHOD_NAMES[key], "error": str(e)[:200]})
                print(f"   {pe.METHOD_NAMES[key]} failed: {str(e)[:120]}")
        time.sleep(args.pause)

    write_outputs(rows, questions, [pe.METHOD_NAMES[k] for k, _ in methods])


FIELDS = ["question", "type", "expected", "method", "answer", "cited_policy", "declined", "right_policy",
          "declined_correctly", "invented_details", "unsupported", "support_check", "seconds",
          "tokens_total", "model", "error"]


def rescore():
    """Re-run only the support check on saved answers, without calling Gemini again."""
    policies = {p["title"]: p for p in pe.load_policies()}
    with open(pe.ROOT / "data" / "eval_questions.csv", newline="", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))
    with open(RESULTS / "eval_results.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r.get("error"):
            continue
        for key in ("declined", "right_policy", "declined_correctly"):
            r[key] = r[key] == "True"
        r["seconds"] = float(r["seconds"])
        r["tokens_total"] = int(r["tokens_total"])
        cited = [policies[t] for t in r["cited_policy"].split("; ") if t in policies]
        support = pe.support_check({"declined": r["declined"], "policies": cited, "answer": r["answer"]}, r["question"])
        answerable = r["expected"] != "NONE"
        r["invented_details"] = support["invented"]
        r["unsupported"] = support["unsupported"] or ((not answerable) and not r["declined"])
        r["support_check"] = support["label"]
    methods = list(dict.fromkeys(r["method"] for r in rows))
    write_outputs(rows, questions, methods)


def write_outputs(rows, questions, method_names):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "eval_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    n_answerable = sum(1 for q in questions if q["expected_policy"] != "NONE")
    n_none = len(questions) - n_answerable
    summary = []
    for name in method_names:
        ok = [r for r in rows if r["method"] == name and not r.get("error")]
        failed = sum(1 for r in rows if r["method"] == name and r.get("error"))
        times = sorted(r["seconds"] for r in ok) or [0]
        summary.append({
            "Method": name,
            f"Right policy (of {n_answerable})": sum(r["right_policy"] for r in ok),
            f"Declined correctly (of {n_none})": sum(r["declined_correctly"] for r in ok),
            f"Invented details (of {len(questions)})": sum(r["invented_details"] for r in ok),
            f"Unsupported answers (of {len(questions)})": sum(r["unsupported"] for r in ok),
            "Median response time (s)": pe.fmt_seconds(times[len(times) // 2]),
            "Avg tokens per question": pe.fmt_int(round(sum(r["tokens_total"] for r in ok) / max(len(ok), 1))),
            "Failed calls": failed,
        })
    with open(RESULTS / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    per_q = []
    for q in questions:
        line = {"Question": q["question"],
                "Expected policy": q["expected_policy"].replace("|", " or ").replace("NONE", "None (should decline)")}
        for name in method_names:
            r = next((x for x in rows if x["question"] == q["question"] and x["method"] == name), None)
            if r is None or r.get("error"):
                cell = "Error"
            elif r["declined"]:
                cell = "Declined"
            else:
                cell = r["cited_policy"] or "No valid policy"
                if r["unsupported"]:
                    cell += " (unsupported)"
            line[name] = cell
        per_q.append(line)
    with open(RESULTS / "per_question.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_q[0].keys()))
        w.writeheader()
        w.writerows(per_q)

    print("\nSummary")
    for s in summary:
        print(" | ".join(f"{k}: {v}" for k, v in s.items()))


if __name__ == "__main__":
    main()
