"""Company Policy Assistant: compares three ways of answering policy questions.

Run locally:  py -m streamlit run app.py
"""
import csv
from pathlib import Path

import pandas as pd
import streamlit as st

import policy_engine as pe

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"

st.set_page_config(page_title="Company Policy Assistant", layout="wide")

st.markdown(
    """<style>
    .block-container {max-width: 1200px; padding-top: 2rem;}
    .method-note {color: #5b6470; font-size: 0.9rem; margin-top: -0.6rem;}
    .policy-box {border-left: 3px solid #2f5d8a; padding: 0.4rem 0.8rem; background: rgba(47,93,138,0.06); margin-bottom: 0.5rem;}
    </style>""",
    unsafe_allow_html=True,
)

METHOD_NOTES = {
    "rules": "Keyword matching with weighted words and a synonym list. No model call. Returns the policy text itself.",
    "no_index": "Gemini receives all 98 policies in the prompt and is told to answer only from them.",
    "index": "The question is embedded and compared with a stored embedding of each policy. Gemini receives only the three closest policies.",
}


def md(text):
    """Escape characters Streamlit would otherwise treat as LaTeX or markup."""
    return str(text).replace("$", "\\$")


def api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return pe.get_secret("GEMINI_API_KEY")


@st.cache_resource
def load_everything():
    from google import genai

    policies = pe.load_policies()
    return {
        "client": genai.Client(api_key=api_key()),
        "policies": policies,
        "rules": pe.RulesIndex(policies),
        "index": pe.load_index(policies),
    }


def run_all(question, res):
    runs = [
        ("rules", lambda: pe.rules_based(question, res["rules"])),
        ("no_index", lambda: pe.llm_no_index(res["client"], question, res["policies"])),
        ("index", lambda: pe.llm_with_index(res["client"], question, res["policies"], res["index"])),
    ]
    out = []
    for key, fn in runs:
        try:
            out.append(fn())
        except Exception as e:
            out.append({"method": key, "error": str(e)})
    return out


def show_result(col, r):
    col.subheader(pe.METHOD_NAMES[r["method"]])
    col.markdown(f"<div class='method-note'>{METHOD_NOTES[r['method']]}</div>", unsafe_allow_html=True)
    if r.get("error"):
        col.error(f"This method failed: {r['error'][:300]}")
        return
    col.markdown("**Answer**")
    col.write(md(r["answer"]))
    col.markdown("**Relevant policy**")
    if r["policies"]:
        for p in r["policies"]:
            col.markdown(
                f"<div class='policy-box'><b>{p['title']}</b><br>{p['category']}, {p['department']} department"
                f"<br><i>{md(p['text'])}</i></div>",
                unsafe_allow_html=True,
            )
    else:
        col.write("None found")
    tokens = r["tokens"]
    token_line = (
        "0 (no model call)"
        if tokens["total"] == 0
        else f"{pe.fmt_int(tokens['total'])} ({pe.fmt_int(tokens['input'])} in, {pe.fmt_int(tokens['output'])} out)"
    )
    if r.get("embedding_calls"):
        token_line += ", plus one embedding call"
    col.markdown(
        f"**Response time:** {pe.fmt_seconds(r['seconds'])} s  \n"
        f"**Tokens:** {token_line}  \n"
        f"**Support check:** {r['support']['label']}"
    )
    if r["model"]:
        col.caption(f"Model: {r['model']}")
    if r["method"] in ("rules", "index") and r.get("retrieved"):
        label = "Top keyword scores" if r["method"] == "rules" else "Retrieved by the vector index (cosine similarity)"
        with col.expander(label):
            for p, s in r["retrieved"]:
                st.write(f"{p['title']}: {str(s).replace('.', ',')}")


st.title("Company Policy Assistant")
st.write(
    "Ask a question about company policy. Three methods answer it side by side, "
    "each showing its answer, the policy it relied on, response time, token use and "
    "whether the answer is backed by the policy database."
)

tab_ask, tab_compare = st.tabs(["Ask a question", "Comparison"])

with tab_ask:
    with open(ROOT / "data" / "eval_questions.csv", newline="", encoding="utf-8") as f:
        examples = [row["question"] for row in csv.DictReader(f)]
    choice = st.selectbox("Pick a test question or write your own below", ["Write my own"] + examples)
    question = st.text_input("Question", value="" if choice == "Write my own" else choice, max_chars=300)
    if st.button("Compare the three methods", type="primary") and question.strip():
        with st.spinner("Asking all three methods..."):
            st.session_state["last"] = (question.strip(), run_all(question.strip(), load_everything()))
    if "last" in st.session_state:
        q, results = st.session_state["last"]
        st.markdown(f"**Question:** {md(q)}")
        for col, r in zip(st.columns(3), results):
            show_result(col, r)
    st.caption(
        "Support check: an answer counts as unsupported if it cites no policy that exists in the database, "
        "or if it contains a number (days, amounts, limits) that appears neither in the policy it cites nor in the question."
    )

with tab_compare:
    comparison = ROOT / "comparison.md"
    if comparison.exists():
        st.markdown(md(comparison.read_text(encoding="utf-8")))
    if (RESULTS / "summary.csv").exists():
        st.markdown("**Table 1. Results over 26 test questions**")
        st.dataframe(pd.read_csv(RESULTS / "summary.csv", dtype=str), hide_index=True, width="stretch")
    if (RESULTS / "per_question.csv").exists():
        st.markdown("**Table 2. Policy identified per question**")
        st.dataframe(pd.read_csv(RESULTS / "per_question.csv", dtype=str), hide_index=True, width="stretch")
    if not (RESULTS / "summary.csv").exists():
        st.info("No evaluation results yet. Run: py evaluate.py")
