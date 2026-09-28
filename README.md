# Company Policy Assistant

Case 2 of Assignment 1, AI in Accounting and Business, SDU, autumn 2026.

Answers questions about 98 company policies with three methods and compares them:
rules-based keyword search, an LLM with no vector index (all policies in the prompt),
and an LLM with a vector index (only the three closest policies in the prompt).
A Slack bot uses the vector-index method.

Live site: (add link)

## Files

- `policy_engine.py` shared logic for all three methods and the support check
- `app.py` Streamlit website
- `build_index.py` builds `data/policy_index.json` (Gemini embeddings)
- `evaluate.py` runs the 26 test questions in `data/eval_questions.csv` and writes `results/`
- `slack_bot.py` Slack bot (runs locally, Socket Mode)
- `slack_manifest.yml` Slack app settings
- `LOG.md` time and cost

## Run locally

```
py -m pip install -r requirements.txt
copy .streamlit\secrets.toml.example .streamlit\secrets.toml   (then paste your keys)
py build_index.py
py evaluate.py
py -m streamlit run app.py
py slack_bot.py
```

No API key is stored in this repository. The live site reads it from Streamlit Secrets.
