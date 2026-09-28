# Time and cost log

Same format as the other Assignment 1 cases.

## Time

| Date | Task | Time (h) |
|---|---|---|
| 28.09.2026 | Case read-through, engine, website, evaluation script and Slack bot (built with Claude), Gemini key, vector index, evaluation run | |
| | **Total** | |

## Cost

### Token cost per question

Measured over 26 test questions (results/summary.csv): about 2.267 tokens per question for the LLM without
a vector index (all 98 policies in the prompt) and about 295 for the LLM with a vector index (three policies),
plus one embedding call per question. Rules-based search uses no tokens.
Prices are Google's paid-tier list prices per 1M tokens, as checked for the P-card case on 28.09.2026.

| Method | Tokens per question | Cost per question (gemini-3.1-flash-lite, $0,25 in / $1,50 out) | Cost per 1.000 questions |
|---|---|---|---|
| Rules-based search | 0 | $0 | $0 |
| LLM, no vector index | about 2.267 | about $0,0006 | about $0,60 |
| LLM with vector index | about 295 | about $0,00015 | about $0,15 |

Building the index was one batch of 98 embeddings, a one-off cost of well under $0,01 at paid-tier prices.
The key runs on the free tier, so the actual cost was $0. Newer Gemini models can produce internal thinking
tokens that are billed as output, so the real paid cost could be somewhat higher.
