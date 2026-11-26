# Problem: LLM answers that sound right but aren't supported

## The situation

A telecom support assistant answers customer questions from a help article. A customer asks, "Does unused data carry forward on my plan?" The article says it does not. The model replies, "Yes, unused data carries forward to the next day." The answer is fluent, confident and wrong. The customer acts on it, and the company now has a complaint, a refund, or a compliance issue.

Retrieval-augmented systems reduce this, but they don't remove it. Even with the right context in the prompt, models:

- change numbers, dates and names (`Rs 48 crore` stays, but `March 2020` becomes `March 2021`; `264 KB` becomes `512 KB`)
- swap facts *within* the context (the coach, Joseph Mathew, gets credit for striker Anjali Nair's goals)
- flip negations (`not approved for patients below 18` becomes `approved`)
- add plausible details that the context never states ("return pickup is always free")
- overgeneralize (`buildings with four or more flats` becomes `every rented property`)

## The task

Given `(question, context, response)`, decide:

1. **Answer level:** is the response fully supported by the context?
2. **Claim level:** which specific statements are unsupported?

"Supported" means supported *by the given context*. A claim that's true in the real world but absent from the context counts as unsupported, because the system can't vouch for it.

## Approaches to compare

| Approach | What it measures | Cost |
|---|---|---|
| Exact matching | String overlap between response and context | Free |
| Semantic similarity | Embedding similarity per sentence | Embedding calls |
| LLM judge | Holistic groundedness score (1 to 5) | 1 LLM call |
| Claim-level verification | Unsupported claims after decomposition | 1 + N LLM calls |

## What success looks like

- High recall on hallucinated answers *without* flagging faithful paraphrases (low false-positive rate).
- The ability to point at the bad sentence, so a product can highlight it, block it or regenerate it.
- A clear cost and latency picture, so a team can pick a method for its own budget.

## Experiment

1. Build 100 question-context-response examples with gold labels (50 faithful, 50 hallucinated).
2. Generate responses with an LLM (`experiments/generate_responses.py`, live mode).
3. Compare responses against the reference context with each method.
4. Evaluate with an LLM judge.
5. Identify unsupported claims with claim-level verification.
6. Tune thresholds on the dev split and report on the test split.
