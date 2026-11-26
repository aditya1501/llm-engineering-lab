# Datasets

| Dataset | Version | Size | Used by | Built by |
|---|---|---|---|---|
| `hallucination_qa` | v1 | 100 examples (50 faithful, 50 hallucinated; 50 dev / 50 test) | Challenge 01 | `challenges/01-hallucination-detection/experiments/build_dataset.py` |

## hallucination_qa/v1.jsonl

One JSON object per line:

| Field | Type | Meaning |
|---|---|---|
| `id` | str | `b001-f` (faithful) or `b001-h` (hallucinated) |
| `base_id` | str | The faithful and hallucinated versions share a base item |
| `split` | `dev` \| `test` | Split by base item, so a pair never straddles splits |
| `domain` | str | Topic area (finance, medical, software, ...) |
| `question`, `context` | str | The question and the only allowed source of truth |
| `response` | str | The answer to evaluate (concatenation of `claims`) |
| `claims` | list | `{text, supported}` gold claim labels |
| `hallucinated` | bool | True if any claim is unsupported |
| `hallucination_type` | str \| null | `entity_swap`, `number_change`, `date_change`, `in_context_swap`, `negation`, `unsupported_addition`, `overgeneralization` |

How it was made: 50 fictional contexts were written by hand so a model cannot answer from memory. Each has two faithful claims written as paraphrases, plus one hand-written perturbation that makes the answer unsupported. The responses are curated, not sampled from a model, because curated responses come with exact gold labels. The challenge's `generate_responses.py` covers model-generated answers.

Known limitations: small (50 test examples), English only, two-to-three-sentence answers, one unsupported claim per hallucinated answer, and some type buckets have only one to four test examples.
