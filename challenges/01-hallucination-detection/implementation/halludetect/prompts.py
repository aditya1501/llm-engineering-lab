"""Prompt templates. Kept in one file so they can be versioned and diffed.

All prompts wrap inputs in XML-style tags. That makes them easy to parse for
the mock responder and makes prompt injection inside the answer text a bit
harder (see Challenge 05 for the real treatment).
"""

PROMPT_VERSION = "v1"

JUDGE_SYSTEM = (
    "You are a strict fact-checking judge. You decide whether an ANSWER is fully "
    "supported by the CONTEXT. Use only the CONTEXT, never outside knowledge. "
    "A claim that is true in the real world but absent from the CONTEXT is unsupported."
)

JUDGE_TEMPLATE = """Rate how grounded the answer is in the context.

<context>
{context}
</context>

<question>
{question}
</question>

<answer>
{answer}
</answer>

Score groundedness from 1 to 5:
5 = every statement is directly supported by the context
4 = supported, with minor harmless paraphrase or inference
3 = partly supported; some details cannot be verified from the context
2 = contains at least one unsupported or altered detail
1 = mostly unsupported or contradicts the context

Respond with JSON only: {{"groundedness": <1-5>, "reason": "<one sentence>"}}"""

DECOMPOSE_SYSTEM = "You split answers into atomic factual claims."

DECOMPOSE_TEMPLATE = """Split the answer into a list of atomic, self-contained factual claims.
Drop filler words such as "Yes," or "No," but keep the factual content and any negation.

<answer>
{answer}
</answer>

Respond with JSON only: {{"claims": ["<claim 1>", "<claim 2>", ...]}}"""

VERIFY_SYSTEM = JUDGE_SYSTEM

VERIFY_TEMPLATE = """Decide whether the claim is supported by the context.

<context>
{context}
</context>

<claim>
{claim}
</claim>

Labels:
- supported: the context states or directly implies the claim
- contradicted: the context states something incompatible with the claim
- unsupported: the context does not contain enough information

Respond with JSON only: {{"verdict": "supported|contradicted|unsupported", "evidence": "<quote from context or empty>"}}"""
