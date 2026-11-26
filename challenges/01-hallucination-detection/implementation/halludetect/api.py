"""Optional FastAPI service exposing the detectors.

Run:  uvicorn halludetect.api:app --reload
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from halludetect.factory import METHODS, build_detectors
from halludetect.types import Example

app = FastAPI(title="Hallucination detection", version="0.1.0")
_detectors = build_detectors(
    provider=os.environ.get("LLM_PROVIDER", "mock"),
    embedder=os.environ.get("EMBED_PROVIDER", "hashing"),
)
# Default thresholds; override with values calibrated by run_benchmark.py.
THRESHOLDS = {"exact_match": 0.2, "semantic_similarity": 0.5, "llm_judge": 0.5, "claim_verification": 1.0}


class DetectRequest(BaseModel):
    question: str
    context: str = Field(min_length=1)
    response: str = Field(min_length=1)
    method: str = "claim_verification"


class ClaimOut(BaseModel):
    claim: str
    verdict: str
    score: float
    reason: str


class DetectResponse(BaseModel):
    method: str
    score: float
    hallucinated: bool
    threshold: float
    claims: list[ClaimOut] | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "methods": list(METHODS)}


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest) -> DetectResponse:
    if req.method not in _detectors:
        raise HTTPException(400, f"method must be one of {list(METHODS)}")
    ex = Example(id="api", question=req.question, context=req.context, response=req.response)
    d = _detectors[req.method].detect(ex)
    t = THRESHOLDS[req.method]
    claims = [ClaimOut(**vars(c)) for c in d.claims] if d.claims is not None else None
    return DetectResponse(method=req.method, score=d.score, hallucinated=d.score >= t, threshold=t, claims=claims)
