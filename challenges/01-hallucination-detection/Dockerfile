# Runs Challenge 01 tests and the mock benchmark in a clean container.
# Build from the repo root:  docker build -f challenges/01-hallucination-detection/Dockerfile -t llm-lab-01 .
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY shared ./shared
COPY challenges/01-hallucination-detection ./challenges/01-hallucination-detection
COPY datasets/hallucination_qa ./datasets/hallucination_qa
RUN pip install --no-cache-dir -e ".[dev,api]"
CMD ["sh", "-c", "pytest challenges/01-hallucination-detection && python challenges/01-hallucination-detection/experiments/run_benchmark.py --out /tmp/results"]
