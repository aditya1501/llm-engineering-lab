.PHONY: install test lint bench-01 dataset-01 api-01 docker-01

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

dataset-01:
	python challenges/01-hallucination-detection/experiments/build_dataset.py

bench-01:
	python challenges/01-hallucination-detection/experiments/run_benchmark.py

api-01:
	uvicorn halludetect.api:app --reload

docker-01:
	docker build -f challenges/01-hallucination-detection/Dockerfile -t llm-lab-01 .
	docker run --rm llm-lab-01
