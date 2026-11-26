# Adding a challenge

1. Open an issue describing the problem, the baseline, the improved approach and the metric.
2. Create `challenges/NN-name/` with:

```
NN-name/
├── README.md          # architecture, results table (copied from results/), trade-offs, lessons
├── problem.md         # the real-world problem and what success looks like
├── implementation/    # an importable package
├── experiments/       # dataset builder + benchmark runner
├── results/           # written by the runner only
├── tests/
└── Dockerfile
```

3. Register the package in the root `pyproject.toml` (`[tool.setuptools.package-dir]`, `packages`, and the pytest `pythonpath`).
4. Write a mock responder so the benchmark runs offline.
5. Split the dataset into dev and test. Tune on dev.
6. Add the benchmark to CI and commit `results/`.
7. Add the challenge to the table in the main README.

Checklist for the README:

- Problem statement and why it matters in production
- Baseline and improved implementation
- Architecture diagram
- Metrics and how they are computed
- Results table, labelled mock or live
- Trade-offs (quality vs. cost vs. latency vs. complexity)
- Lessons learned, separated into "observed in our runs" and "design reasoning"
