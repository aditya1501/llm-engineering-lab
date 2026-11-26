# Contributing

Thanks for helping. This repo is meant to be a place where engineers can learn by changing things and measuring what happens, so contributions that add measurements are the most valuable kind.

## Ground rules

1. **Never fabricate results.** Every number in a README must come from a file in `results/` that the code wrote. If you didn't run it, don't write it down.
2. **Label every run.** Mock results use tags that start with `mock`. Live results name the provider and model (`live-anthropic-claude-opus-5-5`, for example).
3. **Offline first.** Every new code path needs a mock so `pytest` runs without network or keys.
4. **Tune on dev, report on test.** Don't choose thresholds, prompts or heuristics by looking at test-split numbers.
5. **Keep challenges independent.** A challenge may import from `shared/llm_lab`, never from another challenge.

## Development setup

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## Typical contributions

- **New detector / method:** add it under `implementation/`, register it in `factory.py`, add tests, re-run the benchmark, commit `results/`, and paste the before/after table in the PR.
- **Dataset additions:** edit `experiments/build_dataset.py` (the dataset is generated from it), bump the dataset version if existing examples change, and explain how the new examples were written.
- **Live results:** run with `--provider` and `--tag`, commit the generated `results/<tag>.*` files and, if you can, the replay cassette. Say what it cost.
- **New challenge:** open an issue first. Follow [docs/adding-a-challenge.md](docs/adding-a-challenge.md).

## Pull requests

- Keep PRs focused on one change.
- CI must pass. It runs lint, tests, and checks that committed mock results match a fresh run.
- Fill in the PR template, including whether numbers are mock-run or live.
