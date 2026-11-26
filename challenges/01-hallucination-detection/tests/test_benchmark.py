import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "run_benchmark.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_benchmark", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_benchmark_end_to_end(tmp_path):
    report = _load().main(["--out", str(tmp_path)])
    assert report["run"]["mode"] == "mock"
    assert set(report["methods"]) == {"exact_match", "semantic_similarity", "llm_judge", "claim_verification"}
    saved = json.loads((tmp_path / "mock-run.json").read_text())
    assert saved["methods"].keys() == report["methods"].keys()
    assert "Mock run" in (tmp_path / "mock-run.md").read_text()


def test_benchmark_is_deterministic(tmp_path):
    mod = _load()
    a = mod.main(["--out", str(tmp_path / "a")])
    b = mod.main(["--out", str(tmp_path / "b")])
    for name in a["methods"]:
        assert a["methods"][name]["test"] == b["methods"][name]["test"]


def test_mock_cannot_be_labelled_live(tmp_path):
    import pytest

    with pytest.raises(SystemExit):
        _load().main(["--out", str(tmp_path), "--tag", "live-gpt"])
