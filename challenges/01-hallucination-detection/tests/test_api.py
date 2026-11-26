import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from halludetect.api import app  # noqa: E402

client = TestClient(app)
CTX = "The Vayu EV has a 40 kWh battery and a certified range of 480 km."


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_detect_flags_bad_claim():
    r = client.post("/detect", json={"question": "Range?", "context": CTX, "response": "Its range is 520 km."})
    body = r.json()
    assert r.status_code == 200 and body["hallucinated"] is True
    assert body["claims"][0]["verdict"] == "unsupported"


def test_unknown_method():
    r = client.post("/detect", json={"question": "q", "context": CTX, "response": "x", "method": "vibes"})
    assert r.status_code == 400
