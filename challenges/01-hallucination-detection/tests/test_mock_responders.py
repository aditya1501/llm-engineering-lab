import json

from halludetect.mock_responders import analyse, decompose_responder, verify_responder
from halludetect.prompts import DECOMPOSE_TEMPLATE, VERIFY_TEMPLATE

CTX = "It is not approved for patients under 18. The Kochi Blasters won the 2019 title."


def test_negation_mismatch_detected():
    assert analyse("It is approved for patients under 18.", CTX)["negation_mismatch"]
    assert not analyse("It is not approved for patients under 18.", CTX)["negation_mismatch"]


def test_won_is_not_a_negation():
    assert not analyse("The Kochi Blasters won the 2019 title.", CTX)["negation_mismatch"]


def test_new_number_is_unsupported():
    out = json.loads(verify_responder(VERIFY_TEMPLATE.format(context=CTX, claim="The Kochi Blasters won in 2020."), None))
    assert out["verdict"] == "unsupported"


def test_decompose_strips_particles_and_keeps_dr():
    out = json.loads(decompose_responder(DECOMPOSE_TEMPLATE.format(answer="No, it was led by Dr. Levi. She left."), None))
    assert out["claims"] == ["It was led by Dr. Levi.", "She left."]
