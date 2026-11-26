from collections import Counter

from halludetect.dataset import load_dataset


def test_size_and_balance():
    ds = load_dataset()
    assert len(ds) == 100
    assert Counter(e.hallucinated for e in ds) == {True: 50, False: 50}
    assert Counter(e.split for e in ds) == {"dev": 50, "test": 50}


def test_ids_unique_and_pairs_share_split():
    ds = load_dataset()
    assert len({e.id for e in ds}) == 100
    by_base = {}
    for e in ds:
        by_base.setdefault(e.base_id, set()).add(e.split)
    assert all(len(s) == 1 for s in by_base.values())


def test_labels_consistent_with_claims():
    for e in load_dataset():
        unsupported = [c for c in e.claims if not c.supported]
        assert e.response == " ".join(c.text for c in e.claims)
        if e.hallucinated:
            assert len(unsupported) == 1 and e.hallucination_type
        else:
            assert not unsupported and e.hallucination_type is None
