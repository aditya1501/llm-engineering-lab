"""Binary classification metrics implemented without sklearn, so they are easy to read."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class BinaryMetrics:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def accuracy(self) -> float:
        n = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / n if n else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(precision=self.precision, recall=self.recall, f1=self.f1, accuracy=self.accuracy)
        return d


def confusion(y_true: list[bool], y_pred: list[bool]) -> BinaryMetrics:
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    tn = sum((not t) and (not p) for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    return BinaryMetrics(tp, fp, tn, fn)


def roc_auc(y_true: list[bool], scores: list[float]) -> float:
    """Probability a random positive scores higher than a random negative (ties count 0.5)."""
    pos = [s for s, t in zip(scores, y_true) if t]
    neg = [s for s, t in zip(scores, y_true) if not t]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def best_threshold(y_true: list[bool], scores: list[float]) -> float:
    """Threshold maximising F1 on the given data (use on a dev split only)."""
    candidates = sorted(set(scores))
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        f1 = confusion(y_true, [s >= t for s in scores]).f1
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t
