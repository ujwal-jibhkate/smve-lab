"""Hand-computed checks for the ranking metrics."""

import math

import pytest

from smve_lab.metrics import (
    ap_at_k,
    dcg,
    evaluate_run,
    first_relevant_rank,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# Relevant docs are "a" and "c"; the system ranks: x, a, y, c, z
RANKED = ["x", "a", "y", "c", "z"]
QREL = {"a": 1, "c": 1}


def test_dcg():
    assert dcg([1, 0, 1]) == pytest.approx(1 / math.log2(2) + 1 / math.log2(4))
    assert dcg([]) == 0.0


def test_ndcg():
    # DCG = 1/log2(3) + 1/log2(5); IDCG = 1/log2(2) + 1/log2(3)
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(RANKED, QREL, 5) == pytest.approx(expected)
    assert ndcg_at_k(["a", "c"], QREL, 2) == pytest.approx(1.0)
    assert ndcg_at_k(RANKED, QREL, 1) == 0.0


def test_ndcg_ideal_is_truncated_at_k():
    # 3 relevant docs but k=1: a perfect top-1 must still score 1.0.
    assert ndcg_at_k(["a"], {"a": 1, "b": 1, "c": 1}, 1) == pytest.approx(1.0)


def test_precision_recall():
    assert precision_at_k(RANKED, QREL, 2) == pytest.approx(1 / 2)
    assert precision_at_k(RANKED, QREL, 5) == pytest.approx(2 / 5)
    assert recall_at_k(RANKED, QREL, 2) == pytest.approx(1 / 2)
    assert recall_at_k(RANKED, QREL, 5) == pytest.approx(1.0)


def test_precision_divides_by_k_even_if_run_is_short():
    assert precision_at_k(["a"], QREL, 10) == pytest.approx(0.1)


def test_mrr_and_ap():
    assert mrr_at_k(RANKED, QREL, 5) == pytest.approx(1 / 2)
    assert mrr_at_k(RANKED, QREL, 1) == 0.0
    # AP = (P@2 + P@4) / 2 = (1/2 + 2/4) / 2
    assert ap_at_k(RANKED, QREL, 5) == pytest.approx(0.5)


def test_first_relevant_rank():
    assert first_relevant_rank(RANKED, QREL) == 2
    assert first_relevant_rank(["x", "y"], QREL) is None


def test_evaluate_run_scores_missing_queries_as_zero():
    qrels = {"q1": QREL, "q2": {"b": 1}}
    df = evaluate_run({"q1": RANKED}, qrels, ks=[5])
    assert list(df.index) == ["q1", "q2"]
    assert df.loc["q2", "ndcg@5"] == 0.0
    assert df.loc["q1", "recall@5"] == pytest.approx(1.0)
