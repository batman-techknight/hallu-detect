from unittest.mock import MagicMock

import pytest

from app.core.semantic_entropy import SemanticEntropyEstimator


class FakeNLI:
    """Two texts entail each other iff they share the same first word."""

    def entailment_prob(self, premise: str, hypothesis: str) -> float:
        return 1.0 if premise.split()[0] == hypothesis.split()[0] else 0.0


@pytest.fixture
def estimator():
    return SemanticEntropyEstimator(nli=FakeNLI(), http_client=MagicMock())


def test_all_identical_samples_have_zero_entropy(estimator):
    samples = ["Paris is the capital"] * 5
    clusters = estimator._cluster_by_entailment(samples)
    assert len(set(clusters)) == 1
    assert estimator._normalized_entropy(clusters) == 0.0


def test_fully_split_samples_have_max_entropy(estimator):
    samples = ["Paris is the capital", "London is bigger", "Tokyo has trains", "Rome is old"]
    clusters = estimator._cluster_by_entailment(samples)
    assert len(set(clusters)) == len(samples)
    assert estimator._normalized_entropy(clusters) == pytest.approx(1.0)


def test_partial_agreement_gives_intermediate_entropy(estimator):
    samples = ["Paris is nice", "Paris is lovely", "London is bigger"]
    clusters = estimator._cluster_by_entailment(samples)
    entropy = estimator._normalized_entropy(clusters)
    assert 0.0 < entropy < 1.0


def test_single_sample_has_zero_entropy(estimator):
    assert estimator._normalized_entropy([0]) == 0.0
