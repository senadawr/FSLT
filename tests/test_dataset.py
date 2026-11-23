import pytest

from ml.dataset import SampleMetadata, split_by_signer


def record(sample_id: str, signer_id: str | None) -> SampleMetadata:
    return SampleMetadata(sample_id, signer_id, None, "HELLO", None, "test", 4, "fsl-landmarks-v1", sample_id)


def test_split_by_signer_keeps_unknown_signers_explicit():
    result = split_by_signer([record("a", "s1"), record("b", "s2"), record("c", None)], {"s1"}, {"s2"})
    assert [item.sample_id for item in result["validation"]] == ["a"]
    assert [item.sample_id for item in result["test"]] == ["b"]
    assert [item.sample_id for item in result["unassigned"]] == ["c"]


def test_split_by_signer_rejects_overlap():
    with pytest.raises(ValueError, match="validation and test"):
        split_by_signer([], {"s1"}, {"s1"})
