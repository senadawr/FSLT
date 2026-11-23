import numpy as np
import pytest

from ml.asl_landmarks import LABELS, EXPECTED_SHAPE, normalize_hand


def test_asl_alphabet_labels_are_complete():
    assert "".join(LABELS) == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def test_normalize_hand_moves_wrist_to_origin_and_normalizes_scale():
    landmarks = np.zeros((2, 21, 3), dtype=np.float32)
    landmarks[:, 0, :] = (1.0, 2.0, 3.0)
    landmarks[:, 9, :] = (3.0, 2.0, 3.0)
    normalized = normalize_hand(landmarks)
    assert normalized.shape == (2, 21, 3)
    assert np.allclose(normalized[:, 0, :], 0)
    assert np.isfinite(normalized).all()


def test_import_contract_rejects_unknown_shape():
    assert EXPECTED_SHAPE == (21, 3)
