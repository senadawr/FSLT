import numpy as np
import pytest

from ml.feature_schema import FEATURE_DIMENSION, LANDMARK_COUNT, FeatureFrame, validate_frame


def test_valid_frame_uses_fixed_contract():
    validate_frame(FeatureFrame(
        "fsl-landmarks-v1", 1, np.zeros(FEATURE_DIMENSION), np.ones(LANDMARK_COUNT)
    ))


def test_invalid_shape_is_rejected():
    with pytest.raises(ValueError, match="feature values"):
        validate_frame(FeatureFrame(
            "fsl-landmarks-v1", 1, np.zeros(3), np.ones(LANDMARK_COUNT)
        ))
