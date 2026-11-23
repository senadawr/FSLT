import numpy as np

from ml.features import FEATURE_DIMENSION, build_selected_features


def test_feature_builder_normalizes_shape_and_presence():
    raw = np.zeros((2, 64, 1665), dtype=np.float32)
    pose = raw.reshape(2, 64, -1, 3)
    pose[:, :, 478 + 11, :] = 1
    pose[:, :, 478 + 12, :] = 1
    features = build_selected_features(raw)
    assert features.shape == (2, 64, FEATURE_DIMENSION)
    assert np.isfinite(features).all()
    assert features[:, :, 225:300].shape == (2, 64, 75)
