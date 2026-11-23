from dataclasses import dataclass
import numpy as np

FEATURE_SCHEMA_VERSION = "fsl-landmarks-v1"
LANDMARK_COUNT = 553
FEATURE_DIMENSION = LANDMARK_COUNT * 3


@dataclass(frozen=True)
class FeatureFrame:
    schema_version: str
    timestamp_ms: int
    features: np.ndarray
    presence: np.ndarray


def validate_frame(frame: FeatureFrame) -> None:
    if frame.schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported feature schema: {frame.schema_version}")
    if frame.features.shape != (FEATURE_DIMENSION,):
        raise ValueError(f"Expected {FEATURE_DIMENSION} feature values")
    if frame.presence.shape != (LANDMARK_COUNT,):
        raise ValueError(f"Expected {LANDMARK_COUNT} presence values")
    if not np.isfinite(frame.features).all() or not np.isfinite(frame.presence).all():
        raise ValueError("Feature values must be finite")
