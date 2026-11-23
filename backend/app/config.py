from functools import lru_cache
import os
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_path: str | None
    model_name: str
    model_version: str
    feature_schema_version: str
    confidence_threshold: float
    max_sequence_length: int


@lru_cache
def get_settings() -> Settings:
    threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
    if not 0 < threshold <= 1:
        raise ValueError("CONFIDENCE_THRESHOLD must be greater than 0 and at most 1")
    configured_path = os.getenv("MODEL_PATH")
    if configured_path:
        model_path = configured_path
    else:
        project_root = Path(__file__).resolve().parents[2]
        candidate = project_root / "artifacts" / "isolated_sign" / "fsl105" / "gru_best.keras"
        model_path = str(candidate) if candidate.is_file() else None
    return Settings(
        model_path=model_path,
        model_name=os.getenv("MODEL_NAME", "improved-gru-isolated-sign"),
        model_version=os.getenv("MODEL_VERSION", "experimental-gru-v1"),
        feature_schema_version=os.getenv("FEATURE_SCHEMA_VERSION", "fsl-landmarks-v1"),
        confidence_threshold=threshold,
        max_sequence_length=int(os.getenv("MAX_SEQUENCE_LENGTH", "64")),
    )
