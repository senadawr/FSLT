from typing import Literal
from pydantic import BaseModel, Field


FEATURE_DIMENSION = 1659
LANDMARK_COUNT = 553


class LandmarkFrame(BaseModel):
    schema_version: str = Field(min_length=1)
    timestamp_ms: int = Field(ge=0)
    features: list[float] = Field(min_length=FEATURE_DIMENSION, max_length=1665)
    presence: list[float] = Field(min_length=LANDMARK_COUNT, max_length=LANDMARK_COUNT)


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    supported_signs: list[str]
    training_status: Literal["trained", "untrained", "unavailable"]
    recognition_mode: Literal["isolated", "continuous"]
    feature_schema_version: str


class DatasetInfo(BaseModel):
    datasets: list[dict[str, str | int]]
    note: str


class InferenceResponse(BaseModel):
    prediction: str | None
    confidence: float
    prediction_status: Literal["unavailable", "uncertain", "recognized"]
    recognized_glosses: list[str]
    sequence_complete: bool
    model_version: str
    timestamp_ms: int
    error: str | None = None


class ASLHandFrame(BaseModel):
    schema_version: str = "asl-hand-v1"
    timestamp_ms: int = Field(ge=0)
    hand: list[float] = Field(min_length=63, max_length=63)


class PoseTemplate(BaseModel):
    label: str = Field(min_length=1, max_length=100)
    landmarks: list[list[float]] = Field(min_length=21, max_length=21)
    samples: list[list[list[float]]] | None = None
    saved_at: str = Field(min_length=1)


class PoseTemplatesPayload(BaseModel):
    templates: list[PoseTemplate] = Field(max_length=1000)
