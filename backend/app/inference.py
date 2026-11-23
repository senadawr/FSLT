from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ml.features import build_selected_features
except ImportError:
    build_selected_features = None

from .config import Settings
from .schemas import InferenceResponse, LandmarkFrame, ModelInfo


@dataclass
class Prediction:
    label: str | None
    confidence: float


class InferenceService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.frames: deque[LandmarkFrame] = deque(maxlen=settings.max_sequence_length)
        self.model: Any | None = None
        self.labels: list[str] = []
        self._load_model_once()

    def _load_model_once(self) -> None:
        if not self.settings.model_path:
            return
        model_path = Path(self.settings.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Configured model does not exist: {model_path}")
        try:
            from tensorflow import keras
        except ImportError as exc:
            raise RuntimeError("TensorFlow is required to load a configured model") from exc
        self.model = keras.models.load_model(model_path)
        labels_path = model_path.with_suffix(".labels.json")
        if labels_path.is_file():
            self.labels = json.loads(labels_path.read_text(encoding="utf-8"))
        else:
            generic_labels = model_path.parent / f"{self.settings.model_name}.labels.json"
            architecture_labels = model_path.parent / f"{model_path.stem.removesuffix('_best').removesuffix('_final')}.labels.json"
            for candidate in (generic_labels, architecture_labels):
                if candidate.is_file():
                    self.labels = json.loads(candidate.read_text(encoding="utf-8"))
                    break

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            model_name=self.settings.model_name,
            model_version=self.settings.model_version,
            supported_signs=self.labels,
            training_status="trained" if self.model is not None else "unavailable",
            recognition_mode="isolated",
            feature_schema_version=self.settings.feature_schema_version,
        )

    def add_frame(self, frame: LandmarkFrame) -> InferenceResponse:
        if frame.schema_version != self.settings.feature_schema_version:
            return self._response(frame, "unavailable", error="Unsupported feature schema version")
        self.frames.append(frame)
        if self.model is None:
            return self._response(frame, "unavailable", error="No trained model is configured")
        prediction = self._predict()
        if prediction.label is None or prediction.confidence < self.settings.confidence_threshold:
            return self._response(frame, "uncertain", prediction.confidence)
        return self._response(frame, "recognized", prediction.confidence, prediction.label)

    def _predict(self) -> Prediction:
        values = np.asarray([frame.features for frame in self.frames], dtype=np.float32)
        input_shape = getattr(self.model, "input_shape", None)
        feature_dimension = input_shape[-1] if input_shape and len(input_shape) >= 3 else None
        if feature_dimension == 770 and build_selected_features is not None:
            padded = np.zeros((64, values.shape[1]), dtype=np.float32)
            padded[-len(values):] = values
            values = build_selected_features(padded[None, ...])[0]
        output = np.asarray(self.model.predict(values[None, ...], verbose=0))[0]
        index = int(np.argmax(output))
        confidence = float(output[index])
        label = self.labels[index] if index < len(self.labels) else None
        return Prediction(label, confidence)

    def _response(
        self,
        frame: LandmarkFrame,
        status: str,
        confidence: float = 0.0,
        prediction: str | None = None,
        error: str | None = None,
    ) -> InferenceResponse:
        return InferenceResponse(
            prediction=prediction,
            confidence=confidence,
            prediction_status=status,
            recognized_glosses=[prediction] if prediction else [],
            sequence_complete=False,
            model_version=self.settings.model_version,
            timestamp_ms=frame.timestamp_ms,
            error=error,
        )
