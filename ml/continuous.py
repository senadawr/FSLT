from dataclasses import dataclass


@dataclass(frozen=True)
class ContinuousAnnotation:
    video_id: str
    signer_id: str | None
    session_id: str | None
    glosses: tuple[str, ...]
    translation_fil: str | None
    translation_en: str | None
    start_ms: int | None = None
    end_ms: int | None = None
    non_manual_markers: tuple[str, ...] = ()


def build_ctc_encoder(input_shape: tuple[int | None, int], token_count: int):
    """Build the untrained continuous gloss model; weights require real CTC data."""
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("Install the ML training extras before building a model") from exc
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=True))(inputs)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(token_count + 1, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="bigru_ctc_gloss_encoder")


class GlossToTextTranslator:
    def __init__(self) -> None:
        self.available = False

    def translate(self, glosses: list[str], language: str) -> str:
        if not self.available:
            raise RuntimeError("Gloss-to-text translation model is not available")
        raise NotImplementedError("Load a validated parallel FSL translation model")
