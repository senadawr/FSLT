def build_recurrent_classifier(
    input_shape: tuple[int | None, int],
    class_count: int,
    architecture: str = "gru",
    dropout: float = 0.3,
):
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("Install the ML training extras before building a model") from exc
    if architecture not in {"gru", "lstm"}:
        raise ValueError("architecture must be 'gru' or 'lstm'")
    recurrent = keras.layers.GRU if architecture == "gru" else keras.layers.LSTM
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Masking(mask_value=0.0)(inputs)
    x = recurrent(128, return_sequences=True)(x)
    x = keras.layers.Dropout(dropout)(x)
    x = recurrent(64)(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(class_count, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name=f"{architecture}_isolated_classifier")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
