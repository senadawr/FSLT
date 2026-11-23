from pathlib import Path
import argparse
import json
import sys

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load(artifact_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.load(artifact_dir / f"{split}_landmarks.npy"),
        np.load(artifact_dir / f"{split}_labels.npy"),
    )


def train(artifact_dir: Path, architecture: str, epochs: int, batch_size: int) -> None:
    x_train, y_train = load(artifact_dir, "train")
    x_validation, y_validation = load(artifact_dir, "validation")
    x_test, y_test = load(artifact_dir, "test")
    inputs = keras.Input(shape=(21, 3))
    if architecture == "gru":
        x = keras.layers.GRU(128, return_sequences=True)(inputs)
        x = keras.layers.Dropout(0.25)(x)
        x = keras.layers.GRU(64)(x)
    else:
        x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(inputs)
        x = keras.layers.Dropout(0.25)(x)
        x = keras.layers.LSTM(64)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.25)(x)
    outputs = keras.layers.Dense(26, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name=f"{architecture}_asl_alphabet")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    checkpoint = artifact_dir / f"{architecture}_best.keras"
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=8,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_accuracy",
                factor=0.5,
                patience=3,
            ),
            keras.callbacks.ModelCheckpoint(
                str(checkpoint),
                monitor="val_accuracy",
                save_best_only=True,
            ),
        ],
        shuffle=True,
    )
    model.save(artifact_dir / f"{architecture}_final.keras")
    np.save(artifact_dir / f"{architecture}_history.npy", history.history)
    predictions = np.argmax(model.predict(x_test, verbose=0), axis=1)
    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(26),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=np.arange(26))
    np.savetxt(artifact_dir / f"{architecture}_confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    (artifact_dir / f"{architecture}_evaluation.json").write_text(
        json.dumps({
            "architecture": architecture,
            "test_accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "test_count": len(y_test),
            "unique_signers_available": False,
            "dataset": "ASL Alphabet Hand Landmarks",
            "license": "CC0 Public Domain",
        }, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "test_accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "checkpoint": str(checkpoint),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/asl_alphabet"))
    parser.add_argument("--architecture", choices=("gru", "lstm"), default="gru")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    train(args.artifacts, args.architecture, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
