import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.models import build_recurrent_classifier
from ml.features import build_selected_features, augment_temporal_spatial


def load(split_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    labels = np.load(split_dir / f"{split}_labels.npy")
    paths = np.load(split_dir / f"{split}_paths.npy")
    sequences = np.asarray([np.load(path) for path in paths])
    return sequences, labels


def train(artifact_dir: Path, architecture: str, epochs: int, batch_size: int) -> None:
    x_train, y_train = load(artifact_dir, "train")
    x_test, y_test = load(artifact_dir, "test")
    x_train = build_selected_features(x_train)
    x_test = build_selected_features(x_test)
    rng = np.random.default_rng(2026)
    x_train_augmented = augment_temporal_spatial(x_train, rng)
    x_train = np.concatenate((x_train, x_train_augmented), axis=0)
    y_train = np.concatenate((y_train, y_train), axis=0)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(105),
        y=y_train,
    )
    class_weight = {index: float(weight) for index, weight in enumerate(class_weights)}
    model = build_recurrent_classifier(
        input_shape=(x_train.shape[1], x_train.shape[2]),
        class_count=105,
        architecture=architecture,
        dropout=0.35,
    )
    checkpoint = artifact_dir / f"{architecture}_best.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=2),
        keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_accuracy", save_best_only=True),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        class_weight=class_weight,
    )
    model.save(artifact_dir / f"{architecture}_final.keras")
    np.save(artifact_dir / f"{architecture}_history.npy", history.history)
    predictions = np.argmax(model.predict(x_test, verbose=0), axis=1)
    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(105),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=np.arange(105))
    np.savetxt(artifact_dir / f"{architecture}_confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    (artifact_dir / f"{architecture}_evaluation.json").write_text(
        json.dumps({
            "architecture": architecture,
            "feature_dimension": int(x_train.shape[2]),
            "test_clip_count": int(len(y_test)),
            "unique_signers_available": False,
            "split_claim": "publisher train/test split; clip-level only",
            "accuracy": float(report["accuracy"]),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "per_class": {str(key): value for key, value in report.items() if key.isdigit()},
        }, indent=2),
        encoding="utf-8",
    )
    labels = []
    with (artifact_dir.parent.parent.parent / "data" / "fsl-105" / "labels.csv").open(
        newline="", encoding="utf-8-sig"
    ) as file:
        labels = [row["label"] for row in csv.DictReader(file)]
    (artifact_dir / f"{architecture}.labels.json").write_text(
        json.dumps(labels),
        encoding="utf-8",
    )
    print(f"Saved best model to {checkpoint}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/isolated_sign/fsl105"))
    parser.add_argument("--architecture", choices=("gru", "lstm"), default="gru")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train(args.artifacts, args.architecture, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
