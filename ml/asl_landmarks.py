from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import json

import numpy as np
from sklearn.model_selection import train_test_split


LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
EXPECTED_SHAPE = (21, 3)
FLAT_SHAPE = (63,)


@dataclass(frozen=True)
class ASLLandmarkImport:
    landmarks: np.ndarray
    labels: np.ndarray
    signer_ids: np.ndarray
    source: str
    license: str


def import_asl_alphabet(dataset_dir: Path) -> ASLLandmarkImport:
    root = dataset_dir / "landmarks"
    if not root.is_dir():
        root = dataset_dir
    landmarks = []
    labels = []
    for label_index, label in enumerate(LABELS):
        class_dir = root / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing ASL landmark class directory: {class_dir}")
        files = sorted(class_dir.glob("*.npy"))
        if not files:
            raise ValueError(f"No .npy landmark samples found for {label}")
        for file in files:
            landmark = np.load(file, allow_pickle=False)
            if landmark.shape == FLAT_SHAPE:
                landmark = landmark.reshape(EXPECTED_SHAPE)
            elif landmark.shape != EXPECTED_SHAPE:
                raise ValueError(
                    f"{file}: expected shape {EXPECTED_SHAPE} or {FLAT_SHAPE}, got {landmark.shape}"
                )
            if not np.isfinite(landmark).all():
                raise ValueError(f"{file}: contains non-finite landmark values")
            landmarks.append(landmark.astype(np.float32))
            labels.append(label_index)
    return ASLLandmarkImport(
        landmarks=np.asarray(landmarks, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int32),
        signer_ids=np.asarray(["unavailable"] * len(labels), dtype=object),
        source="Boris Graudt ASL Alphabet Hand Landmarks",
        license="CC0 Public Domain",
    )


def normalize_hand(landmarks: np.ndarray) -> np.ndarray:
    wrist = landmarks[:, 0:1, :]
    scale = np.linalg.norm(landmarks[:, 9, :] - landmarks[:, 0, :], axis=1)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return (landmarks - wrist) / scale[:, None, None]


def prepare_splits(
    dataset_dir: Path,
    output_dir: Path,
    seed: int = 2026,
) -> dict[str, int]:
    imported = import_asl_alphabet(dataset_dir)
    normalized = normalize_hand(imported.landmarks)
    train_indices, test_indices = train_test_split(
        np.arange(len(normalized)),
        test_size=0.2,
        random_state=seed,
        stratify=imported.labels,
    )
    train_indices, validation_indices = train_test_split(
        train_indices,
        test_size=0.125,
        random_state=seed,
        stratify=imported.labels[train_indices],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "train_landmarks.npy", normalized[train_indices])
    np.save(output_dir / "train_labels.npy", imported.labels[train_indices])
    np.save(output_dir / "validation_landmarks.npy", normalized[validation_indices])
    np.save(output_dir / "validation_labels.npy", imported.labels[validation_indices])
    np.save(output_dir / "test_landmarks.npy", normalized[test_indices])
    np.save(output_dir / "test_labels.npy", imported.labels[test_indices])
    np.save(output_dir / "train_signers.npy", imported.signer_ids[train_indices])
    np.save(output_dir / "validation_signers.npy", imported.signer_ids[validation_indices])
    np.save(output_dir / "test_signers.npy", imported.signer_ids[test_indices])
    (output_dir / "labels.json").write_text(
        json.dumps(list(LABELS)),
        encoding="utf-8",
    )
    (output_dir / "dataset.json").write_text(
        json.dumps({
            "source": imported.source,
            "license": imported.license,
            "sample_count": len(normalized),
            "class_count": len(LABELS),
            "signer_ids_available": False,
            "train_count": len(train_indices),
            "validation_count": len(validation_indices),
            "test_count": len(test_indices),
        }, indent=2),
        encoding="utf-8",
    )
    return {
        "sample_count": len(normalized),
        "train_count": len(train_indices),
        "validation_count": len(validation_indices),
        "test_count": len(test_indices),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/asl_alphabet"))
    args = parser.parse_args()
    print(json.dumps(prepare_splits(args.dataset_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
