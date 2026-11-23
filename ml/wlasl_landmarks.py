from dataclasses import dataclass
from pathlib import Path
import argparse
import json
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np


MAX_SUBSET_GLOSSES = 2000


@dataclass(frozen=True)
class WLASLImport:
    landmarks: dict[str, np.ndarray]
    labels: dict[str, int]
    train_ids: set[str]
    validation_ids: set[str]
    test_ids: set[str]
    signer_ids: dict[str, int]


def load_metadata(metadata_path: Path) -> WLASLImport:
    glosses = json.loads(metadata_path.read_text(encoding="utf-8"))
    labels = {entry["gloss"]: index for index, entry in enumerate(glosses)}
    landmarks: dict[str, np.ndarray] = {}
    train_ids: set[str] = set()
    validation_ids: set[str] = set()
    test_ids: set[str] = set()
    signer_ids: dict[str, int] = {}
    for entry in glosses:
        for instance in entry["instances"]:
            video_id = str(instance["video_id"])
            if instance["split"] == "train":
                train_ids.add(video_id)
            elif instance["split"] == "val":
                validation_ids.add(video_id)
            elif instance["split"] == "test":
                test_ids.add(video_id)
            signer_ids[video_id] = int(instance["signer_id"])
    return WLASLImport(
        landmarks=landmarks,
        labels=labels,
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
        signer_ids=signer_ids,
    )


def extract_hand(png_path: Path) -> np.ndarray:
    image = cv2.imread(str(png_path))
    if image is None:
        return np.zeros((42, 3), dtype=np.float32)
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.25,
    )
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        left = np.zeros((21, 3), dtype=np.float32)
        right = np.zeros((21, 3), dtype=np.float32)
        for index, landmarks in enumerate(result.multi_hand_landmarks or []):
            handedness = result.multi_handedness or []
            label = handedness[index].classification[0].label.lower() if index < len(handedness) else ""
            values = np.asarray(
                [[point.x, point.y, point.z] for point in landmarks.landmark],
                dtype=np.float32,
            )
            if label == "left":
                left = values
            elif label == "right":
                right = values
        return np.concatenate((left, right), axis=0)
    finally:
        hands.close()


def normalize_hand_pair(hand_pair: np.ndarray) -> np.ndarray:
    result = hand_pair.copy()
    for hand_index in (0, 1):
        hand = result[hand_index * 21:(hand_index + 1) * 21]
        if np.linalg.norm(hand) <= 1e-6:
            continue
        wrist = hand[0:1]
        scale = np.linalg.norm(hand[9] - hand[0])
        if scale > 1e-6:
            result[hand_index * 21:(hand_index + 1) * 21] = (hand - wrist) / scale
    return result


def prepare(
    dataset_dir: Path,
    output_dir: Path,
    limit_glosses: int | None = None,
) -> dict[str, int]:
    metadata_path = dataset_dir / "WLASL" / "info.json"
    imported = load_metadata(metadata_path)
    frame_root = dataset_dir / "WLASL" / "frames"
    selected_glosses = list(imported.labels.keys())[:limit_glosses]
    selected_label_map = {gloss: index for index, gloss in enumerate(selected_glosses)}

    split_data: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    split_labels: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    split_signers: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    detected: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for gloss in selected_glosses:
        gloss_dir = frame_root / gloss
        if not gloss_dir.is_dir():
            continue
        for png_path in sorted(gloss_dir.glob("*.png")):
            video_id = png_path.stem
            if video_id not in imported.signer_ids:
                continue
            split = "train" if video_id in imported.train_ids else "val" if video_id in imported.validation_ids else "test"
            landmarks = normalize_hand_pair(extract_hand(png_path))
            if np.linalg.norm(landmarks) > 1e-6:
                detected[split] += 1
            split_data[split].append(landmarks)
            split_labels[split].append(selected_label_map[gloss])
            split_signers[split].append(imported.signer_ids[video_id])

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        np.save(output_dir / f"{split}_landmarks.npy", np.asarray(split_data[split], dtype=np.float32))
        np.save(output_dir / f"{split}_labels.npy", np.asarray(split_labels[split], dtype=np.int32))
        np.save(output_dir / f"{split}_signers.npy", np.asarray(split_signers[split], dtype=np.int32))
    (output_dir / "labels.json").write_text(
        json.dumps(selected_glosses),
        encoding="utf-8",
    )
    (output_dir / "dataset.json").write_text(
        json.dumps({
            "source": "WLASL v0.3 processed Kaggle mirror",
            "license": "C-UDA-1.0 computational use",
            "gloss_count": len(selected_glosses),
            "train_count": len(split_labels["train"]),
            "validation_count": len(split_labels["val"]),
            "test_count": len(split_labels["test"]),
            "detected_hand_counts": detected,
            "signer_count": len(set(sum(split_signers.values(), []))),
            "scope": "static representative-frame isolated signs",
        }, indent=2),
        encoding="utf-8",
    )
    return {
        "gloss_count": len(selected_glosses),
        "train_count": len(split_labels["train"]),
        "validation_count": len(split_labels["val"]),
        "test_count": len(split_labels["test"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/wlasl-processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/wlasl"))
    parser.add_argument("--limit-glosses", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset_dir, args.output_dir, args.limit_glosses), indent=2))


if __name__ == "__main__":
    main()
