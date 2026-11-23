import csv
import json
from dataclasses import dataclass
from pathlib import Path
import argparse

import cv2
import numpy as np
import mediapipe as mp

from sklearn.model_selection import train_test_split


MAX_FRAMES = 64
HAND_MODEL_FRAMES = 21


@dataclass(frozen=True)
class SkeletalWordImport:
    video_paths: list[Path]
    glosses: list[str]
    source: str
    license: str


def import_words(dataset_dir: Path) -> SkeletalWordImport:
    words_path = dataset_dir / "words.txt"
    videos_dir = dataset_dir / "videos" / "videos"
    words = [line.strip() for line in words_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    videos = []
    glosses = []
    for word in words:
        candidates = list(videos_dir.glob(f"{word}.mp4"))
        if not candidates and word == "don't want":
            candidates = list(videos_dir.glob("donot want.mp4"))
        if not candidates:
            raise FileNotFoundError(f"No skeletal video for word: {word}")
        videos.append(candidates[0])
        glosses.append(word)
    return SkeletalWordImport(
        video_paths=videos,
        glosses=glosses,
        source="Skeletal Sign Language Videos - ASL",
        license="MIT",
    )


def extract_hand_from_skeleton_video(video_path: Path) -> np.ndarray:
    """Extract dedicated MediaPipe Hands landmarks from pre-rendered skeleton frames."""
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.25,
    )
    frames = []
    capture = cv2.VideoCapture(str(video_path))
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            left = np.zeros((21, 3), dtype=np.float32)
            right = np.zeros((21, 3), dtype=np.float32)
            if result.multi_hand_landmarks:
                handedness = result.multi_handedness or []
                for index, landmarks in enumerate(result.multi_hand_landmarks):
                    label = handedness[index].classification[0].label.lower() if index < len(handedness) else ""
                    values = np.asarray(
                        [[point.x, point.y, point.z] for point in landmarks.landmark],
                        dtype=np.float32,
                    )
                    if label == "left":
                        left = values
                    elif label == "right":
                        right = values
            frames.append(np.concatenate((left, right), axis=0))
    finally:
        capture.release()
        hands.close()
    if not frames:
        return np.zeros((MAX_FRAMES, 42, 3), dtype=np.float32)
    indices = np.linspace(0, len(frames) - 1, num=MAX_FRAMES).astype(int)
    return np.asarray([frames[index] for index in indices], dtype=np.float32)


def prepare(dataset_dir: Path, output_dir: Path, limit: int | None = None, seed: int = 2026) -> dict[str, int]:
    imported = import_words(dataset_dir)
    video_paths = imported.video_paths
    glosses = imported.glosses
    if limit:
        video_paths = video_paths[:limit]
        glosses = glosses[:limit]
    sequences = np.asarray([extract_hand_from_skeleton_video(path) for path in video_paths], dtype=np.float32)
    label_map = {gloss: index for index, gloss in enumerate(glosses)}
    labels = np.asarray([label_map[gloss] for gloss in glosses], dtype=np.int32)
    counts = np.bincount(labels)
    stratify_labels = labels if counts.min() >= 2 else None
    train_indices, test_indices = train_test_split(
        np.arange(len(sequences)),
        test_size=0.2,
        random_state=seed,
        stratify=stratify_labels,
    )
    train_indices, validation_indices = train_test_split(
        train_indices,
        test_size=0.125,
        random_state=seed,
        stratify=None,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "train_sequences.npy", sequences[train_indices])
    np.save(output_dir / "train_labels.npy", labels[train_indices])
    np.save(output_dir / "validation_sequences.npy", sequences[validation_indices])
    np.save(output_dir / "validation_labels.npy", labels[validation_indices])
    np.save(output_dir / "test_sequences.npy", sequences[test_indices])
    np.save(output_dir / "test_labels.npy", labels[test_indices])
    (output_dir / "labels.json").write_text(json.dumps(glosses), encoding="utf-8")
    (output_dir / "dataset.json").write_text(json.dumps({
        "source": imported.source,
        "license": imported.license,
        "word_count": len(glosses),
        "train_count": len(train_indices),
        "validation_count": len(validation_indices),
        "test_count": len(test_indices),
        "signer_ids_available": False,
        "continuous_sequences": False,
        "scope": "isolated whole-word skeleton videos",
    }, indent=2), encoding="utf-8")
    return {
        "word_count": len(glosses),
        "train_count": len(train_indices),
        "validation_count": len(validation_indices),
        "test_count": len(test_indices),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/asl-words-skeletal"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/asl_words"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset_dir, args.output_dir, args.limit), indent=2))


if __name__ == "__main__":
    main()
