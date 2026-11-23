import csv
import zipfile
from pathlib import Path
import argparse

import cv2
import mediapipe as mp
import numpy as np


FEATURE_DIMENSION = 1665
LANDMARK_COUNT = 553
MAX_FRAMES = 64


def frame_to_features(results) -> tuple[np.ndarray, np.ndarray]:
    features = np.zeros(FEATURE_DIMENSION, dtype=np.float32)
    presence = np.zeros(LANDMARK_COUNT, dtype=np.uint8)
    offset = 0

    groups = (
        (getattr(results, "face_landmarks", None), 478),
        (getattr(results, "pose_landmarks", None), 33),
        (getattr(results, "left_hand_landmarks", None), 21),
        (getattr(results, "right_hand_landmarks", None), 21),
    )
    landmark_index = 0
    for landmarks, count in groups:
        for index in range(count):
            if landmarks is not None and index < len(landmarks.landmark):
                point = landmarks.landmark[index]
                features[offset:offset + 3] = (point.x, point.y, point.z)
                presence[landmark_index] = 1
            offset += 3
            landmark_index += 1
    return features, presence


def extract_video(path: Path, holistic) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if len(frames) >= MAX_FRAMES:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            features, _ = frame_to_features(results)
            frames.append(features)
    finally:
        capture.release()
    if not frames:
        return np.zeros((1, FEATURE_DIMENSION), dtype=np.float32)
    indices = np.linspace(0, len(frames) - 1, num=MAX_FRAMES).astype(int)
    sampled = np.asarray([frames[index] for index in indices], dtype=np.float32)
    return sampled


def prepare(dataset_dir: Path, output_dir: Path, limit: int | None = None) -> None:
    mp_holistic = mp.solutions.holistic
    output_dir.mkdir(parents=True, exist_ok=True)
    mp_holistic_instance = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    try:
        for split in ("train", "test"):
            rows = []
            with (dataset_dir / f"{split}.csv").open(newline="", encoding="utf-8-sig") as file:
                rows = list(csv.DictReader(file))
            if limit:
                rows = rows[:limit]
            samples = []
            labels = []
            for index, row in enumerate(rows, start=1):
                relative = Path(row["vid_path"].replace("\\", "/"))
                video = dataset_dir / relative
                if not video.is_file():
                    raise FileNotFoundError(f"Missing extracted video: {video}")
                features = extract_video(video, mp_holistic_instance)
                sample_path = output_dir / split / f"{index:05d}.npy"
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(sample_path, features)
                samples.append(str(sample_path))
                labels.append(int(row["id_label"]))
                if index % 25 == 0:
                    print(f"{split}: {index}/{len(rows)}")
            np.save(output_dir / f"{split}_labels.npy", np.asarray(labels, dtype=np.int32))
            np.save(output_dir / f"{split}_paths.npy", np.asarray(samples))
    finally:
        mp_holistic_instance.close()


def extract_archive(dataset_dir: Path) -> None:
    archive = dataset_dir / "clips.zip"
    with zipfile.ZipFile(archive) as zipped:
        bad = zipped.testzip()
        if bad:
            raise RuntimeError(f"Corrupt ZIP entry: {bad}")
        zipped.extractall(dataset_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/fsl-105"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/isolated_sign/fsl105"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--extract-archive", action="store_true")
    args = parser.parse_args()
    if args.extract_archive:
        extract_archive(args.dataset_dir)
    prepare(args.dataset_dir, args.output_dir, args.limit)


if __name__ == "__main__":
    main()
