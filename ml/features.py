from dataclasses import dataclass
import numpy as np


FACE_LANDMARK_COUNT = 478
POSE_LANDMARK_COUNT = 33
HAND_LANDMARK_COUNT = 21
MAX_FRAMES = 64
SELECTED_LANDMARK_COUNT = POSE_LANDMARK_COUNT + 2 * HAND_LANDMARK_COUNT
FEATURE_DIMENSION = SELECTED_LANDMARK_COUNT * 3 * 3 + SELECTED_LANDMARK_COUNT + 20


@dataclass(frozen=True)
class FeatureStats:
    feature_dimension: int
    selected_landmarks: int


def build_selected_features(sequences: np.ndarray) -> np.ndarray:
    if (
        sequences.ndim != 3
        or sequences.shape[1] != MAX_FRAMES
        or sequences.shape[2] not in (1659, 1665)
    ):
        raise ValueError(
            f"Expected (samples, {MAX_FRAMES}, 1659 or 1665), got {sequences.shape}"
        )
    output = np.zeros((sequences.shape[0], MAX_FRAMES, FEATURE_DIMENSION), dtype=np.float32)
    pose_indices = list(range(FACE_LANDMARK_COUNT, FACE_LANDMARK_COUNT + POSE_LANDMARK_COUNT))
    left_indices = list(range(
        FACE_LANDMARK_COUNT + POSE_LANDMARK_COUNT,
        FACE_LANDMARK_COUNT + POSE_LANDMARK_COUNT + HAND_LANDMARK_COUNT,
    ))
    right_indices = list(range(
        FACE_LANDMARK_COUNT + POSE_LANDMARK_COUNT + HAND_LANDMARK_COUNT,
        FACE_LANDMARK_COUNT + POSE_LANDMARK_COUNT + 2 * HAND_LANDMARK_COUNT,
    ))
    selected_indices = pose_indices + left_indices + right_indices

    for sample_index, sequence in enumerate(sequences):
        coords = sequence.reshape(MAX_FRAMES, -1, 3)
        selected = coords[:, selected_indices, :].copy()
        pose = coords[:, pose_indices, :].copy()
        shoulders = pose[:, [11, 12], :].copy()
        valid_shoulders = np.all(np.abs(shoulders) > 1e-6, axis=2)
        shoulder_means = shoulders.mean(axis=1)[:, None, :]
        center = np.where(
            (valid_shoulders.mean(axis=1)[:, None, None] > 0.5),
            shoulder_means,
            np.zeros((MAX_FRAMES, 1, 3), dtype=np.float32),
        )
        shoulder_distance = np.linalg.norm(shoulders[:, 0, :] - shoulders[:, 1, :], axis=1)
        scale = np.where(shoulder_distance > 1e-5, shoulder_distance, 1.0)

        normalized = (selected - center) / scale[:, None, None]
        presence = (np.linalg.norm(selected, axis=2) > 1e-6).astype(np.float32)
        normalized *= presence[:, :, None]
        velocity = np.zeros_like(normalized)
        velocity[1:] = (normalized[1:] - normalized[:-1]) * presence[1:, :, None] * presence[:-1, :, None]
        acceleration = np.zeros_like(normalized)
        acceleration[1:] = velocity[1:] - velocity[:-1]

        hand_distances = np.zeros((MAX_FRAMES, 20), dtype=np.float32)
        left_hand = normalized[:, POSE_LANDMARK_COUNT:POSE_LANDMARK_COUNT + 21, :]
        right_hand = normalized[:, POSE_LANDMARK_COUNT + 21:, :]
        fingertips = (4, 8, 12, 16, 20)
        for offset, hand in enumerate((left_hand, right_hand)):
            column = offset * 10
            distance_index = 0
            for first in range(len(fingertips)):
                for second in range(first + 1, len(fingertips)):
                    p1 = hand[:, fingertips[first], :]
                    p2 = hand[:, fingertips[second], :]
                    hand_distances[:, column + distance_index] = np.linalg.norm(p1 - p2, axis=1)
                    distance_index += 1

        output[sample_index] = np.concatenate(
            (
                normalized.reshape(MAX_FRAMES, -1),
                presence,
                velocity.reshape(MAX_FRAMES, -1),
                acceleration.reshape(MAX_FRAMES, -1),
                hand_distances,
            ),
            axis=1,
        )
    return output


def augment_temporal_spatial(sequences: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    augmented = sequences.copy()
    coordinate_columns = SELECTED_LANDMARK_COUNT * 3
    coordinates = augmented[:, :, :coordinate_columns]
    scale = rng.uniform(0.92, 1.08, size=(len(sequences), 1, 1))
    coordinates *= scale.astype(np.float32)
    coordinates += rng.normal(0.0, 0.008, size=coordinates.shape).astype(np.float32)
    augmented[:, :, :coordinate_columns] = coordinates
    return augmented
