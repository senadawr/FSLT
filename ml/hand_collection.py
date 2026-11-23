import json
from pathlib import Path
import argparse

import numpy as np


def convert_json(json_path: Path, output_path: Path) -> None:
    session = json.loads(json_path.read_text(encoding="utf-8"))
    frames = session["frames"]
    sequence = np.zeros((len(frames), 42, 3), dtype=np.float32)
    presence = np.zeros((len(frames), 42), dtype=np.uint8)
    timestamps = np.zeros(len(frames), dtype=np.int64)
    for index, frame in enumerate(frames):
        timestamps[index] = frame["timestamp_ms"]
        if frame["left_hand"] is not None:
            sequence[index, :21] = np.asarray(frame["left_hand"], dtype=np.float32)
            presence[index, :21] = np.asarray(frame["left_presence"], dtype=np.uint8)
        if frame["right_hand"] is not None:
            sequence[index, 21:] = np.asarray(frame["right_hand"], dtype=np.float32)
            presence[index, 21:] = np.asarray(frame["right_presence"], dtype=np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        landmarks=sequence,
        presence=presence,
        timestamps=timestamps,
        sign_label=np.asarray(session["sign_label"]),
        signer_id=np.asarray(session["signer_id"] or "unavailable"),
        session_id=np.asarray(session["session_id"]),
        schema_version=np.asarray(session["schema_version"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hand_collection"))
    args = parser.parse_args()
    for input_path in args.inputs:
        convert_json(input_path, args.output_dir / f"{input_path.stem}.npz")


if __name__ == "__main__":
    main()
