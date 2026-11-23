import json
from pathlib import Path

from ml.hand_collection import convert_json


def test_hand_collection_json_converts_to_npz(tmp_path: Path):
    source = tmp_path / "hello.json"
    source.write_text(json.dumps({
        "schema_version": "fsl-hand-collection-v1",
        "session_id": "test-session",
        "signer_id": None,
        "sign_label": "HELLO",
        "frames": [{
            "timestamp_ms": 100,
            "left_hand": [[0.1, 0.2, 0.0] for _ in range(21)],
            "right_hand": None,
            "left_presence": [1] * 21,
            "right_presence": [0] * 21,
        }],
    }), encoding="utf-8")
    output = tmp_path / "hello.npz"
    convert_json(source, output)
    assert output.is_file()

