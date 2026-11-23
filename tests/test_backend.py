from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas import FEATURE_DIMENSION, LANDMARK_COUNT


def test_health_and_untrained_model_info():
    client = TestClient(app)
    assert client.get("/api/health").json() == {"status": "ok"}
    info = client.get("/api/model/info").json()
    assert info["training_status"] == "trained"
    assert info["feature_schema_version"] == "fsl-landmarks-v1"


def test_websocket_rejects_bad_payload_and_handles_experimental_model():
    client = TestClient(app)
    with client.websocket_connect("/ws/inference") as websocket:
        websocket.send_json({"schema_version": "wrong"})
        invalid = websocket.receive_json()
        assert invalid["error"] == "Invalid landmark frame"

        websocket.send_json({
            "schema_version": "fsl-landmarks-v1",
            "timestamp_ms": 10,
            "features": [0.0] * FEATURE_DIMENSION,
            "presence": [0.0] * LANDMARK_COUNT,
        })
        response = websocket.receive_json()
        assert response["prediction_status"] in {"uncertain", "recognized"}
        assert response["prediction"] is None
