import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from .config import get_settings
from .inference import InferenceService
from .schemas import ASLHandFrame, DatasetInfo, LandmarkFrame, PoseTemplatesPayload

settings = get_settings()
inference = InferenceService(settings)
app = FastAPI(title="FSLT Inference API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):517[3-9]",
    allow_methods=["GET", "PUT"],
    allow_headers=["*"],
)

pose_templates_path = Path(__file__).resolve().parents[2] / "artifacts" / "pose_templates.json"


def ensure_pose_templates_file() -> None:
    pose_templates_path.parent.mkdir(parents=True, exist_ok=True)
    if not pose_templates_path.is_file():
        pose_templates_path.write_text(
            json.dumps({"templates": []}, indent=2),
            encoding="utf-8",
        )


ensure_pose_templates_file()


@app.get("/api/poses", response_model=PoseTemplatesPayload)
def get_pose_templates() -> PoseTemplatesPayload:
    ensure_pose_templates_file()
    return PoseTemplatesPayload.model_validate_json(
        pose_templates_path.read_text(encoding="utf-8"),
    )


@app.put("/api/poses", response_model=PoseTemplatesPayload)
def save_pose_templates(payload: PoseTemplatesPayload) -> PoseTemplatesPayload:
    pose_templates_path.parent.mkdir(parents=True, exist_ok=True)
    pose_templates_path.write_text(
        json.dumps(payload.model_dump(), indent=2),
        encoding="utf-8",
    )
    return payload


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/model/info")
def model_info():
    return inference.model_info()


@app.get("/api/dataset/info", response_model=DatasetInfo)
def dataset_info() -> DatasetInfo:
    return DatasetInfo(
        datasets=[],
        note="No dataset has been imported. See docs/dataset_guide.md.",
    )


@app.websocket("/ws/inference")
async def inference_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            try:
                frame = LandmarkFrame.model_validate(payload)
                response = inference.add_frame(frame)
            except ValidationError as exc:
                await websocket.send_json({"error": "Invalid landmark frame", "details": exc.errors()})
                continue
            await websocket.send_json(response.model_dump())
    except WebSocketDisconnect:
        return


asl_model = None
asl_labels: list[str] = []
try:
    from tensorflow import keras
    from pathlib import Path
    import json

    asl_path = Path("artifacts/asl_alphabet/gru_best.keras")
    if asl_path.is_file():
        asl_model = keras.models.load_model(asl_path)
        labels_path = asl_path.parent / "labels.json"
        if labels_path.is_file():
            asl_labels = json.loads(labels_path.read_text(encoding="utf-8"))
except Exception:
    asl_model = None


@app.websocket("/ws/asl")
async def asl_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            try:
                frame = ASLHandFrame.model_validate(payload)
            except ValidationError as exc:
                await websocket.send_json({"error": "Invalid ASL hand frame", "details": exc.errors()})
                continue
            if asl_model is None:
                await websocket.send_json({
                    "prediction": None,
                    "confidence": 0.0,
                    "status": "unavailable",
                    "model_version": "asl-gru-not-loaded",
                })
                continue
            import numpy as np

            raw = np.asarray(frame.hand, dtype=np.float32).reshape(21, 3)
            wrist = raw[0:1, :]
            scale = float(np.linalg.norm(raw[9] - raw[0]))
            scale = scale if scale > 1e-6 else 1.0
            normalized = (raw - wrist) / scale
            probabilities = np.asarray(
                asl_model.predict(normalized[None, ...], verbose=0)
            )[0]
            index = int(np.argmax(probabilities))
            confidence = float(probabilities[index])
            await websocket.send_json({
                "prediction": asl_labels[index] if index < len(asl_labels) else None,
                "confidence": confidence,
                "status": "recognized" if confidence >= 0.75 else "uncertain",
                "model_version": "experimental-asl-gru-v1",
            })
    except WebSocketDisconnect:
        return
