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
    try:
        payload = PoseTemplatesPayload.model_validate_json(
            pose_templates_path.read_text(encoding="utf-8"),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Invalid pose template file at {pose_templates_path}: {exc}"
        ) from exc
    return payload


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


@app.websocket("/ws/asl")
async def asl_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.receive_json()
            await websocket.send_json({
                "prediction": None,
                "confidence": 0.0,
                "status": "unavailable",
                "model_version": "pose-templates-no-legacy-model",
            })
    except WebSocketDisconnect:
        return
