# FSLT: Filipino Sign Language Recognition and Translation

FSLT is a prototype for real-time Filipino Sign Language recognition. It is
structured to support both isolated-sign recognition and continuous gloss
recognition without presenting untrained models as translations.

## Status

- The browser prototype extracts holistic landmarks locally and sends only
  versioned landmark features to the backend. The Tasks Vision model is loaded
  from the configured `VITE_HOLISTIC_MODEL_URL` (or its documented default).
- The backend validates frames, maintains a bounded sequence buffer, and
  reports model availability and confidence-aware predictions.
- Improved isolated GRU/LSTM training builders, feature engineering, class
  weighting, per-class evaluation, and Dataset A metadata utilities are
  included.
- The initial improved GRU achieved 5.16% test accuracy and 3.89% macro F1;
  the improved LSTM achieved 1.41% test accuracy and 0.29% macro F1. These
  results are clip-level and do not establish unseen-signer performance.
- Continuous CTC and gloss-to-text infrastructure is intentionally marked
  **not trained** until suitable continuous FSL annotations and parallel
  translation data are supplied.
- The browser can save one normalized 21-point hand pose per user-defined label
  through the local backend into `artifacts/pose_templates.json`, then match
  live poses with a distance-based confidence score without uploading webcam
  footage.
- A trained ASL A–Z alphabet GRU is available experimentally from the CC0
  10,508-sample public landmark dataset. It is static-isolated alphabet only,
  and its 99.95% test accuracy is not an unseen-signer result.

## Layout

- `frontend/` - React + TypeScript + Vite webcam client
- `backend/` - FastAPI and WebSocket inference service
- `ml/` - feature validation, dataset metadata, model builders, evaluation
- `docs/` - architecture, schema, datasets, training, and continuous signing
- `tests/` - backend and feature-contract tests

## Requirements

- Node.js 20+
- Python 3.11 or 3.12
- A modern browser with webcam support

## Run locally

### One command on Windows

From the project root:

```powershell
.\start.ps1
```

This opens the website and starts the local frontend and backend. Press
`Ctrl+C` to stop the services launched by the script.

### Backend

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
uvicorn backend.app.main:app --reload
```

### Frontend

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

The frontend defaults to `ws://localhost:8000/ws/inference`. Set
`VITE_INFERENCE_WS_URL` to use another backend.

Run these commands from the `frontend` directory because
`frontend\package.json` is the npm project file; the repository root does not
contain a `package.json`.

## Datasets and model weights

The project does not download or fabricate datasets. Use the documented
importer interfaces in [`docs/dataset_guide.md`](docs/dataset_guide.md) after
reviewing the source dataset licenses and actual file layouts. No trained
weights are included; the API explicitly reports that inference is unavailable
until a compatible model is configured.

See [`docs/architecture.md`](docs/architecture.md) and
[`docs/continuous_translation.md`](docs/continuous_translation.md) for the
current boundaries and limitations.
