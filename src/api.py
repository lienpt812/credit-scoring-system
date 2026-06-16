from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import FEATURE_COLUMNS, MODEL_PATH, ROOT_DIR
from src.modeling import predict_risk, train_and_evaluate
from src.schemas import CreditApplication, PredictionResponse


FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(title="Credit Scoring API", version="0.1.0")
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/train")
def train_model() -> dict:
    return train_and_evaluate()


@app.post("/predict", response_model=PredictionResponse)
def predict(application: CreditApplication) -> dict:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="Model not found. Run python -m src.train first.")
    row = application.model_dump(by_alias=True)
    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    return predict_risk(df)
