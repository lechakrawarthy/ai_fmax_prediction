"""FastAPI backend.  Run: uvicorn api.main:app --reload   (interactive docs at /docs)"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fmax import __version__
from fmax import config as C
from fmax.service import FmaxService

app = FastAPI(title="Fmax Prediction API", version=__version__,
              description="Per-device Fmax prediction, calibrated search window and OOD fallback decision.")


class Unit(BaseModel):
    ro_01: float
    ro_02: float
    ro_03: float
    ro_04: float
    ro_05: float
    ro_06: float
    ro_07: float
    ro_08: float
    ro_09: float
    ro_10: float
    iddq_norm: float
    vmin_norm: float
    die_x: float
    die_y: float
    radius_sq: float


class SimulateRequest(BaseModel):
    unit: Unit
    fmax_true_mhz: float = Field(description="Hidden true Fmax, used only to drive the simulated tester.")
    seed: int = 0


@lru_cache(maxsize=1)
def service() -> FmaxService:
    try:
        return FmaxService()
    except FileNotFoundError:
        raise HTTPException(503, "No model bundle. Run `python -m fmax.pipeline` first.")


def _clean(v):
    if isinstance(v, (np.generic,)):
        v = v.item()
    return None if isinstance(v, float) and not np.isfinite(v) else v


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": (C.ARTIFACT_DIR / "model_bundle.joblib").exists()}


@app.get("/model")
def model_info():
    return service().meta


@app.get("/metrics")
def metrics():
    path = C.ARTIFACT_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "No metrics yet. Run `python -m fmax.pipeline` first.")
    return json.loads(path.read_text())


@app.post("/predict")
def predict(units: list[Unit]):
    if not units:
        raise HTTPException(422, "Provide at least one unit.")
    out = service().predict(pd.DataFrame([u.model_dump() for u in units]))
    return [{k: _clean(v) for k, v in row.items()} for row in out.to_dict(orient="records")]


@app.post("/simulate")
def simulate(req: SimulateRequest):
    return service().simulate(req.unit.model_dump(), req.fmax_true_mhz, req.seed)
