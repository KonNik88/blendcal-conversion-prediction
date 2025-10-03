# api/app/main.py
from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from typing import Any, Dict, Literal, Optional
import io
import json
import time
from pathlib import Path

import pandas as pd
import numpy as np

from .artifacts_loader import Artifacts
from .config import MODEL_VERSION, ROOT
from .logger import log_request, log_response, log_error, new_request_id


class PredictRequest(BaseModel):
    features: Dict[str, Any]
    mode: Optional[str] = None


def _resolve_threshold(mode: Optional[str], thresholds: Dict[str, float]) -> float:
    """
    Приводит режим к единому виду (hi-recall ~= hi_recall) и достаёт порог.
    Если режим не найден — откатываемся к bestF1, далее к 0.5.
    """
    mode_norm = (mode or "bestF1").replace("-", "_").lower()
    thr_map = {k.replace("-", "_").lower(): v for k, v in thresholds.items()}
    return float(thr_map.get(mode_norm, thr_map.get("bestf1", 0.5)))


app = FastAPI(title="BlendCAL API", version="1.0.2")

# CORS для UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    info_path = ROOT / "MODEL_INFO.json"
    payload = {"version": "dev", "model_info_path": str(info_path), "exists": info_path.exists()}
    try:
        if info_path.exists():
            payload = json.loads(info_path.read_text(encoding="utf-8"))
            payload["model_info_path"] = str(info_path)
            payload["exists"] = True
        else:
            payload["note"] = "MODEL_INFO.json not found"
    except Exception as e:
        payload["error"] = f"{type(e).__name__}: {e}"

    # не роняем /version, если модели не загрузились
    try:
        art = Artifacts.get()
        payload["active_models"] = [name for name, _ in art.active]
    except Exception as e:
        payload["active_models"] = []
        payload["active_models_error"] = f"{type(e).__name__}: {e}"

    return payload


@app.get("/features")
def features():
    art = Artifacts.get()
    return {
        "num_cols": art.prep.num_cols,
        "cat_cols": art.prep.cat_cols,
        "clip_cols": art.prep.clip_cols,
        "feature_order": art.feature_order,
    }


@app.get("/debug")
def debug():
    art = Artifacts.get()
    weights = {name: float(w) for name, w in art.active}  # из списка (имя, вес)
    return {
        "thresholds": art.thresholds,
        "weights": weights,
        "active_models": list(weights.keys()),
    }


@app.post("/predict")
def predict(req: PredictRequest, mode: Literal["bestF1", "hi-recall"] = "bestF1"):
    art = Artifacts.get()
    mode_in = (req.mode or mode)

    request_id = new_request_id()
    t0 = time.perf_counter()
    payload = {"features": req.features, "mode": mode_in}
    log_request("/predict", payload, request_id=request_id)

    try:
        # Препроцесс → DataFrame → обязательно в numpy (без имён колонок!)
        X_df = art.prep.transform_payload(req.features)
        try:
            X = X_df.to_numpy(dtype=float, copy=False)
        except Exception:
            X = np.asarray(X_df, dtype=float)

        proba = float(art.predict_proba_blend(X)[0])
        thr = _resolve_threshold(mode_in, art.thresholds)
        y_hat = int(proba >= thr)

        t_ms = (time.perf_counter() - t0) * 1000.0
        resp = {
            "proba": proba,
            "y_hat": y_hat,
            "threshold_used": thr,
            "threshold_mode": mode_in,
        }
        log_response(
            "/predict",
            resp,
            request_id=request_id,
            status="ok",
            t_ms=t_ms,
            model_version=MODEL_VERSION,
            active_models=[name for name, _ in art.active],
            threshold_mode=mode_in,
            threshold_used=thr,
        )
        return resp

    except Exception as e:
        t_ms = (time.perf_counter() - t0) * 1000.0
        log_error("/predict", request_id, error=f"{type(e).__name__}: {e}", t_ms=t_ms)
        return JSONResponse(status_code=500, content={"error": str(e), "request_id": request_id})


@app.post("/predict_debug")
def predict_debug(req: PredictRequest, mode: Optional[str] = None):
    art = Artifacts.get()
    mode_in = req.mode or mode

    X_df = art.prep.transform_payload(req.features)
    try:
        X = X_df.to_numpy(dtype=float, copy=False)
    except Exception:
        X = np.asarray(X_df, dtype=float)

    parts: Dict[str, Dict[str, float]] = {}
    for name, w in art.active:
        if name == "cat":
            p = float(art._proba_cat(X)[0])
        elif name == "xgb":
            p = float(art._proba_xgb(X)[0])
        elif name == "lgb":
            p = float(art._proba_lgb(X)[0])
        else:
            continue
        parts[name] = {"proba": p, "weight": float(w), "weighted": float(w) * p}

    proba_blend = float(art.predict_proba_blend(X)[0])
    thr = _resolve_threshold(mode_in, art.thresholds)
    return {
        "mode": mode_in,
        "threshold_used": thr,
        "proba_blend": proba_blend,
        "y_hat": int(proba_blend >= thr),
        "parts": parts,
    }


@app.get("/cat_top")
def cat_top(col: str, k: int = 10):
    art = Artifacts.get()
    fmap = art.prep.freq_maps.get(col)
    if fmap is None:
        return JSONResponse(status_code=404, content={"error": f"unknown categorical column: {col}"})
    ser = fmap.sort_values(ascending=False).head(max(1, int(k)))
    return {"col": col, "top": ser.index.tolist(), "scores": [float(x) for x in ser.values]}


@app.post("/predict_batch")
async def predict_batch(
    file: UploadFile = File(...),
    mode: Optional[str] = Form("bestF1"),
):
    art = Artifacts.get()

    request_id = new_request_id()
    t0 = time.perf_counter()

    content = await file.read()
    df_raw = pd.read_csv(io.BytesIO(content))

    # Препроцесс → numpy
    X_df = art.prep.transform_df(df_raw)
    try:
        X = X_df.to_numpy(dtype=float, copy=False)
    except Exception:
        X = np.asarray(X_df, dtype=float)

    probas = art.predict_proba_blend(X)
    thr = _resolve_threshold(mode, art.thresholds)
    y_hat = (probas >= thr).astype(int)

    t_ms = (time.perf_counter() - t0) * 1000.0
    n_rows = int(getattr(df_raw, "shape", (0, 0))[0])
    log_response(
        "/predict_batch",
        payload={"n_rows": n_rows},
        request_id=request_id,
        status="ok",
        t_ms=t_ms,
        model_version=MODEL_VERSION,
        active_models=[name for name, _ in art.active],
        threshold_mode=mode,
        threshold_used=thr,
        extra={
            "proba_min": float(probas.min()) if n_rows else None,
            "proba_mean": float(probas.mean()) if n_rows else None,
            "proba_max": float(probas.max()) if n_rows else None,
        },
    )

    # CSV-ответ
    out = df_raw.copy()
    out["proba"] = probas
    out["y_hat"] = y_hat

    buf = io.StringIO()
    out.to_csv(buf, index=False)
    buf.seek(0)

    out_name = f'{Path(file.filename).stem}__scored.csv'
    headers = {"Content-Disposition": f'attachment; filename="{out_name}"'}
    return Response(content=buf.getvalue(), media_type="text/csv", headers=headers)
