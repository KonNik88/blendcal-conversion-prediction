from __future__ import annotations
import json, time, os, uuid
from pathlib import Path
from typing import Any, Dict, Optional
from .config import LOG_DIR, MODEL_VERSION

# Log files
REQ_LOG = LOG_DIR / "requests.jsonl"
RESP_LOG = LOG_DIR / "responses.jsonl"

# Rotation settings (simple size-based rotation)
LOG_MAX_MB = float(os.getenv("LOG_MAX_MB", 50))
LOG_BACKUPS = int(os.getenv("LOG_BACKUPS", 3))

def _rotate_if_needed(path: Path) -> None:
    try:
        if not path.exists():
            return
        max_bytes = int(LOG_MAX_MB * 1024 * 1024)
        if path.stat().st_size <= max_bytes:
            return
        # rotate: .N <- .N-1 <- ... <- .1 <- current
        for i in range(LOG_BACKUPS, 0, -1):
            src = path.with_suffix(path.suffix + f".{i}")
            dst = path.with_suffix(path.suffix + f".{i+1}")
            if src.exists():
                if i == LOG_BACKUPS:
                    try:
                        src.unlink(missing_ok=True)
                    except Exception:
                        pass
                else:
                    try:
                        src.rename(dst)
                    except Exception:
                        pass
        # move current -> .1
        path.rename(path.with_suffix(path.suffix + ".1"))
    except Exception:
        # rotation failures must be non-fatal
        pass

def new_request_id() -> str:
    return uuid.uuid4().hex

def log_request(endpoint: str, payload: Dict[str, Any], *, request_id: Optional[str] = None, extra: Optional[Dict[str, Any]] = None):
    rec = {
        "ts": time.time(),
        "endpoint": endpoint,
        "request_id": request_id,
        "payload": payload,
    }
    if extra:
        rec.update(extra)
    _rotate_if_needed(REQ_LOG)
    with REQ_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def log_response(
    endpoint: str,
    payload: Dict[str, Any],
    *,
    request_id: Optional[str] = None,
    status: str = "ok",
    t_ms: Optional[float] = None,
    model_version: Optional[str] = None,
    active_models: Optional[list] = None,
    threshold_mode: Optional[str] = None,
    threshold_used: Optional[float] = None,
    error: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    rec = {
        "ts": time.time(),
        "endpoint": endpoint,
        "request_id": request_id,
        "status": status,
        "t_ms": t_ms,
        "model_version": model_version or MODEL_VERSION,
        "active_models": active_models,
        "threshold_mode": threshold_mode,
        "threshold_used": threshold_used,
        "payload": payload,
    }
    if error is not None:
        rec["error"] = str(error)
    if extra:
        rec.update(extra)
    _rotate_if_needed(RESP_LOG)
    with RESP_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def log_error(endpoint: str, request_id: Optional[str], error: str, *, t_ms: Optional[float] = None, extra: Optional[Dict[str, Any]] = None):
    log_response(
        endpoint,
        payload={},
        request_id=request_id,
        status="error",
        t_ms=t_ms,
        error=error,
        extra=extra,
    )
