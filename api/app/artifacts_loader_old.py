# api/app/artifacts_loader.py
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import xgboost as xgb
import lightgbm as lgb

from .config import (
    CAT_PATH, XGB_PATH, LGB_PATH,               # legacy pkl (опционально)
    XGB_JSON, XGB_CAL, LGB_TXT, LGB_CAL,        # устойчивые артефакты
    WEIGHTS_PATH, THRESHOLDS_PATH,
    PREP_META_PATH, FREQ_MAPS_PATH,
)
from .preprocessor import Preprocessor

log = logging.getLogger("blendcal")
log.setLevel(logging.INFO)

def _load_blend_weights(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    d = d.get("weights", d)  # поддержка train-формата
    keymap = {
        "CatBoost_calibrated_isotonic": "cat",
        "CatBoost_calibrated": "cat",
        "XGBoost_calibrated": "xgb",
        "LightGBM_calibrated": "lgb",
        "cat": "cat", "xgb": "xgb", "lgb": "lgb",
    }
    out = {}
    for k, v in d.items():
        kk = keymap.get(k)
        if kk:
            out[kk] = float(v)
    s = sum(out.values()) or 1.0
    # нормируем на всякий случай, чтобы w_cat + w_xgb + w_lgb = 1
    return {k: v / s for k, v in out.items()}

def _load_thresholds(path: Path) -> dict:
    t = json.loads(path.read_text(encoding="utf-8"))
    best = t.get("bestF1", t.get("thr_bestF1_valid", 0.5))
    hire = t.get("hi_recall", t.get("thr_highRecall_valid", 0.5))
    return {"bestF1": float(best), "hi_recall": float(hire)}

def _read_json(path, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"Failed to read JSON '{path.name}': {e!r}. Using default: {default}")
        return dict(default)


def _pick_weight(weights: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    """Возвращает первый найденный ключ из списка с float-преобразованием."""
    for k in keys:
        if k in weights:
            try:
                return float(weights[k])
            except Exception:
                pass
    return float(default)


class Artifacts:
    _instance: Optional["Artifacts"] = None

    def __init__(self):
        # --- Веса/пороги (как есть из файлов)
        self.weights: Dict[str, float] = _read_json(WEIGHTS_PATH, {})
        self.thresholds: Dict[str, float] = _read_json(THRESHOLDS_PATH, {"bestF1": 0.5})

        # === НОРМАЛИЗАЦИЯ ВЕСОВ ===
        # Поддерживаем формат {"weights": {...}, "note": "..."} → берём вложенный словарь
        if isinstance(self.weights, dict) and "weights" in self.weights and isinstance(self.weights["weights"], dict):
            self.weights = self.weights["weights"]

        # === НОРМАЛИЗАЦИЯ ПОРОГОВ ===
        # Поддерживаем твой формат: thr_bestF1_valid / thr_highRecall_valid, а также hi-recall/hi_recall и т.п.
        t = dict(self.thresholds) if isinstance(self.thresholds, dict) else {}

        def _pick_thr(candidates: List[str], default: Optional[float] = None) -> Optional[float]:
            for k in candidates:
                if k in t:
                    try:
                        return float(t[k])
                    except Exception:
                        pass
            return default

        bestF1_thr = _pick_thr(
            ["bestF1", "bestf1", "thr_bestF1", "thr_bestF1_valid"],
            default=0.5
        )
        hi_recall_thr = _pick_thr(
            ["hi-recall", "hi_recall", "thr_highRecall", "thr_highRecall_valid"],
            default=0.5
        )

        # Приводим к плоскому виду, который использует API
        self.thresholds = {"bestF1": bestF1_thr, "hi-recall": hi_recall_thr}

        # --- Препроцессор
        self.prep = Preprocessor(PREP_META_PATH, FREQ_MAPS_PATH)
        self.feature_order = self.prep.feature_order

        # --- CatBoost (pkl)
        self.model_cat = None
        try:
            self.model_cat = joblib.load(CAT_PATH)
            log.info(f"CatBoost: loaded {CAT_PATH.name}")
        except Exception as e:
            log.warning(f"CatBoost load failed: {e!r}")

        # --- XGB: ROBUST-FIRST (booster+calibrator) -> legacy .pkl
        self.model_xgb = None  # калиброванная sklearn-модель (если удастся)
        self.xgb_booster: Optional[xgb.Booster] = None
        self.xgb_cal = None

        try:
            if XGB_JSON.exists() and XGB_CAL.exists():
                bst = xgb.Booster()
                bst.load_model(str(XGB_JSON))
                self.xgb_booster = bst
                self.xgb_cal = joblib.load(XGB_CAL)
                log.info(f"XGB: loaded booster={XGB_JSON.name}, calibrator={XGB_CAL.name}")
            elif XGB_PATH.exists():
                # fallback: возможны ошибки из-за __main__.* обёрток
                self.model_xgb = joblib.load(XGB_PATH)
                log.info(f"XGB: loaded legacy calibrated {XGB_PATH.name}")
            else:
                log.warning("XGB: no artifacts found (neither robust nor legacy).")
        except Exception as e:
            log.warning(f"XGB load failed: {e!r}")

        # --- LGB: ROBUST-FIRST (booster+calibrator) -> legacy .pkl
        self.model_lgb = None
        self.lgb_booster: Optional[lgb.Booster] = None
        self.lgb_cal = None

        try:
            if LGB_TXT.exists() and LGB_CAL.exists():
                self.lgb_booster = lgb.Booster(model_file=str(LGB_TXT))
                self.lgb_cal = joblib.load(LGB_CAL)
                log.info(f"LGB: loaded booster={LGB_TXT.name}, calibrator={LGB_CAL.name}")
            elif LGB_PATH.exists():
                self.model_lgb = joblib.load(LGB_PATH)
                log.info(f"LGB: loaded legacy calibrated {LGB_PATH.name}")
            else:
                log.warning("LGB: no artifacts found (neither robust nor legacy).")
        except Exception as e:
            log.warning(f"LGB load failed: {e!r}")

        # --- Сбор активных моделей и нормировка весов
        # Поддерживаем разные соглашения в ключах весов (включая «длинные» из твоего файла)
        w_cat = _pick_weight(
            self.weights,
            ["cat", "catboost", "CatBoost", "cat_boost", "CatBoost_calibrated_isotonic"],
            default=0.366
        )
        w_xgb = _pick_weight(
            self.weights,
            ["xgb", "xgboost", "XGB", "XGBoost", "XGBoost_calibrated"],
            default=0.402
        )
        w_lgb = _pick_weight(
            self.weights,
            ["lgb", "lightgbm", "LGBM", "LightGBM", "LightGBM_calibrated"],
            default=0.232
        )

        active: List[Tuple[str, float]] = []

        if self.model_cat is not None:
            active.append(("cat", w_cat))

        # xgb активируем, если есть хоть один путь: legacy-модель или (booster AND calibrator)
        if (self.model_xgb is not None) or (self.xgb_booster is not None and self.xgb_cal is not None):
            active.append(("xgb", w_xgb))
        elif w_xgb > 0:
            log.warning("XGB weight > 0, but no usable XGB artifacts were loaded.")

        if (self.model_lgb is not None) or (self.lgb_booster is not None and self.lgb_cal is not None):
            active.append(("lgb", w_lgb))
        elif w_lgb > 0:
            log.warning("LGB weight > 0, but no usable LGB artifacts were loaded.")

        if not active:
            # Бэкап: если доступен только CatBoost — используем его,
            # иначе действительно нечем предсказывать
            if self.model_cat is not None:
                active = [("cat", 1.0)]
                log.warning("No XGB/LGB available. Falling back to CatBoost only.")
            else:
                raise RuntimeError("No models available for inference (Cat/XGB/LGB missing).")

        # Нормировка весов
        w_sum = sum(w for _, w in active)
        if w_sum <= 0:
            # если все веса нулевые/отрицательные — распределим поровну
            eq = 1.0 / len(active)
            self.active = [(n, eq) for n, _ in active]
        else:
            self.active = [(n, w / w_sum) for n, w in active]

        log.info(f"Active models: {self.active}")

    @classmethod
    def get(cls) -> "Artifacts":
        if cls._instance is None:
            cls._instance = Artifacts()
        return cls._instance

    # --- Приватные вычислители вероятностей ---
    def _proba_cat(self, X: np.ndarray) -> np.ndarray:
        # CatBoost всегда sklearn-совместимый
        return self.model_cat.predict_proba(X)[:, 1]

    def _proba_xgb(self, X: np.ndarray) -> np.ndarray:
        if self.model_xgb is not None:
            return self.model_xgb.predict_proba(X)[:, 1]
        # booster-путь: X -> DMatrix -> proba -> isotonic.predict
        dmat = xgb.DMatrix(X)
        p_raw = self.xgb_booster.predict(dmat)  # shape ~ (n,) или (n,1)
        p_raw = np.asarray(p_raw).reshape(-1)
        return self.xgb_cal.predict(p_raw)

    def _proba_lgb(self, X: np.ndarray) -> np.ndarray:
        if self.model_lgb is not None:
            return self.model_lgb.predict_proba(X)[:, 1]
        # Booster.predict для бинарной задачи — вероятности (raw_score=False по умолчанию)
        p_raw = self.lgb_booster.predict(X)  # shape ~ (n,)
        p_raw = np.asarray(p_raw).reshape(-1)
        return self.lgb_cal.predict(p_raw)

    # --- Публичный бленд ---
    def predict_proba_blend(self, X: np.ndarray) -> np.ndarray:
        parts = []
        for name, w in self.active:
            if name == "cat":
                p = self._proba_cat(X)
            elif name == "xgb":
                p = self._proba_xgb(X)
            elif name == "lgb":
                p = self._proba_lgb(X)
            else:
                continue
            parts.append(w * p)
        return np.sum(parts, axis=0)
