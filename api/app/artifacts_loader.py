# api/app/artifacts_loader.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

# XGBoost/LightGBM могут отсутствовать при сборке — подстрахуемся
try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None  # type: ignore

try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover
    lgb = None  # type: ignore

from .config import (
    CAT_PATH, XGB_PATH, LGB_PATH,               # legacy pkl (опционально)
    XGB_JSON, XGB_CAL, LGB_TXT, LGB_CAL,        # «робастные» артефакты
    WEIGHTS_PATH, THRESHOLDS_PATH,
    PREP_META_PATH, FREQ_MAPS_PATH,
)
from .preprocessor import Preprocessor

log = logging.getLogger("blendcal")
log.setLevel(logging.INFO)


# --- Compat: sklearn CalibratedClassifierCV / _CalibratedClassifier ----------
def _patch_calibrated_estimator_compat(model) -> Any:
    """
    Делает модель, сохранённую на другой версии sklearn, совместимой с текущей.
    У внутренних объектов _CalibratedClassifier добавляет alias .estimator -> .base_estimator (если отсутствует),
    а также подставляет .classes_ для базовой модели, если это требуется.
    Ничего не ломает, если уже всё ок.
    """
    try:
        # Случай CalibratedClassifierCV
        if hasattr(model, "calibrated_classifiers_"):
            for cc in getattr(model, "calibrated_classifiers_", []):
                # у новых версий есть base_estimator; у старых — estimator
                if not hasattr(cc, "estimator") and hasattr(cc, "base_estimator"):
                    try:
                        setattr(cc, "estimator", getattr(cc, "base_estimator"))
                    except Exception:
                        pass
                # иногда базовой модели нужна classes_
                if hasattr(model, "classes_"):
                    try:
                        base = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
                        if base is not None and not hasattr(base, "classes_"):
                            setattr(base, "classes_", getattr(model, "classes_"))
                    except Exception:
                        pass
        return model
    except Exception as e:
        log.warning(f"Compat patch for calibrated model failed: {e!r}")
        return model


# --------------------------- Helpers: weights & thresholds ---------------------------

def _load_blend_weights(path: Path) -> Dict[str, float]:
    """
    Читает веса бленда из JSON.
    Поддерживает оба варианта:
      { "weights": {"CatBoost_calibrated_isotonic":0.36,"XGBoost_calibrated":0.40,"LightGBM_calibrated":0.23} }
      { "cat":0.36, "xgb":0.40, "lgb":0.23 }
    Возвращает нормированные по сумме веса c ключами {cat,xgb,lgb}.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"Blend weights not found/failed to read '{path}': {e!r}. Using empty weights.")
        return {}

    d = raw.get("weights", raw) if isinstance(raw, dict) else {}
    keymap = {
        "CatBoost_calibrated_isotonic": "cat",
        "CatBoost_calibrated": "cat",
        "cat": "cat",
        "XGBoost_calibrated": "xgb",
        "xgb": "xgb",
        "LightGBM_calibrated": "lgb",
        "lgb": "lgb",
        "lightgbm": "lgb",
    }
    out: Dict[str, float] = {}
    for k, v in (d.items() if isinstance(d, dict) else []):
        kk = keymap.get(k, k.lower())
        if kk not in ("cat", "xgb", "lgb"):
            continue
        try:
            out[kk] = float(v)
        except Exception:
            pass
    s = sum(out.values())
    if s <= 0:
        return {}
    return {k: v / s for k, v in out.items()}


def _load_thresholds(path: Path) -> Dict[str, float]:
    """
    Читает пороги. Поддерживает:
      { "bestF1": 0.12, "hi_recall": 0.056 }
      { "thr_bestF1_valid": 0.12, "thr_highRecall_valid": 0.056 }
    Возвращает словарь с дублями-синонимами: {"bestF1": ..., "hi_recall": ..., "hi-recall": ...}
    """
    try:
        t = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(t, dict):
            raise ValueError("thresholds json is not a dict")
    except Exception as e:
        log.warning(f"Thresholds not found/failed to read '{path}': {e!r}. Using defaults 0.5.")
        t = {}

    best = t.get("bestF1", t.get("thr_bestF1_valid", 0.5))
    hire = t.get("hi_recall", t.get("hi-recall", t.get("thr_highRecall_valid", 0.5)))
    try:
        best = float(best)
    except Exception:
        best = 0.5
    try:
        hire = float(hire)
    except Exception:
        hire = 0.5

    return {"bestF1": best, "hi_recall": hire, "hi-recall": hire}


# ------------------------------------ Loader ------------------------------------

class Artifacts:
    """
    Загружает препроцессор, модели и метаданные бленда.
    Приоритет XGB/LGB: (Booster + Isotonic) → legacy Calibrated .pkl → недоступно.
    CatBoost ожидаем как sklearn-совместимый калиброванный классификатор (.pkl).
    """
    _instance: Optional["Artifacts"] = None

    def __init__(self) -> None:
        # --- Препроцессор и порядок признаков
        self.prep = Preprocessor(PREP_META_PATH, FREQ_MAPS_PATH)
        self.feature_order = self.prep.feature_order

        # --- Пороговые режимы
        self.thresholds: Dict[str, float] = _load_thresholds(THRESHOLDS_PATH)

        # --- Веса бленда (нормируем позже только по активным моделям)
        self._weights_all: Dict[str, float] = _load_blend_weights(WEIGHTS_PATH)

        # --- CatBoost (sklearn-совместимый pkl)
        self.model_cat = None
        try:
            if CAT_PATH.exists():
                self.model_cat = joblib.load(CAT_PATH)
                self.model_cat = _patch_calibrated_estimator_compat(self.model_cat)
                # sanity: должна быть predict_proba
                if not hasattr(self.model_cat, "predict_proba"):
                    raise TypeError("Loaded CatBoost calibrated model has no predict_proba")
                log.info(f"CatBoost: loaded {CAT_PATH.name} (compat-patched)")
            else:
                log.info("CatBoost: artifact not found.")
        except Exception as e:
            log.warning(f"CatBoost load failed: {e!r}")
            self.model_cat = None

        # --- XGBoost: booster+isotonic (предпочтительно) или legacy .pkl
        self.model_xgb = None  # legacy CalibratedClassifierCV (если понадобится)
        self.xgb_booster = None
        self.xgb_cal = None
        try:
            if xgb is not None and XGB_JSON.exists() and XGB_CAL.exists():
                bst = xgb.Booster()
                bst.load_model(str(XGB_JSON))
                self.xgb_booster = bst
                self.xgb_cal = joblib.load(XGB_CAL)
                # sanity
                if not hasattr(self.xgb_cal, "predict") or not callable(getattr(self.xgb_cal, "predict")):
                    raise TypeError("XGB calibrator has no callable .predict")
                log.info(f"XGB: loaded booster={XGB_JSON.name}, calibrator={XGB_CAL.name}")
            elif XGB_PATH.exists():
                self.model_xgb = joblib.load(XGB_PATH)
                self.model_xgb = _patch_calibrated_estimator_compat(self.model_xgb)
                if not hasattr(self.model_xgb, "predict_proba"):
                    raise TypeError("Legacy XGB model has no predict_proba")
                log.info(f"XGB: loaded legacy calibrated {XGB_PATH.name}")
            else:
                log.info("XGB: artifacts not found.")
        except Exception as e:
            log.warning(f"XGB load failed: {e!r}")
            self.model_xgb = None
            self.xgb_booster = None
            self.xgb_cal = None

        # --- LightGBM: booster+isotonic (предпочтительно) или legacy .pkl
        self.model_lgb = None  # legacy CalibratedClassifierCV (если понадобится)
        self.lgb_booster = None
        self.lgb_cal = None
        try:
            if lgb is not None and LGB_TXT.exists() and LGB_CAL.exists():
                self.lgb_booster = lgb.Booster(model_file=str(LGB_TXT))  # type: ignore[arg-type]
                self.lgb_cal = joblib.load(LGB_CAL)
                if not hasattr(self.lgb_cal, "predict") or not callable(getattr(self.lgb_cal, "predict")):
                    raise TypeError("LGB calibrator has no callable .predict")
                log.info(f"LGB: loaded booster={LGB_TXT.name}, calibrator={LGB_CAL.name}")
            elif LGB_PATH.exists():
                self.model_lgb = joblib.load(LGB_PATH)
                self.model_lgb = _patch_calibrated_estimator_compat(self.model_lgb)
                if not hasattr(self.model_lgb, "predict_proba"):
                    raise TypeError("Legacy LGB model has no predict_proba")
                log.info(f"LGB: loaded legacy calibrated {LGB_PATH.name}")
            else:
                log.info("LGB: artifacts not found.")
        except Exception as e:
            log.warning(f"LGB load failed: {e!r}")
            self.model_lgb = None
            self.lgb_booster = None
            self.lgb_cal = None

        # --- Определяем активные модели
        active: List[str] = []
        if self.model_cat is not None:
            active.append("cat")
        if (self.model_xgb is not None) or (self.xgb_booster is not None and self.xgb_cal is not None):
            active.append("xgb")
        if (self.model_lgb is not None) or (self.lgb_booster is not None and self.lgb_cal is not None):
            active.append("lgb")

        if not active:
            raise RuntimeError("No models available for inference (Cat/XGB/LGB missing).")

        self.active_models: List[str] = active

        # --- Нормируем веса по активным моделям, если файл весов пустой/неполный — равные веса
        picked = {k: self._weights_all.get(k, 0.0) for k in active}
        s = sum(picked.values())
        if s <= 0:
            eq = 1.0 / len(active)
            self.active: List[Tuple[str, float]] = [(k, eq) for k in active]
            log.info(f"No/invalid weights → using equal weights: {self.active}")
        else:
            self.active = [(k, picked[k] / s) for k in active]
            log.info(f"Active models with weights: {self.active}")

    # ------------------------------- Singleton -------------------------------

    @classmethod
    def get(cls) -> "Artifacts":
        if cls._instance is None:
            cls._instance = Artifacts()
        return cls._instance

    # ---------------------------- Thresholds access ---------------------------

    def threshold_for_mode(self, mode: str) -> float:
        m = (mode or "").strip().lower().replace("-", "_")
        if m in ("bestf1", "best_f1"):
            return float(self.thresholds.get("bestF1", 0.5))
        if m in ("hirecall", "hi_recall"):
            return float(self.thresholds.get("hi_recall", self.thresholds.get("hi-recall", 0.5)))
        # дефолт
        return float(self.thresholds.get("bestF1", 0.5))

    # -------------------------- Private helpers -------------------------------

    @staticmethod
    def _to_2d(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return X if X.ndim == 2 else X.reshape(-1, X.shape[-1])

    @staticmethod
    def _apply_calibrator(cal, raw: np.ndarray, name: str) -> np.ndarray:
        """Безопасно вызвать калибратор (IsotonicRegression 1D). При ошибке вернём raw."""
        try:
            pred = cal.predict(raw)  # type: ignore[call-arg]
            pred = np.asarray(pred, dtype=float).ravel()
            return pred
        except Exception as e:
            log.warning(f"{name} calibrator predict failed: {e!r}; using raw probabilities.")
            return raw

    # -------------------------- Private proba calculators ---------------------

    def _proba_cat(self, X: np.ndarray) -> np.ndarray:
        if self.model_cat is None:
            raise RuntimeError("CatBoost model is not loaded")
        X = self._to_2d(X)
        # sklearn-совместимый классификатор
        p = self.model_cat.predict_proba(X)[:, 1]
        return np.asarray(p, dtype=float)

    def _proba_xgb(self, X: np.ndarray) -> np.ndarray:
        X = self._to_2d(X)
        # legacy CalibratedClassifierCV
        if self.model_xgb is not None:
            p = self.model_xgb.predict_proba(X)[:, 1]
            return np.asarray(p, dtype=float)

        # booster + isotonic calibrator
        if self.xgb_booster is None or self.xgb_cal is None or xgb is None:
            raise RuntimeError("XGB artifacts not loaded")

        # DMatrix без имён фич, и отключаем проверку имён при predict
        dmat = xgb.DMatrix(np.asarray(X, dtype=float), feature_names=None)
        raw = np.asarray(self.xgb_booster.predict(dmat, validate_features=False)).ravel()
        raw = np.clip(raw, 1e-9, 1 - 1e-9)
        cal = np.asarray(self.xgb_cal.predict(raw), dtype=float)
        return cal

    def _proba_lgb(self, X: np.ndarray) -> np.ndarray:
        X = self._to_2d(X)
        # legacy CalibratedClassifierCV
        if self.model_lgb is not None:
            p = self.model_lgb.predict_proba(X)[:, 1]
            return np.asarray(p, dtype=float)

        # booster + isotonic calibrator
        if self.lgb_booster is None or self.lgb_cal is None:
            raise RuntimeError("LGB artifacts not loaded")
        raw = np.asarray(self.lgb_booster.predict(X)).ravel()
        raw = np.clip(raw, 1e-9, 1 - 1e-9)
        cal = self._apply_calibrator(self.lgb_cal, raw, "LGB")
        return cal

    # ------------------------------- Public blend -----------------------------

    def predict_proba_blend(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает blended вероятность класса 1.
        """
        parts: List[np.ndarray] = []
        for name, w in self.active:
            if name == "cat":
                p = self._proba_cat(X)
            elif name == "xgb":
                p = self._proba_xgb(X)
            elif name == "lgb":
                p = self._proba_lgb(X)
            else:
                continue
            parts.append(w * np.asarray(p, dtype=float))
        if not parts:
            raise RuntimeError("No active models to blend")
        return np.sum(parts, axis=0)
