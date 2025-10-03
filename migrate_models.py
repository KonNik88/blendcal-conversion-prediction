# migrate_models.py  — запускать в lightautoml_env!
from pathlib import Path
import sys, joblib, json
import numpy as np

# 1) --- ШИМЫ для классов из __main__ (чтобы распаковать pickle) ---
# Эти классы только помогают загрузить объект; реальная модель будет внутри.

class BoosterSklearnWrapper:
    def __init__(self, *a, **kw): pass
    def __setstate__(self, state):
        # распакуем как есть
        self.__dict__.update(state)
    # Найти реальный XGB-классификатор/бустер внутри объекта
    def get_booster(self):
        # прямой метод
        if hasattr(self, "booster") and hasattr(self.booster, "save_model"):
            return self.booster
        # Поиск по распространённым атрибутам-обёрткам
        for name in ["model", "clf", "estimator", "wrapped", "base", "xgb"]:
            obj = getattr(self, name, None)
            if obj is None:
                continue
            # XGBClassifier
            if hasattr(obj, "get_booster"):
                return obj.get_booster()
            # Сам бустер
            if hasattr(obj, "save_model"):
                return obj
        # Если сам объект — XGBClassifier
        if hasattr(self, "get_booster"):
            return self.get_booster()
        raise AttributeError("Не удалось найти XGB Booster внутри BoosterSklearnWrapper")

class LGBWrapper:
    def __init__(self, *a, **kw): pass
    def __setstate__(self, state):
        self.__dict__.update(state)
    def get_booster(self):
        # прямой
        if hasattr(self, "booster_"):
            return self.booster_
        # Частые контейнеры
        for name in ["model", "clf", "estimator", "wrapped", "base", "lgb"]:
            obj = getattr(self, name, None)
            if obj is None:
                continue
            if hasattr(obj, "booster_"):
                return obj.booster_
            if hasattr(obj, "save_model") and hasattr(obj, "num_trees"):
                return obj  # уже Booster
        if hasattr(self, "booster_"):
            return self.booster_
        raise AttributeError("Не удалось найти LGB Booster внутри LGBWrapper")

# Подкинем шима-классы в __main__, чтобы unpickle прошёл
sys.modules["__main__"].BoosterSklearnWrapper = BoosterSklearnWrapper
sys.modules["__main__"].LGBWrapper = LGBWrapper

# 2) --- Пути проекта/артефактов ---
ROOT = Path(r"D:\ML\LS\final\final_project")
MODELS = ROOT / "artifacts" / "models"
MODELS.mkdir(parents=True, exist_ok=True)

# СТАРЫЕ pkl (калиброванные)
XGB_OLD = MODELS / "XGBoost_calibrated.pkl"
LGB_OLD = MODELS / "LightGBM_calibrated.pkl"

# НОВЫЕ устойчивые артефакты
XGB_JSON = MODELS / "XGB.booster.json"      # нативный XGB Booster
XGB_CAL  = MODELS / "XGB_isotonic.pkl"      # изотонический калибратор

LGB_TXT  = MODELS / "LGB.booster.txt"       # нативный LGB Booster
LGB_CAL  = MODELS / "LGB_isotonic.pkl"      # изотонический калибратор

# 3) --- Миграция XGB ---
def export_xgb():
    print("[XGB] loading old calibrated model:", XGB_OLD)
    old = joblib.load(XGB_OLD)

    # Попытаемся извлечь base_estimator из CalibratedClassifierCV
    from sklearn.calibration import CalibratedClassifierCV
    if isinstance(old, CalibratedClassifierCV):
        base = old.base_estimator
    else:
        base = getattr(old, "base_estimator", old)

    # Достаём бустер
    booster = None
    # 3.1 прямые пути
    if hasattr(base, "get_booster"):
        booster = base.get_booster()
    elif hasattr(base, "booster"):
        booster = base.booster
    # 3.2 внутри возможных атрибутов
    if booster is None:
        for name in ["model", "clf", "estimator", "wrapped", "base", "xgb"]:
            obj = getattr(base, name, None)
            if obj is None:
                continue
            if hasattr(obj, "get_booster"):
                booster = obj.get_booster()
                break
            if hasattr(obj, "save_model"):
                booster = obj
                break
    if booster is None:
        raise RuntimeError("Не нашёл XGB Booster внутри калиброванной модели")

    # Сохраняем бустер в JSON
    booster.save_model(str(XGB_JSON))
    print("[XGB] saved booster ->", XGB_JSON)

    # Извлекаем изотонический калибратор
    # В sklearn 0.24 структура: calibrated_classifiers_[i].calibrators_[0]
    try:
        cc = old.calibrated_classifiers_[0]
        iso = cc.calibrators_[0]
    except Exception:
        # иногда calibrators хранится в .calibrator  или списком по классам
        iso = getattr(old, "calibrator", None) or getattr(old, "calibrators_", None)
        if isinstance(iso, (list, tuple)):
            iso = iso[0]
        if iso is None:
            raise RuntimeError("Не удалось извлечь изотонический калибратор для XGB")
    joblib.dump(iso, XGB_CAL)
    print("[XGB] saved isotonic calibrator ->", XGB_CAL)

# 4) --- Миграция LGB ---
def export_lgb():
    from sklearn.calibration import CalibratedClassifierCV
    import lightgbm as lgb
    import inspect

    print("[LGB] loading old calibrated model:", LGB_OLD)
    old = joblib.load(LGB_OLD)

    # 1) достаём "базовую" модель из CalibratedClassifierCV
    if isinstance(old, CalibratedClassifierCV):
        base = old.base_estimator
    else:
        base = getattr(old, "base_estimator", old)

    # 2) рекурсивный поиск бустера LightGBM
    seen = set()
    def find_lgb_booster(obj):
        oid = id(obj)
        if oid in seen:
            return None
        seen.add(oid)

        # a) уже Booster?
        try:
            import lightgbm as lgb
            if isinstance(obj, lgb.Booster):
                return obj
        except Exception:
            pass

        # b) LGBMClassifier / LGBMRegressor с booster_?
        if hasattr(obj, "booster_"):
            return obj.booster_

        # c) частые контейнеры sklearn:
        for attr in ("best_estimator_", "estimator", "base_estimator", "final_estimator", "model", "clf", "est", "wrapped", "base", "lgb"):
            if hasattr(obj, attr):
                res = find_lgb_booster(getattr(obj, attr))
                if res is not None:
                    return res

        # d) Pipeline.steps / feature_union / column_transformer
        if hasattr(obj, "steps") and isinstance(obj.steps, (list, tuple)):
            for name, step in obj.steps:
                res = find_lgb_booster(step)
                if res is not None:
                    return res
        if hasattr(obj, "transformer_list") and isinstance(obj.transformer_list, (list, tuple)):
            for name, step in obj.transformer_list:
                res = find_lgb_booster(step)
                if res is not None:
                    return res

        # e) перебор по __dict__/итерируемым
        try:
            for k, v in vars(obj).items():
                res = find_lgb_booster(v)
                if res is not None:
                    return res
        except Exception:
            pass

        if isinstance(obj, (list, tuple, set)):
            for v in obj:
                res = find_lgb_booster(v)
                if res is not None:
                    return res
        if isinstance(obj, dict):
            for v in obj.values():
                res = find_lgb_booster(v)
                if res is not None:
                    return res

        return None

    booster = find_lgb_booster(base)
    if booster is None:
        raise RuntimeError("Не нашёл LGB Booster внутри калиброванной модели (обойдены вложенные контейнеры).")

    # 3) сохраняем бустер
    booster.save_model(str(LGB_TXT))
    print("[LGB] saved booster ->", LGB_TXT)

    # 4) извлекаем изотонический калибратор
    # В 0.24 поле 'calibrators_' (deprecated), в новых — 'calibrators'
    iso = None
    try:
        cc = old.calibrated_classifiers_[0]
        iso = cc.calibrators_[0] if hasattr(cc, "calibrators_") else cc.calibrators[0]
    except Exception:
        # запасные варианты
        iso = getattr(old, "calibrator", None) or getattr(old, "calibrators_", None) or getattr(old, "calibrators", None)
        if isinstance(iso, (list, tuple)):
            iso = iso[0]
    if iso is None:
        raise RuntimeError("Не удалось извлечь изотонический калибратор для LGB")

    joblib.dump(iso, LGB_CAL)
    print("[LGB] saved isotonic calibrator ->", LGB_CAL)

if __name__ == "__main__":
    export_xgb()
    export_lgb()
    print("Done. Теперь используй XGB.booster.json / XGB_isotonic.pkl / LGB.booster.txt / LGB_isotonic.pkl для инференса.")
