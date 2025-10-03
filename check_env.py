# check_env.py
import sys, json
import numpy as np, pandas as pd

print("Python:", sys.version)
import sklearn, xgboost, lightgbm, catboost
print("sklearn:", sklearn.__version__)
print("xgboost:", xgboost.__version__)
print("lightgbm:", lightgbm.__version__)
print("catboost:", catboost.__version__)

# пути под себя
from api.app.artifacts_loader import Artifacts
from api.app.config import PREP_META_PATH, FREQ_MAPS_PATH, WEIGHTS_PATH, THRESHOLDS_PATH

print("\n[DRY-RUN] loading artifacts...")
art = Artifacts.get()
print("feature_order:", len(art.feature_order))

# микропейлоад — заведомо «бедный»: препроцессор сам добьёт недостающие
payload = {
    "utm_source": "referral",
    "device_category": "desktop",
    "geo_city": "Moscow",
    "visit_hour": 13,
    "visit_dow": 2,
    "events_in_session": 3,
}

print("[DRY-RUN] transform & predict...")
X = art.prep.transform_payload(payload)
p = float(art.predict_proba_blend(X)[0])
print("proba:", round(p, 6))

thr_map = {k.lower(): v for k,v in json.load(open(THRESHOLDS_PATH, "r", encoding="utf-8")).items()}
thr = float(thr_map.get("bestf1", 0.5))
print("y_hat:", int(p >= thr))
print("\nOK: environment is COMPATIBLE.")
