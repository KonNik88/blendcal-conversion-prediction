import os
from pathlib import Path

ROOT = Path(os.getenv("FINAL_PROJECT_ROOT", r"D:\ML\LS\final\final_project"))

ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR    = ARTIFACTS_DIR / "models"
PREP_DIR      = ARTIFACTS_DIR / "prep"
LOG_DIR       = ARTIFACTS_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- старые PKL (оставляем для обратной совместимости) ---
CAT_PATH = MODELS_DIR / "CatBoost_calibrated_isotonic.pkl"
XGB_PATH = MODELS_DIR / "XGBoost_calibrated.pkl"
LGB_PATH = MODELS_DIR / "LightGBM_calibrated.pkl"

# --- НОВЫЕ устойчивые артефакты (после миграции) ---
XGB_JSON = MODELS_DIR / "XGB.booster.json"      # нативный XGB Booster
XGB_CAL  = MODELS_DIR / "XGB_isotonic.pkl"      # изотонический калибратор

LGB_TXT  = MODELS_DIR / "LGB.booster.txt"       # нативный LGB Booster
LGB_CAL  = MODELS_DIR / "LGB_isotonic.pkl"      # изотонический калибратор

# --- прочие артефакты ---
WEIGHTS_PATH    = ARTIFACTS_DIR / "BlendCAL_weights.json"
THRESHOLDS_PATH = ARTIFACTS_DIR / "BlendCAL_thresholds.json"

PREP_META_PATH  = PREP_DIR / "prep_params.json"
FREQ_MAPS_PATH  = PREP_DIR / "freq_maps.json"

# --- версия ---
VERSION_FILE  = ROOT / "VERSION"
MODEL_VERSION = os.getenv(
    "MODEL_VERSION",
    VERSION_FILE.read_text(encoding="utf-8").strip() if VERSION_FILE.exists() else "dev"
)
