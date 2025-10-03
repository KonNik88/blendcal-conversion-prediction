import os, sys
from pathlib import Path

PROJECT_ROOT = os.environ.get("FINAL_PROJECT_ROOT", "/opt/airflow/project")

for p in [PROJECT_ROOT, str(Path(PROJECT_ROOT) / "api")]:
    if p not in sys.path:
        sys.path.insert(0, p)

def predict_batch(features_path: str, output_csv: str):
    # ЛЕНИВЫЕ импорты → парсер DAG не падает, если libs не стоят
    import pandas as pd
    from app.artifacts_loader import Artifacts

    art = Artifacts.get()
    X_raw = pd.read_parquet(features_path)

    X_df = art.prep.transform_df(X_raw)
    X_np = X_df.to_numpy(dtype=float, copy=False)

    probas = art.predict_proba_blend(X_np)
    thr = art.threshold_for_mode("bestF1")
    y_hat = (probas >= thr).astype(int)

    from pathlib import Path
    import pandas as pd
    out = pd.DataFrame({
        "session_id": X_raw.get("session_id"),
        "proba": probas,
        "y_hat": y_hat
    })
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

