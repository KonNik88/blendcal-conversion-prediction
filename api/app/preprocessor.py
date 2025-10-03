# api/app/preprocessor.py
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

class Preprocessor:
    def __init__(self, prep_meta_path: Path, freq_maps_path: Path):
        meta = json.loads(Path(prep_meta_path).read_text(encoding="utf-8"))

        self.num_cols: List[str] = meta["num_cols"]
        self.cat_cols: List[str] = meta["cat_cols"]
        self.clip_cols: List[str] = meta["clip_cols"]
        self.no_clip = set(meta.get("no_clip", []))

        self.num_medians = pd.Series({k: float(v) for k, v in meta["num_medians"].items()}, dtype=float)
        self.quantiles = {k: (float(v["lo"]), float(v["hi"])) for k, v in meta["quantiles"].items()}

        fm = json.loads(Path(freq_maps_path).read_text(encoding="utf-8"))
        self.freq_maps: Dict[str, pd.Series] = {c: pd.Series(m) for c, m in fm.items()}

        self.feature_order = self.num_cols + self.cat_cols

    def _ensure_all_features(self, X: pd.DataFrame) -> pd.DataFrame:
        for c in self.feature_order:
            if c not in X.columns:
                X[c] = 0.0
        return X[self.feature_order].astype(float)

    def transform_df(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        X = df_raw.copy()

        # --- numeric: всегда работаем с Series и создаём отсутствующие столбцы
        for c in self.num_cols:
            if c not in X.columns:
                X[c] = np.nan
            s = pd.to_numeric(X[c], errors="coerce")
            X[c] = s.fillna(self.num_medians.get(c, 0.0)).astype(float)

        # --- clipping хвостов по train-квантилям
        for c in self.clip_cols:
            lo, hi = self.quantiles[c]
            # столбец гарантированно есть (см. цикл выше), значит это Series
            X[c] = X[c].clip(lo, hi)

        # --- categorical → frequency encoding (unseen -> 0.0)
        for c in self.cat_cols:
            if c not in X.columns:
                X[c] = "__MISSING__"
            s = X[c].astype(str).fillna("__MISSING__")
            fmap = self.freq_maps[c]
            X[c] = s.map(fmap).astype(float).fillna(0.0)

        # --- порядок и тип
        return self._ensure_all_features(X)

    def transform_payload(self, payload: Dict[str, Any]) -> pd.DataFrame:
        return self.transform_df(pd.DataFrame([payload]))
