# dags/blendcal_inference.py
from __future__ import annotations

import os
import sys
import shutil
import logging
import datetime as dt
from datetime import timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, get_current_context

# --- Логи (видно в Airflow UI)
log = logging.getLogger("airflow.task")
if not log.handlers:
    logging.basicConfig(level=logging.INFO)

# --- Корень проекта внутри КОНТЕЙНЕРА
PROJECT_ROOT = os.environ.get("FINAL_PROJECT_ROOT", "/opt/airflow/project")

# --- PYTHONPATH для импортов модулей проекта
for p in [
    PROJECT_ROOT,
    str(Path(PROJECT_ROOT) / "modules"),
    str(Path(PROJECT_ROOT) / "api"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

# --- Импорты бизнес-логики (оставляю твои имена функций)
from extract_csv import ingest_sessions_csv, ingest_heats_csv
from prepare import build_featureset
from ensemble import predict_batch

# --- Полезные пути
DATA_DIR = f"{PROJECT_ROOT}/data"
LANDING = f"{DATA_DIR}/landing"
STAGING = f"{DATA_DIR}/staging"
PRED_DIR = f"{DATA_DIR}/predictions"
RAW_DIR = f"{DATA_DIR}/raw"

def _ensure_dirs() -> None:
    for p in (LANDING, STAGING, PRED_DIR):
        Path(p).mkdir(parents=True, exist_ok=True)

# --- Обёртки, чтобы добавить mkdir + логи, не меняя твою логику
def _task_ingest_sessions(src_csv: str, dst_parquet: str, **_):
    _ensure_dirs()
    log.info("ingest_sessions: %s -> %s", src_csv, dst_parquet)
    ingest_sessions_csv(src_csv=src_csv, dst_parquet=dst_parquet)

def _task_ingest_heats(src_csv: str, dst_parquet: str, **_):
    _ensure_dirs()
    log.info("ingest_heats: %s -> %s", src_csv, dst_parquet)
    ingest_heats_csv(src_csv=src_csv, dst_parquet=dst_parquet)

def _task_prepare_features(sessions_path: str, heats_path: str, cars_path, output_path: str, **_):
    _ensure_dirs()
    log.info("prepare_features: sessions=%s heats=%s -> %s", sessions_path, heats_path, output_path)
    build_featureset(
        sessions_path=sessions_path,
        heats_path=heats_path,
        cars_path=cars_path,
        output_path=output_path,
    )

def _task_predict_ensemble(features_path: str, output_csv: str, **_):
    _ensure_dirs()
    log.info("predict_ensemble: %s -> %s", features_path, output_csv)
    predict_batch(features_path=features_path, output_csv=output_csv)

def _task_archive_outputs():
    """Копирует preds.csv в подпапку с run_id, чтобы не перетёрся результат предыдущего запуска."""
    _ensure_dirs()
    ctx = get_current_context()
    run_id = ctx.get("run_id")
    src = Path(PRED_DIR) / "preds.csv"
    if not src.exists():
        log.info("archive_outputs: nothing to archive (no preds.csv)")
        return
    dst_dir = Path(PRED_DIR) / f"run_id={run_id}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / "preds.csv")
    log.info("archive_outputs: saved %s", dst_dir / "preds.csv")

# --- Аргументы DAG
default_args = dict(
    owner="airflow",
    start_date=dt.datetime(2025, 8, 1),  # уже в прошлом — ок для ручного запуска
    retries=0,
)

with DAG(
    dag_id="blendcal_inference_csv_once",
    default_args=default_args,
    schedule_interval=None,   # запуск вручную через Trigger
    catchup=False,
    max_active_runs=1,
    tags=["blendcal", "inference"],
) as dag:

    t_sessions = PythonOperator(
        task_id="ingest_sessions",
        python_callable=_task_ingest_sessions,
        op_kwargs=dict(
            src_csv=f"{RAW_DIR}/ga_sessions.csv",
            dst_parquet=f"{LANDING}/sessions.parquet",
        ),
        execution_timeout=timedelta(minutes=10),
    )

    t_heats = PythonOperator(
        task_id="ingest_heats",
        python_callable=_task_ingest_heats,
        op_kwargs=dict(
            src_csv=f"{RAW_DIR}/ga_hits.csv",
            dst_parquet=f"{LANDING}/heats.parquet",
        ),
        execution_timeout=timedelta(minutes=30),  # hits большой CSV
    )

    t_prepare = PythonOperator(
        task_id="prepare_features",
        python_callable=_task_prepare_features,
        op_kwargs=dict(
            sessions_path=f"{LANDING}/sessions.parquet",
            heats_path=f"{LANDING}/heats.parquet",
            cars_path=None,
            output_path=f"{STAGING}/features.parquet",
        ),
        execution_timeout=timedelta(hours=1),     # самый тяжёлый шаг
        retries=0,
    )

    t_predict = PythonOperator(
        task_id="predict_ensemble",
        python_callable=_task_predict_ensemble,
        op_kwargs=dict(
            features_path=f"{STAGING}/features.parquet",
            output_csv=f"{PRED_DIR}/preds.csv",
        ),
        execution_timeout=timedelta(minutes=10),
    )

    t_archive = PythonOperator(
        task_id="archive_outputs",
        python_callable=_task_archive_outputs,
        # выполняем всегда, даже если предикт упал: архиватор безопасный
        trigger_rule="all_done",
        execution_timeout=timedelta(minutes=2),
    )

    [t_sessions, t_heats] >> t_prepare >> t_predict >> t_archive
