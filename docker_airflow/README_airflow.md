# BlendCAL Airflow Pipeline

## Требования
- Docker Desktop (Windows)
- Ресурсы Docker: CPU ≥ 4, RAM ≥ 8–12 GB (настрой в Settings → Resources)
- ~25 GB свободно (данные + образы)

## Структура
- `docker_airflow/` — `Dockerfile`, `docker-compose.yml`
- `dags/` — DAG `blendcal_inference.py`
- `project/modules/` — `extract_csv.py`, `prepare.py`, `ensemble.py`
- `api/app/` — прeпроцессор и загрузка артефактов (`artifacts_loader.py`, `preprocessor.py`, `config.py`)
- `project/data/raw/` — положить исходные CSV:
  - `ga_sessions.csv`
  - `ga_hits.csv`
- `project/artifacts/` — положить артефакты:
  - `prep/prep_params.json`
  - `prep/freq_maps.json`
  - (при наличии) `BlendCAL_weights.json`, `BlendCAL_thresholds.json`, модели `CatBoost_*`, `LGB.booster.txt`, `XGB.booster.json` и калибровки

> На Windows пути в `docker-compose.yml` указываются в виде `//d/…`. Внутри контейнера пути — `/opt/airflow/project/...`.

## Запуск
```powershell
cd docker_airflow
docker compose down -v
docker compose up airflow-init
docker compose up -d webserver scheduler ```

Airflow UI: http://localhost:8080
 (логин/пароль: admin/admin).

## Что делает DAG

1 - ingest_sessions → project/data/landing/sessions.parquet

2 - ingest_heats → project/data/landing/heats.parquet

3 - prepare_features → project/data/staging/features.parquet

4 - predict_ensemble → project/data/predictions/preds.csv

5 - archive_outputs → копия предиктов в project/data/predictions/run_id=.../preds.csv

## Проверка результатов
```powershell
# размеры файлов
docker exec -it blendcal-airflow-scheduler-1 ls -lh /opt/airflow/project/data/{landing,staging,predictions}

# sanity-чек для предсказаний
docker exec -it blendcal-airflow-scheduler-1 bash -lc "python - << 'PY'
import pandas as pd
p='/opt/airflow/project/data/predictions/preds.csv'
df=pd.read_csv(p)
print('rows:',len(df),'nunique:',df.session_id.nunique())
print('proba min/max/mean:',df.proba.min(),df.proba.max(),df.proba.mean())
print('y_hat value_counts:',df.y_hat.value_counts().to_dict())
PY" ```

## Типичные проблемы
- FileNotFoundError: /opt/airflow/project/artifacts/...
  Проверь, что в docker-compose.yml смонтирован том:
  - //d/ML/LS/final/final_project/artifacts:/opt/airflow/project/artifacts

- 403 при открытии логов в UI
  Проверь AIRFLOW__WEBSERVER__SECRET_KEY в docker-compose.yml.

- prepare_features крутится долго
  Это нормально (большие данные). Таймаут в DAG стоит 1 час. Для диагностики смотри прогресс‑логи в UI (prepare: ...).

## Чистый запуск
```powershell
# если нужно заново прогнать и очистить прошлые артефакты
Remove-Item ..\project\data\landing\* -Force
Remove-Item ..\project\data\staging\* -Force
Remove-Item ..\project\data\predictions\* -Force ```

## Локальный smoke‑test без Airflow
```powershell
docker exec -it blendcal-airflow-scheduler-1 bash -lc "python - << 'PY'
from app.artifacts_loader import Artifacts
import pandas as pd
X=pd.read_parquet('/opt/airflow/project/data/staging/features.parquet')
art=Artifacts.get(); proba=art.predict_proba(X[:1000])
print('ok:', len(proba), float(proba.mean())) ```


PY"