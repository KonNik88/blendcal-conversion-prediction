# BlendCAL — предсказание конверсии автосессий

## О проекте
**BlendCAL** — учебный end-to-end проект по машинному обучению (Skillbox, ML-специализация).  
Задача: по данным веб-сессий предсказать, совершит ли пользователь целевое действие (конверсию).  

Мы построили **полный ML-pipeline**:  
1. Подготовка данных и фич (sessions + hits).  
2. Обучение ансамбля (CatBoost + XGBoost + LightGBM с калибровкой).  
3. Сервис FastAPI для инференса.  
4. Streamlit UI для удобного тестирования.  
5. Airflow DAG для оркестрации (ingest → features → predict → archive).  
6. Docker-окружение для всего пайплайна.  

---

## Структура проекта
final_project/
│
├─ dags/ # Airflow DAG (blendcal_inference.py)
├─ project/
│ ├─ modules/ # ETL и фичепайплайн
│ │ ├─ extract_csv.py
│ │ ├─ prepare.py
│ │ └─ ensemble.py
│ ├─ data/ # raw/landing/staging/predictions
│ └─ artifacts/ # prep_params.json, freq_maps.json, модели, калибровки
│
├─ screenshots # скриншоты UI, API, DAG
│
├─ api/app/ # FastAPI код: main.py, artifacts_loader.py, preprocessor.py
├─ app/ # Streamlit UI (streamlit_app.py)
│
├─ docker_airflow/ # Dockerfile + docker-compose для Airflow
├─ docker-compose.yml # для API + UI
├─ requirements-api.txt
├─ requirements-ui.txt
├─ MODEL_INFO.json
├─ VERSION
└─ README_airflow.md # отдельное руководство по Airflow

## Технологии
- **ML**: CatBoost, XGBoost, LightGBM (взвешенный ансамбль + изотоническая калибровка).  
- **Препроцессинг**: медианные импуты, квантильный клиппинг, frequency encoding, синусы/косинусы.  
- **API**: FastAPI, Pydantic, Uvicorn.  
- **UI**: Streamlit.  
- **Оркестратор**: Airflow (PythonOperator, Docker stack).  
- **Контейнеризация**: Docker, docker-compose.  

## Запуск

### 1. API + UI локально в Docker
```bash
docker compose up --build
FastAPI: http://localhost:8000/docs

Streamlit: http://localhost:8501

2. Airflow DAG
powershell

cd docker_airflow
docker compose down -v
docker compose up airflow-init
docker compose up -d webserver scheduler
Airflow UI: http://localhost:8080 (логин/пароль: admin/admin)

Результаты
ROC-AUC: 0.86

F1 macro: 0.75

Holdout период: 2021-11 → 2021-12

Метрики и артефакты зафиксированы в MODEL_INFO.json.

Скриншоты
Streamlit UI
FastAPI
Airflow DAG

Полезные ссылки
README Airflow — отдельные инструкции по пайплайну.
MODEL_INFO.json — паспорт модели.

Авторы
Константин Никифоров
ML-специализация, Skillbox (2025)