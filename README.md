# BlendCAL — Web-Session Conversion Prediction

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-1.38-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="Airflow" src="https://img.shields.io/badge/Apache%20Airflow-2.x-017CEE?logo=apacheairflow&logoColor=white">
  <img alt="Docker" src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-0.24-FA9F1C?logo=scikitlearn&logoColor=white">
  <img alt="CatBoost" src="https://img.shields.io/badge/CatBoost-1.2.7-FFCC00">
  <img alt="XGBoost" src="https://img.shields.io/badge/XGBoost-2.1.4-EB5E28">
  <img alt="LightGBM" src="https://img.shields.io/badge/LightGBM-4.6.0-80C342">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
</p>

## Overview
**BlendCAL** is an end‑to‑end ML project for predicting whether a web session will convert.  
The pipeline covers: feature engineering, an ensemble of **CatBoost + XGBoost + LightGBM** with **isotonic calibration**, a **FastAPI** inference service, a **Streamlit** UI, an **Airflow** DAG for orchestration, and full **Docker** setup.

> Course project (Skillbox, ML specialization).

---

## Data volume
- `ga_sessions.csv` — **1,860,042** rows  
- `ga_hits.csv` — **15,726,470** rows

---

## Project structure
```
final_project/
├─ dags/                          # Airflow DAG (blendcal_inference.py)
├─ project/
│  ├─ modules/                    # ETL & feature pipeline
│  │  ├─ extract_csv.py
│  │  ├─ prepare.py
│  │  └─ ensemble.py
│  ├─ data/                       # raw / landing / staging / predictions
│  └─ artifacts/                  # prep_params.json, freq_maps.json, models, calibration
├─ api/app/                       # FastAPI: main.py, artifacts_loader.py, preprocessor.py
├─ app/                           # Streamlit UI (streamlit_app.py)
├─ docker_airflow/                # Dockerfile + docker-compose for Airflow
├─ docker-compose.yml             # API + UI
├─ requirements-api.txt
├─ requirements-ui.txt
├─ MODEL_INFO.json
├─ VERSION
└─ docker_airflow/README_airflow.md
```

---

## Tech stack
- **ML**: CatBoost, XGBoost, LightGBM (weighted ensemble + isotonic calibration)  
- **Preprocessing**: median imputation, quantile clipping, frequency encoding, cyclic time features (sin/cos)  
- **API**: FastAPI, Pydantic, Uvicorn  
- **UI**: Streamlit  
- **Orchestration**: Airflow (PythonOperator, Docker stack)  
- **Containerization**: Docker, docker‑compose

---

## Quickstart

### 1) API + UI (Docker)
```bash
docker compose up --build
```
- FastAPI Swagger: http://localhost:8000/docs  
- Streamlit: http://localhost:8501

### 2) Airflow (Docker)
See [`docker_airflow/README_airflow.md`](docker_airflow/README_airflow.md) for details. TL;DR:
```powershell
cd docker_airflow
docker compose down -v
docker compose up airflow-init
docker compose up -d webserver scheduler
```
- Airflow UI: http://localhost:8080 (admin / admin)

---

## Results
- ROC‑AUC: **0.86**  
- F1 macro: **0.75**  
- Holdout period: **2021‑11 → 2021‑12**  

Model metadata & artifacts are recorded in `MODEL_INFO.json`.

---

## Useful links
- Airflow how‑to: [`docker_airflow/README_airflow.md`](docker_airflow/README_airflow.md)  
- Model passport: [`MODEL_INFO.json`](MODEL_INFO.json)

---

## Author
- **Konstantin Nikiforov** — Skillbox ML specialization (2025)

## License
This project is licensed under the **MIT License**. See [`LICENSE`](LICENSE) for details.
