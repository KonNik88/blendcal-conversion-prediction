# базовый образ с Python 3.9
FROM apache/airflow:2.9.2-python3.9

ENV PIP_NO_CACHE_DIR=1

# Научный стек в тех же версиях, что в API
# (складывается с артефактами и калибрацией из sklearn 0.24.2)
RUN pip install --no-cache-dir \
    numpy==1.21.6 \
    pandas==1.3.5 \
    scipy==1.7.3 \
    joblib==1.4.2 \
    scikit-learn==0.24.2 \
    # parquet-движок, совместимый с pandas 1.3.x
    pyarrow==6.0.1 \
    # модели
    xgboost==2.1.4 \
    lightgbm==4.6.0 \
    catboost==1.2.7
