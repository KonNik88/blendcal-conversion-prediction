# BlendCAL: Docker quickstart

## Dev mode (build from sources)

1) Project structure:
   - api/app/...
   - app/streamlit_app.py
   - artifacts/ (models, prep, BlendCAL_*.json, logs/)
   - MODEL_INFO.json
   - VERSION

2) Build & run:
   ```powershell
   docker compose up --build ```
3) Open:
API: http://localhost:8000/docs
UI: http://localhost:8501

4) Notes:
The API reads artifacts from FINAL_PROJECT_ROOT/artifacts. Compose mounts ./artifacts into /app/artifacts.
Logger writes jsonl lines into artifacts/logs/{requests,responses}.jsonl (mounted volume).
Update thresholds/weights without rebuilding: edit files under ./artifacts and the API will pick them up on next request.

## Release mode (use prebuilt images)

1) Load release images:
cd D:\ML\LS\final\final_project
docker load -i .\release\images\blendcal_1.0.2.tar

2) Run with release compose:
docker compose -f docker-compose.release.yml up -d
docker compose -f docker-compose.release.yml ps

3) Expected:
network blendcal-net created
blendcal-api → Healthy on port 8000
blendcal-ui → Up on port 8501

4) Smoke tests:

Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/version
Invoke-RestMethod http://localhost:8000/features | ConvertTo-Json -Depth 5

5)Verify prediction (example):

$features = @{
  visit_number=4; visit_hour=20; visit_dow=3; is_weekend=0;
  is_organic=0; is_paid=1; is_social=0; is_email=0; is_push=0; is_sms=0; is_perf=1;
  events_in_session=140; unique_event_types=20; n_brands=6; n_models=10;
  any_car_detail=1; is_big_city=1; is_top_brand=1; has_model=1; is_returning_user=1;
  utm_medium="cpc"; device_category="desktop"; geo_country="Russia"; geo_city="Moscow";
  traffic_group="performance"; car_brand="Toyota"; car_model="Camry"; brand_model="Toyota_Camry"
}
$body = @{ features = $features; mode = "bestF1" } | ConvertTo-Json -Depth 7
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType 'application/json' -Body $body

6) Verify batch predict:
curl.exe -X POST http://localhost:8000/predict_batch -F "file=@tests/batch_demo.csv" -F "mode=bestF1" -o tests/batch_scored.csv
Get-Content tests\batch_scored.csv | Select -First 5

Release archive integrity
File: release/images/blendcal_1.0.2.tar
SHA256: 17ED495DBCD8661DE4B6DEE53D72B44D1EEC9DA30FD8CB4F59B50846FDD3AEB7
Check with:
Get-FileHash .\release\images\blendcal_1.0.2.tar -Algorithm SHA256
Hash must match the value above.

Troubleshooting
Error: training data did not have the following fields ...
→ причина: запущен старый образ (1.0.0) или compose ссылается не на тот тег.

Решение:
# 1) Остановить релизные контейнеры
docker compose -f docker-compose.release.yml down

# 2) Проверить, на какие теги ссылается compose (должно быть 1.0.2)
docker compose -f docker-compose.release.yml config | Select-String "image:"

# 3) Снести старые образы (если остались)
docker image rm -f `
  final_project-api:1.0.0 final_project-ui:1.0.0 `
  final_project-api:1.0.1 final_project-ui:1.0.1

# 4) На всякий случай перезалить нужные (если их нет/повреждены)
docker load -i .\release\images\blendcal_1.0.2.tar

# 5) Поднять 1.0.2
docker compose -f docker-compose.release.yml up -d
docker compose -f docker-compose.release.yml ps

Batch upload fails in PowerShell:
Не используйте Invoke-WebRequest -Form. Используйте curl.exe как в примерах.
Порты 8000/8501 заняты:
Проверьте кто слушает:
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
или остановите старые контейнеры:
docker compose -f docker-compose.release.yml down
UI не видит API:
Убедитесь, что в docker-compose.release.yml для сервиса ui прописана переменная окружения:
BLENDCAL_API=http://api:8000
и сеть blendcal-net активна.edit files under ./artifacts and the API will pick them up on next request.