# BlendCAL release

## Быстрый старт

1. Установите **Docker** и **Docker Compose**.  
2. Загрузите образы:
   ```powershell
   docker load -i release/images/blendcal_1.0.2.tar
3. Убедитесь, что в корне проекта есть папка ./artifacts
(с моделями, порогами, препроцессором, логами).

4. Запустите контейнеры:
docker compose -f docker-compose.release.yml up -d
docker compose -f docker-compose.release.yml ps

Ожидается:
- сеть blendcal-net создана
- blendcal-api → Healthy (порт 8000)
- blendcal-ui → Up (порт 8501)

5. Откройте сервисы:
- API: http://localhost:8000/docs
- UI: http://localhost:8501

## Проверка API

Быстрые smoke-тесты:
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/version
Invoke-RestMethod http://localhost:8000/features | ConvertTo-Json -Depth 5

Проверка предсказания (пример):
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

Проверка пакетного предсказания:
curl.exe -X POST http://localhost:8000/predict_batch -F "file=@tests/batch_demo.csv" -F "mode=bestF1" -o tests/batch_scored.csv
Get-Content tests\batch_scored.csv | Select -First 5

Release archive integrity
- File: release/images/blendcal_1.0.2.tar
- SHA256: 17ED495DBCD8661DE4B6DEE53D72B44D1EEC9DA30FD8CB4F59B50846FDD3AEB7

Проверить совпадение можно так:
- Get-FileHash .\release\images\blendcal_1.0.2.tar -Algorithm SHA256

Логи
- Запросы пишутся в: artifacts/logs/requests.jsonl
- Ответы пишутся в: artifacts/logs/responses.jsonl
- Формат JSONL, каждая строка — отдельный запрос/ответ.

Troubleshooting
- Ошибка: training data did not have the following fields ...
  Причина: запущен старый образ (1.0.0) или compose ссылается на не тот тег.
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

- Batch upload fails в PowerShell
  Не используйте Invoke-WebRequest -Form. Вместо этого используйте curl.exe.

- Порты 8000/8501 заняты
  Проверить:
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

Остановить старые контейнеры:
docker compose -f docker-compose.release.yml down

- UI не видит API
Убедитесь, что в docker-compose.release.yml для сервиса ui прописано:
BLENDCAL_API=http://api:8000
и сеть blendcal-net активна.

Дополнительно
Для подробного dev-режима и отладки см. dockerreadme.md

## Release 1.0.2 (2025‑08‑23)
- Images: final_project-api:1.0.2, final_project-ui:1.0.2
- TAR: release/images/blendcal_1.0.2.tar
- SHA256: <вставьте из Get-FileHash>
- Compose: docker-compose.release.yml (теги 1.0.2)
- Smoke: /health, /version, /features, /predict, /predict_batch — OK