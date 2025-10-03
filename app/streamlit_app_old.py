import os
import json
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

def _safe_rerun():
    # Streamlit ≥ 1.40
    try:
        import streamlit as st
        st.rerun()
        return
    except Exception:
        pass
    # Старые версии (на всякий случай)
    try:
        st.experimental_rerun()
    except Exception:
        pass

def number_or_none(col, name, initial=None, help_text=None):
    raw = col.text_input(
        name,
        value=("" if initial is None else str(initial)),
        help=help_text,
        placeholder="число",
    ).strip().replace(",", ".")
    if raw == "":
        return None
    try:
        v = float(raw)
        return int(v) if v.is_integer() else v
    except Exception:
        st.warning(f"«{name}»: не удалось преобразовать «{raw}» в число — оставляю пустым.")
        return None

# ---------- Paths & defaults ----------
API_DEFAULT = os.getenv("BLENDCAL_API", "http://localhost:8000")
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO1 = ASSETS_DIR / "logo1.png"  # car
LOGO2 = ASSETS_DIR / "logo2.jpg"  # ML

st.set_page_config(page_title="BlendCAL — предсказание конверсии", layout="wide")

# ---------- Light theming via CSS ----------
st.markdown(
    """
<style>
:root {
  --primary: #5B67F1;
  --bg: #FFFFFF;
  --bg2: #F6F7FF;
  --text: #1A1B22;
  --muted: #556;
}
html, body, [data-testid="stAppViewContainer"] {
  background-color: var(--bg);
}
.main .block-container { padding-top: 1rem; }
.header {
  display: flex; align-items: center; gap: 16px;
  padding: 8px 0 10px 0; border-bottom: 1px solid #eaeaea;
}
.header h1 { margin: 0; font-size: 1.6rem; }
.badge {
  background: #EEF; border: 1px solid #DDE; padding: 2px 10px; border-radius: 999px;
  font-size: 12px; color: var(--muted);
}
.smallcaps { font-variant: all-small-caps; letter-spacing: .6px; color: var(--muted); }
hr.soft { border: none; border-top: 1px solid #ececf1; margin: .6rem 0 1.0rem 0; }
.card {
  background: var(--bg2); border: 1px solid #E6E8FF; border-radius: 12px; padding: 10px 12px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Header with two logos ----------
col_logo1, col_title, col_logo2 = st.columns([1, 5, 1], vertical_alignment="center")
with col_logo1:
    if LOGO1.exists():
        st.image(str(LOGO1), use_container_width=True)
with col_title:
    st.markdown(
        '<div class="header">'
        '<h1>🔮 BlendCAL — предсказание конверсии</h1>'
        '<span class="badge">UI over FastAPI</span>'
        "</div>",
        unsafe_allow_html=True,
    )
with col_logo2:
    if LOGO2.exists():
        st.image(str(LOGO2), use_container_width=True)

st.markdown('<hr class="soft"/>', unsafe_allow_html=True)

# ---------- Sidebar: API, threshold, model passport ----------
with st.sidebar:
    st.header("Настройки")
    api_url = st.text_input("API URL", value=API_DEFAULT, help="Адрес FastAPI сервиса")
    MODE_OPTIONS = ["bestF1", "hi_recall"]
    mode = st.radio("Режим порога", MODE_OPTIONS, index=0, help="Порог классификации")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Проверить API", use_container_width=True):
            try:
                ping = requests.get(f"{api_url}/health", timeout=5).json()
                st.success(ping)
            except Exception as e:
                st.error(e)
    with colB:
        refresh = st.button("Обновить версию", use_container_width=True)

    # Паспорт модели из /version (читает MODEL_INFO.json + active_models)
    model_info = {}
    try:
        model_info = requests.get(f"{api_url}/version", timeout=8).json()
    except Exception as e:
        if refresh:
            st.warning(f"Версия недоступна: {e}")

    if model_info:
        st.markdown("### Паспорт модели")
        v = model_info.get("version", "—")
        author = model_info.get("author", model_info.get("owner", "—"))
        date = model_info.get("date", "—")
        desc = model_info.get("description", "")
        active = model_info.get("active_models", None)  # если добавлено в API
        st.write(f"**Версия:** {v}")
        st.write(f"**Автор:** {author}")
        st.write(f"**Дата:** {date}")
        if active:
            st.write(f"**Ансамбль:** {', '.join(active)}")
        if desc:
            st.caption(desc)

# ---------- Helper: download CSV template from /features ----------
with st.expander("Как вводить данные?"):
    st.markdown(
        """
- **Single**: заполни поля и/или вставь **JSON** (utm_*, device_*, geo_*, поведенческие).
- **Batch CSV**: загрузи CSV с «сырыми» колонками (как в train). На выходе получишь `proba` и `y_hat`.
- Не уверен в составе колонок — скачай **шаблон CSV** (строка-заголовок, без данных).
        """
    )
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        if st.button("Скачать шаблон CSV", use_container_width=True):
            try:
                feats = requests.get(f"{api_url}/features", timeout=10).json()
                order = feats.get("feature_order", [])
                if not order:
                    st.warning("API не вернул feature_order.")
                else:
                    header = ",".join(order)
                    st.download_button(
                        "⬇️ template.csv",
                        data=(header + "\n").encode("utf-8"),
                        file_name="template.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(e)
    with tcol2:
        st.markdown(
            '<div class="card">'
            '<div class="smallcaps">Подсказка</div>'
            'Пустые значения допустимы — препроцессор заполнит медианами/0.0 и обрежет хвосты по квантилям.'
            "</div>",
            unsafe_allow_html=True,
        )

# ---------- Tabs ----------
tab_single, tab_batch, tab_about = st.tabs(["Single", "Batch CSV", "Описание"])

# === TAB 1: SINGLE ===
with tab_single:
    st.subheader("Single предсказание")
    st.caption("Заполни поля ниже и/или подмешай JSON. Чем больше полей — тем точнее. Пропуски заимпутятся.")

    # --- схема признаков из API ---
    try:
        feats = requests.get(f"{api_url}/features", timeout=10).json()
        num_cols = feats.get("num_cols", [])
        cat_cols = feats.get("cat_cols", [])
        feature_order = feats.get("feature_order", [])
    except Exception as e:
        st.error(f"Не удалось получить /features: {e}")
        num_cols, cat_cols, feature_order = [], [], []

    LABELS = {
        "utm_source": "Источник трафика (utm_source)",
        "utm_medium": "Тип трафика / медиум (utm_medium)",
        "utm_campaign": "Кампания (utm_campaign)",
        "traffic_group": "Группа трафика (агрегация)",
        "device_category": "Тип устройства",
        "geo_country": "Страна",
        "geo_city": "Город",
        "visit_number": "№ визита пользователя",
        "visit_hour": "Час визита (0–23)",
        "visit_dow": "День недели (0–6)",
        "events_in_session": "Событий в сессии",
        "unique_event_types": "Уникальных типов событий",
        "n_brands": "Сколько брендов замечено",
        "n_models": "Сколько моделей замечено",
        "car_brand": "Бренд авто",
        "car_model": "Модель авто",
        "brand_model": "Бренд+модель (склейка)",
        "is_top_brand": "Топ-бренд? (0/1)",
        "has_model": "Указана модель? (0/1)",
        "is_big_city": "Крупный город (Мск/СПб)? (0/1)",
        "is_organic": "Органический трафик? (0/1)",
        "is_paid": "Платный трафик? (0/1)",
        "is_social": "Соцсети? (0/1)",
        "is_email": "Email? (0/1)",
        "is_push": "Push? (0/1)",
        "is_sms": "SMS? (0/1)",
        "is_perf": "Performance? (0/1)",
        "hour_sin": "sin(2π·hour/24)",
        "hour_cos": "cos(2π·hour/24)",
        "dow_sin": "sin(2π·dow/7)",
        "dow_cos": "cos(2π·dow/7)",
    }

    HELP = {
        "utm_medium": "Канал: cpc/banner/email/organic/referral/(none)/push/sms/social…",
        "utm_source": "Источник: google/yandex/… (если есть).",
        "traffic_group": "Агрегированная группа трафика: search/banner/social/email/push/sms/referral/perf…",
        "device_category": "desktop / mobile / tablet.",
        "geo_country": "Страна (как в train).",
        "geo_city": "Город (редкие — other_city).",
        "visit_hour": "Час визита [0–23].",
        "visit_dow": "День недели [0–6], где 0 — понедельник (как в train).",
        "events_in_session": "Число событий в сессии (≥0).",
        "unique_event_types": "Число уникальных типов событий (≥0).",
        "n_brands": "Сколько разных брендов замечено (≥0).",
        "n_models": "Сколько разных моделей замечено (≥0).",
        "car_brand": "Извлечённый бренд авто или other_brand.",
        "car_model": "Извлечённая модель авто или other_model.",
        "brand_model": "Склейка brand_model: toyota_camry / other_brand_other_model.",
        "is_top_brand": "Флаг топ-бренда (0/1).",
        "has_model": "Указана конкретная модель? (0/1).",
        "is_big_city": "Москва/СПб (и обл.) — 1, иначе 0.",
        "is_organic": "Органический трафик (0/1).",
        "is_paid": "Платный (0/1).",
    }

    with st.expander("📚 Справочник признаков (что это такое)"):
        rows = []
        for c in (cat_cols + num_cols):
            rows.append({
                "Признак": c,
                "Название": LABELS.get(c, c),
                "Описание": HELP.get(c, "—"),
                "Тип": ("числовой" if c in num_cols else "категория/флаг")
            })
        if rows:
            df_help = pd.DataFrame(rows, columns=["Признак", "Название", "Описание", "Тип"])
            st.dataframe(df_help, use_container_width=True, hide_index=True)

    CHOICES = {
        "device_category": ["desktop", "mobile", "tablet"],
        "utm_medium": ["cpc", "banner", "email", "organic", "referral", "(none)", "push", "sms", "social"],
        "traffic_group": ["search", "banner", "social", "email", "push", "sms", "referral", "perf"],
    }
    FLAG_COLS = {c for c in cat_cols if c.startswith("is_")}

    # --- пресеты для быстрой проверки ---
    def preset_high_cr():
        # Обновлённый «тёплый» пресет (давал y_hat=1 в debug)
        return {
            "utm_medium": "cpc",
            "traffic_group": "performance" if "performance" in CHOICES.get("traffic_group", []) else "perf",
            "device_category": "mobile",
            "geo_country": "Russia",
            "geo_city": "Moscow",
            "is_big_city": 1,

            "visit_hour": 14,
            "visit_dow": 2,

            "events_in_session": 80,
            "unique_event_types": 30,
            "visit_number": 20,
            "n_brands": 10,
            "n_models": 8,

            "car_brand": "other_brand",
            "car_model": "other_model",
            "brand_model": "other_brand_other_model",

            "is_top_brand": 1,
            "has_model": 1,

            "is_organic": 0,
            "is_paid": 1,
            "is_social": 0,
            "is_email": 0,
            "is_push": 1,
            "is_sms": 1,
            "is_perf": 1,
        }

    def preset_low_cr():
        return {
            "device_category": "tablet",
            "utm_medium": "social",
            "traffic_group": "social",
            "geo_country": "Russia",
            "geo_city": "other_city",
            "is_big_city": 0,
            "visit_hour": 2,
            "visit_dow": 6,
            "events_in_session": 1,
            "unique_event_types": 1,
            "visit_number": 1,
            "n_brands": 0,
            "n_models": 0,
            "car_brand": "other_brand",
            "car_model": "other_model",
            "brand_model": "other_brand_other_model",
            "is_top_brand": 0,
            "has_model": 0,
            "is_organic": 0,
            "is_paid": 1,
            "is_social": 1,
            "is_email": 0,
            "is_push": 0,
            "is_sms": 0,
            "is_perf": 0,
        }

    # --- state и пресеты ---
    if "form_values" not in st.session_state:
        st.session_state.form_values = {}

    col_p1, col_p2, col_p3, col_p4 = st.columns([1,1,2,2])
    with col_p1:
        if st.button("⚡ High‑CR пресет"):
            st.session_state.form_values = {**st.session_state.form_values, **preset_high_cr()}
            _safe_rerun()
    with col_p2:
        if st.button("🧊 Low‑CR пресет"):
            st.session_state.form_values = {**st.session_state.form_values, **preset_low_cr()}
            _safe_rerun()

    # ---- Auto‑warm: подтянуть top-категории из /cat_top и «разогреть» числа
    CAT_COLS_FOR_WARM = ["utm_medium", "traffic_group", "device_category", "geo_city", "car_brand", "car_model", "brand_model"]

    def get_top(api: str, col: str, k: int = 1):
        try:
            r = requests.get(f"{api}/cat_top", params={"col": col, "k": k}, timeout=10)
            r.raise_for_status()
            data = r.json()
            top = data.get("top") or []
            return top[0] if top else None
        except Exception:
            return None

    with col_p3:
        if st.button("⚡ Auto‑warm from /cat_top"):
            changed = 0
            for c in CAT_COLS_FOR_WARM:
                v = get_top(api_url, c, 1)
                if v is not None:
                    st.session_state.form_values[c] = v
                    changed += 1
            # усилить поведенческие
            warm_nums = {"events_in_session": 80, "unique_event_types": 30, "visit_number": 20, "n_brands": 10, "n_models": 8}
            st.session_state.form_values.update(warm_nums)
            st.success(f"Подставлены top‑категории ({changed}) и тёплые числовые значения.")
            _safe_rerun()
    with col_p4:
        auto_trig = st.checkbox("Автосчитать hour_sin/hour_cos и dow_sin/dow_cos из hour/dow", value=True)

    # --- JSON «подмешать» ---
    with st.expander("Дополнительно: подмешать JSON"):
        allow_json_override = st.checkbox(
            "Разрешить JSON переопределять режим порога (mode)", value=False
        )
        example_json = '{\n  "features": {"events_in_session": 3, "utm_source": "google"}\n}'
        extra_json = st.text_area("features JSON", value=example_json, height=140)
        up = st.file_uploader("или загрузить JSON-файл", type=["json"])
        extra_dict = {}
        if up is not None:
            try:
                extra_dict = json.loads(up.getvalue())
                st.success("JSON загружен.")
            except Exception as e:
                st.error(f"Ошибка JSON: {e}")
        try:
            if extra_json.strip():
                extra_dict.update(json.loads(extra_json))
        except Exception as e:
            st.warning(f"Игнорирую текстовый JSON: {e}")

    # --- собственно форма ---
    with st.form("single_form"):
        # Группируем поля на блоки для удобства
        block1 = ["utm_source", "utm_medium", "traffic_group", "device_category"]
        block2 = ["geo_country", "geo_city", "is_big_city"]
        block3 = ["visit_hour", "visit_dow", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        block4 = ["events_in_session", "unique_event_types", "visit_number", "n_brands", "n_models"]
        block5 = ["car_brand", "car_model", "brand_model", "is_top_brand", "has_model"]
        known = set(block1 + block2 + block3 + block4 + block5)
        rest = [c for c in (cat_cols + num_cols) if c not in known]

        def put_input(col, name):
            initial = st.session_state.form_values.get(name, None)
            label = LABELS.get(name, name)
            base_help = HELP.get(name, "Признак из train. Можно оставить пустым — заимпутится/закодируется.")
            help_text = f"{base_help}\n\nКод признака: `{name}`"

            if name in FLAG_COLS:
                val = col.selectbox(label, [None, 0, 1],
                                    index=[None, 0, 1].index(initial) if initial in (None, 0, 1) else 0,
                                    help=help_text)
            elif name in CHOICES:
                opts = [None] + CHOICES[name]
                val = col.selectbox(label, opts,
                                    index=opts.index(initial) if initial in opts else 0,
                                    help=help_text)
            elif name in num_cols:
                val = number_or_none(col, label, initial=initial, help_text=help_text)
            else:
                val = col.text_input(label, value=initial if isinstance(initial, str) else "",
                                     help=help_text, placeholder="строка")
            if val == "" or val is None:
                return None
            return int(val) if (name in FLAG_COLS and val in (0, 1)) else val

        st.markdown("### Трафик")
        c1, c2, c3, c4 = st.columns(4)
        for nm, col in zip(block1, [c1, c2, c3, c4]):
            if nm in (cat_cols + num_cols): st.session_state.form_values[nm] = put_input(col, nm)

        st.markdown("### Гео")
        c1, c2, c3 = st.columns(3)
        for nm, col in zip(block2, [c1, c2, c3]):
            if nm in (cat_cols + num_cols): st.session_state.form_values[nm] = put_input(col, nm)

        st.markdown("### Время")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        for nm, col in zip(block3, [c1, c2, c3, c4, c5, c6]):
            if nm in (cat_cols + num_cols): st.session_state.form_values[nm] = put_input(col, nm)

        st.markdown("### Поведение")
        c1, c2, c3, c4, c5 = st.columns(5)
        for nm, col in zip(block4, [c1, c2, c3, c4, c5]):
            if nm in (cat_cols + num_cols): st.session_state.form_values[nm] = put_input(col, nm)

        st.markdown("### Авто")
        c1, c2, c3, c4, c5 = st.columns(5)
        for nm, col in zip(block5, [c1, c2, c3, c4, c5]):
            if nm in (cat_cols + num_cols): st.session_state.form_values[nm] = put_input(col, nm)

        if rest:
            st.markdown("### Прочие признаки")
            cols = st.columns(4)
            for i, nm in enumerate(rest):
                st.session_state.form_values[nm] = put_input(cols[i % 4], nm)

        # Соберём итоговый payload fv
        fv = {k: v for k, v in st.session_state.form_values.items() if v is not None}

        # Автосчёт синусов/косинусов
        if auto_trig:
            import math
            h = fv.get("visit_hour")
            d = fv.get("visit_dow")
            if h is not None:
                fv["hour_sin"] = math.sin(2*math.pi*float(h)/24.0)
                fv["hour_cos"] = math.cos(2*math.pi*float(h)/24.0)
            if d is not None:
                fv["dow_sin"] = math.sin(2*math.pi*float(d)/7.0)
                fv["dow_cos"] = math.cos(2*math.pi*float(d)/7.0)

        # Подмешаем JSON
        ej = {}
        json_mode = None
        if isinstance(extra_dict, dict):
            ej = extra_dict.get("features", extra_dict)  # если есть обёртка, берём её содержимое
            json_mode = extra_dict.get("mode")
            if allow_json_override and json_mode:
                mm = str(json_mode).replace("-", "_").lower()
                if mm in ("bestf1", "best_f1"):
                    mode = "bestF1"
                elif mm in ("hi_recall", "hi-recall"):
                    mode = "hi_recall"

        fv.update({k: v for k, v in ej.items() if v is not None})

        # --- Диагностика заполнения ---
        with st.expander("⚙️ Диагностика заполнения"):
            req = feature_order or (cat_cols + num_cols)
            from_form = {k for k, v in st.session_state.form_values.items() if v is not None}
            from_json = set(extra_dict.keys()) if isinstance(extra_dict, dict) else set()
            present = set(fv.keys())
            missing = [c for c in req if c not in present]
            extra = [c for c in present if c not in req]

            st.write(f"Заполнено: **{len(present)}** из **{len(req)}** требуемых признаков.")
            if missing:
                st.warning("Не заполнены (будет импутация в API): " + ", ".join(missing[:25]) + (
                    " …" if len(missing) > 25 else ""))
            else:
                st.success("Все требуемые признаки присутствуют (импутация не понадобится).")

            if extra:
                st.info("Лишние поля (игнорируются моделью): " + ", ".join(extra[:25]) + (" …" if len(extra) > 25 else ""))

            both = sorted(list(from_form & from_json))
            if both:
                st.caption("⚠️ Поля присутствуют и в форме, и в JSON. По текущей логике **JSON перезаписывает форму**:")
                st.code(", ".join(both), language="text")

            wrong_numeric = [c for c in (present & set(num_cols)) if not isinstance(fv.get(c), (int, float))]
            if wrong_numeric:
                st.error("Числовые поля с нечисловыми значениями: " + ", ".join(wrong_numeric[:25]) + (
                    " …" if len(wrong_numeric) > 25 else ""))

        # Кнопки
        cL, cM, cR, cD = st.columns([1, 1, 1, 1])
        with cL:
            refresh_diag = st.form_submit_button("Обновить диагностику", use_container_width=True)
        with cM:
            submitted = st.form_submit_button("Предсказать", type="primary", use_container_width=True)
        with cR:
            clear = st.form_submit_button("Очистить", use_container_width=True)
            if clear:
                st.session_state.form_values = {}
                _safe_rerun()
        with cD:
            dbg_req = st.form_submit_button("🔎 Debug prediction", use_container_width=True)

    # Выполним запросы после формы
    if 'submitted' in locals() and submitted:
        try:
            payload = {"features": fv, "mode": mode}
            resp = requests.post(f"{api_url}/predict", json=payload, timeout=25)
            if resp.status_code != 200:
                st.error(f"{resp.status_code}: {resp.text}")
            else:
                data = resp.json()
                m1, m2, m3 = st.columns(3)
                m1.metric("Вероятность", f"{data['proba']:.3f}")
                m2.metric("Порог", f"{data['threshold_used']:.3f}")
                m3.metric("Класс", "1 (конверсия)" if int(data["y_hat"]) == 1 else "0")
                with st.expander("Отправленные признаки (payload)"):
                    st.json(fv)
        except Exception as e:
            st.exception(e)

    if 'dbg_req' in locals() and dbg_req:
        try:
            payload = {"features": fv, "mode": mode}
            r = requests.post(f"{api_url}/predict_debug", json=payload, timeout=30)
            r.raise_for_status()
            dbg = r.json()
            st.write("**Threshold used:**", dbg.get("threshold_used"))
            st.write("**Blended proba:**", dbg.get("proba_blend"))
            st.write("**Class (y_hat):**", dbg.get("y_hat"))
            parts = dbg.get("parts", {})
            if parts:
                df = pd.DataFrame.from_dict(parts, orient="index")
                df.index.name = "model"
                st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Debug failed: {e}")

# === TAB 2: BATCH CSV ===
with tab_batch:
    st.subheader("Batch CSV")
    st.caption("Загрузи CSV с исходными полями (как в train). На выходе получишь CSV с proba и y_hat.")
    f = st.file_uploader("CSV файл", type=["csv"])
    if f is not None and st.button("Скорить CSV", use_container_width=True):
        try:
            files = {"file": (f.name, f.getvalue(), "text/csv")}
            data = {"mode": mode}
            resp = requests.post(f"{api_url}/predict_batch", files=files, data=data, timeout=180)
            if resp.status_code != 200:
                st.error(f"{resp.status_code}: {resp.text}")
            else:
                st.success("Готово! Скачай результат ниже.")
                out_name = f"{f.name.rsplit('.', 1)[0]}__scored.csv"
                st.download_button("⬇️ Скачать результат", data=resp.content, file_name=out_name, mime="text/csv")
                try:
                    df_preview = pd.read_csv(StringIO(resp.content.decode("utf-8"))).head(50)
                    st.dataframe(df_preview, use_container_width=True, hide_index=True)
                except Exception:
                    pass
        except Exception as e:
            st.exception(e)

# === TAB 3: ABOUT ===
with tab_about:
    st.subheader("Описание")
    st.markdown("""
**BlendCAL** — ансамбль CatBoost + XGBoost + LightGBM (калиброванные вероятности, взвешенное усреднение).
- Метрика приоритета: **PR-AUC (AP)**; доп.: ROC-AUC, F1, Top-K.
- Пороговые режимы: **bestF1** (баланс) и **hi-recall** (высокая полнота).
- Препроцессинг: медианные импуты, квантильный клиппинг, frequency-encoding категорий, фиксированный порядок признаков.
- Сервис: **FastAPI** (`/predict`, `/predict_batch`, `/features`, `/version`), UI — **Streamlit**.
""")
    with st.expander("Исходные поля датасета (sessions & hits)"):
        st.markdown("""
**GA Sessions (ga_sessions.pkl)**  
`session_id` — ID визита; `client_id` — ID посетителя; `visit_date` — дата визита; `visit_time` — время визита;  
`visit_number` — порядковый номер визита; `utm_source` — источник; `utm_medium` — медиум;  
`utm_campaign` — кампания; `utm_keyword` — ключевое слово; `device_category` — тип устройства;  
`device_os` — ОС; `device_brand` — марка устройства; `device_model` — модель;  
`device_screen_resolution` — разрешение; `device_browser` — браузер; `geo_country` — страна; `geo_city` — город.

**GA Hits (ga_hits.pkl)**  
`session_id` — ID визита; `hit_date` — дата события; `hit_time` — время; `hit_number` — № события;  
`hit_type` — тип события; `hit_referer` — источник события; `hit_page_path` — страница;  
`event_category` — тип действия; `event_action` — действие; `event_label` — тег; `event_value` — значение.
""")
    st.info(
        "Под капотом: артефакты в `artifacts/`, веса ансамбля и рабочие пороги из `BlendCAL_*.json`. "
        "Если не уверены в составе признаков — скачайте шаблон CSV во вкладке выше."
    )
