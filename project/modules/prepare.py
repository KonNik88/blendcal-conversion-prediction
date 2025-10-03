from __future__ import annotations
import re, json
import pandas as pd
import numpy as np
from pathlib import Path

def _mode_or_na(x: pd.Series):
    x = x.dropna()
    if x.empty: return pd.NA
    m = x.mode()
    return m.iloc[0] if not m.empty else pd.NA

def _clean_token(s):
    if pd.isna(s): return pd.NA
    s = re.sub(r"[^a-z0-9\-]", "", str(s).lower())
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s if s else pd.NA

def _parse_cars_from_paths(ga_hits: pd.DataFrame) -> pd.DataFrame:
    paths_noq = (
        ga_hits["hit_page_path"].astype("string")
        .str.split("?", n=1, expand=True)[0]
        .str.lower()
    )
    pat_main = r"/cars/(?:all/)?(?P<brand>[a-z0-9\-]+)/(?P<model>[a-z0-9\-]+)(?:/[a-z0-9\-]+)?"
    pat_brand_only = r"/cars/(?:all/)?(?P<brand_only>[a-z0-9\-]+)/?$"

    extracted = paths_noq.str.extract(pat_main)
    extracted_brand_only = paths_noq.str.extract(pat_brand_only)

    brand = extracted["brand"].fillna(extracted_brand_only["brand_only"]).map(_clean_token).astype("string")
    model = extracted["model"].map(_clean_token).astype("string")

    cars_parsed = pd.DataFrame({
        "session_id": ga_hits["session_id"].astype("string"),
        "car_brand": brand,
        "car_model": model,
    })
    cars_by_session = (
        cars_parsed.groupby("session_id", as_index=False)
        .agg(
            car_brand=("car_brand", _mode_or_na),
            car_model=("car_model", _mode_or_na),
            n_brands=("car_brand", lambda s: s.dropna().nunique()),
            n_models=("car_model", lambda s: s.dropna().nunique()),
            any_car_detail=("car_model", lambda s: int(s.notna().any())),
        )
    )
    return cars_by_session

def _build_sessions_time_features(ga_sessions: pd.DataFrame) -> pd.DataFrame:
    # коэрсим отдельно дату и время, затем склеиваем строкой и снова коэрсим
    d = pd.to_datetime(ga_sessions["visit_date"], errors="coerce")
    t = pd.to_datetime(ga_sessions["visit_time"].astype(str)
                       .str.extract(r'(\d{1,2}:\d{2}:\d{2})', expand=False),
                       format="%H:%M:%S", errors="coerce").dt.time
    t = pd.Series(t).fillna(pd.to_datetime("00:00:00").time())
    ga_sessions["visit_datetime"] = pd.to_datetime(
        d.dt.date.astype("string") + " " + pd.Series(t).astype("string"),
        errors="coerce"
    )
    ga_sessions["visit_hour"] = ga_sessions["visit_datetime"].dt.hour.astype("Int32")
    ga_sessions["visit_dow"]  = ga_sessions["visit_datetime"].dt.dayofweek.astype("Int32")
    ga_sessions["is_weekend"] = ga_sessions["visit_dow"].isin([5,6]).astype("int8")
    return ga_sessions

def _build_hits_time_features(ga_hits: pd.DataFrame) -> pd.DataFrame:
    DAY_SECS = 24*3600
    # безопасно приводим типы
    ga_hits["hit_date"]   = pd.to_datetime(ga_hits["hit_date"], errors="coerce")
    ga_hits["hit_time"]   = pd.to_numeric(ga_hits["hit_time"], errors="coerce").round().astype("Int64")
    ga_hits["hit_number"] = pd.to_numeric(ga_hits["hit_number"], errors="coerce").round().astype("Int64")

    ga_hits["hit_day_offset"] = (ga_hits["hit_time"] // DAY_SECS).astype("Int64")
    ga_hits["hit_sec_in_day"] = (ga_hits["hit_time"] %  DAY_SECS).astype("Int64")
    ga_hits["hit_datetime"] = (
        ga_hits["hit_date"]
        + pd.to_timedelta(ga_hits["hit_day_offset"].fillna(0).astype(int), unit="D")
        + pd.to_timedelta(ga_hits["hit_sec_in_day"].fillna(0).astype(int), unit="s")
    )
    ga_hits["hit_hour"] = ga_hits["hit_datetime"].dt.hour.astype("Int32")
    ga_hits["hit_dow"]  = ga_hits["hit_datetime"].dt.dayofweek.astype("Int32")
    ga_hits["is_weekend_hit"] = ga_hits["hit_dow"].isin([5,6]).astype("int8")
    return ga_hits

def _aggregate_hits(ga_hits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    TARGET_ACTIONS = {
        "sub_car_claim_click","sub_car_claim_submit_click","sub_open_dialog_click",
        "sub_custom_question_submit_click","sub_call_number_click","sub_callback_submit_click",
        "sub_submit_success","sub_car_request_submit_click",
    }
    target_by_session = (
        ga_hits.assign(target=lambda df: df["event_action"].isin(TARGET_ACTIONS).astype(int))
               .groupby("session_id", as_index=False)["target"].max()
    )
    events_agg = (
        ga_hits.groupby("session_id")
               .agg(events_in_session=("event_action","count"),
                    unique_event_types=("event_action","nunique"))
               .reset_index()
    )
    return target_by_session, events_agg

def _traffic_features(df: pd.DataFrame) -> pd.DataFrame:
    df["is_organic"] = df["utm_medium"].isin(["organic","referral","(none)"]).astype(int)
    df["is_paid"]    = 1 - df["is_organic"]

    def group_medium(m):
        if m in ["organic","referral","(none)"]: return "organic"
        elif m in ["cpc","cpm","cpv","cpa"]:     return "performance"
        elif m in ["smm","fb_smm","vk_smm","ok_smm","tg","stories","blogger_channel","blogger_stories","blogger_header","social"]:
            return "social"
        elif m=="email": return "email"
        elif m=="push":  return "push"
        elif m=="sms":   return "sms"
        elif m=="banner":return "banner"
        else:            return "other"
    df["traffic_group"] = df["utm_medium"].map(group_medium)
    df["is_social"] = (df["traffic_group"]=="social").astype(int)
    df["is_email"]  = (df["traffic_group"]=="email").astype(int)
    df["is_push"]   = (df["traffic_group"]=="push").astype(int)
    df["is_sms"]    = (df["traffic_group"]=="sms").astype(int)
    df["is_perf"]   = (df["traffic_group"]=="performance").astype(int)
    return df

def _geo_auto_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    df["geo_city"] = df["geo_city"].replace("(not set)","other_city")
    top_countries = df["geo_country"].value_counts().head(10).index
    df["geo_country"] = df["geo_country"].where(df["geo_country"].isin(top_countries), "other_country")
    top_cities = df["geo_city"].value_counts().head(10).index
    df["geo_city"] = df["geo_city"].where(df["geo_city"].isin(top_cities), "other_city")

    big_cities = {"Moscow","Saint Petersburg","Novosibirsk","Yekaterinburg","Kazan"}
    df["is_big_city"] = df["geo_city"].isin(big_cities).astype("int8")

    def clean_car_value(x):
        if pd.isna(x): return pd.NA
        s = str(x).strip().lower()
        if re.search(r"\d", s): return pd.NA
        return s
    df["car_brand"] = df["car_brand"].apply(clean_car_value)
    df["car_model"] = df["car_model"].apply(clean_car_value)

    top_brands = df["car_brand"].value_counts().head(10).index
    df["car_brand"] = df["car_brand"].where(df["car_brand"].isin(top_brands), "other_brand")

    popular_models = df["car_model"].value_counts()[lambda x: x>=100].index
    df["car_model"] = df["car_model"].where(df["car_model"].isin(popular_models), "other_model")

    df["brand_model"] = df["car_brand"].fillna("none") + "_" + df["car_model"].fillna("none")
    df["is_top_brand"] = df["car_brand"].isin(top_brands).astype("int8")
    df["has_model"] = (~df["car_model"].isna() & (df["car_model"]!="other_model")).astype("int8")
    return df

def _counts_clip_log_and_cyc(df: pd.DataFrame) -> pd.DataFrame:
    medium_map = {
        "cpc":"cpc","yandex_cpc":"cpc","google_cpc":"cpc",
        "cpv":"cpm","cpm":"cpm","CPM":"cpm","cpa":"cpm",
        "banner":"banner","smartbanner":"banner",
        "social":"social","smm":"social","fb_smm":"social","vk_smm":"social","ok_smm":"social",
        "tg":"social","stories":"social","blogger_channel":"social","blogger_stories":"social","blogger_header":"social",
        "organic":"organic","(none)":"organic","referral":"organic",
        "email":"email","push":"push","sms":"sms",
    }
    df["utm_medium_norm"] = df["utm_medium"].map(lambda x: medium_map.get(str(x), "other")).astype("string")

    def p99_clip(series: pd.Series) -> float:
        return float(series.quantile(0.99))

    for col in ["visit_number","events_in_session","unique_event_types"]:
        df[f"log1p_{col}"] = np.log1p(df[col].astype(float))
        hi = p99_clip(df[col])
        df[f"{col}_clip"] = df[col].clip(upper=hi)

    df["hour_sin"] = np.sin(2*np.pi*df["visit_hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["visit_hour"]/24)
    df["dow_sin"]  = np.sin(2*np.pi*df["visit_dow"]/7)
    df["dow_cos"]  = np.cos(2*np.pi*df["visit_dow"]/7)
    return df

def build_featureset(sessions_path: str, heats_path: str, cars_path: str|None, output_path: str):
    import numpy as np
    import pandas as pd
    # читаем parquet, названия колонок должны соответствовать CSV
    ga_sessions = pd.read_parquet(sessions_path)
    ga_hits     = pd.read_parquet(heats_path)

    # дедуп по session_id
    ga_sessions = ga_sessions.drop_duplicates(subset=["session_id"], keep="last")
    ga_sessions["session_id"] = ga_sessions["session_id"].astype("string")
    assert ga_sessions["session_id"].is_unique

    # время
    ga_sessions = _build_sessions_time_features(ga_sessions)
    ga_hits     = _build_hits_time_features(ga_hits)

    # цели и агрегаты
    target_by_session, events_agg = _aggregate_hits(ga_hits)

    # авто из путей
    cars_by_session = _parse_cars_from_paths(ga_hits)

    # merge
    df = (
        ga_sessions
        .merge(target_by_session, on="session_id", how="left")
        .merge(events_agg,       on="session_id", how="left")
        .merge(cars_by_session,  on="session_id", how="left")
    )

    # базовые NA/типы
    df["target"] = df["target"].fillna(0).astype("int8")
    df[["events_in_session","unique_event_types"]] = (
        df[["events_in_session","unique_event_types"]].fillna(0).astype("Int64")
    )

    # трафик + geo/auto очистка
    df = _traffic_features(df)
    df = _geo_auto_cleanup(df)

    # удалить лишнее — как в ноуте
    drop_cols = [
        "client_id","visit_date","visit_time","utm_source","utm_campaign","utm_adcontent","utm_keyword",
        "device_os","device_brand","device_model","device_screen_resolution","device_browser"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # клиппинг/лог/циклические
    df = _counts_clip_log_and_cyc(df)

    # вернуть id и аккуратно привести ключи к строке
    df["session_id"] = df["session_id"].astype(str)

    # финально — сохранить parquet
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
