import pandas as pd

def _csv_to_parquet(src_csv: str, dst_parquet: str) -> None:
    df = pd.read_csv(src_csv, low_memory=False)
    df.to_parquet(dst_parquet, index=False)

def ingest_sessions_csv(src_csv: str, dst_parquet: str) -> None:
    _csv_to_parquet(src_csv, dst_parquet)

def ingest_heats_csv(src_csv: str, dst_parquet: str) -> None:
    _csv_to_parquet(src_csv, dst_parquet)
