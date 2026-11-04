"""Bước 1: Combined_Plant_Data_Updated.csv -> data/processed/features.csv.

Thay cho file Combined_Plant_Data_Updated_New_Feature_2.csv mà notebook đọc ở ô 24 (file này không có
trong repo). Các bước, theo Mục 5.1-5.2 báo cáo:
  1. Bỏ 4 dòng không khớp thời tiết (Plant 1, 2020-06-03 14:00)   -> 136,472 dòng
  2. HOUR = giờ của DATE_TIME (0-23)
  3. IS DAYTIME = 1 nếu 06:00 <= giờ:phút <= 17:45
  4. TEMP DIFF = MODULE_TEMPERATURE - AMBIENT_TEMPERATURE

    python -m src.build_features
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.common import load_config, p

WEATHER = ["SOURCE_KEY_weather", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]


def build(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
    df = df.dropna(subset=WEATHER).reset_index(drop=True)
    minutes = df["DATE_TIME"].dt.hour * 60 + df["DATE_TIME"].dt.minute
    df["HOUR"] = df["DATE_TIME"].dt.hour
    df["IS DAYTIME"] = ((minutes >= cfg["data"]["daytime_start_min"]) &
                        (minutes <= cfg["data"]["daytime_end_min"])).astype(int)
    df["TEMP DIFF"] = df["MODULE_TEMPERATURE"] - df["AMBIENT_TEMPERATURE"]
    return df


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=cfg["paths"]["merged_csv"])
    ap.add_argument("--out", default=cfg["paths"]["features_csv"])
    a = ap.parse_args()

    raw = pd.read_csv(p(a.inp))
    df = build(raw, cfg)
    out = p(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, lineterminator="\n")      # cố định LF để checksum giống nhau trên mọi HĐH
    by_plant = df.groupby("PLANT_ID").size().to_dict()
    print(f"Đọc {len(raw):,} dòng; bỏ {len(raw) - len(df)} dòng thiếu thời tiết -> {len(df):,} dòng {by_plant}")
    print(f"Đã ghi {out}  (mong đợi 136,472 dòng: 68,774 Plant 1 + 67,698 Plant 2)")


if __name__ == "__main__":
    main()
