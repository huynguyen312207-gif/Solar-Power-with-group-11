"""Bước 0 (tùy chọn): 4 file CSV gốc Kaggle -> Combined_Plant_Data_Updated.csv.

Tương ứng ô 1-2 của notebook: gộp Generation + Weather theo (DATE_TIME, PLANT_ID) kiểu left join,
ghép 2 nhà máy, chia DC_POWER của Plant 1 (4135001) cho 10.
Repo đã có sẵn Combined_Plant_Data_Updated.csv, nên chỉ cần chạy bước này khi muốn tái tạo từ dữ liệu gốc.

    python -m src.make_dataset --raw-dir data/raw --out data/interim/Combined_Plant_Data_Updated.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.common import load_config, p

RAW_FILES = {
    "gen1": "Plant_1_Generation_Data.csv",
    "weather1": "Plant_1_Weather_Sensor_Data.csv",
    "gen2": "Plant_2_Generation_Data.csv",
    "weather2": "Plant_2_Weather_Sensor_Data.csv",
}


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=cfg["paths"]["raw_dir"])
    ap.add_argument("--out", default="data/interim/Combined_Plant_Data_Updated.csv")
    a = ap.parse_args()

    raw = {k: pd.read_csv(p(a.raw_dir) / v) for k, v in RAW_FILES.items()}
    # Plant 1 Generation dùng định dạng dd-mm-yyyy HH:MM -> khai báo tường minh; 3 file còn lại là ISO
    raw["gen1"]["DATE_TIME"] = pd.to_datetime(raw["gen1"]["DATE_TIME"], format="%d-%m-%Y %H:%M")
    for k in ("weather1", "gen2", "weather2"):
        raw[k]["DATE_TIME"] = pd.to_datetime(raw[k]["DATE_TIME"])

    merged = []
    for g, w in (("gen1", "weather1"), ("gen2", "weather2")):
        merged.append(pd.merge(raw[g], raw[w], on=["DATE_TIME", "PLANT_ID"], how="left",
                               suffixes=("", "_weather")))
    df = pd.concat(merged, ignore_index=True)

    fix = df["PLANT_ID"] == cfg["data"]["dc_fix_plant"]
    df.loc[fix, "DC_POWER"] = df.loc[fix, "DC_POWER"] / cfg["data"]["dc_fix_factor"]

    out = p(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Đã ghi {out} — {len(df):,} dòng x {df.shape[1]} cột "
          f"(mong đợi 136,476 x 11; 4 dòng thiếu thời tiết ở Plant 1, 2020-06-03 14:00)")


if __name__ == "__main__":
    main()
