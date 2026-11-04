"""Suy luận trên dữ liệu mới: ước lượng AC_POWER của 1 inverter từ thời tiết đo cùng thời điểm.

    python -m src.predict --model XGBoost --inverter P2-05 --time "2020-06-15 12:00" \
        --irradiation 0.85 --ambient 32.5 --module 55.0

Hỗ trợ Linear Regression, Random Forest, XGBoost (đầu vào 1 thời điểm). LSTM cần chuỗi 16 bước
(4 giờ) nên không có trong lệnh này.
"""
from __future__ import annotations

import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.common import load_config, p


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="XGBoost", choices=["LinearRegression", "RandomForest", "XGBoost"])
    ap.add_argument("--ckpt-dir", default=cfg["paths"]["reference_dir"])
    ap.add_argument("--inverter", required=True, help="P1-01..P1-22, P2-01..P2-22 (xem inverter_ids.csv)")
    ap.add_argument("--time", required=True, help='"YYYY-MM-DD HH:MM"')
    ap.add_argument("--irradiation", type=float, required=True)
    ap.add_argument("--ambient", type=float, required=True, help="AMBIENT_TEMPERATURE (°C)")
    ap.add_argument("--module", type=float, required=True, help="MODULE_TEMPERATURE (°C)")
    a = ap.parse_args()
    ck = p(a.ckpt_dir)

    ids = pd.read_csv(p(cfg["paths"]["inverter_ids"]))
    row = ids[ids.inverter_id == a.inverter]
    if row.empty:
        raise SystemExit(f"Không có inverter {a.inverter}; xem {cfg['paths']['inverter_ids']}")
    key, plant = row.SOURCE_KEY.item(), int(row.PLANT_ID.item())
    t = pd.Timestamp(a.time)
    minutes = t.hour * 60 + t.minute
    x = {"AMBIENT_TEMPERATURE": a.ambient, "MODULE_TEMPERATURE": a.module, "IRRADIATION": a.irradiation,
         "HOUR": t.hour, "TEMP DIFF": a.module - a.ambient,
         "IS DAYTIME": int(cfg["data"]["daytime_start_min"] <= minutes <= cfg["data"]["daytime_end_min"]),
         "SOURCE_KEY_ENCODED": int(LabelEncoder().fit(ids.SOURCE_KEY).transform([key])[0]),
         "PLANT_ID_4136001": int(plant == 4136001)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaler = joblib.load(ck / "standard_scaler.pkl")
        model, feats = joblib.load(ck / f"model_{a.model}.pkl")
    X = pd.DataFrame([x])
    X[cfg["features"]["scale"]] = scaler.transform(X[cfg["features"]["scale"]])
    for f in feats:                                       # cột one-hot SK_* cho Linear Regression
        if f.startswith("SK_"):
            X[f] = int(f == f"SK_{key}")
    y = float(np.clip(model.predict(X[feats]), 0, None)[0])
    print(json.dumps({"model": a.model, "inverter": a.inverter, "SOURCE_KEY": key, "plant": plant,
                      "time": str(t), "AC_POWER_kW": round(y, 2)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
