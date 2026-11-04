"""Mã hóa, chia tập 70/15/15 theo thời gian, chuẩn hóa, đánh dấu dòng bất thường (ô 28-32 của notebook)."""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.common import p


def prepare(cfg: dict, features_csv: str | None = None, scaler: StandardScaler | None = None) -> dict:
    """Trả về dict gồm S (train/val/test đã scale), danh sách cột, scaler, bảng ánh xạ SOURCE_KEY.

    scaler=None -> fit StandardScaler trên train (như notebook); truyền scaler đã lưu để dùng lại.
    """
    fcfg, dcfg = cfg["features"], cfg["data"]
    df = pd.read_csv(p(features_csv or cfg["paths"]["features_csv"]))
    target = fcfg["target"]

    # SOURCE_KEY_weather trùng 1-1 với PLANT_ID -> bỏ
    le = LabelEncoder()
    df["SOURCE_KEY_ENCODED"] = le.fit_transform(df["SOURCE_KEY"])
    mapping = (pd.DataFrame({"SOURCE_KEY": le.classes_, "SOURCE_KEY_ENCODED": range(len(le.classes_))})
               .merge(df[["SOURCE_KEY", "PLANT_ID"]].drop_duplicates(), on="SOURCE_KEY"))
    df = pd.get_dummies(df, columns=["PLANT_ID", "SOURCE_KEY"], prefix=["PLANT_ID", "SK"],
                        drop_first=True, dtype=int)
    df = df.drop(columns=["SOURCE_KEY_weather"])
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
    sk_cols = [c for c in df.columns if c.startswith("SK_")]            # 43 cột one-hot

    feats = list(fcfg["tabular"])
    drop_lr = set(fcfg.get("linear_regression_drop", [])) | {"SOURCE_KEY_ENCODED"}
    feats_lr = [f for f in feats if f not in drop_lr] + sk_cols

    # Chia TRƯỚC khi scale, theo thời gian, không shuffle; cắt theo số dòng rồi làm tròn tới mốc thời gian
    df = df.sort_values(["DATE_TIME", "SOURCE_KEY_ENCODED"]).reset_index(drop=True)
    f_val, f_test = dcfg["split_fracs"]
    cut_val = df["DATE_TIME"].iloc[int(len(df) * f_val)]
    cut_test = df["DATE_TIME"].iloc[int(len(df) * f_test)]
    S = {"train": df[df.DATE_TIME < cut_val].copy(),
         "val": df[(df.DATE_TIME >= cut_val) & (df.DATE_TIME < cut_test)].copy(),
         "test": df[df.DATE_TIME >= cut_test].copy()}

    for d in S.values():
        d["IRR_RAW"] = d["IRRADIATION"]                                    # giữ bản chưa scale
        d["IS_ANOMALY"] = ((d.IRR_RAW > dcfg["anomaly_irradiation"]) & (d[target] == 0)).astype(int)

    scale_cols = list(fcfg["scale"])
    if scaler is None:
        scaler = StandardScaler().fit(S["train"][scale_cols])               # fit CHỈ trên train
    for d in S.values():
        d[scale_cols] = scaler.transform(d[scale_cols])

    return {"S": S, "features": feats, "features_lr": feats_lr, "sk_cols": sk_cols,
            "scale_cols": scale_cols, "scaler": scaler, "mapping": mapping, "target": target,
            "cut_val": cut_val, "cut_test": cut_test}


def split_summary(prep: dict) -> pd.DataFrame:
    """Bảng 2 của báo cáo: khoảng thời gian, số dòng, tỷ lệ, số dòng bất thường của mỗi tập."""
    S = prep["S"]
    n = sum(len(d) for d in S.values())
    return pd.DataFrame([{"Set": k, "start": d.DATE_TIME.min(), "end": d.DATE_TIME.max(), "rows": len(d),
                          "pct": round(100 * len(d) / n, 1), "anomalous_rows": int(d.IS_ANOMALY.sum())}
                         for k, d in S.items()])
