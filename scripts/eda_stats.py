"""Số liệu Mục 3-4 của báo cáo + Bảng 1, 2, 3, 7 và hệ số Linear Regression (Mục 7.4).

    python scripts/eda_stats.py                                   # dùng checkpoint gốc cho Bảng 3, 7
    python scripts/eda_stats.py --model-dir outputs/retrain       # dùng kết quả huấn luyện lại

Đầu ra: outputs/eda/ (table1_daily_yield.csv, table2_split.csv, table3_tuning.csv, table7_importance.csv,
lr_coefficients.csv, dataset_stats.json)
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import load_config, p  # noqa: E402
from src.data import prepare, split_summary  # noqa: E402

warnings.filterwarnings("ignore")


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=cfg["paths"]["reference_dir"])
    ap.add_argument("--out", default=f"{cfg['paths']['outputs']}/eda")
    a = ap.parse_args()
    out = p(a.out); out.mkdir(parents=True, exist_ok=True)
    md = p(a.model_dir)
    stats = {}

    # ---------- Mục 3: mô tả dữ liệu (trước khi bỏ 4 dòng NaN)
    m = pd.read_csv(p(cfg["paths"]["merged_csv"]), parse_dates=["DATE_TIME"])
    full = pd.date_range(m.DATE_TIME.min().normalize(), m.DATE_TIME.max(), freq="15min")
    for pl, g in m.groupby("PLANT_ID"):
        tp = g.DATE_TIME.drop_duplicates().sort_values()
        gap = tp.diff().max()
        stats[f"plant_{pl}"] = {"generation_rows": int(len(g)), "time_points": int(len(tp)),
                                "time_points_possible": int(len(full)),
                                "longest_gap_hours": gap.total_seconds() / 3600,
                                "longest_gap_ends": str(tp[tp.diff() == gap].iloc[0]),
                                "inverters": int(g.SOURCE_KEY.nunique())}

    f = pd.read_csv(p(cfg["paths"]["features_csv"]))
    stats["cleaned_rows"] = int(len(f))
    for pl, g in f.groupby("PLANT_ID"):
        s = stats[f"plant_{pl}"]
        s["cleaned_rows"] = int(len(g))
        for c in ("AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"):
            s[f"{c}_range"] = [round(float(g[c].min()), 2), round(float(g[c].max()), 2)]
        # Mục 4.1: tương quan Pearson
        corr = g[["AC_POWER", "IRRADIATION", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE", "DAILY_YIELD"]].corr()
        s["pearson_with_AC_POWER"] = corr["AC_POWER"].drop("AC_POWER").round(2).to_dict()
        s["pearson_IRRADIATION_MODULE_TEMPERATURE"] = round(float(g.IRRADIATION.corr(g.MODULE_TEMPERATURE)), 2)
        # Mục 4.2: hiệu suất AC/DC (sau khi sửa DC_POWER), trung vị theo inverter khi có nắng
        dc = g[g.DC_POWER > 0]
        eff = dc.groupby("SOURCE_KEY").AC_POWER.sum() / dc.groupby("SOURCE_KEY").DC_POWER.sum()
        s["ac_dc_efficiency_range"] = [round(float(eff.min()), 4), round(float(eff.max()), 4)]
        s["ac_dc_ratio_before_fix"] = round(float(dc.AC_POWER.sum() / (dc.DC_POWER.sum() *
                                       (cfg["data"]["dc_fix_factor"] if pl == cfg["data"]["dc_fix_plant"] else 1))), 3)
        med = g[g.IRRADIATION > 0].groupby("SOURCE_KEY").AC_POWER.median()
        s["median_of_inverter_medians_kW"] = round(float(med.median()), 1)
        s["inverters_below_85pct_median"] = int((med < 0.85 * med.median()).sum())
        s["anomalous_records"] = int(((g.IRRADIATION > cfg["data"]["anomaly_irradiation"]) &
                                      (g.AC_POWER == 0)).sum())
    (out / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------- Bảng 1: theo nhóm DAILY_YIELD
    grp = pd.cut(f.DAILY_YIELD, [-np.inf, 3000, 6000, np.inf], right=False,
                 labels=["Low (< 3,000)", "Normal (3,000-5,999)", "High (>= 6,000)"])
    t1 = f.groupby(grp, observed=True).agg(Records=("AC_POWER", "size"), IRRADIATION=("IRRADIATION", "mean"),
                                           MODULE_TEMP=("MODULE_TEMPERATURE", "mean"),
                                           AC_POWER=("AC_POWER", "mean"))
    t1.round({"IRRADIATION": 4, "MODULE_TEMP": 2, "AC_POWER": 2}).to_csv(out / "table1_daily_yield.csv")

    # ---------- Bảng 2: chia tập
    prep = prepare(cfg)
    t2 = split_summary(prep)
    t2.to_csv(out / "table2_split.csv", index=False)

    # ---------- Bảng 3: kết quả thử cấu hình (validation)
    if (md / "tuning_tabular.csv").exists():
        tt, ts = pd.read_csv(md / "tuning_tabular.csv"), pd.read_csv(md / "tuning_ts.csv")
    else:                                   # checkpoint gốc: lưu trong model_comparison.xlsx
        tt = pd.read_excel(md / "model_comparison.xlsx", sheet_name="Tuning_Tabular")
        ts = pd.read_excel(md / "model_comparison.xlsx", sheet_name="Tuning_ARIMA_SARIMA")
    t3 = pd.concat([tt.assign(plant=np.nan), ts], ignore_index=True)
    t3.round(4).to_csv(out / "table3_tuning.csv", index=False)

    # ---------- Bảng 7: độ quan trọng đặc trưng
    rf, rf_f = joblib.load(md / "model_RandomForest.pkl")
    xgb, xgb_f = joblib.load(md / "model_XGBoost.pkl")
    t7 = pd.DataFrame({"feature": rf_f, "RandomForest": rf.feature_importances_,
                       "XGBoost": pd.Series(xgb.feature_importances_, index=xgb_f).reindex(rf_f).values})
    t7.sort_values("XGBoost", ascending=False).round(3).to_csv(out / "table7_importance.csv", index=False)

    # ---------- Mục 7.4: hệ số Linear Regression (48 cột, OLS tất định -> fit lại cho đúng tập cột báo cáo)
    from sklearn.linear_model import LinearRegression
    S = prep["S"]
    tr = S["train"][S["train"].IS_ANOMALY == 0]
    lr = LinearRegression().fit(tr[prep["features_lr"]], tr[prep["target"]])
    coef = pd.Series(lr.coef_, index=prep["features_lr"], name="coef_kW_per_std")
    coef.round(2).to_csv(out / "lr_coefficients.csv")

    pd.set_option("display.width", 200); pd.set_option("display.max_columns", None)
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    print("\nBảng 1\n", t1.round(4))
    print("\nBảng 2\n", t2.to_string(index=False))
    print("\nBảng 3 (val RMSE)\n", t3[["model", "plant", "params", "order", "seasonal_order", "RMSE"]]
          .round(1).to_string(index=False))
    print("\nBảng 7\n", t7.sort_values("XGBoost", ascending=False).round(3).to_string(index=False))
    print("\nHệ số LR (5 biến thời tiết/thời gian)\n", coef.iloc[:5].round(2).to_string())
    print(f"\nĐã ghi vào {out}")


if __name__ == "__main__":
    main()
