"""Bước 3b: tính Bảng 4, 5, 6 và phân tích inverter P2-01 từ các file dự đoán test, rồi so với báo cáo.

    python -m src.evaluate --pred-dir outputs/checkpoint_eval              # so khớp tuyệt đối (mặc định)
    python -m src.evaluate --pred-dir outputs/retrain --tol retrain       # so với ngưỡng khi huấn luyện lại

Đầu vào: pred_test_tabular.csv, pred_test_LSTM.csv, pred_test_ARIMA.csv, pred_test_SARIMA.csv
Đầu ra (<pred-dir>/results/): table4_inverter.csv, table5_plant.csv, table6_per_plant.csv,
p2_01_analysis.csv, plant_level_predictions.csv, metrics_test.json, check_vs_report.csv
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
import yaml

from src.common import ALL_MODELS, INV_MODELS, NAMES, evaluate, load_config, p


def load_preds(d):
    tab = pd.read_csv(d / "pred_test_tabular.csv", parse_dates=["DATE_TIME"])
    lstm = pd.read_csv(d / "pred_test_LSTM.csv", parse_dates=["DATE_TIME"])
    ts = pd.concat([pd.read_csv(d / f"pred_test_{m}.csv", parse_dates=["DATE_TIME"]) for m in ("ARIMA", "SARIMA")])
    # Mức inverter: chỉ so trên phần giao (LSTM bỏ vài mẫu đầu mỗi đoạn dữ liệu liên tục)
    inv = tab.merge(lstm, on=["DATE_TIME", "SOURCE_KEY_ENCODED"], how="inner")
    return inv, ts


def to_plant(inv, ts, target="AC_POWER", keep=None):
    """Trung bình các inverter của mỗi nhà máy tại mỗi mốc 15 phút + ghép dự đoán ARIMAX/SARIMAX."""
    plant = inv.groupby(["DATE_TIME", "PLANT_ID_4136001"]).agg(
        y_true=(target, "mean"), IRR_RAW=("IRR_RAW", "mean"),
        **{m: (m, "mean") for m in INV_MODELS}).reset_index()
    for m in ("ARIMA", "SARIMA"):
        plant = plant.merge(ts[ts.model == m][["DATE_TIME", "PLANT_ID_4136001", "y_pred"]]
                            .rename(columns={"y_pred": m}), on=["DATE_TIME", "PLANT_ID_4136001"], how="inner")
    if keep is not None:                                   # giữ đúng các mốc của Bảng 5
        plant = plant.merge(keep, on=["DATE_TIME", "PLANT_ID_4136001"], how="inner")
    return plant


def r2_by_plant(plant, models):
    return {pl: {m: evaluate(g.y_true, g[m])["R2"] for m in models} for pl, g in plant.groupby("PLANT_ID_4136001")}


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=f"{cfg['paths']['outputs']}/checkpoint_eval")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tol", default="exact", choices=["exact", "retrain", "none"])
    a = ap.parse_args()
    d = p(a.pred_dir)
    out = p(a.out) if a.out else d / "results"
    out.mkdir(parents=True, exist_ok=True)

    inv, ts = load_preds(d)
    # ---- Bảng 4: mức inverter
    t4 = pd.DataFrame([{"Model": NAMES[m], **evaluate(inv.AC_POWER, inv[m], inv.IRR_RAW > 0)}
                       for m in INV_MODELS]).set_index("Model")
    # ---- Bảng 5: mức nhà máy, 6 mô hình
    plant = to_plant(inv, ts)
    keep = plant[["DATE_TIME", "PLANT_ID_4136001"]]
    t5 = pd.DataFrame([{"Model": NAMES[m], **evaluate(plant.y_true, plant[m], plant.IRR_RAW > 0)}
                       for m in ALL_MODELS]).set_index("Model").sort_values("RMSE_day")
    # ---- Bảng 6: R2 theo nhà máy; "without anomalous" = bỏ dòng inverter bất thường TRƯỚC khi lấy trung bình
    clean = to_plant(inv[inv.IS_ANOMALY == 0], ts, keep=keep)
    r_all, r_clean = r2_by_plant(plant, ALL_MODELS), r2_by_plant(clean, ALL_MODELS)
    t6 = pd.DataFrame([{"Model": NAMES[m], "Plant 1 (all)": r_all[0][m], "Plant 2 (all)": r_all[1][m],
                        "Plant 1 (w/o anomalous)": r_clean[0][m], "Plant 2 (w/o anomalous)": r_clean[1][m]}
                       for m in ALL_MODELS]).set_index("Model").loc[[NAMES[m] for m in
                       ["LSTM", "ARIMA", "LinearRegression", "XGBoost", "RandomForest", "SARIMA"]]]
    # ---- Mục 7.6: inverter P2-01 (SOURCE_KEY Qf4GUc1pJu5T6c6)
    ids = pd.read_csv(p(cfg["paths"]["inverter_ids"]))
    key = ids.loc[ids.inverter_id == "P2-01", "SOURCE_KEY"].item()
    from sklearn.preprocessing import LabelEncoder
    enc = int(LabelEncoder().fit(ids.SOURCE_KEY).transform([key])[0])
    no_p201 = to_plant(inv[inv.SOURCE_KEY_ENCODED != enc], ts, keep=keep)
    both = to_plant(inv[(inv.SOURCE_KEY_ENCODED != enc) & (inv.IS_ANOMALY == 0)], ts, keep=keep)
    anom = inv[inv.IS_ANOMALY == 1]
    p201 = pd.DataFrame([
        {"scenario": "all records", "LSTM_R2_plant2": r_all[1]["LSTM"]},
        {"scenario": "without P2-01", "LSTM_R2_plant2": r2_by_plant(no_p201, ["LSTM"])[1]["LSTM"]},
        {"scenario": "without anomalous records", "LSTM_R2_plant2": r_clean[1]["LSTM"]},
        {"scenario": "without both", "LSTM_R2_plant2": r2_by_plant(both, ["LSTM"])[1]["LSTM"]}])

    t4.round(4).to_csv(out / "table4_inverter.csv")
    t5.round(4).to_csv(out / "table5_plant.csv")
    t6.round(4).to_csv(out / "table6_per_plant.csv")
    p201.round(4).to_csv(out / "p2_01_analysis.csv", index=False)
    plant.to_csv(out / "plant_level_predictions.csv", index=False)
    metrics = {"n_inverter_records": int(len(inv)), "n_plant_time_points": int(len(plant)),
               "table4": t4.to_dict("index"), "table5": t5.to_dict("index"), "table6": t6.to_dict("index"),
               "p2_01": dict(zip(p201.scenario, p201.LSTM_R2_plant2)),
               "anomalous_test_records": {"total": int(len(anom)),
                                          "plant1": int((anom.PLANT_ID_4136001 == 0).sum()),
                                          "plant2": int((anom.PLANT_ID_4136001 == 1).sum()),
                                          "P2-01": int((anom.SOURCE_KEY_ENCODED == enc).sum())}}
    (out / "metrics_test.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.set_option("display.width", 200); pd.set_option("display.max_columns", None)
    print(f"Bảng 4 — mức inverter ({len(inv):,} bản ghi)\n{t4.round(4)}\n")
    print(f"Bảng 5 — mức nhà máy ({len(plant)} mốc)\n{t5.round(4)}\n")
    print(f"Bảng 6 — R2 theo nhà máy\n{t6.round(4)}\n")
    print(f"Mục 7.6 — LSTM, R2 Plant 2\n{p201.round(4).to_string(index=False)}")
    print(f"Dòng bất thường trong test (phần giao 4 mô hình): {metrics['anomalous_test_records']}\n")

    if a.tol != "none":
        check(metrics, a.tol, out)
    print(f"Kết quả lưu ở {out}")


def check(metrics: dict, tol_name: str, out) -> None:
    """So từng số với configs/expected_results.yaml (số trong báo cáo) theo ngưỡng đã khai báo."""
    exp = yaml.safe_load(open(p("configs/expected_results.yaml"), encoding="utf-8"))
    tol = exp["tolerance"][tol_name]
    rows = []
    for table, models in exp["report"].items():
        for model, vals in models.items():
            for k, v in vals.items():
                got = metrics[table][model][k] if table != "p2_01" else metrics[table][model]
                if k.startswith("R2") or table in ("table6", "p2_01"):
                    ok, lim = abs(got - v) <= tol["r2_abs"], tol["r2_abs"]
                else:
                    lim = max(tol["err_abs"], tol["err_rel"] * abs(v))
                    ok = abs(got - v) <= lim
                rows.append({"table": table, "model": model, "metric": k, "report": v,
                             "reproduced": round(got, 4), "diff": round(got - v, 4), "tolerance": lim,
                             "status": "PASS" if ok else "FAIL"})
    df = pd.DataFrame(rows)
    df.to_csv(out / "check_vs_report.csv", index=False)
    n_fail = int((df.status == "FAIL").sum())
    print(f"Đối chiếu với báo cáo (ngưỡng '{tol_name}'): {len(df) - n_fail}/{len(df)} PASS")
    if n_fail:
        print(df[df.status == "FAIL"].to_string(index=False))


if __name__ == "__main__":
    main()
