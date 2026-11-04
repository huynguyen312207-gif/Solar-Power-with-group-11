"""Bước 2: huấn luyện lại từ đầu 6 mô hình và lưu checkpoint + dự đoán trên tập test.

    python -m src.train                                  # cả 6 mô hình, seed theo config (tái tạo báo cáo)
    python -m src.train --models RandomForest XGBoost    # chỉ vài mô hình
    python -m src.train --seed 123 --out outputs/seed_123

Đầu ra (mặc định outputs/retrain/): model_*.pkl, model_LSTM.pt, pred_test_*.csv, tuning_*.csv,
history_lstm.csv, split_summary.csv, standard_scaler.pkl, source_key_mapping.csv, run_info.json
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd

from src.common import ALL_MODELS, load_config, p, set_seed
from src.data import prepare, split_summary
from src.models import (build_windows, expand_grid, fit_tabular, fit_timeseries_plant, lstm_predict,
                        train_lstm)


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--seed", type=int, default=None,
                    help="ghi đè seed numpy/torch VÀ random_state của RF/XGB (mặc định: theo config)")
    ap.add_argument("--out", default=f"{cfg['paths']['outputs']}/retrain")
    ap.add_argument("--device", default=None, help="cuda | cpu (mặc định: cuda nếu có)")
    ap.add_argument("--max-epochs", type=int, default=None, help="giới hạn epoch LSTM (chạy thử nhanh)")
    a = ap.parse_args()
    if a.max_epochs:
        cfg["models"]["LSTM"]["max_epochs"] = a.max_epochs

    seed = cfg["seed"] if a.seed is None else a.seed
    rs_rf = cfg["models"]["RandomForest"]["random_state"] if a.seed is None else a.seed
    rs_xgb = cfg["models"]["XGBoost"]["random_state"] if a.seed is None else a.seed
    out = p(a.out); out.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    t_all = time.time()

    prep = prepare(cfg)
    S, target = prep["S"], prep["target"]
    joblib.dump(prep["scaler"], out / "standard_scaler.pkl")
    prep["mapping"].to_csv(out / "source_key_mapping.csv", index=False)
    split_summary(prep).to_csv(out / "split_summary.csv", index=False)
    print(split_summary(prep).to_string(index=False))

    train = S["train"][S["train"].IS_ANOMALY == 0]                 # train sạch: bỏ dòng bất thường
    val, test = S["val"], S["test"]
    info = {"seed": seed, "rf_random_state": rs_rf, "xgb_random_state": rs_xgb, "models": a.models,
            "python": platform.python_version(), "platform": platform.platform()}

    # ---------------- mô hình dạng bảng (mức inverter)
    tab_names = [m for m in ("LinearRegression", "RandomForest", "XGBoost") if m in a.models]
    if tab_names:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from xgboost import XGBRegressor
        pred_tab = test[["DATE_TIME", "SOURCE_KEY_ENCODED", "PLANT_ID_4136001", target, "IRR_RAW",
                         "IS_ANOMALY"]].copy()
        tuning = []
        for name in tab_names:
            print(f"\n== {name}")
            t = time.time()
            if name == "LinearRegression":
                # So sánh mã hóa SOURCE_KEY: one-hot (chọn) vs label (Bảng 3 báo cáo)
                cands = [({}, prep["features_lr"]), ({}, prep["features"])]
                builder = LinearRegression
            elif name == "RandomForest":
                c = cfg["models"]["RandomForest"]
                cands = [({**g, "n_jobs": -1, "random_state": rs_rf}, prep["features"])
                         for g in expand_grid(c["grid"])]
                builder = RandomForestRegressor
            else:
                c = cfg["models"]["XGBoost"]
                cands = [({**g, "subsample": c["subsample"], "colsample_bytree": c["colsample_bytree"],
                           "random_state": rs_xgb, "n_jobs": -1}, prep["features"])
                         for g in expand_grid(c["grid"])]
                builder = XGBRegressor
            m, feats, pred, tun = fit_tabular(name, builder, cands, train, val, test, target)
            pred_tab[name] = pred
            tuning += tun
            joblib.dump((m, feats), out / f"model_{name}.pkl")
            info[f"time_{name}_s"] = round(time.time() - t, 1)
        pred_tab.to_csv(out / "pred_test_tabular.csv", index=False)
        pd.DataFrame(tuning).to_csv(out / "tuning_tabular.csv", index=False)

    # ---------------- ARIMAX / SARIMAX (mức nhà máy, mỗi nhà máy 1 mô hình)
    ts_tuning = []
    for name in [m for m in ("ARIMA", "SARIMA") if m in a.models]:
        print(f"\n== {name}")
        t = time.time()
        c = cfg["models"][name]
        data = {"train": train, "val": val, "test": test}
        preds = []
        for plant in (0, 1):
            kw, rv, tun, pred = fit_timeseries_plant(name, plant, c["candidates"], c["freq"], data, target,
                                                     cfg["features"]["exog"])
            print(f"  Plant {plant + 1}: chọn {kw} -> val RMSE={rv['RMSE']:.2f}")
            ts_tuning += tun
            preds.append(pred)
        pd.concat(preds).to_csv(out / f"pred_test_{name}.csv", index=False)
        info[f"time_{name}_s"] = round(time.time() - t, 1)
    if ts_tuning:
        pd.DataFrame(ts_tuning).to_csv(out / "tuning_ts.csv", index=False)

    # ---------------- LSTM (mức inverter, cửa sổ 4 giờ)
    if "LSTM" in a.models:
        import torch
        print("\n== LSTM")
        t = time.time()
        set_seed(seed)                                   # độc lập với thứ tự chạy các mô hình trước
        device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        lc = cfg["models"]["LSTM"]
        X, I, y, M, mask, y_scale = build_windows(S, cfg["features"]["sequence"], target, lc["window"])
        print("  thiết bị:", device, "| số mẫu:", {k: int(v.sum()) for k, v in mask.items()})
        net, state, hist, Xt, It = train_lstm(X, I, y, mask, y_scale, lc, device)
        torch.save(state, out / "model_LSTM.pt")
        hist.to_csv(out / "history_lstm.csv", index=False)
        pred = M.loc[mask["test"], ["DATE_TIME", "SOURCE_KEY_ENCODED"]].copy()
        pred["LSTM"] = lstm_predict(net, Xt, It, mask["test"], y_scale, device)
        pred.to_csv(out / "pred_test_LSTM.csv", index=False)
        info.update({"lstm_device": device, "lstm_best_epoch": int(np.argmin(hist.RMSE)) + 1,
                     "lstm_epochs_run": int(len(hist)), "lstm_y_scale": y_scale,
                     "time_LSTM_s": round(time.time() - t, 1), "torch": torch.__version__})

    info["time_total_s"] = round(time.time() - t_all, 1)
    (out / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nXong sau {info['time_total_s'] / 60:.1f} phút. Kết quả ở {out}")
    print(f"Đánh giá: python -m src.evaluate --pred-dir {a.out}")


if __name__ == "__main__":
    main()
