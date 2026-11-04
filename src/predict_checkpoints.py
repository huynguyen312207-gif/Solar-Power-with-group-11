"""Bước 3a: dự đoán tập test từ checkpoint đã huấn luyện sẵn (KHÔNG huấn luyện lại).

    python -m src.predict_checkpoints                       # đọc checkpoint gốc trong paths.reference_dir
    python -m src.predict_checkpoints --ckpt-dir outputs/retrain --out outputs/retrain_from_ckpt

- Linear Regression / Random Forest / XGBoost: model_*.pkl (joblib, lưu kèm danh sách cột).
- LSTM: model_LSTM.pt (state_dict PyTorch).
- ARIMAX / SARIMAX: notebook không lưu mô hình statsmodels, chỉ lưu dự đoán -> sao chép pred_test_ARIMA.csv,
  pred_test_SARIMA.csv từ thư mục checkpoint (muốn tạo lại thì chạy src.train --models ARIMA SARIMA).
"""
from __future__ import annotations

import argparse
import shutil
import warnings

import joblib
import numpy as np

from src.common import load_config, p, set_seed
from src.data import prepare
from src.models import build_windows, lstm_predict, make_lstm


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=cfg["paths"]["reference_dir"])
    ap.add_argument("--out", default=f"{cfg['paths']['outputs']}/checkpoint_eval")
    a = ap.parse_args()
    ck, out = p(a.ckpt_dir), p(a.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                   # cảnh báo khác phiên bản khi unpickle
        scaler = joblib.load(ck / "standard_scaler.pkl")
    prep = prepare(cfg, scaler=scaler)
    refit = prepare(cfg)["scaler"]
    assert np.allclose(refit.mean_, scaler.mean_) and np.allclose(refit.scale_, scaler.scale_), \
        "Scaler trong checkpoint khác scaler fit lại trên train -> dữ liệu đầu vào không khớp"
    S, target = prep["S"], prep["target"]
    test = S["test"]

    pred_tab = test[["DATE_TIME", "SOURCE_KEY_ENCODED", "PLANT_ID_4136001", target, "IRR_RAW",
                     "IS_ANOMALY"]].copy()
    for name in ("LinearRegression", "RandomForest", "XGBoost"):
        m, feats = joblib.load(ck / f"model_{name}.pkl")
        pred_tab[name] = np.clip(m.predict(test[feats]), 0, None)
        print(f"{name:17s} <- {ck.name}/model_{name}.pkl ({len(feats)} cột)")
    pred_tab.to_csv(out / "pred_test_tabular.csv", index=False)

    import torch
    lc = cfg["models"]["LSTM"]
    X, I, y, M, mask, y_scale = build_windows(S, cfg["features"]["sequence"], target, lc["window"])
    net = make_lstm(X.shape[2], lc)
    net.load_state_dict(torch.load(ck / "model_LSTM.pt", map_location="cpu"))
    pred = M.loc[mask["test"], ["DATE_TIME", "SOURCE_KEY_ENCODED"]].copy()
    pred["LSTM"] = lstm_predict(net, torch.tensor(X), torch.tensor(I), mask["test"], y_scale, "cpu")
    pred.to_csv(out / "pred_test_LSTM.csv", index=False)
    print(f"LSTM              <- {ck.name}/model_LSTM.pt ({int(mask['test'].sum())} mẫu test)")

    for name in ("ARIMA", "SARIMA"):
        shutil.copy(ck / f"pred_test_{name}.csv", out / f"pred_test_{name}.csv")
        print(f"{name:17s} <- {ck.name}/pred_test_{name}.csv (sao chép, không có checkpoint statsmodels)")
    print(f"\nĐã ghi dự đoán vào {out}. Tiếp theo: python -m src.evaluate --pred-dir {a.out}")


if __name__ == "__main__":
    main()
