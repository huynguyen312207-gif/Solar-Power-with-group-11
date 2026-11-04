"""Mục 5.3 báo cáo (số cây): Random Forest 200 cây và XGBoost 42 cây so với cấu hình đã chọn.

    python scripts/ablation_trees.py

Mong đợi (báo cáo): RF 200 cây -> val R2 không đổi (0.7809); XGBoost 42 cây -> val RMSE tốt nhất 157.2,
lệch ban đêm ~34 kW, MAE test mức inverter 71.62 (so với 156.4 / <1 kW / 44.22 khi dùng 800 cây).
Đầu ra: outputs/ablation/ablation_trees.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import evaluate, load_config, p  # noqa: E402
from src.data import prepare  # noqa: E402
from src.models import expand_grid, fit_tabular  # noqa: E402


def main() -> None:
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    cfg = load_config()
    prep = prepare(cfg)
    S, target, feats = prep["S"], prep["target"], prep["features"]
    train = S["train"][S["train"].IS_ANOMALY == 0]
    val, test = S["val"], S["test"]
    # chỉ so trên các bản ghi test mà LSTM cũng có (như Bảng 4)
    lstm = pd.read_csv(p(cfg["paths"]["reference_dir"]) / "pred_test_LSTM.csv", parse_dates=["DATE_TIME"])
    common = test.reset_index().merge(lstm[["DATE_TIME", "SOURCE_KEY_ENCODED"]],
                                      on=["DATE_TIME", "SOURCE_KEY_ENCODED"])["index"]
    rows = []
    rf_cfg, xc = cfg["models"]["RandomForest"], cfg["models"]["XGBoost"]
    runs = [("RandomForest", RandomForestRegressor, n,
             [({**g, "n_estimators": n, "n_jobs": -1, "random_state": rf_cfg["random_state"]}, feats)
              for g in expand_grid({k: v for k, v in rf_cfg["grid"].items() if k != "n_estimators"})])
            for n in (42, 200)]
    runs += [("XGBoost", XGBRegressor, n,
              [({**g, "n_estimators": n, "subsample": xc["subsample"], "colsample_bytree": xc["colsample_bytree"],
                 "random_state": xc["random_state"], "n_jobs": -1}, feats)
               for g in expand_grid({k: v for k, v in xc["grid"].items() if k != "n_estimators"})])
             for n in (42, 800)]
    for name, builder, n, cands in runs:
        m, f, pred, tun = fit_tabular(name, builder, cands, train, val, test, target, log=lambda *_: None)
        tun = pd.DataFrame(tun).sort_values("RMSE")
        t = test.assign(pred=pred).loc[common]
        rows.append({"model": name, "n_estimators": n, "best_params": tun.params.iloc[0],
                     "val_RMSE_best": tun.RMSE.iloc[0], "val_R2_best": tun.R2.iloc[0],
                     "val_RMSE_range": f"{tun.RMSE.min():.1f}-{tun.RMSE.max():.1f}",
                     "test_inv_MAE": evaluate(t[target], t.pred)["MAE"],
                     "night_offset_kW": float(t.loc[t.IRR_RAW == 0, "pred"].mean())})
        print(rows[-1])
    out = p(f"{cfg['paths']['outputs']}/ablation"); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(out / "ablation_trees.csv", index=False)
    print(f"Đã ghi {out / 'ablation_trees.csv'}")


if __name__ == "__main__":
    main()
