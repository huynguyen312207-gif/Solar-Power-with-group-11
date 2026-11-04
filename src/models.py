"""Định nghĩa và huấn luyện 6 mô hình (ô 34-46 của notebook), tách ra để dùng lại khi đánh giá checkpoint."""
from __future__ import annotations

import itertools
import time
import warnings

import numpy as np
import pandas as pd

from src.common import evaluate

# ----------------------------------------------------------------------------- mô hình dạng bảng


def expand_grid(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def fit_tabular(name, builder, candidates, train, val, test, target, log=print):
    """candidates: list of (params, feats). Chọn theo RMSE validation, dự đoán test (cắt âm về 0)."""
    tuning, best = [], None
    for params, feats in candidates:
        t = time.time()
        m = builder(**params).fit(train[feats], train[target])
        r = evaluate(val[target], m.predict(val[feats]), val.IRR_RAW > 0)
        shown = {k: v for k, v in params.items() if k not in ("n_jobs", "random_state")}
        tuning.append({"model": name, "params": str(shown), "n_features": len(feats), **r,
                       "fit_s": round(time.time() - t, 1)})
        log(f"  {name} {shown} ({len(feats)} cột) -> val RMSE={r['RMSE']:.2f}  ({time.time() - t:.0f}s)")
        if best is None or r["RMSE"] < best[1]["RMSE"]:
            best = (m, r, feats)
    m, rv, feats = best
    pred = np.clip(m.predict(test[feats]), 0, None)
    return m, feats, pred, tuning


# ----------------------------------------------------------------------------- ARIMAX / SARIMAX (mức nhà máy)


def plant_series(d: pd.DataFrame, plant: int, target: str, exog: list, freq: str = "15min") -> pd.DataFrame:
    """AC_POWER trung bình mỗi inverter của 1 nhà máy tại mỗi mốc 15 phút; nội suy mốc thiếu."""
    s = d[d.PLANT_ID_4136001 == plant].groupby("DATE_TIME")[[target] + exog + ["IRR_RAW"]].mean()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="15min")).interpolate(limit_direction="both")
    if freq != "15min":
        s = s.resample(freq).mean().interpolate(limit_direction="both")
    return s


def fit_timeseries_plant(name, plant, candidates, freq, data, target, exog):
    """Fit 1 nhà máy: chọn cấu hình theo RMSE validation (trên lưới 15 phút), nạp val, dự báo test."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    warnings.filterwarnings("ignore")
    orig = {k: plant_series(data[k], plant, target, exog, "15min") for k in ("train", "val", "test")}
    res = {k: plant_series(data[k], plant, target, exog, freq) for k in ("train", "val", "test")}

    def upsample(pred, idx_from, idx_to):
        return pd.Series(pred, index=idx_from).reindex(idx_to, method="ffill").ffill().bfill()

    best, tuning = None, []
    for c in candidates:
        t = time.time()
        kw = {"order": tuple(c["order"]), "seasonal_order": tuple(c["seasonal_order"])}
        fit = SARIMAX(res["train"][target], exog=res["train"][exog], **kw, trend="c",
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
        pv = np.clip(fit.forecast(len(res["val"]), exog=res["val"][exog]).values, 0, None)
        pv = upsample(pv, res["val"].index, orig["val"].index)
        r = evaluate(orig["val"][target], pv, orig["val"].IRR_RAW > 0)
        tuning.append({"model": name, "plant": plant, "order": str(kw["order"]),
                       "seasonal_order": str(kw["seasonal_order"]), "AIC": fit.aic, **r,
                       "fit_s": round(time.time() - t, 1)})
        if best is None or r["RMSE"] < best[1]["RMSE"]:
            best = (fit, r, kw)
    fit, rv, kw = best
    # Nạp dữ liệu val vào trạng thái, KHÔNG ước lượng lại tham số, rồi dự báo toàn bộ test
    fit2 = fit.append(res["val"][target], exog=res["val"][exog], refit=False)
    pt = np.clip(fit2.forecast(len(res["test"]), exog=res["test"][exog]).values, 0, None)
    pt = upsample(pt, res["test"].index, orig["test"].index)
    pred = pd.DataFrame({"DATE_TIME": orig["test"].index, "PLANT_ID_4136001": plant, "model": name,
                         "y_true": orig["test"][target].values, "y_pred": pt.values,
                         "IRR_RAW": orig["test"].IRR_RAW.values})
    return kw, rv, tuning, pred


# ----------------------------------------------------------------------------- LSTM


def build_windows(S: dict, seq_feats: list, target: str, win: int):
    """Cửa sổ trượt 'win' bước cho từng inverter; bỏ cửa sổ vắt qua khoảng dữ liệu bị mất."""
    parts = []
    for k, d in S.items():
        d = d.copy()
        d["SPLIT"] = k
        h = d.DATE_TIME.dt.hour + d.DATE_TIME.dt.minute / 60
        d["HOUR_SIN"], d["HOUR_COS"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
        parts.append(d)
    dl = pd.concat(parts).sort_values(["SOURCE_KEY_ENCODED", "DATE_TIME"]).reset_index(drop=True)
    Xs, ids, meta = [], [], []
    for inv, g in dl.groupby("SOURCE_KEY_ENCODED"):
        F = g[seq_feats].to_numpy(np.float32)
        ok = np.diff(g.DATE_TIME.to_numpy()) == np.timedelta64(15, "m")
        run = np.concatenate([[0], np.cumsum(~ok)])
        for e in range(win - 1, len(g)):
            if run[e] == run[e - win + 1]:
                Xs.append(F[e - win + 1:e + 1]); ids.append(inv); meta.append(g.index[e])
    X, I = np.stack(Xs), np.array(ids)
    M = dl.loc[meta].reset_index(drop=True)
    y = M[target].to_numpy(np.float32)
    y_scale = float(M.loc[M.SPLIT == "train", target].max())         # chuẩn hóa target theo max của train
    mask = {k: (M.SPLIT == k).to_numpy().copy() for k in ("train", "val", "test")}
    mask["train"] &= (M.IS_ANOMALY == 0).to_numpy()
    return X, I, y, M, mask, y_scale


def make_lstm(n_feat: int, cfg_lstm: dict, n_inv: int = 44):
    import torch.nn as nn
    import torch

    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            h, e = cfg_lstm["hidden"], cfg_lstm["embedding"]
            self.lstm = nn.LSTM(n_feat, h, num_layers=cfg_lstm["layers"], batch_first=True,
                                dropout=cfg_lstm["dropout"])
            self.emb = nn.Embedding(n_inv, e)                           # nhúng mã inverter
            self.head = nn.Sequential(nn.Linear(h + e, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x, inv):
            out, _ = self.lstm(x)
            return self.head(torch.cat([out[:, -1], self.emb(inv)], 1)).squeeze(1)

    return LSTMNet()


def lstm_predict(net, Xt, It, m, y_scale, device):
    import torch
    net.eval()
    out, idx_all = [], np.where(m)[0]
    with torch.no_grad():
        for i in range(0, len(idx_all), 4096):
            idx = idx_all[i:i + 4096]
            out.append(net(Xt[idx].to(device), It[idx].to(device)).cpu().numpy())
    return np.clip(np.concatenate(out) * y_scale, 0, None)


def train_lstm(X, I, y, mask, y_scale, cfg_lstm, device, log=print):
    import torch
    import torch.nn as nn
    Xt, It = torch.tensor(X), torch.tensor(I)
    net = make_lstm(X.shape[2], cfg_lstm).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg_lstm["lr"])
    yt = torch.tensor(y / y_scale)
    tr_idx = np.where(mask["train"])[0]
    bs, best, best_state, wait, hist = cfg_lstm["batch_size"], 1e9, None, 0, []
    for ep in range(1, cfg_lstm["max_epochs"] + 1):
        t = time.time()
        net.train()
        np.random.shuffle(tr_idx)                                         # xáo trộn mẫu trong train
        for i in range(0, len(tr_idx), bs):
            b = tr_idx[i:i + bs]
            loss = nn.functional.mse_loss(net(Xt[b].to(device), It[b].to(device)), yt[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        rv = evaluate(y[mask["val"]], lstm_predict(net, Xt, It, mask["val"], y_scale, device))
        hist.append({"epoch": ep, "train_loss": loss.item(), **rv, "epoch_s": round(time.time() - t, 1)})
        log(f"  epoch {ep:2d}  val RMSE={rv['RMSE']:.2f}  ({time.time() - t:.0f}s)")
        if rv["RMSE"] < best - cfg_lstm["min_delta"]:
            best, wait = rv["RMSE"], 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg_lstm["patience"]:
                log("  early stopping")
                break
    net.load_state_dict(best_state)
    return net, best_state, pd.DataFrame(hist), Xt, It
