"""Hàm dùng chung: đọc cấu hình, đặt seed, tính metric, đường dẫn."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]          # thư mục gốc repo


def load_config(path: str | os.PathLike = "configs/default.yaml") -> dict:
    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def p(rel: str | os.PathLike) -> Path:
    """Đường dẫn tương đối (tính từ gốc repo) -> đường dẫn tuyệt đối."""
    return ROOT / rel


def set_seed(seed: int = 42) -> None:
    """Đặt seed cho mọi nguồn ngẫu nhiên dùng trong dự án."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def evaluate(y_true, y_pred, daytime_mask=None) -> dict:
    """RMSE, MAE, R2 (sklearn.metrics); thêm *_day nếu có mặt nạ ban ngày (IRRADIATION gốc > 0)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    res = {"RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
           "MAE": float(mean_absolute_error(y_true, y_pred)),
           "R2": float(r2_score(y_true, y_pred))}
    if daytime_mask is not None:
        m = np.asarray(daytime_mask, dtype=bool)
        res.update({"RMSE_day": float(np.sqrt(mean_squared_error(y_true[m], y_pred[m]))),
                    "MAE_day": float(mean_absolute_error(y_true[m], y_pred[m])),
                    "R2_day": float(r2_score(y_true[m], y_pred[m]))})
    return res


NAMES = {"LinearRegression": "Linear Regression", "RandomForest": "Random Forest", "XGBoost": "XGBoost",
         "ARIMA": "ARIMAX", "SARIMA": "SARIMAX", "LSTM": "LSTM"}
INV_MODELS = ["LinearRegression", "RandomForest", "XGBoost", "LSTM"]
ALL_MODELS = ["LinearRegression", "RandomForest", "XGBoost", "ARIMA", "SARIMA", "LSTM"]
