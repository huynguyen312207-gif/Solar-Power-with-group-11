"""Huấn luyện lại Random Forest, XGBoost, LSTM với nhiều seed và tổng hợp mean ± std (Mục 9 REPRODUCE.md).

    python scripts/run_seeds.py --seeds 42 123 2024

Linear Regression (OLS), ARIMAX, SARIMAX không có thành phần ngẫu nhiên nên không chạy lại theo seed;
để có Bảng 5 đầy đủ, mỗi lần chạy dùng dự đoán của 3 mô hình này từ outputs/retrain (chạy src.train trước).
Lưu ý: --seed ghi đè random_state của RF/XGB, nên seed 42 ở đây dùng XGBoost random_state=42
(báo cáo dùng 800) -> kết quả XGBoost của seed 42 khác một chút so với outputs/retrain.

Đầu ra: outputs/seeds/seed_<s>/ và outputs/seeds/seeds_summary.csv, seeds_mean_std.csv
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import evaluate, NAMES  # noqa: E402
from src.evaluate import load_preds, to_plant  # noqa: E402

SEED_MODELS = ["RandomForest", "XGBoost", "LSTM"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    ap.add_argument("--base", default="outputs/retrain", help="nơi lấy dự đoán LR/ARIMAX/SARIMAX")
    ap.add_argument("--out", default="outputs/seeds")
    ap.add_argument("--skip-train", action="store_true", help="chỉ tổng hợp lại từ các thư mục đã có")
    a = ap.parse_args()
    base, out = ROOT / a.base, ROOT / a.out
    rows = []
    for s in a.seeds:
        d = out / f"seed_{s}"
        if not a.skip_train:
            subprocess.run([sys.executable, "-m", "src.train", "--models", *SEED_MODELS, "--seed", str(s),
                            "--out", str(d.relative_to(ROOT))], cwd=ROOT, check=True)
        # ghép LR + ARIMAX/SARIMAX (tất định) từ lần chạy cơ sở
        tab = pd.read_csv(d / "pred_test_tabular.csv")
        tab["LinearRegression"] = pd.read_csv(base / "pred_test_tabular.csv")["LinearRegression"].values
        tab.to_csv(d / "pred_test_tabular.csv", index=False)
        for m in ("ARIMA", "SARIMA"):
            shutil.copy(base / f"pred_test_{m}.csv", d / f"pred_test_{m}.csv")
        inv, ts = load_preds(d)
        plant = to_plant(inv, ts)
        info = json.loads((d / "run_info.json").read_text(encoding="utf-8"))
        for m in SEED_MODELS:
            ri = evaluate(inv.AC_POWER, inv[m], inv.IRR_RAW > 0)
            rp = evaluate(plant.y_true, plant[m], plant.IRR_RAW > 0)
            row = {"seed": s, "model": NAMES[m], **{f"inv_{k}": v for k, v in ri.items()},
                   **{f"plant_{k}": v for k, v in rp.items()}}
            if m == "LSTM":
                row["lstm_best_epoch"] = info.get("lstm_best_epoch")
            rows.append(row)
    df = pd.DataFrame(rows)
    df.round(4).to_csv(out / "seeds_summary.csv", index=False)
    cols = ["inv_RMSE", "inv_MAE", "inv_R2", "plant_RMSE", "plant_MAE", "plant_R2", "plant_R2_day"]
    agg = df.groupby("model")[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg.round(4).to_csv(out / "seeds_mean_std.csv")
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", None)
    print(df[["seed", "model"] + cols].round(4).to_string(index=False))
    print("\nmean ± std:")
    for m, r in agg.iterrows():
        print(f"  {m:14s} " + "  ".join(f"{c}={r[c + '_mean']:.4f}±{r[c + '_std']:.4f}" for c in cols))
    print(f"\nĐã ghi {out / 'seeds_summary.csv'} và {out / 'seeds_mean_std.csv'}")


if __name__ == "__main__":
    main()
