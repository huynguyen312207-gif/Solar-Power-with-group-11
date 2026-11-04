"""Kiểm tra checksum SHA-256 của dữ liệu và checkpoint (chạy được trên Windows/macOS/Linux).

    python scripts/check_sha256.py              # kiểm tra mọi file có trong SHA256SUMS
    python scripts/check_sha256.py --write      # (dành cho nhóm) tạo lại SHA256SUMS
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "SHA256SUMS"
REF = "model_training-20261019T083032Z-1-001/model_training"
FILES = ["Combined_Plant_Data_Updated.csv", "inverter_ids.csv", "data/processed/features.csv"] + [
    f"{REF}/{n}" for n in ("standard_scaler.pkl", "model_LinearRegression.pkl", "model_RandomForest.pkl",
                           "model_XGBoost.pkl", "model_LSTM.pt", "pred_test_tabular.csv", "pred_test_LSTM.csv",
                           "pred_test_ARIMA.csv", "pred_test_SARIMA.csv", "model_comparison.xlsx")]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.write:
        LIST.write_text("".join(f"{sha256(ROOT / f)}  {f}\n" for f in FILES), encoding="utf-8")
        print(f"Đã ghi {LIST}")
        return
    bad = 0
    for line in LIST.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        f = ROOT / name
        if not f.exists():
            print(f"MISSING  {name}" + ("  (tạo bằng: python -m src.build_features)" if "processed" in name else ""))
            bad += 1
        elif sha256(f) != digest:
            print(f"FAIL     {name}")
            bad += 1
        else:
            print(f"OK       {name}")
    print("Tất cả khớp." if bad == 0 else f"{bad} file thiếu hoặc không khớp.")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
