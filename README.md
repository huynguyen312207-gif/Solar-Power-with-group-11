# Solar Power with Group 11

Estimating Solar Inverter AC Power from Weather Data: Linear, Tree-Based, ARIMAX and LSTM Models Compared
(ADY201m, Group 11, FPT University).

- **Hướng dẫn tái lập kết quả:** xem [`REPRODUCE.md`](REPRODUCE.md).
- Chạy nhanh (Python 3.12):

```bash
pip install -r requirements.txt
python -m src.build_features
python -m src.predict_checkpoints
python -m src.evaluate --pred-dir outputs/checkpoint_eval   # -> 88/88 PASS
```

- Notebook gốc (Colab): `ADY201m_Solar_Power.ipynb`
- Ánh xạ mã inverter P1-01..P2-22 ↔ SOURCE_KEY: `inverter_ids.csv`
