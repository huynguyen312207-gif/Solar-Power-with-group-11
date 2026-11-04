# Hướng dẫn tái lập — Ước lượng công suất AC của biến tần điện mặt trời từ dữ liệu thời tiết

Tài liệu này giúp người chưa từng tham gia dự án chạy lại toàn bộ pipeline và thu được đúng các số liệu trong báo cáo *“Estimating Solar Inverter AC Power from Weather Data: Linear, Tree-Based, ARIMAX and LSTM Models Compared”* (file `final.pdf`, 15 trang). Mọi lệnh dưới đây chạy từ **thư mục gốc của repo**.

---

## 0. Thông tin chung

| Mục | Nội dung |
|---|---|
| Đề tài (EN) | Estimating Solar Inverter AC Power from Weather Data: Linear, Tree-Based, ARIMAX and LSTM Models Compared |
| Đề tài (VI) | Ước lượng công suất AC của biến tần điện mặt trời từ dữ liệu thời tiết: so sánh mô hình tuyến tính, mô hình cây, ARIMAX và LSTM |
| Học phần / học kỳ | ADY201m — Fall 2026 — Trường Đại học FPT, TP. Hồ Chí Minh |
| GVHD | Lê Nhật Tùng, Lê Thu |
| Repo | https://github.com/huynguyen312207-gif/Solar-Power-with-group-11 |
| Phiên bản ứng với báo cáo | tag `v1.0-final` — mã commit: `git rev-list -n 1 v1.0-final` (bản DOCX/PDF nộp kèm ghi rõ: `<mã commit>`) |
| Dữ liệu | Có sẵn trong repo: `Combined_Plant_Data_Updated.csv` (17 MB, 136,476 dòng). Nguồn gốc: Kaggle (Mục 5) |
| Checkpoint | Có sẵn trong repo: `model_training-20261019T083032Z-1-001/model_training/` (34 MB) |
| Cập nhật | 04/11/2026 — tài liệu v1.0 |

| Thành viên | MSSV | Email | Vai trò trong pipeline |
|---|---|---|---|
| Nguyễn Gia Huy | SE210697 | huynguyen312207@gmail.com | Phân tích bằng SQL và Python |
| Nguyễn Anh Khôi | SE210598 | khoi28562@gmail.com | Xác định bài toán, câu hỏi nghiên cứu; hiểu và làm sạch dữ liệu |
| Trần Đức Cường | SE191071 | duccuong271205@gmail.com | Trực quan hóa dữ liệu, mô hình hồi quy |

> Người chấm: sau khi clone hãy `git checkout v1.0-final` để chạy đúng phiên bản ứng với báo cáo.

---

## 1. Quick Start (từ máy trống đến Bảng 4–6, khoảng 5 phút)

Cần Python 3.12, 3.13 hoặc 3.14 và Git. Khối lệnh dưới đây **không huấn luyện lại**; nó nạp checkpoint có sẵn và tính lại Bảng 4, 5, 6.

```bash
git clone https://github.com/huynguyen312207-gif/Solar-Power-with-group-11.git
cd Solar-Power-with-group-11
git checkout v1.0-final
python3 -m venv .venv && source .venv/bin/activate         # Windows: py -m venv .venv  rồi  .venv\Scripts\activate
pip install -r requirements.txt                             # 3–5 phút
python -m src.build_features                                # 5 giây  -> data/processed/features.csv
python scripts/check_sha256.py                              # kiểm tra checksum dữ liệu + checkpoint
python -m src.predict_checkpoints                           # 10 giây -> outputs/checkpoint_eval/
python -m src.evaluate --pred-dir outputs/checkpoint_eval   # 5 giây
```

**Kết quả mong đợi** (dòng cuối của lệnh cuối):

```
Đối chiếu với báo cáo (ngưỡng 'exact'): 88/88 PASS
```

Trong đó, Bảng 5 in ra có dòng `LSTM  87.2901  36.2311  0.9282  115.8034  63.1299  0.8687`, khớp Bảng 5 trong báo cáo (RMSE 87.29, R² 0.9282). Toàn bộ 88 con số của Bảng 4, 5, 6 và Mục 7.6 được tự động so với `configs/expected_results.yaml`; chi tiết ở `outputs/checkpoint_eval/results/check_vs_report.csv`.

Nếu có `make` (Linux/macOS/Git Bash), có thể rút gọn: `make setup`, kích hoạt `.venv`, rồi `make reproduce` (thêm Hình 2–4 và Bảng 1, 2, 3, 7).

**Chạy trên Google Colab:** `!git clone ...`, `%cd Solar-Power-with-group-11`, `!git checkout v1.0-final`, `!pip install -r requirements.txt`, chọn *Runtime → Restart session*, rồi chạy 4 lệnh `python ...` ở trên với tiền tố `!`.

---

## 2. Cấu trúc thư mục

```
Solar-Power-with-group-11/
├── ADY201m_Solar_Power.ipynb      # notebook gốc trên Colab (EDA, SQL, huấn luyện) — tham khảo, không cần chạy
├── Combined_Plant_Data_Updated.csv# dữ liệu đã gộp 2 nhà máy + sửa DC_POWER (đầu vào của pipeline)
├── inverter_ids.csv               # ánh xạ mã P1-01..P2-22 trong báo cáo <-> SOURCE_KEY
├── model_training-20261019T083032Z-1-001/model_training/
│   ├── model_LinearRegression.pkl, model_RandomForest.pkl, model_XGBoost.pkl, model_LSTM.pt   # checkpoint
│   ├── standard_scaler.pkl, source_key_mapping.csv
│   ├── train_scaled.csv, val_scaled.csv, test_scaled.csv     # 3 tập đã chia + chuẩn hóa (cố định)
│   ├── pred_test_tabular.csv, pred_test_LSTM.csv, pred_test_ARIMA.csv, pred_test_SARIMA.csv
│   └── model_comparison.xlsx      # bảng kết quả + kết quả thử cấu hình + lịch sử LSTM do notebook xuất
├── configs/
│   ├── default.yaml               # MỌI siêu tham số, đường dẫn, seed
│   └── expected_results.yaml      # số liệu trong báo cáo + ngưỡng sai số chấp nhận
├── src/
│   ├── common.py                  # đọc config, set_seed(), hàm metric
│   ├── make_dataset.py            # (tùy chọn) 4 file Kaggle -> Combined_Plant_Data_Updated.csv
│   ├── build_features.py          # bỏ 4 dòng NaN, tạo HOUR, IS DAYTIME, TEMP DIFF
│   ├── data.py                    # mã hóa, chia 70/15/15 theo thời gian, chuẩn hóa, đánh dấu bất thường
│   ├── models.py                  # 6 mô hình (LR, RF, XGBoost, ARIMAX, SARIMAX, LSTM)
│   ├── train.py                   # huấn luyện lại từ đầu
│   ├── predict_checkpoints.py     # dự đoán test từ checkpoint có sẵn
│   ├── evaluate.py                # Bảng 4, 5, 6, Mục 7.6 + tự đối chiếu với báo cáo
│   └── predict.py                 # suy luận cho 1 bản ghi mới
├── scripts/
│   ├── eda_stats.py               # Mục 3–4, Bảng 1, 2, 3, 7, hệ số Linear Regression
│   ├── make_figures.py            # Hình 2, 3, 4
│   ├── run_seeds.py               # chạy nhiều seed -> mean ± std
│   ├── ablation_trees.py          # Mục 5.3 (số cây RF/XGBoost)
│   └── check_sha256.py            # kiểm tra checksum
├── data/raw/README.md             # cách tải 4 file gốc Kaggle (tùy chọn)
├── SHA256SUMS, requirements.txt, Makefile, LICENSE, .gitignore, .gitattributes
└── REPRODUCE.md
```

Thư mục sinh ra khi chạy (không commit): `data/processed/` (đặc trưng), `outputs/` (dự đoán, bảng, hình, log).

Notebook gốc dùng đường dẫn Google Drive (`/content/drive/MyDrive/Group11_ADY201m/...`) và đọc file `Combined_Plant_Data_Updated_New_Feature_2.csv` không có trong repo. Các script trong `src/` tách đúng logic của notebook (ô 1–2 và 24–50) ra thành lệnh chạy được ở bất kỳ đâu; các khác biệt nhỏ được liệt kê ở Mục 11.

---

## 3. Yêu cầu phần cứng và phần mềm

| Hạng mục | Môi trường nhóm đã dùng (tạo checkpoint) | Môi trường kiểm chứng tài liệu này | Tối thiểu khuyến nghị |
|---|---|---|---|
| Nền tảng | Google Colab, tháng 10/2026 | Linux x86_64 (Ubuntu, glibc 2.39) | Linux / Windows 10+ / macOS 13+ |
| CPU / RAM | Colab mặc định | 2 vCPU, 7 GB RAM | 2 nhân, 8 GB |
| GPU | NVIDIA T4 (chỉ LSTM dùng GPU) | Không có GPU | Không bắt buộc |
| Python | Colab mặc định (không ghi lại) | 3.12.3; Quick Start kiểm thêm trên 3.13.18 và 3.14.6 | 3.12–3.14 |
| Thư viện chính | scikit-learn 1.6.1, xgboost 3.4.1 (đọc từ checkpoint); các gói khác không ghi lại | như `requirements.txt` | như `requirements.txt` |
| Dung lượng đĩa | — | repo khoảng 60 MB + venv 7 GB (torch kèm CUDA) | 2 GB nếu cài torch bản CPU |
| Đánh giá từ checkpoint | — | 20 giây | — |
| Huấn luyện lại 6 mô hình | ARIMAX 1.3 phút, SARIMAX 1.6 phút, LSTM ~1 giây/epoch (T4) | 3.6 phút tổng (LSTM 7 giây/epoch trên CPU) | — |
| 3 seed (RF, XGBoost, LSTM) | — | 9 phút | — |

Dự án là ML truyền thống kết hợp một mô hình DL nhỏ (LSTM 2 lớp, 64 đơn vị); toàn bộ pipeline chạy được trên CPU. RandomForest và XGBoost dùng `n_jobs=-1` (mọi nhân CPU), nên thời gian tỉ lệ nghịch với số nhân.

---

## 4. Cài đặt môi trường

Cách cài: `venv` + `requirements.txt` (khóa phiên bản chính xác).

```bash
python3 -m venv .venv                     # Python 3.12, 3.13 hoặc 3.14
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`scikit-learn==1.6.1` và `xgboost==3.4.1` là đúng phiên bản đã tạo các file `.pkl` trên Colab (đọc được từ metadata của checkpoint); không tự nâng cấp hai gói này. `requirements.txt` tự chọn bản numpy/pandas/matplotlib có sẵn cho từng phiên bản Python (3.12, 3.13, 3.14). Riêng Python 3.14 chưa có scikit-learn 1.6.1 nên dùng 1.7.2; đã kiểm chứng checkpoint vẫn cho đúng 88/88 số liệu. Máy không có GPU có thể cài torch bản CPU (nhẹ hơn khoảng 5 GB) **trước** khi cài phần còn lại:

```bash
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Kiểm tra cài đặt:

```bash
python -c "import sklearn, xgboost, statsmodels, torch, pandas; print(sklearn.__version__, xgboost.__version__, statsmodels.__version__, torch.__version__.split('+')[0], pandas.__version__)"
# Mong đợi: Python 3.12 -> 1.6.1 3.4.1 0.14.5 2.9.1 2.2.2 | 3.13 -> ... 2.2.3 | 3.14 -> 1.7.2 3.4.1 0.14.5 2.9.1 2.3.3
```

Dự án không dùng biến môi trường, API key hay dịch vụ trả phí.

---

## 5. Dữ liệu

### 5.1 Nguồn và giấy phép

*Solar Power Generation Data*, tác giả Ani Kannal, Kaggle (2020): https://www.kaggle.com/datasets/anikannal/solar-power-generation-data. Giấy phép ghi trên Kaggle: “Data files © Original Authors”. Nhóm truy cập ngày 15/09/2026. Dữ liệu ghi nhận từ 2 nhà máy điện mặt trời ở Ấn Độ, 15/05–17/06/2020, chu kỳ 15 phút, không chứa thông tin cá nhân.

### 5.2 Mô tả

| | Plant 1 (4135001) | Plant 2 (4136001) |
|---|---|---|
| File Generation (dòng) | 68,778 | 67,698 |
| File Weather Sensor (dòng) | 3,182 | 3,259 |
| Inverter / cảm biến thời tiết | 22 / 1 | 22 / 1 |
| Mốc thời gian có dữ liệu Generation | 3,158 / 3,264 (khoảng trống dài nhất 9 giờ, đêm 20–21/5) | 3,259 / 3,264 |
| AMBIENT_TEMPERATURE (°C) | 20.40–35.25 | 20.94–39.18 |
| MODULE_TEMPERATURE (°C) | 18.14–65.55 | 20.27–66.64 |
| IRRADIATION (không rõ đơn vị) | 0–1.22 | 0–1.10 |
| Bản ghi bất thường (IRRADIATION > 0.3 và AC_POWER = 0) | 62 | 3,723 |

Biến mục tiêu: `AC_POWER` (kW), hồi quy liên tục. Các file gốc không có ô trống.

### 5.3 Cách lấy dữ liệu và checksum

Repo đã có sẵn file đã gộp `Combined_Plant_Data_Updated.csv`, nên **không cần tải gì thêm**. Kiểm tra đúng file:

```bash
python scripts/check_sha256.py
# Mong đợi: 13 dòng "OK" rồi "Tất cả khớp."  (Linux/macOS cũng có thể dùng: sha256sum -c SHA256SUMS)
```

| File | SHA-256 |
|---|---|
| `Combined_Plant_Data_Updated.csv` | `d9f841e92c1f835b7f7d633f1b14ac0de63a22a1df8237f9b72d5cd50a2eb738` |
| `data/processed/features.csv` (sinh ra ở 5.4) | `58c789d520f4c34b882c34ec41b9a84752bb883075b3d19e245664423b9eb539` |

Checksum của 10 file checkpoint và dự đoán nằm trong `SHA256SUMS`.

*(Tùy chọn)* Tái tạo file đã gộp từ 4 file gốc: tải dữ liệu theo `data/raw/README.md` (cần tài khoản Kaggle), rồi chạy `python -m src.make_dataset`. Lệnh ghi ra `data/interim/Combined_Plant_Data_Updated.csv` (136,476 dòng × 11 cột).

### 5.4 Tiền xử lý

| # | Bước | Mã nguồn | Số dòng còn lại |
|---|---|---|---|
| 1 | Đọc 4 file; DATE_TIME của Plant 1 Generation theo định dạng `dd-mm-yyyy HH:MM` | `src/make_dataset.py` | — |
| 2 | Left join Generation + Weather theo (DATE_TIME, PLANT_ID), ghép 2 nhà máy | `src/make_dataset.py` | 136,476 |
| 3 | Chia DC_POWER của Plant 1 cho 10 (tỷ lệ AC/DC 0.10 → 0.98; DC_POWER không dùng làm đầu vào) | `src/make_dataset.py` | 136,476 |
| 4 | Bỏ 4 dòng không có thời tiết (Plant 1, 03/06/2020 14:00) | `src/build_features.py` | 136,472 (68,774 + 67,698) |
| 5 | Tạo `HOUR`, `IS DAYTIME` (06:00–17:45), `TEMP DIFF` = MODULE − AMBIENT | `src/build_features.py` | 136,472 |
| 6 | Label Encoding SOURCE_KEY (RF, XGBoost, embedding LSTM); one-hot SOURCE_KEY 43 cột (LR); one-hot PLANT_ID; bỏ SOURCE_KEY_weather | `src/data.py` | 136,472 |
| 7 | Chia tập (5.5), StandardScaler **fit trên train** cho AMBIENT, MODULE, IRRADIATION, HOUR, TEMP DIFF | `src/data.py` | 136,472 |
| 8 | Đánh dấu dòng bất thường (IRRADIATION gốc > 0.3 và AC_POWER = 0); **bỏ khỏi train**, giữ trong val/test | `src/data.py` | train còn 92,704 |
| 9 | LSTM: cửa sổ 16 bước (4 giờ) theo từng inverter, thêm HOUR_SIN/HOUR_COS, bỏ cửa sổ vắt qua khoảng trống | `src/models.py` | 85,662 / 20,460 / 20,174 mẫu |

Lệnh: `python -m src.build_features` (bước 4–5); bước 6–9 chạy tự động bên trong `src.train` và `src.predict_checkpoints`.

### 5.5 Chia tập

Chia 70/15/15 **theo thời gian, không xáo trộn**: sắp xếp theo (DATE_TIME, SOURCE_KEY_ENCODED), cắt tại dòng thứ 70% và 85%, làm tròn tới mốc thời gian (Bảng 2 báo cáo). Không có thành phần ngẫu nhiên.

| Tập | Khoảng thời gian | Dòng | Tỷ lệ | Dòng bất thường |
|---|---|---|---|---|
| Train | 15/05/2020 00:00 – 08/06/2020 06:45 | 95,508 | 70.0% | 2,804 |
| Validation | 08/06/2020 07:00 – 13/06/2020 03:00 | 20,460 | 15.0% | 493 |
| Test | 13/06/2020 03:15 – 17/06/2020 23:45 | 20,504 | 15.0% | 488 |

File chia tập cố định: `model_training-.../train_scaled.csv`, `val_scaled.csv`, `test_scaled.csv` (do notebook xuất). Mỗi lần chạy, `src.train` ghi lại bảng trên vào `split_summary.csv`. Tập test chỉ dùng để báo cáo, không dùng để chọn siêu tham số.

---

## 6. Cấu hình và siêu tham số

Toàn bộ siêu tham số, đường dẫn và seed nằm trong `configs/default.yaml`; mã nguồn không hard-code giá trị nào. Mọi mô hình chọn cấu hình theo **RMSE trên tập validation** bằng grid search (thử hết các tổ hợp). Kết quả thử nằm trong `tuning_tabular.csv` và `tuning_ts.csv` (khi huấn luyện lại), hoặc trong các sheet `Tuning_*` của `model_comparison.xlsx` (checkpoint gốc).

| Mô hình | Cấu hình đã thử | Chọn | Val RMSE (kW) |
|---|---|---|---|
| Linear Regression (OLS) | SOURCE_KEY one-hot vs label | one-hot, 48 cột | 162.3 (label: 163.8) |
| Random Forest | 42 cây; max_depth ∈ {10, 16}; min_samples_leaf ∈ {5, 20} | depth 10, leaf 20 | 163.2 |
| XGBoost | 800 cây; max_depth ∈ {4, 6, 8}; learning_rate ∈ {0.05, 0.1}; subsample = colsample_bytree = 0.8 | depth 8, lr 0.05 | 156.4 |
| ARIMAX (mỗi nhà máy 1 mô hình, chuỗi 15 phút) | (p,d,q) ∈ {(1,0,1), (2,0,1), (2,0,2), (1,1,1)}; exog: IRRADIATION, MODULE, AMBIENT; trend = 'c' | P1 (1,0,1); P2 (2,0,2) | 39.8 / 127.8 |
| SARIMAX (chuỗi theo giờ, s = 24) | (1,0,1)(1,0,0)₂₄; (1,0,1)(0,1,1)₂₄; (2,0,1)(1,0,1)₂₄ | P1, P2: (1,0,1)(0,1,1)₂₄ | 73.8 / 150.6 |
| LSTM | 2 lớp × 64, dropout 0.2, embedding inverter 4 chiều, head 68→32→1 (ReLU); Adam lr 1e-3; batch 256; tối đa 40 epoch; early stopping patience 6, min_delta 0.1 kW; loss MSE trên AC_POWER / max(train) | epoch tốt nhất 9 (dừng ở 15) | 152.4 |

Mọi dự đoán âm được cắt về 0. Không dùng mixed precision hay gradient accumulation. Không dùng mô hình pretrained.

---

## 7. Huấn luyện

Huấn luyện lại cả 6 mô hình từ đầu, với seed theo config (khớp báo cáo):

```bash
python -m src.build_features            # nếu chưa chạy
python -m src.train --out outputs/retrain
python -m src.evaluate --pred-dir outputs/retrain --tol retrain
# Mong đợi: "Đối chiếu với báo cáo (ngưỡng 'retrain'): 88/88 PASS"
```

Huấn luyện riêng từng mô hình, kể cả baseline Linear Regression:

```bash
python -m src.train --models LinearRegression --out outputs/lr
python -m src.train --models RandomForest XGBoost --out outputs/trees
python -m src.train --models ARIMA SARIMA --out outputs/ts
python -m src.train --models LSTM --device cpu --out outputs/lstm      # --device cuda nếu có GPU
```

Đầu ra trong thư mục `--out`: checkpoint (`model_*.pkl`, `model_LSTM.pt`), dự đoán test (`pred_test_*.csv`), kết quả thử cấu hình (`tuning_*.csv`), lịch sử LSTM theo epoch (`history_lstm.csv`), `split_summary.csv`, `standard_scaler.pkl` và `run_info.json` (seed, thiết bị, phiên bản, thời gian chạy từng mô hình).

Thời gian đo được trên 2 vCPU, không GPU: LR 0.2 s, RF 12 s, XGBoost 16 s, ARIMAX 25 s, SARIMAX 54 s, LSTM 106 s (15 epoch); tổng **3.6 phút**. Vì toàn bộ chạy dưới 15 phút trên CPU, không cần cấu hình rút gọn; `make smoke` (LSTM 2 epoch, không có ARIMAX/SARIMAX) vẫn có sẵn để kiểm tra nhanh pipeline trong khoảng 1 phút.

LSTM lưu trạng thái tốt nhất theo val RMSE (không lưu epoch cuối) và không hỗ trợ `--resume`, vì một lần huấn luyện chỉ mất khoảng 2 phút. ARIMAX/SARIMAX: notebook gốc không lưu mô hình statsmodels, chỉ lưu dự đoán; muốn tạo lại thì chạy `src.train --models ARIMA SARIMA` (mất khoảng 1.5 phút, kết quả tất định).

---

## 8. Đánh giá và suy luận

**Đánh giá từ checkpoint, không huấn luyện lại:**

```bash
python -m src.predict_checkpoints          # nạp model_*.pkl, model_LSTM.pt -> outputs/checkpoint_eval/pred_test_*.csv
python -m src.evaluate --pred-dir outputs/checkpoint_eval
```

Metric: RMSE, MAE, R² tính bằng `sklearn.metrics` (`mean_squared_error` rồi lấy căn, `mean_absolute_error`, `r2_score`). Biến thể `*_day` chỉ tính trên các bản ghi ban ngày, định nghĩa là IRRADIATION chưa chuẩn hóa > 0. Hai mức đánh giá:

- **Mức inverter (Bảng 4):** 4 mô hình LR, RF, XGBoost, LSTM trên 20,174 bản ghi test mà cả 4 cùng có (LSTM bỏ 15 bước đầu mỗi đoạn liên tục).
- **Mức nhà máy (Bảng 5, 6):** trung bình dự đoán của các inverter trong một nhà máy tại mỗi mốc 15 phút, ghép với ARIMAX/SARIMAX trên 917 mốc chung. Cột “without anomalous” ở Bảng 6 bỏ các bản ghi inverter bất thường **trước** khi lấy trung bình, và giữ đúng 917 mốc đó.

Đầu ra trong `outputs/checkpoint_eval/results/`: `table4_inverter.csv`, `table5_plant.csv`, `table6_per_plant.csv`, `p2_01_analysis.csv`, `plant_level_predictions.csv`, `metrics_test.json`, `check_vs_report.csv`.

**Suy luận cho dữ liệu mới** (LR, RF, XGBoost; một thời điểm của một inverter):

```bash
python -m src.predict --model XGBoost --inverter P2-05 --time "2020-06-15 12:00" --irradiation 0.85 --ambient 32.5 --module 55.0
# Mong đợi: {"model": "XGBoost", "inverter": "P2-05", "SOURCE_KEY": "4UPUqMRk7TRMgml", "plant": 4136001, "time": "2020-06-15 12:00:00", "AC_POWER_kW": 1144.69}
```

LSTM cần chuỗi 4 giờ thời tiết liên tiếp nên không có trong lệnh này. Dự án không có demo web hoặc API.

---

## 9. Kiểm soát ngẫu nhiên và tính tất định

| Nguồn ngẫu nhiên | Giá trị trong báo cáo | Nơi đặt |
|---|---|---|
| `numpy` (xáo trộn mẫu train của LSTM) | 42 | `seed` trong config → `set_seed()` |
| `torch` (khởi tạo trọng số, dropout LSTM) | 42 | `seed` → `set_seed()` |
| Random Forest `random_state` | 42 | `models.RandomForest.random_state` |
| XGBoost `random_state` | 800 | `models.XGBoost.random_state` |
| Chia tập, Linear Regression, ARIMAX, SARIMAX | không ngẫu nhiên | — |

`src/common.py::set_seed()` đặt `random`, `numpy`, `torch`, `PYTHONHASHSEED`, `cudnn.deterministic=True`, `cudnn.benchmark=False`. Notebook gốc chỉ gọi `np.random.seed(42)` và `torch.manual_seed(42)`, không bật cờ cuDNN tất định và không dùng `torch.use_deterministic_algorithms`. Vì vậy LSTM chạy lại trên GPU khác hoặc trên CPU sẽ lệch nhẹ so với checkpoint. Cờ `--seed S` ghi đè **cả** seed numpy/torch **và** `random_state` của RF/XGBoost.

**Kết quả nhiều seed** (`python scripts/run_seeds.py --seeds 42 123 2024`, CPU; LR/ARIMAX/SARIMAX tất định nên dùng chung một lần chạy). Kết quả từng seed ở `outputs/seeds/seeds_summary.csv`.

| Mô hình | Inverter RMSE | Inverter R² | Plant RMSE | Plant R² | Plant R²_day | Báo cáo (plant R²) |
|---|---|---|---|---|---|---|
| LSTM | 144.91 ± 0.78 | 0.8239 ± 0.0019 | 87.32 ± 0.92 | 0.9281 ± 0.0015 | 0.8687 ± 0.0028 | 0.9282 |
| XGBoost | 153.28 ± 0.25 | 0.8029 ± 0.0006 | 99.31 ± 1.04 | 0.9071 ± 0.0019 | 0.8301 ± 0.0036 | 0.9070 |
| Random Forest | 154.44 ± 0.55 | 0.8000 ± 0.0014 | 103.79 ± 0.85 | 0.8985 ± 0.0017 | 0.8144 ± 0.0030 | 0.8993 |

*(mean ± std qua 3 seed, std mẫu.)* Số trong báo cáo đều nằm trong khoảng mean ± 1 std. Độ lệch giữa các seed (≈ 0.002 R²) cùng cỡ với khoảng cách giữa ARIMAX (0.9088), Linear Regression (0.9082) và XGBoost (0.9070). Điều này củng cố nhận xét trong báo cáo rằng khác biệt giữa các mô hình này chưa đủ để kết luận.

**Ngưỡng sai số chấp nhận** (khai báo trong `configs/expected_results.yaml`):

- Đánh giá từ checkpoint (`--tol exact`): phải khớp tới chữ số làm tròn của báo cáo, tức R² ± 0.00005 và RMSE/MAE ± 0.005 kW.
- Huấn luyện lại (`--tol retrain`): R² ± 0.01 (1 điểm %), RMSE/MAE ± 5%. Lần chạy kiểm chứng (CPU, seed theo config) đạt 88/88 PASS. Ví dụ, R² mức nhà máy của LSTM là 0.9292 so với 0.9282, đúng bằng con số “CPU-only run” mà báo cáo nêu ở Mục 7.8.

---

## 10. Ánh xạ kết quả ↔ báo cáo

Các lệnh dùng `outputs/checkpoint_eval` cần chạy `python -m src.predict_checkpoints` trước. Lệnh `make reproduce` chạy tất cả các dòng đánh dấu ✓.

| Bảng/Hình/Mục trong báo cáo | Lệnh tái tạo | Config | File output | Giá trị chính |
|---|---|---|---|---|
| Hình 1 — sơ đồ pipeline | vẽ tay bằng TikZ trong file LaTeX, không sinh từ dữ liệu | — | — | — |
| Mục 3 — mô tả dữ liệu ✓ | `python scripts/eda_stats.py` | default.yaml | `outputs/eda/dataset_stats.json` | 136,472 dòng; 3,158/3,259 mốc |
| Hình 2 — AC power vs irradiation ✓ | `python scripts/make_figures.py` | default.yaml | `outputs/figures/fig2_ac_vs_irradiation.pdf` | bất thường 62 / 3,723 |
| Mục 4.1 — tương quan Pearson ✓ | `python scripts/eda_stats.py` | default.yaml | `outputs/eda/dataset_stats.json` | 0.99 / 0.78 |
| Bảng 1 — theo nhóm DAILY_YIELD ✓ | `python scripts/eda_stats.py` | default.yaml | `outputs/eda/table1_daily_yield.csv` | 69,759 / 32,663 / 34,050 |
| Mục 4.2 — hiệu suất AC/DC, trung vị ✓ | `python scripts/eda_stats.py` | default.yaml | `outputs/eda/dataset_stats.json` | 0.9776–0.9792; 353 kW (P2) |
| Bảng 2 — chia tập ✓ | `python scripts/eda_stats.py` | default.yaml | `outputs/eda/table2_split.csv` | 95,508 / 20,460 / 20,504 |
| Bảng 3 — cấu hình đã thử ✓ | `python scripts/eda_stats.py` (hoặc `src.train`) | default.yaml | `outputs/eda/table3_tuning.csv`; LSTM: sheet `History_LSTM` trong `model_comparison.xlsx` (hoặc `history_lstm.csv` khi huấn luyện lại) | XGB 156.4; LSTM 152.4 (epoch 9) |
| Mục 5.3 — số cây RF/XGBoost | `python scripts/ablation_trees.py` | default.yaml | `outputs/ablation/ablation_trees.csv` | RF 200 cây val R² 0.7809 |
| Bảng 4 — mức inverter ✓ | `python -m src.evaluate --pred-dir outputs/checkpoint_eval` | expected_results.yaml | `.../results/table4_inverter.csv` | LSTM RMSE 144.88 |
| Bảng 5 — mức nhà máy ✓ | như trên | expected_results.yaml | `.../results/table5_plant.csv` | LSTM RMSE 87.29, R² 0.9282 |
| Hình 3 — dự đoán vs thực tế, Plant 2 ✓ | `python scripts/make_figures.py` | — | `outputs/figures/fig3_pred_vs_actual_plant2.pdf` | — |
| Bảng 6 — R² theo nhà máy ✓ | `python -m src.evaluate ...` | expected_results.yaml | `.../results/table6_per_plant.csv` | LSTM P2 0.7900 → 0.9410 |
| Mục 7.4 — hệ số Linear Regression ✓ | `python scripts/eda_stats.py` | default.yaml | `outputs/eda/lr_coefficients.csv` | IRRADIATION 365.86 |
| Bảng 7 — độ quan trọng đặc trưng ✓ | `python scripts/eda_stats.py` | — | `outputs/eda/table7_importance.csv` | 0.994 / 0.777 |
| Hình 4 — phần dư LSTM theo inverter ✓ | `python scripts/make_figures.py` | — | `outputs/figures/fig4_lstm_residuals_plant2.pdf`, `fig4_residual_summary.csv` | P2-01: trung bình −167 kW |
| Mục 7.6 — bỏ P2-01 / dòng bất thường ✓ | `python -m src.evaluate ...` | expected_results.yaml | `.../results/p2_01_analysis.csv` | 0.8173; 0.9410; 0.9414 |
| Mục 7.8 — chạy lại trên CPU | `python -m src.train --device cpu` | default.yaml | `outputs/retrain/results/` | LSTM plant R² 0.9292 |
| (Bổ sung) mean ± std 3 seed | `python scripts/run_seeds.py` | default.yaml | `outputs/seeds/seeds_mean_std.csv` | Mục 9 tài liệu này |

---

## 11. Hạn chế và xử lý sự cố

**Phần không tái lập hoàn toàn được, hoặc khác với báo cáo:**

1. **Bước gộp từ 4 file Kaggle gốc** (`src/make_dataset.py`) chưa được chạy trên file Kaggle thật trong lần kiểm chứng này, vì tải từ Kaggle cần tài khoản. Logic đã được kiểm tra khứ hồi: tách `Combined_Plant_Data_Updated.csv` thành 4 file đúng định dạng gốc rồi gộp lại cho ra file giống hệt. Pipeline chính bắt đầu từ file đã gộp có trong repo và có checksum.
2. **LSTM trên phần cứng khác.** Checkpoint tạo trên GPU T4 nên đánh giá từ checkpoint khớp tuyệt đối. Huấn luyện lại trên CPU cho R² mức nhà máy 0.9292 thay vì 0.9282 (nằm trong ngưỡng ở Mục 9). Đây là chênh lệch số học giữa CPU và GPU, cộng với việc notebook không bật cờ cuDNN tất định.
3. **SARIMAX.** Mục 7.8 báo cáo rằng một lần chạy chỉ dùng CPU đã chọn bậc SARIMAX khác (R² 0.8653). Trong lần kiểm chứng này (statsmodels 0.14.5, CPU), SARIMAX chọn đúng bậc như báo cáo và cho R² 0.8889. Bộ tối ưu của statsmodels có thể hội tụ khác nhau tùy BLAS và phiên bản thư viện.
4. **Mục 4.2, trung vị Plant 1.** Script cho trung vị các trung vị inverter khi IRRADIATION > 0 là 565.6 kW ở Plant 1, còn báo cáo ghi 586 kW. Giá trị Plant 2 (353 kW) và nhận xét “không inverter Plant 1 nào dưới 85% trung vị” vẫn khớp. Con số 586 kW không truy được về đoạn mã nào trong notebook.
5. **Mục 5.3, XGBoost 42 cây.** Huấn luyện lại cho val RMSE 157.0, độ lệch ban đêm 33.9 kW và MAE test 70.84 kW; báo cáo ghi 157.2, ~34 kW và 71.62 kW. Chênh lệch nhỏ, nhiều khả năng do khác phiên bản thư viện hoặc số luồng tính toán, và không đổi kết luận của Mục 5.3 (800 cây tốt hơn 42 cây).
6. **Khác biệt nhỏ giữa notebook và script.** (a) Checkpoint Linear Regression gốc có 50 cột, thêm `TEMP DIFF` và `PLANT_ID_4136001`. Hai cột này là tổ hợp tuyến tính của các cột khác, nên dự đoán trùng với bản 48 cột mô tả trong báo cáo (sai khác < 1e-6 kW); `src.train` dùng 48 cột cho khớp với Mục 5.2 và hệ số ở Mục 7.4. (b) File `Combined_Plant_Data_Updated_New_Feature_2.csv` mà notebook đọc có thêm cột `DAY OF WEEK` không mô hình nào dùng; `src.build_features` không tạo cột này, và đã kiểm tra rằng scaler và số dòng train khớp với checkpoint. (c) Notebook chạy ARIMAX/SARIMAX song song bằng `multiprocessing.Pool(2)`; script chạy tuần tự để chạy được trên Windows, kết quả không đổi.
7. **Một lần chia tập.** Báo cáo dùng một lần chia theo thời gian, không cross-validation, không kiểm định thống kê; tài liệu này chỉ bổ sung kết quả 3 seed (Mục 9).

**Lỗi thường gặp:**

| Thông báo lỗi | Nguyên nhân | Cách khắc phục |
|---|---|---|
| `InconsistentVersionWarning: Trying to unpickle estimator ... from version 1.6.1` hoặc `AttributeError` khi nạp `.pkl` | scikit-learn khác 1.6.1 | `pip install scikit-learn==1.6.1 xgboost==3.4.1` |
| `FileNotFoundError: data/processed/features.csv` | chưa tạo đặc trưng | `python -m src.build_features` |
| `ModuleNotFoundError: No module named 'src'` | chạy lệnh không ở thư mục gốc repo, hoặc chạy `python src/train.py` | `cd` về gốc repo và dùng `python -m src.train` |
| `check_sha256.py` báo `FAIL Combined_Plant_Data_Updated.csv` trên Windows | Git đổi LF→CRLF khi clone (repo cũ chưa có `.gitattributes`) | `git config core.autocrlf false`, xóa thư mục rồi clone lại |
| Colab: `numpy.dtype size changed` sau `pip install` | gói đã nạp trước khi hạ phiên bản numpy/pandas | *Runtime → Restart session*, rồi chạy tiếp |
| `pip` không tìm thấy bản phù hợp, hoặc tải file `.tar.gz` rồi đứng rất lâu ở *Preparing metadata* | Python ngoài khoảng 3.12–3.14 nên pip phải tự build từ mã nguồn | dùng Python 3.12–3.14 (`python --version` để kiểm tra) |
| Hết dung lượng khi cài torch | bản PyPI kèm CUDA (~5 GB) | cài bản CPU như Mục 4 |

---

## 12. Giấy phép, trích dẫn, khai báo AI

- **Mã nguồn của nhóm:** MIT License (file `LICENSE`).
- **Dữ liệu:** Solar Power Generation Data (Kaggle, A. Kannal), “Data files © Original Authors”. Dữ liệu không thuộc phạm vi giấy phép MIT của repo; khi dùng lại cần trích dẫn nguồn gốc. Dự án không dùng mô hình pretrained.
- **Thư viện chính:** pandas, NumPy, scikit-learn [Pedregosa et al., 2011], XGBoost [Chen & Guestrin, 2016], statsmodels [Seabold & Perktold, 2010], PyTorch, Matplotlib; DuckDB chỉ dùng trong phần truy vấn SQL của notebook. Đoạn mã tham khảo hoặc chỉnh sửa từ nguồn ngoài: không có; mã do nhóm tự viết, có dùng công cụ AI hỗ trợ như khai báo ở mục dưới.
- **Công cụ AI tạo sinh:** theo báo cáo, nhóm dùng ChatGPT (OpenAI), Claude, Copilot và Google Gemini để kiểm tra ngữ pháp, chỉnh tiếng Anh, hỗ trợ viết mã, phát triển thuật toán và phân tích dữ liệu. Tài liệu `REPRODUCE.md` và các script trong `src/`, `scripts/` được soạn với hỗ trợ của Claude (Anthropic), dựa trên notebook của nhóm. Mọi số liệu được kiểm chứng bằng cách chạy lại và tự đối chiếu với báo cáo (`check_vs_report.csv`). Các thành viên đã kiểm tra lại và chịu trách nhiệm về nội dung.
- **Trích dẫn dự án:**

```bibtex
@misc{group11_ady201m_2026,
  title  = {Estimating Solar Inverter AC Power from Weather Data: Linear, Tree-Based, ARIMAX and LSTM Models Compared},
  author = {Nguyen, Gia Huy and Nguyen, Anh Khoi and Tran, Duc Cuong and Le, Thu and Le, Nhat-Tung},
  year   = {2026},
  note   = {ADY201m Group 11, FPT University},
  howpublished = {\url{https://github.com/huynguyen312207-gif/Solar-Power-with-group-11}}
}
```
