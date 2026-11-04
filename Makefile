# Lệnh rút gọn (Linux/macOS, hoặc Git Bash trên Windows). Mỗi đích chỉ gọi lệnh Python tương ứng
# trong REPRODUCE.md, nên có thể chạy tay từng lệnh nếu không có make.
PY ?= python

.PHONY: setup raw features check eval figures eda reproduce train eval-retrain seeds ablation smoke all clean

setup:            ## tạo venv + cài thư viện đã khóa phiên bản
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

raw:              ## (tùy chọn) 4 file Kaggle trong data/raw -> data/interim/Combined_Plant_Data_Updated.csv
	$(PY) -m src.make_dataset

features:         ## Combined_Plant_Data_Updated.csv -> data/processed/features.csv
	$(PY) -m src.build_features

check: features   ## kiểm tra checksum dữ liệu + checkpoint
	$(PY) scripts/check_sha256.py

eval: features    ## đánh giá từ checkpoint có sẵn -> Bảng 4, 5, 6 (khoảng 1 phút)
	$(PY) -m src.predict_checkpoints
	$(PY) -m src.evaluate --pred-dir outputs/checkpoint_eval

figures: eval     ## Hình 2, 3, 4
	$(PY) scripts/make_figures.py --pred-dir outputs/checkpoint_eval

eda: features     ## Mục 3-4, Bảng 1, 2, 3, 7, hệ số Linear Regression
	$(PY) scripts/eda_stats.py

reproduce: check eval figures eda   ## toàn bộ kết quả trong báo cáo từ checkpoint, không huấn luyện lại

train: features   ## huấn luyện lại 6 mô hình từ đầu (CPU khoảng 15-20 phút)
	$(PY) -m src.train --out outputs/retrain

eval-retrain:     ## đánh giá kết quả huấn luyện lại với ngưỡng 'retrain'
	$(PY) -m src.evaluate --pred-dir outputs/retrain --tol retrain

seeds: train      ## RF, XGBoost, LSTM với seed 42, 123, 2024 -> mean ± std
	$(PY) scripts/run_seeds.py --seeds 42 123 2024

ablation: features ## Mục 5.3: số cây của Random Forest / XGBoost
	$(PY) scripts/ablation_trees.py

smoke: features   ## chạy thử pipeline huấn luyện trong ~2 phút (LSTM 2 epoch, không có ARIMAX/SARIMAX)
	$(PY) -m src.train --models LinearRegression RandomForest XGBoost LSTM --max-epochs 2 --out outputs/smoke

all: reproduce train eval-retrain seeds ablation

clean:
	rm -rf outputs data/processed data/interim
