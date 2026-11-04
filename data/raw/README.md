# data/raw — dữ liệu gốc Kaggle (tùy chọn)

Chỉ cần khi muốn tái tạo `Combined_Plant_Data_Updated.csv` từ đầu (`make raw`). Repo đã có sẵn file đã gộp.

Nguồn: Ani Kannal, *Solar Power Generation Data*, Kaggle — https://www.kaggle.com/datasets/anikannal/solar-power-generation-data
(giấy phép trên Kaggle: "Data files © Original Authors"; nhóm truy cập ngày 2026-09-15).

Đặt 4 file vào thư mục này:

    Plant_1_Generation_Data.csv       (68,778 dòng)
    Plant_1_Weather_Sensor_Data.csv   (3,182 dòng)
    Plant_2_Generation_Data.csv       (67,698 dòng)
    Plant_2_Weather_Sensor_Data.csv   (3,259 dòng)

Tải bằng Kaggle CLI (cần file ~/.kaggle/kaggle.json):

    kaggle datasets download -d anikannal/solar-power-generation-data -p data/raw --unzip
