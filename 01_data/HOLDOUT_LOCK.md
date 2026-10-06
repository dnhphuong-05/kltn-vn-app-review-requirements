# ⚠ HOLD-OUT LOCK

`processed/shopee_holdout.csv` và `processed/lazada_external_test.csv` là tập **khóa**.

KHÔNG dùng cho: training, tuning hyperparameter, chọn ngưỡng, tuning prompt, chọn mô hình,
hay bất kỳ quyết định thiết kế nào. Chỉ mở **một lần** sau khi toàn bộ tuning trên Validation
đã chốt và config/prompt đã đóng băng (`03_configs/frozen/`, `04_prompts/`).

Ghi lại ngày mở và commit hash tại đây:

| tập | ngày mở | commit | người mở |
|---|---|---|---|
|  |  |  |  |
