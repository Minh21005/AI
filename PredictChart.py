import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Cấu hình danh sách 3 mô hình và 5 tham số thời tiết
models = ['lightgbm', 'xgboost', 'ridge']
colors = {'lightgbm': '#2ca02c', 'xgboost': '#ff7f0e', 'ridge': '#1f77b4'}
target_cols = ['temperature', 'moisture', 'solar_radiation', 'wind_speed', 'rain']

col_titles = {
    'temperature': '1. Nhiệt độ (°C)',
    'moisture': '2. Độ ẩm (%)',
    'solar_radiation': '3. Bức xạ mặt trời (W/m²)',
    'wind_speed': '4. Tốc độ gió (m/s)',
    'rain': '5. Lượng mưa (mm)'
}

# Kiểm tra đường dẫn file dữ liệu gốc (ưu tiên file raw_weather_data_1.csv từ benchmark.py)
raw_path = "data/raw_weather_data_1.csv"

if os.path.exists(raw_path):
    df_raw = pd.read_csv(raw_path)
    # Tách 3,000 mẫu cuối (Tập kiểm thử sau 12,000 mẫu Train)
    df_act = df_raw.iloc[12000:].reset_index(drop=True)
else:
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu gốc tại 'data/raw_weather_data_1.csv' hoặc 'data/raw_weather_data.csv'")

time_col = 'timestamp' if 'timestamp' in df_act.columns else df_act.columns[0]

# 2. Khởi tạo Khung biểu đồ 5 hàng (Subplots)
fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
fig.suptitle('SO SÁNH DỰ BÁO CỦA 3 MÔ HÌNH (LIGHTGBM, XGBOOST, RIDGE) TRÊN 5 THAM SỐ THỜI TIẾT', 
             fontsize=13, fontweight='bold', y=0.99)

# Số mẫu hiển thị trên biểu đồ (đồng bộ giữa Actual và Predicted)
n_show = 300

# 3. Lặp qua từng tham số để vẽ biểu đồ
for i, col in enumerate(target_cols):
    ax = axes[i]
    
    # Line 1: Giá trị Thực tế gốc (dùng df_act[col] thay vì df_act[f'actual_{col}'])
    ax.plot(pd.to_datetime(df_act[time_col])[:n_show], df_act[col][:n_show], 
            color='black', linewidth=2, label='Thực tế (Actual)')
    
    # Line 2, 3, 4: Dự báo của LightGBM, XGBoost, Ridge (Đường nét đứt 3 màu)
    for m in models:
        file_path = f"results/result_{m}_1.csv"
        if os.path.exists(file_path):
            df_m = pd.read_csv(file_path)
            ax.plot(pd.to_datetime(df_m['target_time'])[:n_show], df_m[f'predicted_{col}'][:n_show], 
                    label=f'Dự báo ({m.upper()})', color=colors[m], linestyle='--', alpha=0.8)
    
    ax.set_title(col_titles[col], fontsize=11, fontweight='bold', loc='left')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')

plt.xlabel('Thời gian (Target Time)', fontsize=11, fontweight='bold')
plt.tight_layout()

# 4. Lưu ảnh độ phân giải cao
plt.savefig("results/compare_3_models_5_vars.png", dpi=300, bbox_inches='tight')
plt.show()
print("[✔] Đã xuất thành công: 'results/compare_3_models_5_vars.png'")
