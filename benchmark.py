import os
import time
import psutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb

def get_process_memory():
    """Hàm đo dung lượng RAM đang sử dụng (MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# 0. KHỞI TẠO CẤU TRÚC THƯ MỤC
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("================================================================================")
print("          BENCHMARK SYSTEM REPORT: 6 AI MODELS FOR IOT/AGRICULTURE")
print("          Dataset: 15,000 Samples | Auto-exporting Raw & Result CSVs")
print("========================================================================")

# 1. TẠO VÀ LƯU DỮ LIỆU THÔ (RAW DATA)
np.random.seed(42)
n_samples = 15000

timestamps = pd.date_range(start="2026-09-01 00:00:00", periods=n_samples, freq="15min")
df_raw = pd.DataFrame({
    'timestamp': timestamps,
    'temperature': 25 + 7 * np.sin(np.linspace(0, 50, n_samples)) + np.random.normal(0, 1, n_samples),
    'moisture': 60 - 15 * np.sin(np.linspace(0, 50, n_samples)) + np.random.normal(0, 2, n_samples),
    'solar_radiation': np.maximum(0, 500 * np.sin(np.linspace(0, 100, n_samples))) + np.random.normal(0, 10, n_samples),
    'wind_speed': np.abs(np.random.normal(3, 1.5, n_samples)),
    'rain': np.where(np.random.rand(n_samples) > 0.95, np.random.exponential(1.5, n_samples), 0),
    'sensor_status': 0
})

# Chèn ngẫu nhiên 1% lỗi cảm biến để thử nghiệm
anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.01), replace=False)
df_raw.loc[anomaly_indices, 'temperature'] += np.random.choice([-15, 25], size=len(anomaly_indices))

raw_file_path = "data/raw_weather_data_1.csv"#os.path.join("data", "raw_weather_data_1.csv")
df_raw.to_csv(raw_file_path, index_label="raw_sample_id")
print(f"\n[+] Đã tạo và lưu file dữ liệu thô: '{raw_file_path}' ({n_samples} dòng)")

# 2. TIỀN XỬ LÝ DỮ LIỆU (FEATURE ENGINEERING)
df = df_raw.copy()
df['Hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24.0)
df['Hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24.0)
df['Month_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.month / 12.0)
df['Month_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.month / 12.0)

df['Temp_Lag1'] = df['temperature'].shift(1).bfill()
df['Moisture_Lag1'] = df['moisture'].shift(1).bfill()

feature_cols = ['Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 
                'temperature', 'moisture', 'solar_radiation', 'wind_speed', 'rain',
                'Temp_Lag1', 'Moisture_Lag1']

X = df[feature_cols].values
y = df['temperature'].shift(-1).ffill().values  # Dự báo nhiệt độ bước tiếp theo (t+1)

# Chia tập Train/Test theo tỷ lệ 80/20
indices = np.arange(len(df))
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices, test_size=0.2, random_state=42, shuffle=False
)

df_test = df.iloc[idx_test].copy()

# 3. ĐÁNH GIÁ VÀ XUẤT FILE KẾT QUẢ: NHÓM BẤT THƯỜNG (ANOMALY DETECTION)
print("\n=== 2. ĐÁNH GIÁ NỐI TIẾP & XUẤT FILE: NHÓM PHÁT HIỆN BẤT THƯỜNG ===")

anomaly_models = {
    "Isolation_Forest": IsolationForest(n_estimators=100, max_samples=256, contamination=0.03, random_state=42),
    "One_Class_SVM": OneClassSVM(kernel='rbf', gamma='scale', nu=0.03),
    "Local_Outlier_Factor": LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.03)
}

for name, model in anomaly_models.items():
    mem_before = get_process_memory()
    
    # Train
    t0 = time.time()
    model.fit(X_train)
    t_train = (time.time() - t0) * 1000
    
    # Predict
    t0 = time.time()
    preds = model.predict(X_test)
    t_predict_total = (time.time() - t0) * 1000
    latency_per_sample = t_predict_total / len(X_test)
    mem_used = get_process_memory() - mem_before
    
    # Tính điểm anomaly score
    if name == "Isolation_Forest":
        scores = model.decision_function(X_test)
    elif name == "One_Class_SVM":
        scores = model.decision_function(X_test)
    else:  # LOF
        scores = -model.score_samples(X_test)
        
    is_anomaly = np.where(preds == -1, 1, 0)
    
    # Lưu kết quả riêng ra file CSV
    res_df = pd.DataFrame({
        'timestamp': df_test['timestamp'].values,
        'raw_sample_id': idx_test,
        'score': np.round(scores, 4),
        'is_anomaly': is_anomaly,
        'latency_ms': np.round(latency_per_sample, 6)
    })
    
    out_path = os.path.join("results", f"result_{name.lower()}_1.csv")
    res_df.to_csv(out_path, index=False)
    
    print(f"[{name}]")
    print(f"   • Thời gian Train ({len(X_train)} mẫu) : {t_train:.2f} ms")
    print(f"   • Độ trễ suy luận/mẫu (Latency)        : {latency_per_sample:.4f} ms")
    print(f"   • Phát hiện bất thường                 : {np.sum(is_anomaly)}/{len(X_test)} mẫu")
    print(f"   • File kết quả đã xuất                 : '{out_path}'\n")

# 4. ĐÁNH GIÁ VÀ XUẤT FILE KẾT QUẢ: NHÓM DỰ BÁO (FORECASTING)
print("=== 3. ĐÁNH GIÁ NỐI TIẾP & XUẤT FILE: NHÓM DỰ BÁO GIÁ TRỊ ===")

target_cols = ['temperature', 'moisture', 'solar_radiation', 'wind_speed', 'rain']
feature_cols = ['Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 
                'temperature', 'moisture', 'solar_radiation', 'wind_speed', 'rain',
                'Temp_Lag1', 'Moisture_Lag1']

for col in target_cols:
    df[f'target_{col}'] = df[col].shift(-1).ffill()

X = df[feature_cols].values
Y = df[[f'target_{col}' for col in target_cols]].values # Y bây giờ là ma trận 5 cột

indices = np.arange(len(df))
X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(
    X, Y, indices, test_size=0.2, random_state=42, shuffle=False
)

df_test = df.iloc[idx_test].copy()

forecasting_models = {
    "LightGBM": MultiOutputRegressor(
        lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            verbosity=-1,
            force_row_wise=True,
            min_child_samples=10,
            random_state=42
        )
    ),
    "XGBoost": MultiOutputRegressor(
        xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            verbosity=0,
            random_state=42
        )
    ),
    "Ridge": Ridge(alpha=1.0, random_state=42)
}

summary_results = []

for name, model in forecasting_models.items():
    # Train
    t0 = time.time()
    model.fit(X_train, Y_train)
    t_train = (time.time() - t0) * 1000
    
    # Predict
    t0 = time.time()
    preds = model.predict(X_test) # preds sẽ trả ra 5 cột tương ứng với 5 tham số
    latency_per_sample = ((time.time() - t0) * 1000) / len(X_test)
    
    print(f"\n[{name}]")
    print(f"   • Thời gian Train: {t_train:.2f} ms | Latency: {latency_per_sample:.4f} ms/mẫu")
    
    # Đánh giá riêng từng tham số
    for i, col in enumerate(target_cols):
        mae = mean_absolute_error(Y_test[:, i], preds[:, i])
        r2 = r2_score(Y_test[:, i], preds[:, i])
        print(f"      -> {col.upper()}: MAE = {mae:.4f} | R² = {r2:.4f}") 
   
    # Tạo thời gian target_time (t + 15 phút)
    target_times = df_test['timestamp'] + pd.Timedelta(minutes=15)

   # Khởi tạo dictionary chứa dữ liệu cơ bản
    res_data = {
        'timestamp': df_test['timestamp'].values,
        'target_time': target_times.values,
        'latency_ms': np.round(latency_per_sample, 6)
    }
    
   # Dùng vòng lặp để đưa cả 5 tham số (Dự đoán, Thực tế, Sai số) vào CSV
    for i, col in enumerate(target_cols):
        res_data[f'predicted_{col}'] = np.round(preds[:, i], 4)
        res_data[f'actual_{col}'] = np.round(Y_test[:, i], 4)
        res_data[f'error_{col}'] = np.round(np.abs(Y_test[:, i] - preds[:, i]), 4)
        
    res_df = pd.DataFrame(res_data)
    
    out_path = os.path.join("results", f"result_{name.lower()}_1.csv")
    res_df.to_csv(out_path, index=False)
    
    print(f"   • File kết quả đã xuất: '{out_path}'\n")
