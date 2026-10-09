import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. TẠO DỮ LIỆU CẢM BIẾN GIẢ ĐỊNH
# ==========================================
np.random.seed(42)
n_samples = 1000
time_steps = np.arange(n_samples)

# 1. Nhiệt độ (Temp): Có tính chu kỳ mạnh (ngày/đêm, mùa)
temperature = 25 + 10 * np.sin(2 * np.pi * time_steps / 100) + np.random.normal(0, 1, n_samples)

# 2. Độ ẩm (Humidity): Tương quan nghịch với nhiệt độ
humidity = 80 - 0.5 * temperature + np.random.normal(0, 2, n_samples)
humidity = np.clip(humidity, 0, 100) # Đảm bảo độ ẩm trong khoảng 0-100%

# 3. Bức xạ mặt trời (Solar): Tương quan thuận với nhiệt độ
solar_radiation = 500 + 20 * temperature + np.random.normal(0, 15, n_samples)
solar_radiation = np.clip(solar_radiation, 0, None) # Không có bức xạ âm

# 4. Tốc độ gió (Wind Speed): Biến động ngẫu nhiên hơn, có chu kỳ ngắn
wind_speed = 5 + 3 * np.sin(2 * np.pi * time_steps / 30) + np.random.normal(0, 1.5, n_samples)
wind_speed = np.clip(wind_speed, 0, None) # Gió không mang giá trị âm

# 5. Lượng mưa (Precipitation): Xảy ra chủ yếu khi độ ẩm cao (> 80%)
precipitation = np.zeros(n_samples)
rain_indices = humidity > 80
# Lượng mưa tuân theo phân phối mũ (exponential) tại các thời điểm có mưa
precipitation[rain_indices] = np.random.exponential(scale=5.0, size=np.sum(rain_indices))

# Tạo DataFrame
df = pd.DataFrame({
    'Temp': temperature, 
    'Humidity': humidity, 
    'Solar': solar_radiation,
    'Wind': wind_speed,
    'Precipitation': precipitation
})

# Tạo bất thường (Anomalies) trên cột Temp để giả lập lỗi cảm biến nhiệt độ
outlier_indices = np.random.choice(n_samples, size=30, replace=False)
df.loc[outlier_indices, 'Temp'] += np.random.normal(50, 10, size=30) 

# ==========================================
# 2. CHIẾN LƯỢC NHÚNG TỰ HỒI QUY (SLIDING WINDOW)
# ==========================================
def create_autoregressive_features(data, lag_p=3):
    """Tạo vectơ đặc trưng từ các bước trễ cho TẤT CẢ các biến."""
    df_lagged = data.copy()
    for col in data.columns:
        for lag in range(1, lag_p + 1):
            df_lagged[f'{col}_lag_{lag}'] = data[col].shift(lag)
    return df_lagged.dropna().reset_index(drop=True)

lag_p = 3
# Khung dữ liệu giờ đây sẽ bao gồm cả trễ của Gió và Mưa
df_features = create_autoregressive_features(df, lag_p=lag_p)

# Tách biến mục tiêu (Khôi phục Temp) và các biến dự báo (X)
y = df_features['Temp']
X = df_features.drop(columns=['Temp'])

# Chuẩn hóa Z-Score - Rất quan trọng vì Mưa (sparse) và Gió (scale nhỏ) khác biệt lớn với Solar (scale lớn)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. PHÁT HIỆN BẤT THƯỜNG BẰNG LOF
# ==========================================
# Cấu hình LOF
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
outlier_labels = lof.fit_predict(X_scaled)

normal_mask = (outlier_labels == 1)
anomaly_mask = (outlier_labels == -1)

X_train_clean = X_scaled[normal_mask]
y_train_clean = y[normal_mask]

# ==========================================
# 4. KHÔI PHỤC DỮ LIỆU BẰNG RIDGE REGRESSION
# ==========================================
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_clean, y_train_clean)

y_imputed = y.copy()
if np.sum(anomaly_mask) > 0:
    y_imputed[anomaly_mask] = ridge.predict(X_scaled[anomaly_mask])

# ==========================================
# 5. SO SÁNH CÁC PHƯƠNG PHÁP KIỂM CHỨNG (VALIDATION)

# A. Standard K-Fold CV
kfold_random = KFold(n_splits=5, shuffle=True, random_state=42)
mse_random = [mean_squared_error(y_train_clean.iloc[test], Ridge(alpha=1.0).fit(X_train_clean[train], y_train_clean.iloc[train]).predict(X_train_clean[test])) for train, test in kfold_random.split(X_train_clean)]
print(f"1. Standard K-Fold CV (Shuffle=True) MSE: {np.mean(mse_random):.4f} -> (Quá lạc quan)")

# B. Blocked Cross-Validation
kfold_blocked = KFold(n_splits=5, shuffle=False)
mse_blocked = [mean_squared_error(y_train_clean.iloc[test], Ridge(alpha=1.0).fit(X_train_clean[train], y_train_clean.iloc[train]).predict(X_train_clean[test])) for train, test in kfold_blocked.split(X_train_clean)]
print(f"2. Blocked CV (Shuffle=False) MSE:        {np.mean(mse_blocked):.4f} -> (Phản ánh đúng cấu trúc cục bộ)")

# C. hv-Block Cross-Validation
h_buffer = 10 
tscv_hv_block = TimeSeriesSplit(n_splits=5, gap=h_buffer)
mse_hv = [mean_squared_error(y_train_clean.iloc[test], Ridge(alpha=1.0).fit(X_train_clean[train], y_train_clean.iloc[train]).predict(X_train_clean[test])) for train, test in tscv_hv_block.split(X_train_clean)]
print(f"3. hv-Block CV (Có khoảng đệm h={h_buffer}) MSE: {np.mean(mse_hv):.4f} -> (Tối ưu nhất, chống rò rỉ triệt để)")

# D. Standard Out-of-Sample (OOS)
X_train_oos, X_test_oos, y_train_oos, y_test_oos = train_test_split(X_train_clean, y_train_clean, test_size=0.15, shuffle=False)
preds_oos = Ridge(alpha=1.0).fit(X_train_oos, y_train_oos).predict(X_test_oos)
print(f"4. Out-of-Sample Evaluation (15% cuối) MSE: {mean_squared_error(y_test_oos, preds_oos):.4f} -> (Kiểm chứng khả năng dự báo)")