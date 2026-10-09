import requests
import pandas as pd
import numpy as np
import warnings
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

# ==========================================
# 1. ĐỌC & TIỀN XỬ LÝ DỮ LIỆU TỪ FIREBASE
# ==========================================
firebase_url = "https://directionproject-1e798-default-rtdb.firebaseio.com/user/measured_data/field1/measured_data.json"
print(f"Đang tải dữ liệu từ {firebase_url}...\n")
response = requests.get(firebase_url)

if response.status_code != 200:
    raise ConnectionError(f"Lỗi kết nối Firebase: {response.status_code}")

raw_data = response.json()
records = []
for date_key, daily_data in raw_data.items():
    if isinstance(daily_data, dict):
        for time_key, metrics in daily_data.items():
            if isinstance(metrics, dict):
                record = {'Date': date_key, 'RecordID': time_key}
                record.update(metrics)
                records.append(record)

df = pd.DataFrame(records)

if 'time' in df.columns:
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['time'], errors='coerce', utc=True, format='mixed')
else:
    df['Datetime'] = pd.to_datetime(df['Date'], errors='coerce', utc=True, format='mixed')

df = df.sort_values('Datetime').reset_index(drop=True)
df.set_index('Datetime', inplace=True)

# LỌC VÀ ĐỔI TÊN CÁC CỘT DỮ LIỆU THỰC TẾ
# Bỏ qua các cột trống/hỏng (như RADIATION, RAINFALL...)
col_mapping = {
    't': 'Nhiet_Do',
    'h': 'Do_Am',
    'rad': 'Buc_Xa',
    'rai': 'Mua',
    'w': 'Gio'
}

# Chỉ chọn những cột thực sự tồn tại trong dataframe và nằm trong mapping
available_cols = [c for c in col_mapping.keys() if c in df.columns]
df_numeric = df[available_cols].copy()
df_numeric.rename(columns=col_mapping, inplace=True)

df_numeric.ffill(inplace=True)
df_numeric.dropna(inplace=True)

# ==========================================
# 2. HÀM TẠO ĐẶC TRƯNG TỰ HỒI QUY
# ==========================================
def create_autoregressive_features(data, lag_p=3):
    df_lagged = data.copy()
    for col in data.columns:
        for lag in range(1, lag_p + 1):
            df_lagged[f'{col}_lag_{lag}'] = data[col].shift(lag)
    return df_lagged.dropna().reset_index(drop=True)

lag_p = 3
df_features = create_autoregressive_features(df_numeric, lag_p=lag_p)
actual_numeric_cols = df_numeric.columns.tolist()

# ==========================================
# 3. CHẠY PIPELINE CHO TỪNG BIẾN ĐỘC LẬP
# ==========================================
for target_col in actual_numeric_cols:
    print(f"{'='*50}")
    print(f"PHÂN TÍCH & KHÔI PHỤC BIẾN: {target_col.upper()}")
    print(f"{'='*50}")
    
    y = df_features[target_col]
    X = df_features.drop(columns=[target_col])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    np.random.seed(42)
    X_scaled_noisy = X_scaled + np.random.normal(0, 1e-5, X_scaled.shape)
    
    optimal_neighbors = max(30, min(50, len(X_scaled) // 10))
    
    # SỬ DỤNG CONTAMINATION="AUTO" ĐỂ BẮT BẤT THƯỜNG DỰA TRÊN MẬT ĐỘ THỰC TẾ
    lof = LocalOutlierFactor(n_neighbors=optimal_neighbors, contamination="auto")
    outlier_labels = lof.fit_predict(X_scaled_noisy)
    
    normal_mask = (outlier_labels == 1)
    anomaly_mask = (outlier_labels == -1)
    
    X_train_clean = X_scaled[normal_mask]
    y_train_clean = y[normal_mask]
    
    print(f"-> LOF phát hiện {np.sum(anomaly_mask)} bất thường trên tổng {len(y)} bản ghi.")
    
    if len(X_train_clean) == 0:
        print("   [!] Không đủ dữ liệu sạch để huấn luyện.")
        continue

    # Huấn luyện Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_clean, y_train_clean)
    
    # Validation
    kfold_blocked = KFold(n_splits=5, shuffle=False)
    mse_blocked = [mean_squared_error(y_train_clean.iloc[test], Ridge(alpha=1.0).fit(X_train_clean[train], y_train_clean.iloc[train]).predict(X_train_clean[test])) for train, test in kfold_blocked.split(X_train_clean)]
    
    h_buffer = int(len(X_train_clean) * 0.05)
    tscv_hv_block = TimeSeriesSplit(n_splits=5, gap=h_buffer)
    mse_hv = [mean_squared_error(y_train_clean.iloc[test], Ridge(alpha=1.0).fit(X_train_clean[train], y_train_clean.iloc[train]).predict(X_train_clean[test])) for train, test in tscv_hv_block.split(X_train_clean)]
    
    X_train_oos, X_test_oos, y_train_oos, y_test_oos = train_test_split(X_train_clean, y_train_clean, test_size=0.15, shuffle=False)
    preds_oos = Ridge(alpha=1.0).fit(X_train_oos, y_train_oos).predict(X_test_oos)
    mse_oos = mean_squared_error(y_test_oos, preds_oos)
    
    print(f"   [bl-CV] MSE      : {np.mean(mse_blocked):.4f}")
    print(f"   [hv-Block] MSE   : {np.mean(mse_hv):.4f} (Buffer h={h_buffer})")
    print(f"   [OOS 15%] MSE    : {mse_oos:.4f}\n")
