import pandas as pd
import matplotlib.pyplot as plt

# Cấu hình danh sách các biến cần vẽ
variables = ['Nhiet_Do', 'Do_Am', 'Buc_Xa', 'Mua', 'Gio']
colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']

# ==========================================
# 1. ĐỒ THỊ DỮ LIỆU GỐC (1_firebase_raw_data.csv)
# ==========================================
# Đọc file không dùng index_col để tránh lỗi nếu cột index là số
df_raw = pd.read_csv('1_firebase_raw_data.csv')

# Kiểm tra nếu có cột Datetime thì đặt làm index, nếu không dùng index mặc định
if 'Datetime' in df_raw.columns:
    df_raw['Datetime'] = pd.to_datetime(df_raw['Datetime'])
    df_raw.set_index('Datetime', inplace=True)

fig1, axes1 = plt.subplots(nrows=5, ncols=1, figsize=(15, 12), sharex=True)
fig1.suptitle('1. Dữ Liệu Khí Tượng Gốc Từ Firebase', fontsize=16, fontweight='bold', y=0.98)

for i, col in enumerate(variables):
    if col in df_raw.columns:
        axes1[i].plot(df_raw.index, df_raw[col], color=colors[i], label=col, linewidth=1)
        axes1[i].set_ylabel(col, fontweight='bold')
        axes1[i].legend(loc='upper right')
        axes1[i].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('chart_1_raw_data.png', dpi=300)
plt.close(fig1)
print("Đã lưu đồ thị 1 vào file: 'chart_1_raw_data.png'")

# ==========================================
# 2. ĐỒ THỊ KẾT QUẢ BẤT THƯỜNG (2_anomaly_detection_results.csv)
# ==========================================
df_anomalies = pd.read_csv('2_anomaly_detection_results.csv')

if 'Datetime' in df_anomalies.columns:
    df_anomalies['Datetime'] = pd.to_datetime(df_anomalies['Datetime'])
    df_anomalies.set_index('Datetime', inplace=True)

fig2, axes2 = plt.subplots(nrows=5, ncols=1, figsize=(15, 12), sharex=True)
fig2.suptitle('2. Điểm Dữ Liệu Bất Thường (LOF Detection)', fontsize=16, fontweight='bold', y=0.98)

for i, col in enumerate(variables):
    anomaly_col = f'{col}_Is_Anomaly'
    if anomaly_col in df_anomalies.columns:
        anomalies_only = df_anomalies[df_anomalies[anomaly_col] == True]
        
        axes2[i].scatter(df_anomalies.index, df_anomalies[anomaly_col].astype(int), 
                         color='lightgray', s=1, label='Bình thường')
        axes2[i].scatter(anomalies_only.index, anomalies_only[anomaly_col].astype(int), 
                         color='red', s=10, label=f'Lỗi {col}')
        
        axes2[i].set_ylabel(col, fontweight='bold')
        axes2[i].set_yticks([0, 1])
        axes2[i].set_yticklabels(['Bình thường', 'Bất thường'])
        axes2[i].legend(loc='upper right')
        axes2[i].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('chart_2_anomalies.png', dpi=300)
plt.close(fig2)
print("Đã lưu đồ thị 2 vào file: 'chart_2_anomalies.png'")

# ==========================================
# 3. ĐỒ THỊ DỮ LIỆU SAU KHÔI PHỤC (3_imputed_predictions_results.csv)
# ==========================================
df_imputed = pd.read_csv('3_imputed_predictions_results.csv')

if 'Datetime' in df_imputed.columns:
    df_imputed['Datetime'] = pd.to_datetime(df_imputed['Datetime'])
    df_imputed.set_index('Datetime', inplace=True)

fig3, axes3 = plt.subplots(nrows=5, ncols=1, figsize=(15, 12), sharex=True)
fig3.suptitle('3. Dữ Liệu Đã Được Khôi Phục (Ridge Regression Imputation)', fontsize=16, fontweight='bold', y=0.98)

for i, col in enumerate(variables):
    if col in df_imputed.columns:
        axes3[i].plot(df_imputed.index, df_imputed[col], color=colors[i], label=f'{col} (Imputed)', linewidth=1)
        axes3[i].set_ylabel(col, fontweight='bold')
        axes3[i].legend(loc='upper right')
        axes3[i].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('chart_3_imputed.png', dpi=300)
plt.close(fig3)
print("Đã lưu đồ thị 3 vào file: 'chart_3_imputed.png'")
