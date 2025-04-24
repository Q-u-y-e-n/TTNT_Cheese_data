import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
df = pd.read_csv("cheese_data.csv")

# Kiểm tra và xử lý giá trị thiếu
df = df.dropna(subset=['FlavourEn', 'CharacteristicsEn', 'CategoryTypeEn', 'MilkTypeEn', 'MilkTreatmentTypeEn', 'FatLevel'])

# Các cột phân loại
categorical_columns = ['FlavourEn', 'CharacteristicsEn', 'CategoryTypeEn', 'MilkTypeEn', 'MilkTreatmentTypeEn']

# Khởi tạo và áp dụng LabelEncoder cho các cột phân loại
encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Khởi tạo LabelEncoder cho cột mục tiêu 'FatLevel'
target_encoder = LabelEncoder()
df['FatLevel'] = target_encoder.fit_transform(df['FatLevel'])  # Chuyển đổi 'FatLevel' thành số

# Chọn cột đầu vào (X) và đầu ra (y)
X = df[categorical_columns]
y = df['FatLevel']

# Chuẩn hóa
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (có thể bỏ nếu không đánh giá)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Lưu mô hình, scaler, và encoders
with open("knn_model.pkl", "wb") as f:
    pickle.dump({
        "model": knn_model,
        "scaler": scaler,
        "encoders": encoders,
        "target_encoder": target_encoder  # Lưu target_encoder
    }, f)

print("✅ Đã lưu mô hình, scaler và encoders thành công!")
