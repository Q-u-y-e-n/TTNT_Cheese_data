import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Đọc dữ liệu
df = pd.read_csv('cheese_data.csv', encoding='latin1')

# 2. Giữ lại các cột cần thiết
selected_columns = ['FlavourEn', 'CharacteristicsEn', 'CategoryTypeEn', 'MilkTypeEn', 'MilkTreatmentTypeEn', 'FatLevel']
df = df[selected_columns].dropna()

# 3. Label Encoding cho các cột phân loại
categorical_columns = ['FlavourEn', 'CharacteristicsEn', 'CategoryTypeEn', 'MilkTypeEn', 'MilkTreatmentTypeEn']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 4. Encode target (FatLevel)
target_encoder = LabelEncoder()
df['FatLevel'] = target_encoder.fit_transform(df['FatLevel'])

# 5. Tách X, y
X = df.drop(columns='FatLevel')
y = df['FatLevel']

# 6. Tách tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Huấn luyện mô hình
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 8. Đánh giá
print("\n>>> Classification Report (Train):")
print(classification_report(y_train, model.predict(X_train)))

print("\n>>> Classification Report (Test):")
print(classification_report(y_test, model.predict(X_test)))

# 9. Hiển thị ma trận nhầm lẫn
plt.rcParams["figure.figsize"] = (4, 4)
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title("Confusion Matrix - Test")
plt.show()

# 10. Lưu mô hình và encoders
with open("decision_tree_model.pkl", "wb") as f:
    pickle.dump({
        'model': model,
        'label_encoders': label_encoders,
        'target_encoder': target_encoder
    }, f)

print("\n✅ Đã lưu mô hình vào decision_tree_model.pkl")
