import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Tải mô hình đã huấn luyện
print("Đang tải mô hình...")
try:
    with open("knn_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    
    knn_model = model_data['model']
    scaler = model_data['scaler']
    label_encoders = model_data['encoders']
    print("✅ Đã tải mô hình thành công!")
except Exception as e:
    print(f"❌ Lỗi khi tải mô hình: {e}")
    print("Vui lòng chạy file train.py trước để tạo mô hình.")

# Các cột phân loại
categorical_columns = ['FlavourEn', 'CharacteristicsEn', 'CategoryTypeEn', 'MilkTypeEn', 'MilkTreatmentTypeEn']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu từ form
        input_data = {
            'FlavourEn': request.form['flavour'],
            'CharacteristicsEn': request.form['characteristics'],
            'CategoryTypeEn': request.form['category'],
            'MilkTypeEn': request.form['milk_type'],
            'MilkTreatmentTypeEn': request.form['milk_treatment']
        }

        # Tạo DataFrame
        input_df = pd.DataFrame([input_data])

        # Mã hóa các cột phân loại bằng encoder đã lưu
        for col in categorical_columns:
            if input_df[col][0] not in label_encoders[col].classes_:
                raise ValueError(f"Giá trị '{input_df[col][0]}' không hợp lệ cho '{col}'.")
            input_df[col] = label_encoders[col].transform(input_df[col])

        # Chuẩn hóa
        input_scaled = scaler.transform(input_df)

        # Dự đoán
        prediction = knn_model.predict(input_scaled)[0]

        return render_template('result.html',
                               flavour=input_data['FlavourEn'],
                               characteristics=input_data['CharacteristicsEn'],
                               category=input_data['CategoryTypeEn'],
                               milk_type=input_data['MilkTypeEn'],
                               milk_treatment=input_data['MilkTreatmentTypeEn'],
                               fat_content=prediction)
    except Exception as e:
        return render_template('result.html', error=f"Có lỗi xảy ra: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
