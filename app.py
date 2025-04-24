import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Tải mô hình và label encoders
print("Đang tải mô hình...")
try:
    with open("decision_tree_model.pkl", "rb") as f:
        model_data = pickle.load(f)

    model = model_data['model']
    label_encoders = model_data['label_encoders']
    print("✅ Đã tải mô hình thành công!")
except Exception as e:
    print(f"❌ Lỗi khi tải mô hình: {e}")
    print("⚠️ Vui lòng chạy train.py trước để tạo mô hình.")

# Các cột phân loại (phải trùng tên trong file CSV)
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

        # Tạo DataFrame từ input
        input_df = pd.DataFrame([input_data])

        # Mã hóa các cột bằng label encoder đã lưu
        for col in categorical_columns:
            le = label_encoders[col]
            input_df[col] = le.transform(input_df[col])

        # Dự đoán
        prediction = model.predict(input_df)[0]

        return render_template('result.html',
                               flavour=input_data['FlavourEn'],
                               characteristics=input_data['CharacteristicsEn'],
                               category=input_data['CategoryTypeEn'],
                               milk_type=input_data['MilkTypeEn'],
                               milk_treatment=input_data['MilkTreatmentTypeEn'],
                               fat_content=prediction)
    except Exception as e:
        return render_template('result.html', error=f"Có lỗi xảy ra: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
