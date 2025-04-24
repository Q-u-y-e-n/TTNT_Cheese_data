import pandas as pd
import pickle
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Tải mô hình, scaler, encoders và target_encoder
with open("knn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
scaler = model_data["scaler"]
encoders = model_data["encoders"]
target_encoder = model_data["target_encoder"]  # Tải target_encoder

# Tải lại dữ liệu gốc để lọc động
df_raw = pd.read_csv('cheese_data.csv', encoding='latin1')
df_raw = df_raw[['FlavourEn', 'CharacteristicsEn', 'CategoryTypeEn', 'MilkTypeEn', 'MilkTreatmentTypeEn']].dropna()

@app.route('/')
def index():
    flavour_options = sorted(df_raw['FlavourEn'].unique())
    return render_template('index.html', flavour_options=flavour_options)

@app.route('/get_options', methods=['POST'])
def get_options():
    selected = request.get_json()  # Lấy JSON từ frontend
    filtered = df_raw.copy()

    # Lọc theo dữ liệu đã chọn
    for key, value in selected.items():
        filtered = filtered[filtered[key] == value]

    # Trả về các giá trị có thể tiếp theo
    all_columns = ['FlavourEn', 'CharacteristicsEn', 'CategoryTypeEn', 'MilkTypeEn', 'MilkTreatmentTypeEn']
    result = {}
    for col in all_columns:
        if col not in selected:
            result[col] = sorted(filtered[col].unique())

    return jsonify(result)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'FlavourEn': request.form['flavour'],
            'CharacteristicsEn': request.form['characteristics'],
            'CategoryTypeEn': request.form['category'],
            'MilkTypeEn': request.form['milk_type'],
            'MilkTreatmentTypeEn': request.form['milk_treatment']
        }

        # Mã hóa dữ liệu đầu vào
        input_df = pd.DataFrame([input_data])
        for col in input_data:
            le = encoders[col]
            input_df[col] = le.transform(input_df[col])

        # Chuẩn hóa dữ liệu đầu vào
        input_scaled = scaler.transform(input_df)

        # Dự đoán với mô hình KNN
        prediction = model.predict(input_scaled)[0]

        # Giải mã lại giá trị dự đoán về dạng ban đầu
        fat_content_label = target_encoder.inverse_transform([prediction])[0]  # Sử dụng target_encoder

        return render_template('result.html', **input_data, fat_content=fat_content_label)

    except Exception as e:
        return render_template('result.html', error=f"Có lỗi xảy ra: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
