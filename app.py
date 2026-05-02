from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ======================
# LOAD MODEL & SCALER
# ======================
model = load_model("model.h5")
scaler_X = pickle.load(open("scaler_X.pkl", "rb"))
scaler_y = pickle.load(open("scaler_y.pkl", "rb"))

# ======================
# ROUTE
# ======================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
            float(request.form['curah_hujan']),
            float(request.form['suhu']),
            float(request.form['kelembaban']),
            float(request.form['populasi']),
            float(request.form['kepadatan'])
        ]

        data = np.array([data])
        data_scaled = scaler_X.transform(data)

        pred_scaled = model.predict(data_scaled)
        hasil = scaler_y.inverse_transform(pred_scaled)

        return render_template('index.html', hasil=round(hasil[0][0], 2))

    except:
        return render_template('index.html', hasil="Input tidak valid!")

# ======================
# RUN
# ======================
if __name__ == '__main__':
    app.run(debug=True)