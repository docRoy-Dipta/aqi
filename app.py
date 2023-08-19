from flask import Flask, request, jsonify
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

app = Flask(__name__)

# Load the trained model.
model = tf.keras.models.load_model('lstm_model.h5')

# Load the scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route('/')
def home():
    return "Hello World"


@app.route('/predict', methods=['Post'])
def predict():
    nowcast = request.form.get('nowcast')
    raw = request.form.get('raw')
    date = request.form.get('date')

    input_data = pd.DataFrame({
        'NowCast': [nowcast],
        'RawConc': [raw]
    })

    input_data_scaled = scaler.transform(input_data[['NowCast', 'RawConc']])  # Use the loaded scaler
    input_data_reshaped = input_data_scaled.reshape(1, 2, 1)

    predicted_aqi = model.predict(input_data_reshaped)  # Use the loaded model

    predicted_data = pd.DataFrame({
        'NowCast': [input_data_scaled[0, 0]],
        'RawConc': [input_data_scaled[0, 1]],
        'Predicted_AQI': [predicted_aqi[0, 0]]
    })
    predicted_data_actual_scale = predicted_data.copy()
    predicted_data_actual_scale[['NowCast', 'RawConc']] = scaler.inverse_transform(
        predicted_data[['NowCast', 'RawConc']])
    predicted_aqi_actual_scale = predicted_data_actual_scale['Predicted_AQI'].values[0]

    return jsonify(str(predicted_aqi_actual_scale))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
