from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import math
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

scaler = MinMaxScaler(feature_range=(0,1))

def create_dataset(dataset, time_step=15):
    dataX = []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step)]
        dataX.append(a)
    return np.array(dataX)

model = tf.keras.models.load_model('bitcoin_lstm_model.h5')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        df = pd.read_csv('BTC.csv')
        closedf = df['Close'].values
        
        scaled_data = scaler.fit_transform(closedf.reshape(-1, 1))
        
        last_15_days = scaled_data[-15:]
        X_predict = last_15_days.reshape(1, 15, 1)
        
        scaled_prediction = model.predict(X_predict)
        prediction = scaler.inverse_transform(scaled_prediction)[0][0]
        
        last_price = closedf[-1]
        
        return jsonify({
            'status': 'success',
            'prediction': {
                'date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'price': float(prediction),
                'last_known_price': float(last_price),
                'change_percentage': ((prediction - last_price) / last_price) * 100
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)