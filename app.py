import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments

from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load models and scalers
models = {
    'AAPL': load_model('AAPL_lstm_model.h5'),
    'MSFT': load_model('MSFT_lstm_model.h5'),
    'GOOGL': load_model('GOOGL_lstm_model.h5')
}
scalers = {
    'AAPL': joblib.load('AAPL_scaler.pkl'),
    'MSFT': joblib.load('MSFT_scaler.pkl'),
    'GOOGL': joblib.load('GOOGL_scaler.pkl')
}
scaled_prices = {
    'AAPL': np.load('AAPL_scaled_prices.npy'),
    'MSFT': np.load('MSFT_scaled_prices.npy'),
    'GOOGL': np.load('GOOGL_scaled_prices.npy')
}

@app.route('/')
def home():
    return "Welcome to the Stock Predictor API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    ticker = data.get('ticker')
    features = data.get('features')

    if ticker not in models:
        return jsonify({'error': 'Invalid ticker'}), 400

    model = models[ticker]
    scaler = scalers[ticker]

    # Preprocess and predict
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    return jsonify({'prediction': prediction.tolist()})

@app.route('/plot/<ticker>')
def plot(ticker):
    if ticker not in models:
        return "Invalid ticker", 400

    model = models[ticker]
    scaler = scalers[ticker]
    historical_prices = scaled_prices[ticker]

    # Predict future prices
    future_steps = 30  # Define how many future steps you want to predict
    last_sequence = historical_prices[-60:]  # Last sequence for prediction

    future_predictions = []
    for _ in range(future_steps):
        X = np.array(last_sequence).reshape(1, -1, 1)
        future_price = model.predict(X)
        future_predictions.append(future_price[0, 0])
        last_sequence = np.append(last_sequence[1:], future_price, axis=0)

    future_predictions = np.array(future_predictions)

    # Concatenate historical prices and future predictions
    full_data = np.concatenate([historical_prices, future_predictions.reshape(-1, 1)], axis=0)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(historical_prices, label='Historical Prices')
    plt.plot(range(len(historical_prices), len(full_data)), future_predictions, color='red', label='Future Predictions')
    plt.title(f'{ticker} Price and Future Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

    # Save plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    return f'<img src="data:image/png;base64,{img_base64}"/>'

if __name__ == '__main__':
    app.run(debug=True)
