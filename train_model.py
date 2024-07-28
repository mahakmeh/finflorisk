# Updated train_model.py

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def preprocess_data(data, sequence_length=60):
    prices = data['Close'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_prices)):
        X.append(scaled_prices[i-sequence_length:i, 0])
        y.append(scaled_prices[i, 0])
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler, scaled_prices

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_save_model():
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    for ticker in tickers:
        data = fetch_stock_data(ticker, '2022-01-01', '2023-01-01')
        X, y, scaler, scaled_prices = preprocess_data(data)
        
        model = build_model((X.shape[1], 1))
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X, y, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stopping])
        
        model.save(f'{ticker}_lstm_model.h5')
        joblib.dump(scaler, f'{ticker}_scaler.pkl')
        np.save(f'{ticker}_scaled_prices.npy', scaled_prices)

if __name__ == '__main__':
    train_and_save_model()
