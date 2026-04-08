import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model
import LinearRegression
import matplotlib.pyplot as plt

st.title("📈 Stock Price Trend Predictor")

# User input
stock = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, INFY)", "AAPL")

if stock:
    # Load data
    data = yf.download(stock, start="2020-01-01", end="2024-01-01")

    st.write("### Raw Data")
    st.write(data.tail())

    # Plot closing price
    st.write("### Closing Price Chart")
    fig = plt.figure()
    plt.plot(data['Close'])
    plt.xlabel("Date")
    plt.ylabel("Price")
    st.pyplot(fig)

    # Prepare data for ML
    data['Days'] = np.arange(len(data))
    X = data[['Days']]
    y = data['Close']

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Predict future
    future_days = st.slider("Days to Predict", 1, 30, 5)

    future_X = np.array(range(len(data), len(data) + future_days)).reshape(-1, 1)
    predictions = model.predict(future_X)

    # Show prediction
    st.write("### Predicted Prices")
    st.write(predictions)

    # Plot predictions
    fig2 = plt.figure()
    plt.plot(data['Close'], label="Actual")
    plt.plot(range(len(data), len(data) + future_days), predictions, label="Predicted")
    plt.legend()
    st.pyplot(fig2)
