import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import requests
from textblob import TextBlob

# Streamlit UI Setup
st.title("📈 Stock Market Prediction with XGBoost & Sentiment Analysis")
st.sidebar.header("Stock Selection")
selected_stock = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL")
days_to_predict = st.sidebar.slider("Days to Predict", 1, 30, 7)

# Fetch Stock Data
st.subheader(f"Stock Data for {selected_stock}")
stock_data = yf.download(selected_stock, period="3y", interval="1d")
st.write(stock_data.tail())

# Plot Stock Price
fig, ax = plt.subplots()
ax.plot(stock_data["Close"], label="Closing Price", color='blue')
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.set_title(f"{selected_stock} Stock Price Over Time")
ax.legend()
st.pyplot(fig)

# Prepare Data for XGBoost
stock_data['Prediction'] = stock_data['Close'].shift(-days_to_predict)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(stock_data[['Close']])
X = data_scaled[:-days_to_predict]
y = stock_data['Prediction'].dropna()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
model.fit(X_train, y_train)

# Predict Future Prices
future_prices = model.predict(X_test)
mae = mean_absolute_error(y_test, future_prices)
st.write(f"Mean Absolute Error: {mae:.2f} USD")

# Display Predictions
pred_dates = pd.date_range(start=stock_data.index[-len(future_prices)], periods=len(future_prices))
pred_df = pd.DataFrame({"Date": pred_dates, "Predicted Price": future_prices})
st.subheader("Stock Price Predictions")
st.write(pred_df)

# Plot Predictions
fig2, ax2 = plt.subplots()
ax2.plot(stock_data.index, stock_data["Close"], label="Historical Price", color='blue')
ax2.plot(pred_dates, future_prices, label="Predicted Price", color='red')
ax2.set_xlabel("Date")
ax2.set_ylabel("Price (USD)")
ax2.set_title(f"{selected_stock} Price Prediction")
ax2.legend()
st.pyplot(fig2)

# Fetch News & Sentiment Analysis
st.subheader("Stock Market News & Sentiment Analysis")
news_url = f'https://newsapi.org/v2/everything?q={selected_stock}&apiKey=YOUR_NEWS_API_KEY'
response = requests.get(news_url).json()
news_sentiments = []

if "articles" in response:
    for article in response["articles"][:5]:
        headline = article["title"]
        sentiment = TextBlob(headline).sentiment.polarity
        news_sentiments.append((headline, sentiment))
        st.write(f"📰 {headline}")
        st.write(f"Sentiment Score: {sentiment:.2f}")

st.success("Prediction & Sentiment Analysis Completed! 🚀")
    