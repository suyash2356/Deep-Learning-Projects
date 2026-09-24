import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Trend Predictor (LSTM)", layout="wide")

@st.cache_resource
def load_artifacts():
    model = load_model("saved_model/lstm_stock_model.keras")
    feature_scaler = joblib.load("saved_model/feature_scaler.pkl")
    target_scaler = joblib.load("saved_model/target_scaler.pkl")
    with open("saved_model/config.json") as f:
        config = json.load(f)
    return model, feature_scaler, target_scaler, config

model, feature_scaler, target_scaler, config = load_artifacts()
FEATURES = config["features"]
WINDOW_SIZE = config["window_size"]

st.title("📈 Stock Price Trend Predictor (LSTM)")
st.caption("Upload historical stock data (Date, Open, High, Low, Close, Volume format) to backtest and forecast the next closing price.")

uploaded_file = st.file_uploader("Upload stock CSV", type=["csv", "txt"])

def engineer_features(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['Return'] = df['Close'].pct_change()
    df['MA7'] = df['Close'].rolling(7).mean()
    df['MA21'] = df['Close'].rolling(21).mean()
    df = df.dropna().reset_index(drop=True)
    return df

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    required_cols = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_cols.issubset(raw_df.columns):
        st.error(f"CSV must contain columns: {required_cols}")
    else:
        df = engineer_features(raw_df)

        if len(df) < WINDOW_SIZE + 1:
            st.error(f"Need at least {WINDOW_SIZE + 1} rows of data after preprocessing.")
        else:
            st.success(f"Loaded {len(df)} rows after feature engineering.")
            st.line_chart(df.set_index('Date')['Close'], height=250)

            # ---- Backtest on available history ----
            features_scaled = feature_scaler.transform(df[FEATURES])
            closes = df['Close'].values

            X, base_close, next_close, dates = [], [], [], []
            for i in range(WINDOW_SIZE, len(features_scaled) - 1):
                X.append(features_scaled[i-WINDOW_SIZE:i])
                base_close.append(closes[i-1])
                next_close.append(closes[i])
                dates.append(df['Date'].iloc[i])

            X = np.array(X)
            base_close = np.array(base_close)
            next_close = np.array(next_close)

            if len(X) > 0:
                pred_return_scaled = model.predict(X, verbose=0)
                pred_return = target_scaler.inverse_transform(pred_return_scaled).flatten()
                predicted_price = base_close * (1 + pred_return)

                rmse = np.sqrt(np.mean((next_close - predicted_price) ** 2))
                mae = np.mean(np.abs(next_close - predicted_price))
                naive_rmse = np.sqrt(np.mean((next_close - base_close) ** 2))
                naive_mae = np.mean(np.abs(next_close - base_close))

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Model Performance")
                    st.metric("RMSE", f"${rmse:.2f}", delta=f"{rmse - naive_rmse:+.2f} vs naive", delta_color="inverse")
                    st.metric("MAE", f"${mae:.2f}", delta=f"{mae - naive_mae:+.2f} vs naive", delta_color="inverse")
                with col2:
                    st.subheader("Naive Baseline (yesterday=today)")
                    st.metric("RMSE", f"${naive_rmse:.2f}")
                    st.metric("MAE", f"${naive_mae:.2f}")

                st.caption("ℹ️ Naive baseline predicts tomorrow's price = today's price. Daily stock prices are highly autocorrelated, so beating it is genuinely hard — this comparison keeps the model's performance honest.")

                # ---- Plot last 150 days ----
                fig, ax = plt.subplots(figsize=(14, 5))
                n = min(150, len(dates))
                ax.plot(dates[-n:], next_close[-n:], label="Actual", linewidth=2)
                ax.plot(dates[-n:], predicted_price[-n:], label="Predicted (LSTM)", linewidth=2, alpha=0.8)
                ax.plot(dates[-n:], base_close[-n:], label="Naive Baseline", linestyle="--", alpha=0.6)
                ax.legend()
                ax.set_title("Actual vs Predicted Closing Price (Last 150 Days)")
                ax.set_ylabel("Price ($)")
                plt.xticks(rotation=45)
                st.pyplot(fig)

                # ---- Predict next unseen day ----
                st.subheader("🔮 Forecast Next Trading Day")
                last_window = features_scaled[-WINDOW_SIZE:]
                last_window = last_window.reshape(1, WINDOW_SIZE, len(FEATURES))
                next_return_scaled = model.predict(last_window, verbose=0)
                next_return = target_scaler.inverse_transform(next_return_scaled).flatten()[0]
                last_close = closes[-1]
                forecast_price = last_close * (1 + next_return)

                c1, c2, c3 = st.columns(3)
                c1.metric("Last Known Close", f"${last_close:.2f}")
                c2.metric("Predicted Next Close", f"${forecast_price:.2f}", delta=f"{next_return*100:+.2f}%")
                c3.metric("Predicted Direction", "📈 Up" if next_return > 0 else "📉 Down")
            else:
                st.warning("Not enough data to backtest with the current window size.")
else:
    st.info("👆 Upload a CSV to get started. Expected columns: Date, Open, High, Low, Close, Volume (matches Kaggle 'Huge Stock Market Dataset' format).")