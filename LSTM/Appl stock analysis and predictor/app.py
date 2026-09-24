import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import timedelta

st.set_page_config(page_title="AAPL Trader Toolkit", layout="wide", page_icon="📈")

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    model = load_model("saved_model/trend_classifier_final.keras")
    scaler = joblib.load("saved_model/trend_scaler_final.pkl")
    with open("saved_model/config_final.json") as f:
        config = json.load(f)
    return model, scaler, config

model, scaler, config = load_artifacts()
FEATURES = config["features"]
WINDOW_SIZE = config["window_size"]
CONF_THRESHOLD = config.get("confidence_threshold", 0.65)
TICKER = config["ticker"]

# ---------- Data pipeline ----------
@st.cache_data(ttl=3600)
def fetch_data(ticker):
    df = yf.download(ticker, start="2010-01-01", auto_adjust=True)
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def engineer_features(df):
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['MA7'] = df['Close'].rolling(7).mean()
    df['MA21'] = df['Close'].rolling(21).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    df['RSI'] = calculate_rsi(df)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Volatility21'] = df['Return'].rolling(21).std()
    df['Momentum10'] = df['Close'].pct_change(10)
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_Mid'] = sma20
    df['BB_Upper'] = sma20 + 2*std20
    df['BB_Lower'] = sma20 - 2*std20
    df['Cumulative_Max'] = df['Close'].cummax()
    df['Drawdown'] = (df['Close'] - df['Cumulative_Max']) / df['Cumulative_Max']
    return df.dropna().reset_index(drop=True)

raw_df = fetch_data(TICKER)
df = engineer_features(raw_df)

# ---------- Sidebar ----------
st.sidebar.title("📈 AAPL Trader Toolkit")
st.sidebar.caption("Live data-driven analysis & honest trend signals")
page = st.sidebar.radio("Navigate", ["Overview", "Technical Analysis", "Trend Signal", "Risk Analytics", "About This Tool"])

last_row = df.iloc[-1]
prev_row = df.iloc[-2]
change = last_row['Close'] - prev_row['Close']
change_pct = (change / prev_row['Close']) * 100

st.sidebar.metric("Current Price", f"${last_row['Close']:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
st.sidebar.caption(f"As of {last_row['Date'].date()}")

# ---------- PAGE: Overview ----------
if page == "Overview":
    st.title(f"{TICKER} — Market Overview")

    c1, c2, c3, c4 = st.columns(4)
    fifty_two_wk = df[df['Date'] >= df['Date'].max() - timedelta(days=365)]
    c1.metric("52-Week High", f"${fifty_two_wk['Close'].max():.2f}")
    c2.metric("52-Week Low", f"${fifty_two_wk['Close'].min():.2f}")
    c3.metric("Annualized Volatility", f"{df['Return'].std()*np.sqrt(252)*100:.1f}%")
    c4.metric("Max Drawdown Ever", f"{df['Drawdown'].min()*100:.1f}%")

    st.subheader("Price History")
    range_choice = st.select_slider("Time range", options=["6M","1Y","3Y","5Y","All"], value="1Y")
    days_map = {"6M":182,"1Y":365,"3Y":1095,"5Y":1825,"All":99999}
    plot_df = df[df['Date'] >= df['Date'].max() - timedelta(days=days_map[range_choice])]

    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(plot_df['Date'], plot_df['Close'], color='#1f77b4', linewidth=1.5)
    ax.fill_between(plot_df['Date'], plot_df['Close'], plot_df['Close'].min(), alpha=0.08, color='#1f77b4')
    ax.set_ylabel("Price ($)")
    ax.set_title(f"{TICKER} Closing Price — {range_choice}")
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Trading Volume")
        fig2, ax2 = plt.subplots(figsize=(7,4))
        ax2.bar(plot_df['Date'], plot_df['Volume'], color='gray', width=1)
        st.pyplot(fig2)
    with col2:
        st.subheader("Drawdown from Peak")
        fig3, ax3 = plt.subplots(figsize=(7,4))
        ax3.fill_between(plot_df['Date'], plot_df['Drawdown']*100, 0, color='crimson', alpha=0.4)
        st.pyplot(fig3)

    st.subheader("Seasonality — Avg Daily Return by Month")
    df['Month'] = df['Date'].dt.month_name()
    month_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
    monthly = df.groupby('Month')['Return'].mean().reindex(month_order) * 100
    fig4, ax4 = plt.subplots(figsize=(12,4))
    colors = ['green' if x>0 else 'red' for x in monthly]
    ax4.bar(monthly.index, monthly.values, color=colors)
    ax4.axhline(0, color='black', linewidth=0.8)
    plt.xticks(rotation=45)
    st.pyplot(fig4)

# ---------- PAGE: Technical Analysis ----------
elif page == "Technical Analysis":
    st.title(f"{TICKER} — Technical Analysis")
    recent = df[df['Date'] >= df['Date'].max() - timedelta(days=365)]

    tab1, tab2, tab3, tab4 = st.tabs(["RSI", "MACD", "Bollinger Bands", "Moving Averages"])

    with tab1:
        fig, ax = plt.subplots(figsize=(14,4))
        ax.plot(recent['Date'], recent['RSI'], color='purple')
        ax.axhline(70, color='red', linestyle='--', alpha=0.6)
        ax.axhline(30, color='green', linestyle='--', alpha=0.6)
        st.pyplot(fig)
        rsi_val = last_row['RSI']
        status = "Overbought" if rsi_val>70 else "Oversold" if rsi_val<30 else "Neutral"
        st.info(f"Current RSI: {rsi_val:.2f} → **{status}**")

    with tab2:
        fig, ax = plt.subplots(figsize=(14,4))
        ax.plot(recent['Date'], recent['MACD'], label='MACD', color='blue')
        ax.plot(recent['Date'], recent['MACD_Signal'], label='Signal', color='orange')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.legend()
        st.pyplot(fig)
        macd_status = "Bullish" if last_row['MACD']>last_row['MACD_Signal'] else "Bearish"
        st.info(f"Current MACD Signal: **{macd_status}**")

    with tab3:
        fig, ax = plt.subplots(figsize=(14,5))
        ax.plot(recent['Date'], recent['Close'], color='black', linewidth=1.2, label='Close')
        ax.plot(recent['Date'], recent['BB_Mid'], color='blue', linestyle='--', label='SMA20')
        ax.fill_between(recent['Date'], recent['BB_Upper'], recent['BB_Lower'], alpha=0.2, color='skyblue')
        ax.legend()
        st.pyplot(fig)

    with tab4:
        fig, ax = plt.subplots(figsize=(14,5))
        ax.plot(df['Date'].tail(730), df['Close'].tail(730), color='black', alpha=0.4, label='Close')
        ax.plot(df['Date'].tail(730), df['MA50'].tail(730), color='blue', label='MA50')
        ax.plot(df['Date'].tail(730), df['MA200'].tail(730), color='red', label='MA200')
        ax.legend()
        st.pyplot(fig)
        cross_status = "Golden Cross (Bullish)" if last_row['MA50']>last_row['MA200'] else "Death Cross (Bearish)"
        st.info(f"Current Trend: **{cross_status}**")

# ---------- PAGE: Trend Signal ----------
elif page == "Trend Signal":
    st.title("🔮 AI Trend Signal — with Full Transparency")

    st.warning(
        f"**Honesty disclosure:** This model was rigorously validated using walk-forward "
        f"testing across multiple market regimes. Validated accuracy: "
        f"**{config['validated_accuracy_avg']*100:.1f}%**, vs a majority-class baseline of "
        f"**{config['validated_majority_baseline_avg']*100:.1f}%** "
        f"(beat baseline in {config['folds_beat_baseline']} folds). "
        f"This is a supplementary signal, not a trading recommendation."
    )

    features_scaled = scaler.transform(df[FEATURES])
    latest_window = features_scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, len(FEATURES))
    prob = model.predict(latest_window, verbose=0)[0][0]
    confidence = prob if prob > 0.5 else 1 - prob

    col1, col2, col3 = st.columns(3)
    col1.metric("10-Day Outlook", "📈 UP" if prob>0.5 else "📉 DOWN")
    col2.metric("Model Confidence", f"{confidence*100:.1f}%")

    if confidence < CONF_THRESHOLD:
        col3.metric("Status", "⚪ Weak Signal")
        st.info("Confidence below threshold (65%) — model recommends **no strong directional view** right now. This honest abstention is by design.")
    else:
        col3.metric("Status", "🟢 Signal Active")

    st.subheader("What's driving this?")
    st.write(f"- RSI: {last_row['RSI']:.1f} | MACD: {'Bullish' if last_row['MACD']>last_row['MACD_Signal'] else 'Bearish'} | MA Trend: {'Golden Cross' if last_row['MA50']>last_row['MA200'] else 'Death Cross'}")

# ---------- PAGE: Risk Analytics ----------
elif page == "Risk Analytics":
    st.title(f"{TICKER} — Risk & Volatility Analytics")

    annual_vol = df['Return'].std() * np.sqrt(252)
    sharpe = (df['Return'].mean() * 252) / annual_vol
    var_95 = df['Return'].quantile(0.05)

    c1, c2, c3 = st.columns(3)
    c1.metric("Annualized Volatility", f"{annual_vol*100:.2f}%")
    c2.metric("Sharpe Ratio (approx)", f"{sharpe:.2f}")
    c3.metric("Daily VaR (95%)", f"{var_95*100:.2f}%")

    st.subheader("Rolling 30-Day Volatility")
    df['Rolling_Vol'] = df['Return'].rolling(30).std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(14,4))
    ax.plot(df['Date'], df['Rolling_Vol']*100, color='darkorange')
    ax.set_ylabel("Annualized Vol (%)")
    st.pyplot(fig)

    st.subheader("Return Distribution")
    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.hist(df['Return']*100, bins=100, color='steelblue')
    ax2.axvline(0, color='black')
    st.pyplot(fig2)

# ---------- PAGE: About ----------
else:
    st.title("About This Tool")
    st.markdown(f"""
    This is a **transparent, research-grade stock analysis tool** for {TICKER}, built with 
    honesty as its core design principle.

    **What it does well:**
    - Real-time price, volume, and volatility analytics
    - Industry-standard technical indicators (RSI, MACD, Bollinger Bands, MA crossovers)
    - Risk metrics (Sharpe ratio, Value-at-Risk, rolling volatility)

    **What it's honest about:**
    - The AI trend signal was rigorously walk-forward validated across 16 years of data
      and multiple market regimes. It achieves ~{config['validated_accuracy_avg']*100:.0f}% 
      accuracy — only marginally above a naive majority-class baseline.
    - Confidence-based filtering means the model only speaks up when reasonably confident,
      and clearly says "no signal" otherwise — rather than forcing a guess.
    - **This tool does not predict the market reliably. No tool does.** It's built to inform, 
      not to promise returns.
    """)