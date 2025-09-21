import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="📊 Trendfolge RoboAdvisor", layout="wide")
st.title("📈 Trendfolge RoboAdvisor – Top 450")

# Daten einlesen
@st.cache_data
def load_ticker_list():
    df = pd.read_csv("top_450_ticker_mapped.csv")
    df = df.dropna(subset=["Ticker"])
    return df

ticker_df = load_ticker_list()
tickers = ticker_df["Ticker"].tolist()

# Funktion zur Berechnung der Trendfolge-Indikatoren
def calculate_indicators(ticker):
    try:
        data = yf.download(ticker, period="300d", progress=False)
        if data is None or data.empty:
            return None

        data['Close'] = data['Adj Close']
        gd130 = data['Close'].rolling(window=130).mean()
        gd200 = data['Close'].rolling(window=200).mean()

        latest_close = data['Close'].iloc[-1]
        last_gd130 = gd130.iloc[-1]
        last_gd200 = gd200.iloc[-1]

        dist_gd130 = (latest_close - last_gd130) / last_gd130 * 100
        dist_gd200 = (latest_close - last_gd200) / last_gd200 * 100

        mom260 = (latest_close / data['Close'].iloc[-260] - 1) * 100

        monthly_returns = []
        for i in range(1, 7):
            past_day = -21 * i
            ret = (latest_close / data['Close'].iloc[past_day] - 1) * 100
            monthly_returns.append(ret)
        momjt = np.mean(monthly_returns)

        return round(dist_gd200, 2), round(dist_gd130, 2), round(mom260, 2), round(momjt, 2)
    except Exception:
        return None

# Button zur Analyse
if st.button("🔄 Jetzt analysieren"):
    results = []

    for _, row in ticker_df.iterrows():
        ticker = row["Ticker"]
        name = row["Name"]
        indicators = calculate_indicators(ticker)

        if indicators:
            gd200, gd130, mom260, momjt = indicators
            score = gd200 + gd130 + mom260 + momjt
            results.append({
                "Name": name,
                "Ticker": ticker,
                "GD200": gd200,
                "GD130": gd130,
                "MOM260": mom260,
                "MOMJT": momjt,
                "Score": round(score, 2)
            })
        else:
            results.append({
                "Name": f"{name} (Fehler bei Analyse)",
                "Ticker": ticker,
                "GD200": None,
                "GD130": None,
                "MOM260": None,
                "MOMJT": None,
                "Score": None
            })

    df = pd.DataFrame(results)
    df = df.sort_values(by="Score", ascending=False)
    st.dataframe(df.reset_index(drop=True), use_container_width=True)
