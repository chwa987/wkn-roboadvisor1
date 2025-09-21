import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Einstellungen
st.set_page_config(page_title="📈 WKN RoboAdvisor", layout="wide")
st.title("📊 WKN RoboAdvisor – Trendfolge Ranking")

st.markdown("""
Diese App analysiert Aktien anhand von Trendfolge-Kriterien:

- Abstand zur 200-Tage-Linie (GD200)
- Abstand zur 130-Tage-Linie (GD130 / Relative Stärke)
- Momentum über 260 Tage (MOM260)
- Momentum-Indikator nach Jegadeesh/Titman (MOMJT)
""")

# CSV-Datei einlesen
@st.cache_data
def load_ticker_file(filepath):
    df = pd.read_csv(filepath)
    return df.dropna(subset=["Ticker"])

# Berechnung der Indikatoren
def calculate_indicators(ticker):
    try:
        data = yf.download(ticker, period="300d")
        if data is None or data.empty or len(data) < 260:
            return None

        data['Close'] = data['Adj Close']

        # GD130, GD200
        gd130 = data['Close'].rolling(window=130).mean()
        gd200 = data['Close'].rolling(window=200).mean()

        latest_close = data['Close'].iloc[-1]
        last_gd130 = gd130.iloc[-1]
        last_gd200 = gd200.iloc[-1]

        dist_gd130 = (latest_close - last_gd130) / last_gd130 * 100
        dist_gd200 = (latest_close - last_gd200) / last_gd200 * 100

        # MOM260
        mom260 = (latest_close / data['Close'].iloc[-260] - 1) * 100

        # MOMJT (6 x 21 Tage = ca. 6 Monate)
        monthly_returns = []
        for i in range(1, 7):
            past_day = -21 * i
            ret = (latest_close / data['Close'].iloc[past_day] - 1) * 100
            monthly_returns.append(ret)
        momjt = np.mean(monthly_returns)

        return round(dist_gd200, 2), round(dist_gd130, 2), round(mom260, 2), round(momjt, 2)
    except Exception as e:
        return None

# Upload oder Auswahl der Datei
st.sidebar.header("📁 Datenquelle")
uploaded_file = st.sidebar.file_uploader("Wähle deine CSV-Datei mit Ticker + Namen", type="csv")

if uploaded_file:
    df_input = load_ticker_file(uploaded_file)

    if st.button("🔄 Analyse starten"):
        st.subheader("📉 Trendfolge Analyse")

        result = []
        tickers = df_input["Ticker"].tolist()
        names = df_input["Name"].tolist()

        for name, ticker in zip(names, tickers):
            indicators = calculate_indicators(ticker)
            if indicators:
                gd200, gd130, mom260, momjt = indicators
                score = gd200 + gd130 + mom260 + momjt
                result.append({
                    "Name": name,
                    "Ticker": ticker,
                    "GD200": gd200,
                    "GD130": gd130,
                    "MOM260": mom260,
                    "MOMJT": momjt,
                    "Score": round(score, 2)
                })
            else:
                result.append({
                    "Name": f"{name} (Fehler)",
                    "Ticker": ticker,
                    "GD200": None,
                    "GD130": None,
                    "MOM260": None,
                    "MOMJT": None,
                    "Score": None
                })

        df_result = pd.DataFrame(result)
        df_result = df_result.sort_values(by="Score", ascending=False)
        st.dataframe(df_result.reset_index(drop=True), use_container_width=True)

else:
    st.info("⬅️ Bitte lade eine CSV-Datei mit den Spalten `Name` und `Ticker` hoch.")
