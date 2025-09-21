import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="📈 WKN RoboAdvisor", layout="wide")
st.title("📊 Trendfolge RoboAdvisor – Momentum Ranking")

st.markdown("Lade eine CSV-Datei mit zwei Spalten: **Name** und **Ticker**.")

uploaded_file = st.file_uploader("📁 CSV-Datei hochladen", type="csv")

@st.cache_data
def fetch_ohlc(tickers, start, end):
    try:
        data = yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=True, threads=True)
        prices = {}
        volumes = {}
        for ticker in tickers:
            if ticker in data.columns.get_level_values(0):
                prices[ticker] = data[ticker]["Close"]
                volumes[ticker] = data[ticker]["Volume"]
        return prices, volumes
    except Exception as e:
        st.error(f"Fehler beim Abruf von Yahoo: {e}")
        return {}, {}

def calculate_indicators(prices):
    result = []
    for ticker, series in prices.items():
        if len(series) < 260:
            continue
        latest = series[-1]
        gd130 = series.rolling(130).mean().iloc[-1]
        gd200 = series.rolling(200).mean().iloc[-1]
        mom260 = ((latest / series[-260]) - 1) * 100
        momjt = np.mean([(latest / series[-21 * i] - 1) * 100 for i in range(1, 7)])

        result.append({
            "Ticker": ticker,
            "GD200": round((latest - gd200) / gd200 * 100, 2),
            "GD130": round((latest - gd130) / gd130 * 100, 2),
            "MOM260": round(mom260, 2),
            "MOMJT": round(momjt, 2),
        })
    return result

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "Ticker" not in df.columns:
        st.error("Die CSV-Datei muss eine Spalte 'Ticker' enthalten.")
    else:
        tickers = df["Ticker"].dropna().unique().tolist()
        st.success(f"{len(tickers)} Ticker geladen.")
        if st.button("🔄 Analyse starten"):
            with st.spinner("Daten werden geladen..."):
                start_date = datetime.now() - pd.DateOffset(days=365)
                end_date = datetime.now()
                prices, _ = fetch_ohlc(tickers, start=start_date, end=end_date)
                ranking = calculate_indicators(prices)
                if ranking:
                    df_ranked = pd.DataFrame(ranking)
                    df_ranked["Score"] = df_ranked[["GD200", "GD130", "MOM260", "MOMJT"]].sum(axis=1)
                    df_ranked = df_ranked.sort_values(by="Score", ascending=False)
                    st.dataframe(df_ranked.reset_index(drop=True), use_container_width=True)
                else:
                    st.warning("Keine ausreichenden Kursdaten für Bewertung gefunden.")
else:
    st.info("Bitte lade eine CSV-Datei mit Tickerdaten hoch.")
