import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# =====================
# Funktionen
# =====================

@st.cache_data
def load_ticker_list():
    df = pd.read_csv("top_450_ticker_mapped.csv")
    return df["Ticker"].dropna().unique().tolist()

@st.cache_data
def fetch_ohlc(tickers, start, end):
    close_dict = {}
    vol_dict = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end)
            if df is None or df.empty:
                st.warning(f"⚠️ Keine Daten für {t} gefunden (leer oder None).")
                continue
            closes = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
            vols = df.get("Volume", pd.Series([None] * len(closes), index=closes.index))
            close_dict[t] = closes
            vol_dict[t] = vols
        except Exception as e:
            st.warning(f"❌ Fehler beim Abruf von {t}: {e}")
            continue
    return pd.DataFrame(close_dict), pd.DataFrame(vol_dict)

def calculate_trend_indicators(prices):
    df_indicators = []
    for t in prices.columns:
        try:
            p = prices[t].dropna()
            if len(p) < 260:
                continue
            gd200 = p.iloc[-1] / p.rolling(200).mean().iloc[-1] - 1
            gd130 = p.iloc[-1] / p.rolling(130).mean().iloc[-1] - 1
            mom260 = p.iloc[-1] / p.iloc[-260] - 1
            momjt = (p.iloc[-1] - p.iloc[-21]) + (p.iloc[-22] - p.iloc[-43]) + (p.iloc[-44] - p.iloc[-65])
            score = gd200 + gd130 + mom260 + momjt
            df_indicators.append({
                "Ticker": t,
                "GD200": round(gd200, 4),
                "GD130": round(gd130, 4),
                "MOM260": round(mom260, 4),
                "MOMJT": round(momjt, 4),
                "Score": round(score, 4)
            })
        except Exception as e:
            st.warning(f"Fehler bei Analyse {t}: {e}")
    return pd.DataFrame(df_indicators)

# =====================
# Streamlit UI
# =====================

st.set_page_config(page_title="Trendfolge RoboAdvisor", layout="centered")
st.title("📈 Trendfolge RoboAdvisor – Top 450")

if "tickers" not in st.session_state:
    st.session_state["tickers"] = load_ticker_list()

if st.button("🔄 Jetzt analysieren"):
    with st.spinner("Lade Kursdaten ..."):
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365 * 2)  # ca. 2 Jahre
        prices, volumes = fetch_ohlc(st.session_state["tickers"], start_date, end_date)

    if prices.empty:
        st.error("❌ Keine Kursdaten geladen.")
    else:
        df_result = calculate_trend_indicators(prices)
        if df_result.empty:
            st.warning("⚠️ Keine gültigen Trendindikatoren berechnet.")
        else:
            df_result = df_result.sort_values(by="Score", ascending=False).reset_index(drop=True)
            st.dataframe(df_result)
            csv = df_result.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Ergebnisse als CSV herunterladen", data=csv, file_name="trend_ranking.csv")
