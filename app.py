import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="Momentum-Screener", layout="wide")

# ---------------------------- #
# Daten holen
# ---------------------------- #
@st.cache_data
def fetch_ohlc(tickers, start, end):
    close_dict, vol_dict = {}, {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False)
            if df is None or df.empty:
                print(f"⚠️ Keine Daten für {t}")
                continue
            closes = df["Adj Close"]
            vols = df["Volume"] if "Volume" in df else pd.Series([0]*len(df), index=df.index)
            close_dict[t] = closes
            vol_dict[t] = vols
        except Exception as e:
            print(f"Fehler bei {t}: {e}")
            continue
    return pd.DataFrame(close_dict), pd.DataFrame(vol_dict)

# ---------------------------- #
# Kennzahlen berechnen
# ---------------------------- #
def compute_indicators(prices, volumes):
    results = []
    for t in prices.columns:
        p = prices[t].dropna()
        v = volumes[t].dropna()
        if p.empty:
            continue

        try:
            mom260 = (p.iloc[-1] / p.iloc[-260] - 1) * 100 if len(p) > 260 else np.nan
            momjt = ((p.pct_change().rolling(6).mean() * 100).iloc[-1]) if len(p) > 6 else np.nan
            rs = (p.iloc[-1] / p.mean() - 1) * 100
            vol_score = (v[-20:].mean() / v.mean()) if v.mean() > 0 else 1

            gd20 = p.rolling(20).mean().iloc[-1]
            gd50 = p.rolling(50).mean().iloc[-1]
            gd200 = p.rolling(200).mean().iloc[-1]
            last = p.iloc[-1]

            gd20_signal = "über GD20" if last > gd20 else "unter GD20"
            gd50_signal = "über GD50" if last > gd50 else "unter GD50"
            gd200_signal = "über GD200" if last > gd200 else "unter GD200"

            # Momentum-Score mit Gewichtung
            score = (
                0.3 * (mom260 if not np.isnan(mom260) else 0) +
                0.3 * (momjt if not np.isnan(momjt) else 0) +
                0.2 * (rs if not np.isnan(rs) else 0) +
                0.2 * (vol_score if not np.isnan(vol_score) else 0)
            )

            # Exit-Logik: unter GD50 = Halten/Verkaufen
            if last < gd50:
                status = "Verkaufen"
            elif last < gd200:
                status = "Halten"
            else:
                status = "Kaufen"

            results.append({
                "Ticker": t,
                "Kurs aktuell": round(last, 2),
                "MOM260 (%)": round(mom260, 2) if not np.isnan(mom260) else np.nan,
                "MOMJT (%)": round(momjt, 2) if not np.isnan(momjt) else np.nan,
                "Relative Stärke (%)": round(rs, 2),
                "Volumen-Score": round(vol_score, 2),
                "Abstand GD20 (%)": round((last / gd20 - 1) * 100, 2) if gd20 > 0 else np.nan,
                "GD20-Signal": gd20_signal,
                "Abstand GD50 (%)": round((last / gd50 - 1) * 100, 2) if gd50 > 0 else np.nan,
                "GD50-Signal": gd50_signal,
                "Abstand GD200 (%)": round((last / gd200 - 1) * 100, 2) if gd200 > 0 else np.nan,
                "GD200-Signal": gd200_signal,
                "Momentum-Score": round(score, 2),
                "Handelsempfehlung": status
            })
        except Exception as e:
            print(f"Fehler bei {t}: {e}")
            continue

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by="Momentum-Score", ascending=False).reset_index(drop=True)
    return df

# ---------------------------- #
# Streamlit App
# ---------------------------- #
st.title("📈 Momentum-Screener")

ticker_input = st.text_area("Gib Ticker ein (kommagetrennt):", "AAPL, MSFT, TSLA, NVDA")
tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]

start_date = st.date_input("Startdatum", datetime(2018,1,1))
end_date = st.date_input("Enddatum", datetime.today())

if st.button("Analyse starten"):
    if not tickers:
        st.warning("Bitte mindestens einen Ticker eingeben.")
    else:
        prices, volumes = fetch_ohlc(tickers, start_date, end_date)
        if prices.empty:
            st.error("❌ Keine Kursdaten geladen.")
        else:
            df = compute_indicators(prices, volumes)
            st.subheader("📊 Ergebnisse")
            st.dataframe(df, use_container_width=True)
            st.download_button("⬇️ Ergebnisse als CSV exportieren", df.to_csv(index=False), "ergebnisse.csv")
