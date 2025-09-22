# app.py
# Momentum-Screener für 750-Kosmos (Börseverlag) mit Handlungsempfehlungen

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Momentum-Screener", page_icon="📈", layout="wide")

# ---------------------------- #
#            Utils             #
# ---------------------------- #

@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_ohlc(ticker_list, start, end):
    """Holt OHLCV-Daten; fallback auf Close falls Adj Close fehlt."""
    tickers = [str(t).strip().upper() for t in ticker_list if str(t).strip()]
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()

    try:
        data = yf.download(
            tickers=" ".join(tickers),
            start=start,
            end=end,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception as e:
        st.error(f"Fehler beim Download: {e}")
        return pd.DataFrame(), pd.DataFrame()

    close_dict, vol_dict = {}, {}
    for t in tickers:
        try:
            df = data[t].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
        except Exception:
            df = data.copy()

        if df.empty:
            continue
        closes = (df["Adj Close"] if "Adj Close" in df.columns else df.get("Close")).rename(t)
        vols = df.get("Volume", pd.Series(dtype=float)).rename(t)
        close_dict[t] = closes
        vol_dict[t] = vols

    price = pd.concat(close_dict.values(), axis=1) if close_dict else pd.DataFrame()
    volume = pd.concat(vol_dict.values(), axis=1) if vol_dict else pd.DataFrame()
    price = price.sort_index().dropna(how="all")
    volume = volume.reindex_like(price)
    return price, volume


def pct_change_over_window(series: pd.Series, days: int) -> float:
    s = series.dropna()
    if len(s) < days + 1:
        return np.nan
    return (s.iloc[-1] / s.iloc[-(days+1)] - 1.0) * 100.0


def safe_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(5, window // 5)).mean()


def zscore_last(value: float, mean: float, std: float) -> float:
    if std == 0 or np.isnan(std):
        return 0.0
    return (value - mean) / std


def volume_score(vol_series: pd.Series, lookback=60):
    if vol_series is None or vol_series.dropna().empty:
        return np.nan
    cur = vol_series.dropna().iloc[-1]
    base = vol_series.rolling(lookback, min_periods=max(5, lookback//5)).mean().iloc[-1]
    if base == 0 or pd.isna(base) or pd.isna(cur):
        return np.nan
    return float(np.clip(cur / base, 0.5, 2.0))


def compute_indicators(price_df: pd.DataFrame, volume_df: pd.DataFrame):
    results = []

    mom90_series = pd.Series({t: pct_change_over_window(price_df[t], 90) for t in price_df.columns})
    mu, sigma = mom90_series.mean(), mom90_series.std(ddof=0)

    for t in price_df.columns:
        s = price_df[t].dropna()
        if s.empty or len(s) < 60:
            continue

        last = s.iloc[-1]
        sma50 = safe_sma(s, 50).iloc[-1]
        sma200 = safe_sma(s, 200).iloc[-1]

        mom260 = pct_change_over_window(s, 260)
        momJT  = pct_change_over_window(s, 90)
        rs_90  = mom90_series.get(t, np.nan)
        rs_z   = zscore_last(rs_90, mu, sigma) if not np.isnan(rs_90) else np.nan
        vol_sc = volume_score(volume_df.get(t, pd.Series(dtype=float)), lookback=60)

        def dist(p, m):
            if pd.isna(p) or pd.isna(m) or m == 0:
                return np.nan
            return (p / m - 1.0) * 100.0

        d50, d200 = dist(last, sma50), dist(last, sma200)
        sig50  = "Über GD50"  if last >= sma50 else "Unter GD50"
        sig200 = "Über GD200" if last >= sma200 else "Unter GD200"

        def logp(x):
            if pd.isna(x):
                return np.nan
            return np.sign(x) * np.log1p(abs(x))

        score = (
            0.40 * logp(mom260) +
            0.30 * logp(momJT) +
            0.20 * (0 if pd.isna(rs_z) else rs_z) +
            0.10 * (0 if pd.isna(vol_sc) else (vol_sc - 1.0))
        )
        score = 0.0 if pd.isna(score) else score

        results.append({
            "Ticker": t,
            "Kurs aktuell": last,
            "MOM260 (%)": mom260,
            "MOMJT (%)": momJT,
            "Relative Stärke (%)": rs_90,
            "RS z-Score": rs_z,
            "Volumen-Score": vol_sc,
            "Abstand GD50 (%)": d50,
            "Abstand GD200 (%)": d200,
            "GD50-Signal": sig50,
            "GD200-Signal": sig200,
            "Momentum-Score": float(score),
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df
    df = df.sort_values("Momentum-Score", ascending=False).reset_index(drop=True)
    df["Rank"] = np.arange(1, len(df) + 1)
    return df

# ---------------------------- #
#             UI               #
# ---------------------------- #

st.title("📊 Momentum-Screener für 750-Kosmos")

uploaded = st.file_uploader("Lade deine Kosmos-Datei hoch (CSV oder XLSX)", type=["csv", "xlsx"])
tickers = []
name_map = {}

if uploaded is not None:
    if uploaded.name.endswith(".xlsx"):
        df_in = pd.read_excel(uploaded)
    else:
        df_in = pd.read_csv(uploaded)

    if "Ticker" not in df_in.columns:
        st.error("❌ Datei braucht eine Spalte 'Ticker'.")
        st.stop()
    else:
        tickers = df_in["Ticker"].astype(str).tolist()
        if "Name" in df_in.columns:
            name_map = dict(zip(df_in["Ticker"].astype(str), df_in["Name"].astype(str)))
        st.success(f"{len(tickers)} Ticker aus Datei geladen.")

if not tickers:
    st.info("Bitte Kosmos-Datei hochladen.")
    st.stop()

start_date = st.sidebar.date_input("Startdatum", value=datetime.today() - timedelta(days=900))
end_date   = st.sidebar.date_input("Enddatum", value=datetime.today())

with st.spinner("Lade Kursdaten …"):
    prices, volumes = fetch_ohlc(tickers, start_date, end_date)

if prices.empty:
    st.error("Keine Kursdaten gefunden.")
    st.stop()

df = compute_indicators(prices, volumes)
if df.empty:
    st.warning("Keine Kennzahlen berechnet.")
    st.stop()

df["Name"] = df["Ticker"].map(name_map).fillna(df["Ticker"])
st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Ergebnisse als CSV exportieren", csv, "kosmos_momentum.csv", "text/csv")
