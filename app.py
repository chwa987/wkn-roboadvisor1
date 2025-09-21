--- Fehlerrobustes OHLC-Fetching mit yfinance ---

import yfinance as yf import pandas as pd import streamlit as st

def fetch_ohlc(tickers, start_date, end_date): close_dict = {} vol_dict = {} error_tickers = []

for t in tickers:
    try:
        df = yf.download(t, start=start_date, end=end_date, progress=False)
        if df is None or df.empty:
            error_tickers.append(t)
            continue

        closes = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        vols = df.get("Volume", pd.Series([None] * len(closes), index=closes.index))

        close_dict[t] = closes
        vol_dict[t] = vols
    except Exception as e:
        error_tickers.append(t)
        continue

if error_tickers:
    st.warning(f"⚠️ {len(error_tickers)} Ticker konnten nicht geladen werden: {', '.join(error_tickers[:10])}{'...' if len(error_tickers) > 10 else ''}")

prices = pd.DataFrame(close_dict)
volumes = pd.DataFrame(vol_dict)
return prices, volumes

