# app.py
# Momentum-Screener + Backtest mit Handlungsempfehlungen
# GD50-Exit optional, robustes Monats-Rebalancing

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Momentum RoboAdvisor", page_icon="📈", layout="wide")

# ---------------------------- #
#         Hilfsfunktionen      #
# ---------------------------- #

@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_ohlc(ticker_list, start, end, batch_size=100):
    """Holt OHLCV-Daten robust, in Batches."""
    if isinstance(ticker_list, str):
        tickers = [t.strip() for t in ticker_list.split(",") if t.strip()]
    else:
        tickers = [t.strip() for t in ticker_list if str(t).strip()]
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()

    close_dict, vol_dict = {}, {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(
                tickers=" ".join(batch),
                start=start,
                end=end,
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
        except Exception as e:
            st.error(f"Fehler beim Download: {e}")
            continue

        for t in batch:
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
    start_val = s.iloc[-(days+1)]
    end_val = s.iloc[-1]
    if start_val <= 0 or pd.isna(start_val) or pd.isna(end_val):
        return np.nan
    return (end_val / start_val - 1.0) * 100.0


def safe_sma(series: pd.Series, window: int) -> pd.Series:
    if series is None or series.empty:
        return series
    return series.rolling(window=window, min_periods=max(5, window // 5)).mean()


def zscore_last(value: float, mean: float, std: float) -> float:
    if std is None or std == 0 or np.isnan(std):
        return 0.0
    return (value - mean) / std


def volume_score(vol_series: pd.Series, lookback=60):
    """Volume-Multiplikator: aktuelles Vol / SMA(lookback). Caps (0.5 – 2.0)."""
    if vol_series is None or vol_series.dropna().empty:
        return np.nan
    cur = vol_series.dropna().iloc[-1]
    base = vol_series.rolling(lookback, min_periods=max(5, lookback//5)).mean().iloc[-1]
    if base is None or base == 0 or pd.isna(base) or pd.isna(cur):
        return np.nan
    ratio = cur / base
    return float(np.clip(ratio, 0.5, 2.0))


def compute_indicators(price_df: pd.DataFrame, volume_df: pd.DataFrame):
    """Berechnet alle Kennzahlen inkl. Momentum-Score."""
    results = []

    # Universe-Renditen (130T) für RS
    mom130_universe = {t: pct_change_over_window(price_df[t], 130) for t in price_df.columns}
    mom130_series = pd.Series(mom130_universe).astype(float)
    mu, sigma = mom130_series.mean(), mom130_series.std(ddof=0)

    for t in price_df.columns:
        s = price_df[t].dropna()
        if s.empty or len(s) < 200:
            continue

        last = s.iloc[-1]
        sma50 = safe_sma(s, 50).iloc[-1]
        sma200 = safe_sma(s, 200).iloc[-1]

        mom260 = pct_change_over_window(s, 260)
        mom130 = pct_change_over_window(s, 130)

        rs_130 = mom130_series.get(t, np.nan)
        rs_z  = zscore_last(rs_130, mu, sigma) if not np.isnan(rs_130) else np.nan

        vol_sc = volume_score(volume_df.get(t, pd.Series(dtype=float)), lookback=60)

        def dist(p, m):
            if pd.isna(p) or pd.isna(m) or m == 0:
                return np.nan
            return (p / m - 1.0) * 100.0

        d50  = dist(last, sma50)
        d200 = dist(last, sma200)

        sig50  = "Über GD50"  if (not pd.isna(last) and not pd.isna(sma50)  and last >= sma50)  else "Unter GD50"
        sig200 = "Über GD200" if (not pd.isna(last) and not pd.isna(sma200) and last >= sma200) else "Unter GD200"

        def logp(x):
            if pd.isna(x):
                return np.nan
            return np.sign(x) * np.log1p(abs(x))

        mom_part = 0.40 * logp(mom260) + 0.30 * logp(mom130)
        rs_part  = 0.20 * (0 if pd.isna(rs_z) else rs_z)
        vol_part = 0.10 * (0 if pd.isna(vol_sc) else (vol_sc - 1.0))

        score = mom_part + rs_part + vol_part
        if pd.isna(score):
            score = 0.0

        results.append({
            "Ticker": t,
            "Kurs aktuell": last,
            "MOM260 (%)": mom260,
            "MOM130 (%)":  mom130,
            "Relative Stärke (130T) (%)": rs_130,
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


def monthly_endpoints(idx):
    """Liefert Monatsultimo-Indizes."""
    return idx.to_series().resample("M").last().dropna().tolist()


def run_backtest(price_df, volume_df, start_date, end_date, top_n=10, use_gd50_exit=True):
    """Backtest mit monatlichem Rebalancing."""
    idx = price_df.index
    months = monthly_endpoints(idx)
    months = [d for d in months if idx[0] <= d <= idx[-1] and d >= pd.to_datetime(start_date) and d <= pd.to_datetime(end_date)]

    equity = []
    weights_prev = pd.Series(0.0, index=price_df.columns)
    port_val = 1.0
    trades_log = []

    for m_i, d in enumerate(months[:-1]):
        asof = d
        next_asof = months[m_i+1]

        asof_pos = idx.get_indexer([asof], method="nearest")[0]
        next_pos = idx.get_indexer([next_asof], method="nearest")[0]

        snap = compute_indicators(price_df.iloc[:asof_pos+1], volume_df.iloc[:asof_pos+1])
        if snap.empty:
            equity.append((asof, port_val))
            continue

        pool = snap.copy()
        if use_gd50_exit:
            pool = pool[pool["GD50-Signal"] == "Über GD50"]
        sel = pool.head(top_n)

        new_weights = pd.Series(0.0, index=price_df.columns)
        if not sel.empty:
            w = 1.0 / len(sel)
            new_weights.loc[sel["Ticker"]] = w

        rets = price_df.iloc[asof_pos+1:next_pos+1].pct_change().fillna(0)
        port_rets = (rets * weights_prev).sum(axis=1)
        port_val *= (1.0 + port_rets).prod()

        equity.append((asof, port_val))

        turnover = (new_weights - weights_prev).abs().sum() / 2
        trades_log.append({"Date": asof, "Turnover": turnover})

        weights_prev = new_weights.copy()

    eq_df = pd.DataFrame(equity, columns=["Date", "Equity"]).set_index("Date")
    trades_df = pd.DataFrame(trades_log)
    return eq_df, trades_df


def dot(color: str) -> str:
    return f"<span style='font-size:18px;color:{color}'>●</span>"

# ---------------------------- #
#              UI              #
# ---------------------------- #

st.sidebar.header("⚙️ Einstellungen")
top_n = st.sidebar.number_input("Top-N (Kernpositionen)", min_value=3, max_value=50, value=10, step=1)
reserve_m = st.sidebar.number_input("Reserven (Nachrücker)", min_value=0, max_value=20, value=2, step=1)
start_date = st.sidebar.date_input("Startdatum", value=datetime.today() - timedelta(days=900))
end_date   = st.sidebar.date_input("Enddatum", value=datetime.today())

st.title("📊 Momentum RoboAdvisor – Analyse & Backtest")

uploaded = st.file_uploader("CSV mit **Ticker** und optional **Name** hochladen", type=["csv"])
tickers_txt = st.text_input("Oder Ticker (Yahoo Finance) eingeben:", "AAPL, MSFT, TSLA, NVDA")
portfolio_txt = st.text_input("(Optional) Portfolio-Ticker:", "AAPL")

name_map = {}
if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded)
        if "Ticker" in df_in.columns:
            if "Name" in df_in.columns:
                name_map = dict(zip(df_in["Ticker"].astype(str), df_in["Name"].astype(str)))
            tickers_txt = ", ".join(df_in["Ticker"].astype(str).tolist())
            st.success(f"{len(df_in)} Ticker aus CSV geladen.")
        else:
            st.error("In der CSV muss eine Spalte **Ticker** stehen.")
    except Exception as e:
        st.error(f"CSV konnte nicht gelesen werden: {e}")

tickers = [t.strip().upper() for t in tickers_txt.split(",") if t.strip()]
in_port = set([t.strip().upper() for t in portfolio_txt.split(",") if t.strip()])

if not tickers:
    st.info("Bitte Ticker eingeben oder eine CSV laden.")
    st.stop()

with st.spinner("Lade Kursdaten …"):
    prices, volumes = fetch_ohlc(tickers, start_date, end_date)

if prices.empty:
    st.warning("Keine Kursdaten geladen.")
    st.stop()

df = compute_indicators(prices, volumes)
if df.empty:
    st.warning("Kennzahlen konnten nicht berechnet werden.")
    st.stop()

df["Name"] = df["Ticker"].map(name_map).fillna(df["Ticker"])
df["_GD50_dot"]  = df["GD50-Signal"].apply(lambda s: dot("#16a34a") if s.startswith("Über") else dot("#dc2626"))
df["_GD200_dot"] = df["GD200-Signal"].apply(lambda s: dot("#16a34a") if s.startswith("Über") else dot("#dc2626"))

tab1, tab2, tab3 = st.tabs(["🔬 Analyse", "🧭 Handlungsempfehlungen", "📈 Backtest"])

with tab1:
    st.subheader("Analyse – alle Kennzahlen")
    st.dataframe(df, use_container_width=True)

with tab2:
    st.subheader("Handlungsempfehlungen (GD50/GD200 + Rank)")
    rec_df = df.copy()
    rec_df["GD50"]  = rec_df["_GD50_dot"]
    rec_df["GD200"] = rec_df["_GD200_dot"]
    st.dataframe(rec_df[["Rank", "Ticker", "Name", "Momentum-Score", "GD50", "GD200"]], use_container_width=True)

with tab3:
    st.subheader("Backtest")
    eq_df, trades_df = run_backtest(prices, volumes, start_date, end_date, top_n=top_n, use_gd50_exit=True)
    if not eq_df.empty:
        st.line_chart(eq_df["Equity"])
        st.write("Turnover-Log:", trades_df)

st.caption("Hinweis: Nur für Informations- und Ausbildungszwecke.")
