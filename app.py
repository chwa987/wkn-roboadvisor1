# app.py
# Momentum-Screener + Handlungsempfehlungen + Backtest
# - Relative Stärke: 130 Tage
# - Exposure-Steuerung in 10%-Schritten basierend auf Anteil über GD200 im aktuellen Universum
# - Praktische Umsetzung: Anzahl gehaltene Aktien = round(TopN * Exposure)

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="Momentum RoboAdvisor + Exposure", page_icon="📈", layout="wide")

# ---------------------------- #
#            Utils             #
# ---------------------------- #

@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_ohlc(ticker_list, start, end):
    """Holt OHLCV-Daten; fallback auf Close falls Adj Close fehlt."""
    if isinstance(ticker_list, str):
        tickers = [t.strip() for t in ticker_list.split(",") if t.strip()]
    else:
        tickers = [t.strip() for t in ticker_list if str(t).strip()]
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

        if df is None or df.empty:
            continue
        closes = (df["Adj Close"] if "Adj Close" in df.columns else df.get("Close"))
        vols = df.get("Volume")
        if closes is None or closes.empty:
            continue
        close_dict[t] = closes.rename(t)
        vol_dict[t] = (vols.rename(t) if vols is not None else pd.Series(dtype=float, name=t))

    price = pd.concat(close_dict.values(), axis=1) if close_dict else pd.DataFrame()
    volume = pd.concat(vol_dict.values(), axis=1) if vol_dict else pd.DataFrame()
    price = price.sort_index().dropna(how="all")
    volume = volume.reindex(price.index)
    return price, volume

def pct_change_over_window(series: pd.Series, days: int) -> float:
    s = series.dropna()
    if len(s) < days + 1:
        return np.nan
    start_val = s.iloc[-(days+1)]
    end_val = s.iloc[-1]
    if pd.isna(start_val) or pd.isna(end_val) or start_val <= 0:
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
    if vol_series is None or vol_series.dropna().empty:
        return np.nan
    cur = vol_series.dropna().iloc[-1]
    base = vol_series.rolling(lookback, min_periods=max(5, lookback//5)).mean().iloc[-1]
    if pd.isna(base) or base == 0 or pd.isna(cur):
        return np.nan
    return float(np.clip(cur / base, 0.5, 2.0))

def logp(x):
    if pd.isna(x):
        return np.nan
    return np.sign(x) * np.log1p(abs(x))

# ---------------------------- #
#      Indikatoren / Score    #
# ---------------------------- #

def compute_indicators(price_df: pd.DataFrame, volume_df: pd.DataFrame):
    """
    Berechnet Kennzahlen für das gesamte aktuelle Price-DF (letzter Index).
    - MOM260, MOM130
    - Relative Stärke (130T) als raw %, RS z-Score (cross sectional)
    - Volumen-Score
    - GD50/GD200 Abstände & Signale
    - Momentum-Score (40/30/20/10)
    """
    results = []

    # Universe 130T returns for RS
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
        rs_z = (rs_130 - mu) / sigma if not np.isnan(rs_130) and sigma > 0 else np.nan

        vs = volume_df.get(t, pd.Series(dtype=float)) if isinstance(volume_df, pd.DataFrame) else pd.Series(dtype=float)
        vol_sc = volume_score(vs, lookback=60)

        def dist(p, m):
            if pd.isna(p) or pd.isna(m) or m == 0:
                return np.nan
            return (p / m - 1.0) * 100.0

        d50 = dist(last, sma50)
        d200 = dist(last, sma200)

        sig50 = "Über GD50" if (not pd.isna(last) and not pd.isna(sma50) and last >= sma50) else "Unter GD50"
        sig200 = "Über GD200" if (not pd.isna(last) and not pd.isna(sma200) and last >= sma200) else "Unter GD200"

        score = (
            0.40 * logp(mom260) +
            0.30 * logp(mom130) +
            0.20 * (0 if pd.isna(rs_z) else rs_z) +
            0.10 * (0 if pd.isna(vol_sc) else (vol_sc - 1.0))
        )
        score = 0.0 if pd.isna(score) else float(score)

        results.append({
            "Ticker": t,
            "Kurs aktuell": last,
            "MOM260 (%)": mom260,
            "MOM130 (%)": mom130,
            "Relative Stärke (130T) (%)": rs_130,
            "RS z-Score": rs_z,
            "Volumen-Score": vol_sc,
            "Abstand GD50 (%)": d50,
            "Abstand GD200 (%)": d200,
            "GD50-Signal": sig50,
            "GD200-Signal": sig200,
            "Momentum-Score": score,
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df
    df = df.sort_values("Momentum-Score", ascending=False).reset_index(drop=True)
    df["Rank"] = np.arange(1, len(df) + 1)
    return df

# ---------------------------- #
#      Exposure-Logik         #
# ---------------------------- #

def compute_market_breadth(df_indicators: pd.DataFrame):
    """Berechnet Anteil der Aktien über GD200 (auf verfügbaren Indikatoren)."""
    if df_indicators is None or df_indicators.empty:
        return np.nan
    total = len(df_indicators)
    over = df_indicators["GD200-Signal"].eq("Über GD200").sum()
    return over / total if total > 0 else np.nan

def breadth_to_exposure_tens(breadth: float):
    """
    Convert breadth [0..1] to exposure in 10%-steps (0..100).
    We round to nearest 10%.
    """
    if pd.isna(breadth):
        return 0.0
    v = float(breadth)
    step = int(round(v * 10))  # 0..10
    percent = max(0, min(10, step)) * 10  # 0,10,...100
    return percent / 100.0  # return 0.0..1.0

# ---------------------------- #
#      Reco / Anzeige         #
# ---------------------------- #

def rec_row_dynamic(row, in_port, effective_top_n, reserve=2):
    """Kaufen/Halten/Verkaufen mit dynamischer Anzahl gehaltenen Aktien."""
    t = row["Ticker"]
    rank = row["Rank"]
    over50 = row["GD50-Signal"].startswith("Über")
    over200 = row["GD200-Signal"].startswith("Über")

    if t in in_port:
        if not over50:
            return "🔴 Verkaufen (unter GD50)"
        if rank <= effective_top_n:
            return "🟡 Halten"
        if rank <= effective_top_n + reserve and over200:
            return "🟡 Halten (Reserve)"
        return "🔴 Verkaufen (nicht mehr Top)"
    else:
        if rank <= effective_top_n and over50 and over200:
            return "🟢 Kaufen"
        if rank <= effective_top_n + reserve and over50 and over200:
            return "🟡 Beobachten (Reserve)"
        return "—"

def dot(color: str) -> str:
    return f"<span style='font-size:18px;color:{color}'>●</span>"

# ---------------------------- #
#          Backtest            #
# ---------------------------- #

def monthly_endpoints(idx):
    """Letzter Handelstag je Monat (list of Timestamps)."""
    return idx.to_series().resample("M").last().dropna().tolist()

def run_backtest_dynamic_exposure(price_df, volume_df, start_date, end_date, top_n=10, reserve=2, cost_bps=10.0, slippage_bps=5.0):
    """
    Backtest mit dynamischer Exposure in 10%-Schritten.
    - Rebalancing monatlich (Monatsultimo)
    - Anzahl gehalten = round(top_n * exposure)
    - Returns equity curve and trades log with exposure
    """
    idx = price_df.index
    months = monthly_endpoints(idx)
    months = [d for d in months if d >= pd.to_datetime(start_date) and d <= pd.to_datetime(end_date)]
    if not months or len(months) < 2:
        return pd.DataFrame(), pd.DataFrame()

    equity = []
    trades = []
    port_val = 1.0
    weights_prev = pd.Series(0.0, index=price_df.columns)

    for m_i in range(len(months)-1):
        asof = months[m_i]
        next_asof = months[m_i+1]

        # nearest valid index positions
        asof_pos = idx.get_indexer([asof], method="nearest")[0]
        next_pos = idx.get_indexer([next_asof], method="nearest")[0]

        # snapshot indicators up to asof_pos (use compute_indicators on sliced DF)
        prices_slice = price_df.iloc[:asof_pos+1]
        vols_slice = volume_df.iloc[:asof_pos+1] if isinstance(volume_df, pd.DataFrame) else pd.DataFrame(index=prices_slice.index)
        snap = compute_indicators(prices_slice, vols_slice)
        if snap.empty:
            equity.append((asof, port_val))
            trades.append({"Date": asof, "Exposure": np.nan, "NumHold": 0, "Turnover": 0.0})
            continue

        # compute market breadth & discrete exposure
        breadth = compute_market_breadth(snap)
        exposure = breadth_to_exposure_tens(breadth)  # 0.0..1.0 in 0.1 steps
        num_hold = max(0, int(round(top_n * exposure)))
        # ensure at least 1 if exposure>0
        if exposure > 0 and num_hold == 0:
            num_hold = 1

        pool = snap.copy()
        # filter GD50 if you want buys only when over GD50 (we keep pool but selection will use GD50/GD200 checks)
        pool = pool.sort_values("Momentum-Score", ascending=False).reset_index(drop=True)

        sel = pool.head(num_hold) if num_hold > 0 else pool.iloc[0:0]

        # compute returns from asof_pos+1 .. next_pos (inclusive)
        rets = price_df.iloc[asof_pos+1:next_pos+1].pct_change().fillna(0)
        # portfolio return uses previous weights (weights_prev) to simulate holdings during the month
        if not weights_prev.empty:
            port_rets = (rets * weights_prev).sum(axis=1)
            gross_return = (1.0 + port_rets).prod() - 1.0
        else:
            gross_return = 0.0

        # compute turnover & simple cost model (based on weight change)
        new_weights = pd.Series(0.0, index=price_df.columns)
        if not sel.empty:
            w = 1.0 / len(sel)
            new_weights.loc[sel["Ticker"].values] = w

        turnover = float((new_weights - weights_prev).abs().sum())  # full turnover fraction
        tc = (cost_bps + slippage_bps) / 10000.0
        cost = turnover * tc

        net_return = gross_return - cost
        port_val *= (1.0 + net_return)

        equity.append((next_asof, port_val))
        trades.append({
            "Date": asof,
            "Breadth": breadth,
            "Exposure": int(exposure*100),
            "NumHold": len(sel),
            "Turnover": turnover,
            "GrossRet": gross_return,
            "Cost": cost,
            "NetRet": net_return,
            "PortVal": port_val
        })

        # update weights_prev to new_weights (we assume rebalancing at asof)
        weights_prev = new_weights.copy()

    eq_df = pd.DataFrame(equity, columns=["Date", "Equity"]).set_index("Date")
    trades_df = pd.DataFrame(trades)
    return eq_df, trades_df

# ---------------------------- #
#           UI / Layout        #
# ---------------------------- #

st.sidebar.header("⚙️ Einstellungen")
top_n = st.sidebar.number_input("Top-N (Kernpositionen)", min_value=3, max_value=50, value=10, step=1)
reserve_m = st.sidebar.number_input("Reserven (Nachrücker)", min_value=0, max_value=20, value=2, step=1)
start_date = st.sidebar.date_input("Startdatum (Datenabruf)", value=datetime.today() - timedelta(days=900))
end_date   = st.sidebar.date_input("Enddatum", value=datetime.today())

st.title("📊 Momentum RoboAdvisor – Exposure in 10%-Schritten")

uploaded = st.file_uploader("CSV mit **Ticker** und optional **Name** hochladen", type=["csv"])
tickers_txt = st.text_input("Oder Ticker (Yahoo Finance) eingeben, komma-getrennt:", "AAPL, MSFT, TSLA, NVDA")
portfolio_txt = st.text_input("(Optional) Portfolio-Ticker (für Halten/Verkaufen):", "")

name_map = {}
if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded)
        if "Ticker" not in df_in.columns:
            st.error("In der CSV muss mindestens eine Spalte 'Ticker' enthalten sein.")
        else:
            if "Name" in df_in.columns:
                name_map = dict(zip(df_in["Ticker"].astype(str), df_in["Name"].astype(str)))
            tickers_txt = ", ".join(df_in["Ticker"].astype(str).tolist())
            st.success(f"{len(df_in)} Ticker aus CSV geladen.")
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

# compute latest indicators
df = compute_indicators(prices, volumes)
if df.empty:
    st.warning("Keine Kennzahlen berechnet.")
    st.stop()

# map names
df["Name"] = df["Ticker"].map(name_map).fillna(df["Ticker"])

# dots
df["_GD50_dot"]  = df["GD50-Signal"].apply(lambda s: dot("#16a34a") if s.startswith("Über") else dot("#dc2626"))
df["_GD200_dot"] = df["GD200-Signal"].apply(lambda s: dot("#16a34a") if s.startswith("Über") else dot("#dc2626"))
df["GD50"] = df["_GD50_dot"]
df["GD200"] = df["_GD200_dot"]

# compute breadth & exposure for the current snapshot
breadth_now = compute_market_breadth(df)
exposure_now = breadth_to_exposure_tens(breadth_now)
exposure_percent_now = int(exposure_now * 100)
effective_holdings_now = max(0, int(round(top_n * exposure_now)))
if exposure_now > 0 and effective_holdings_now == 0:
    effective_holdings_now = 1

# show top header info
st.markdown(f"**Universe size:** {len(df)} tickers")
st.markdown(f"**Anteil über GD200 (Breadth):** {breadth_now:.2%}" if not pd.isna(breadth_now) else "**Breadth:** n/a")
st.markdown(f"**Discrete Exposure (10%-steps):** {exposure_percent_now}%")
st.markdown(f"**Aktuelle Anzahl geplanter Holdings (Top-N angepasst):** {effective_holdings_now} (von Top-{top_n})")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔬 Analyse", "🧭 Handlungsempfehlungen", "📈 Backtest"])

with tab1:
    st.subheader("Analyse – alle Kennzahlen")
    df_sorted = df.sort_values("Momentum-Score", ascending=False).reset_index(drop=True)
    df_sorted["Rank"] = np.arange(1, len(df_sorted) + 1)
    show_cols = [
        "Rank","Ticker","Name","Kurs aktuell",
        "MOM260 (%)","MOM130 (%)","Relative Stärke (130T) (%)","RS z-Score","Volumen-Score",
        "Abstand GD50 (%)","Abstand GD200 (%)","GD50-Signal","GD200-Signal","Momentum-Score"
    ]
    st.dataframe(df_sorted[show_cols], use_container_width=True)

with tab2:
    st.subheader("Handlungsempfehlungen – Kaufen / Halten / Verkaufen")
    rec_df = df.sort_values("Momentum-Score", ascending=False).reset_index(drop=True)
    rec_df["Rank"] = np.arange(1, len(rec_df)+1)
    # dynamic effective holdings
    effective_holdings = effective_holdings_now

    rec_df["Handlung"] = rec_df.apply(lambda r: rec_row_dynamic(r, in_port, effective_holdings, reserve=reserve_m), axis=1)

    rec_df["GD50"] = rec_df["_GD50_dot"]
    rec_df["GD200"] = rec_df["_GD200_dot"]

    buy_df = rec_df[rec_df["Handlung"].str.startswith("🟢")].copy()
    hold_df = rec_df[rec_df["Handlung"].str.startswith("🟡")].copy()
    sell_df = rec_df[rec_df["Handlung"].str.startswith("🔴")].copy()

    def render_block(title, frame):
        st.markdown(f"### {title}")
        cols = ["Rank","Ticker","Name","Momentum-Score","GD50","GD200","Handlung"]
        if frame.empty:
            st.write("—")
        else:
            st.write(frame[cols].to_html(escape=False, index=False), unsafe_allow_html=True)

    render_block("🟢 Kaufen", buy_df)
    render_block("🟡 Halten / Beobachten", hold_df)
    render_block("🔴 Verkaufen", sell_df)

with tab3:
    st.subheader("Backtest – dynamische Exposure (monatlich)")
    bt_topn = st.number_input("Backtest Top-N (Basis für 100%)", min_value=3, max_value=50, value=top_n, step=1)
    if st.button("Starte Backtest (monatlich)"):
        with st.spinner("Backtest läuft …"):
            eq_df, trades_df = run_backtest_dynamic_exposure(prices, volumes, start_date, end_date, top_n=bt_topn, reserve=reserve_m)
        if eq_df.empty:
            st.warning("Kein Ergebnis im Backtest.")
        else:
            # plot equity
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(eq_df.index, eq_df["Equity"], label="Equity")
            ax.set_ylabel("Equity (norm.)")
            ax.set_title("Backtest Equity Curve")
            ax.grid(True)
            st.pyplot(fig)

            # show trades log with breadth/exposure
            st.markdown("### Rebalance-Log")
            st.dataframe(trades_df, use_container_width=True)
            st.download_button("📥 Trades (CSV)", trades_df.to_csv(index=False).encode("utf-8"), "backtest_trades.csv", "text/csv")
            st.download_button("📥 Equity (CSV)", eq_df.to_csv().encode("utf-8"), "backtest_equity.csv", "text/csv")

st.caption("Hinweis: Nur zu Informations- und Ausbildungszwecken. Keine Anlageberatung.")
