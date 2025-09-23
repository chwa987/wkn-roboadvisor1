# app.py
# Momentum-Screener + Handlungsempfehlungen + Backtest (Top-N, GD50-Exit)
# Zeitfenster: MOM130 & RS über 130 Tage, MOM260 über 260 Tage

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Momentum-Screener + Backtest", page_icon="📈", layout="wide")

# =========================================================
# Utilities
# =========================================================

@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_ohlc(ticker_list, start, end):
    """Lädt OHLCV-Daten für viele Ticker in einem Rutsch; robust gegen fehlende Spalten."""
    if isinstance(ticker_list, str):
        tickers = [t.strip() for t in ticker_list.split(",") if t.strip()]
    else:
        tickers = [str(t).strip() for t in ticker_list if str(t).strip()]
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
    a, b = s.iloc[-(days+1)], s.iloc[-1]
    if pd.isna(a) or pd.isna(b) or a <= 0:
        return np.nan
    return (b / a - 1.0) * 100.0

def safe_sma(series: pd.Series, window: int) -> pd.Series:
    if series is None or series.empty:
        return series
    return series.rolling(window=window, min_periods=max(5, window // 5)).mean()

def zscore_last(value: float, mean: float, std: float) -> float:
    if std is None or std == 0 or np.isnan(std):
        return 0.0
    return (value - mean) / std

def volume_score(vol_series: pd.Series, lookback=60):
    """Volumen-Multiplikator: aktuelles Vol / SMA(lookback), gecappt 0.5–2.0."""
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

# =========================================================
# Indikatoren/Score (wird von Analyse, Signalen und Backtest genutzt)
# =========================================================
def compute_snapshot_indicators(price_df: pd.DataFrame, volume_df: pd.DataFrame, asof_idx: int):
    """
    Berechnet alle Kennzahlen für einen 'Stichtag' (Indexposition asof_idx) cross-sektional.
    Liefert DataFrame mit Score & Ranks für diesen Tag.
    """
    # Slice bis inkl. asof_idx
    prices = price_df.iloc[:asof_idx+1, :].copy()
    vols   = volume_df.iloc[:asof_idx+1, :].copy() if volume_df is not None and not volume_df.empty else pd.DataFrame(index=prices.index)

    # Universum-Performance 130T
    mom130_series = prices.apply(lambda s: pct_change_over_window(s, 130))
    mu130, sigma130 = mom130_series.mean(), mom130_series.std(ddof=0)

    rows = []
    for t in prices.columns:
        s = prices[t].dropna()
        if s.empty or len(s) < 200:
            continue

        last = s.iloc[-1]
        sma50 = safe_sma(s, 50).iloc[-1]
        sma200 = safe_sma(s, 200).iloc[-1]

        mom260 = pct_change_over_window(s, 260)
        mom130 = pct_change_over_window(s, 130)

        rs_130 = mom130_series.get(t, np.nan)
        rs_z   = (rs_130 - mu130) / sigma130 if not np.isnan(rs_130) and sigma130 > 0 else np.nan

        vs = vols[t] if t in vols.columns else pd.Series(dtype=float, index=prices.index)
        vol_sc = volume_score(vs, 60)

        def dist(p, m):
            if pd.isna(p) or pd.isna(m) or m == 0:
                return np.nan
            return (p / m - 1.0) * 100.0

        d50, d200 = dist(last, sma50), dist(last, sma200)
        sig50  = "Über GD50"  if (not pd.isna(last) and not pd.isna(sma50) and last >= sma50) else "Unter GD50"
        sig200 = "Über GD200" if (not pd.isna(last) and not pd.isna(sma200) and last >= sma200) else "Unter GD200"

        score = (
            0.40 * logp(mom260) +
            0.30 * logp(mom130) +
            0.20 * (0 if pd.isna(rs_z) else rs_z) +
            0.10 * (0 if pd.isna(vol_sc) else (vol_sc - 1.0))
        )
        score = 0.0 if pd.isna(score) else float(score)

        rows.append({
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

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("Momentum-Score", ascending=False).reset_index(drop=True)
    df["Rank"] = np.arange(1, len(df) + 1)
    return df

def compute_indicators_full(price_df: pd.DataFrame, volume_df: pd.DataFrame):
    """Komplette Analyse für 'jetzt' (letzter Index)."""
    return compute_snapshot_indicators(price_df, volume_df, asof_idx=len(price_df.index)-1)

# =========================================================
# Handlungsempfehlungen
# =========================================================
def rec_row(row, in_port, top_n=10, reserve=2):
    t, rank = row["Ticker"], row["Rank"]
    over50  = row["GD50-Signal"].startswith("Über")
    over200 = row["GD200-Signal"].startswith("Über")

    if t in in_port:
        if not over50: return "🔴 Verkaufen (unter GD50)"
        if rank <= top_n: return "🟡 Halten"
        if rank <= top_n + reserve and over200: return "🟡 Halten (Reserve)"
        return "🔴 Verkaufen (nicht mehr Top)"
    else:
        if rank <= top_n and over50 and over200: return "🟢 Kaufen"
        if rank <= top_n + reserve and over50 and over200: return "🟡 Beobachten (Reserve)"
        return "—"

def dot(color: str) -> str:
    return f"<span style='font-size:18px;color:{color}'>●</span>"

# =========================================================
# Backtest
# =========================================================
def monthly_endpoints(index):
    """Letzter Handelstag je Monat."""
    df = pd.DataFrame(index=index)
    return df.resample("M").last().dropna().index

def run_backtest(price_df: pd.DataFrame,
                 volume_df: pd.DataFrame,
                 start_idx: int,
                 top_n: int = 10,
                 reserve: int = 0,
                 use_gd50_exit: bool = True,
                 cost_bps: float = 10.0,   # Transaktionskosten in Basispunkten je Turnover
                 slippage_bps: float = 5.0 # Slippage in Basispunkten je Turnover
                 ):
    """
    Einfache Top-N Momentum-Strategie:
      - Rebalancing: monatlich (Monatsultimo)
      - Auswahl: Momentum-Score (wie Analyse), optional GD50-Exit
      - Gleichgewichtung
      - Kosten: auf Turnover (Ein-/Ausstieg) angewandt
    """
    idx = price_df.index
    months = monthly_endpoints(idx)
    months = [d for d in months if idx.get_loc(d) >= start_idx]  # erst wenn genug Historie da ist

    equity = []
    weights_prev = pd.Series(0.0, index=price_df.columns)  # Vorperiode
    port_val = 1.0

    trades_log = []

    for m_i, d in enumerate(months[:-1]):  # bis vorletzter Monat (Rendite zur nächsten Monatsletzten)
        asof = d
        next_asof = months[m_i+1]
        for m_i, d in enumerate(months[:-1]):
    asof = d
    next_asof = months[m_i+1]

    # robust: falls Monatsultimo nicht im Index existiert (Wochenende/Feiertag)
    asof_pos = idx.get_indexer([asof], method="nearest")[0]
    next_pos = idx.get_indexer([next_asof], method="nearest")[0]

        snap = compute_snapshot_indicators(price_df, volume_df, asof_pos)
        if snap.empty:
            equity.append((asof, port_val))
            continue

        # Selektion Top-N
        pool = snap.copy()
        if use_gd50_exit:
            pool = pool[pool["GD50-Signal"] == "Über GD50"]  # Exit-/Filterbedingung
        sel = pool.head(top_n)

        # Gewichte (gleichgewichtet)
        new_weights = pd.Series(0.0, index=price_df.columns)
        if not sel.empty:
            w = 1.0 / len(sel)
            new_weights.loc[sel["Ticker"].values] = w

        # Turnover & Kosten
        turnover = float((new_weights - weights_prev).abs().sum())
        tc = (cost_bps + slippage_bps) / 10000.0
        cost = turnover * tc

        # Portfolio-Return bis zum nächsten Monatsultimo
        ret_next = price_df.iloc[next_pos, :].div(price_df.iloc[asof_pos, :]) - 1.0
        ret_next = ret_next.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        port_ret_gross = float((new_weights * ret_next).sum())
        port_ret_net = port_ret_gross - cost

        port_val *= (1.0 + port_ret_net)
        equity.append((asof, port_val))

        trades_log.append({
            "Date": asof,
            "Holdings": ", ".join(sel["Ticker"].tolist()) if not sel.empty else "",
            "Turnover": turnover,
            "GrossRet": port_ret_gross,
            "Cost": cost,
            "NetRet": port_ret_net,
            "PortVal": port_val
        })

        weights_prev = new_weights

    eq = pd.DataFrame(equity, columns=["Date", "Equity"]).set_index("Date")
    trades = pd.DataFrame(trades_log)
    return eq, trades

def perf_stats(equity: pd.DataFrame):
def run_backtest(price_df, volume_df, start_date, end_date, top_n=10, use_gd50_exit=True):
    """
    Backtest:
    - Rebalancing: monatlich (Monatsultimo)
    - Auswahl: Momentum-Score (wie Analyse)
    - Gleichgewichtung
    - Kosten: auf Turnover (Ein-/Ausstiege)
    """
    idx = price_df.index
    months = monthly_endpoints(idx)
    # Nur Monate berücksichtigen, die auch >= Startdatum und <= Enddatum sind
    months = [d for d in months if idx[0] <= d <= idx[-1] and d >= pd.to_datetime(start_date) and d <= pd.to_datetime(end_date)]

    equity = []
    weights_prev = pd.Series(0.0, index=price_df.columns)
    port_val = 1.0
    trades_log = []

    for m_i, d in enumerate(months[:-1]):
        asof = d
        next_asof = months[m_i+1]

        # robust: falls Monatsultimo nicht im Index existiert (Wochenende/Feiertag)
        asof_pos = idx.get_indexer([asof], method="nearest")[0]
        next_pos = idx.get_indexer([next_asof], method="nearest")[0]

        snap = compute_snapshot_indicators(price_df.iloc[:asof_pos+1], volume_df.iloc[:asof_pos+1])
        if snap.empty:
            equity.append((asof, port_val))
            continue

        # Top-N Selektion
        pool = snap.copy()
        if use_gd50_exit:
            pool = pool[pool["GD50-Signal"] == "Über GD50"]
        sel = pool.head(top_n)

        # Neue Gewichte
        new_weights = pd.Series(0.0, index=price_df.columns)
        if not sel.empty:
            w = 1.0 / len(sel)
            new_weights.loc[sel["Ticker"]] = w

        # Performance-Berechnung für Periode
        rets = price_df.iloc[asof_pos+1:next_pos+1].pct_change().fillna(0)
        port_rets = (rets * weights_prev).sum(axis=1)
        port_val *= (1.0 + port_rets).prod()

        equity.append((asof, port_val))

        # Turnover loggen
        turnover = (new_weights - weights_prev).abs().sum() / 2
        trades_log.append({"Date": asof, "Turnover": turnover})

        # Update Gewichte
        weights_prev = new_weights.copy()

    eq_df = pd.DataFrame(equity, columns=["Date", "Equity"]).set_index("Date")
    trades_df = pd.DataFrame(trades_log)

    return eq_df, trades_df
    }

# =========================================================
# Sidebar & Eingaben
# =========================================================
st.sidebar.header("⚙️ Einstellungen (Analyse/Signale)")
top_n = st.sidebar.number_input("Top-N (Kernpositionen)", 3, 50, 10, 1)
reserve_m = st.sidebar.number_input("Reserven (Nachrücker)", 0, 20, 2, 1)
start_date = st.sidebar.date_input("Startdatum Datenabruf", value=datetime.today() - timedelta(days=1800))
end_date   = st.sidebar.date_input("Enddatum", value=datetime.today())

st.title("📈 Momentum-Screener + 🧪 Backtest (Top-N, GD50-Exit)")

uploaded = st.file_uploader("CSV mit **Ticker** und optional **Name** hochladen", type=["csv"])
tickers_txt = st.text_input("Oder Ticker eingeben (kommagetrennt, Yahoo Finance):", "AAPL, MSFT, TSLA, NVDA")
portfolio_txt = st.text_input("(Optional) Aktuelle Portfolio-Ticker:", "AAPL")

name_map = {}
if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded)
        if "Ticker" not in df_in.columns:
            st.error("CSV muss eine Spalte **Ticker** enthalten.")
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

# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3 = st.tabs(["🔬 Analyse", "🧭 Handlungsempfehlungen", "🧪 Backtest"])

with tab1:
    st.subheader("Analyse – alle Kennzahlen (heute)")
    df_now = compute_indicators_full(prices, volumes)
    if df_now.empty:
        st.warning("Keine Kennzahlen berechnet.")
    else:
        df_now["Name"] = df_now["Ticker"].map(name_map).fillna(df_now["Ticker"])
        show_cols = [
            "Rank", "Ticker", "Name", "Kurs aktuell",
            "MOM260 (%)", "MOM130 (%)", "Relative Stärke (130T) (%)", "RS z-Score",
            "Volumen-Score", "Abstand GD50 (%)", "Abstand GD200 (%)",
            "GD50-Signal", "GD200-Signal", "Momentum-Score"
        ]
        st.dataframe(df_now[show_cols], use_container_width=True)
        st.download_button("📥 Analyse als CSV", df_now[show_cols].to_csv(index=False).encode("utf-8"),
                           "analyse_momentum.csv", "text/csv")

with tab2:
    st.subheader("Kaufen / Halten / Verkaufen (heute)")
    if 'df_now' not in locals() or df_now.empty:
        st.info("Erst Analyse berechnen (Tab 1).")
    else:
        df_now["_GD50_dot"]  = df_now["GD50-Signal"].apply(lambda s: f"<span style='color:{'#16a34a' if s.startswith('Über') else '#dc2626'}'>●</span>")
        df_now["_GD200_dot"] = df_now["GD200-Signal"].apply(lambda s: f"<span style='color:{'#16a34a' if s.startswith('Über') else '#dc2626'}'>●</span>")
        rec_df = df_now.copy()
        rec_df["Handlung"] = rec_df.apply(lambda r: rec_row(r, in_port, top_n=top_n, reserve=reserve_m), axis=1)
        rec_df = rec_df.sort_values("Rank").reset_index(drop=True)
        rec_df["GD50"]  = rec_df["_GD50_dot"]
        rec_df["GD200"] = rec_df["_GD200_dot"]

        buy_df  = rec_df[rec_df["Handlung"].str.startswith("🟢")].copy()
        hold_df = rec_df[rec_df["Handlung"].str.startswith("🟡")].copy()
        sell_df = rec_df[rec_df["Handlung"].str.startswith("🔴")].copy()

        def render_block(title, frame):
            st.markdown(f"### {title}")
            cols = ["Rank", "Ticker", "Name", "Momentum-Score", "GD50", "GD200", "Handlung"]
            if frame.empty:
                st.write("—")
            else:
                st.write(frame[cols].to_html(escape=False, index=False), unsafe_allow_html=True)

        render_block("🟢 Kaufen", buy_df)
        render_block("🟡 Halten / Beobachten", hold_df)
        render_block("🔴 Verkaufen", sell_df)

with tab3:
    st.subheader("Backtest – Top-N Momentum mit GD50-Exit")
    colA, colB, colC = st.columns(3)
    with colA:
        bt_topn = st.number_input("Top-N", 3, 50, 10, 1)
        bt_use_exit = st.checkbox("GD50-Exit aktiv", value=True)
    with colB:
        bt_cost = st.number_input("Kosten (bps) je Transaktion", 0.0, 100.0, 10.0, 1.0)
        bt_slip = st.number_input("Slippage (bps)", 0.0, 100.0, 5.0, 1.0)
    with colC:
        min_hist_days = st.number_input("Min. Historie (Tage) vor Start", 260, 600, 300, 10)

    # Startindex so wählen, dass die ersten Signale valide sind (mindestens 260 Tage für MOM260)
    start_idx = np.argmax(prices.index >= (prices.index[0] + pd.Timedelta(days=min_hist_days)))
    if st.button("Backtest starten"):
        with st.spinner("Backtest läuft …"):
            eq, trades = run_backtest(
                price_df=prices,
                volume_df=volumes,
                start_idx=start_idx,
                top_n=bt_topn,
                reserve=0,
                use_gd50_exit=bt_use_exit,
                cost_bps=bt_cost,
                slippage_bps=bt_slip
            )
        if eq.empty:
            st.warning("Backtest lieferte keine Ergebnisse.")
        else:
            stats = perf_stats(eq)
            if stats:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("CAGR", f"{stats['CAGR']*100:,.2f}%")
                col2.metric("Volatilität (ann.)", f"{stats['Volatilität (ann.)']*100:,.2f}%")
                col3.metric("Sharpe (ann.)", f"{stats['Sharpe (ann.)']:.2f}")
                col4.metric("Max Drawdown", f"{stats['Max Drawdown']*100:,.2f}%")

            st.line_chart(eq.rename(columns={"Equity": "Depotwert"}))
            st.download_button("📥 Equity-Kurve (CSV)", eq.to_csv().encode("utf-8"),
                               "backtest_equity.csv", "text/csv")

            if not trades.empty:
                st.markdown("### Trades/Turnover")
                st.dataframe(trades, use_container_width=True)
                st.download_button("📥 Trades (CSV)", trades.to_csv(index=False).encode("utf-8"),
                                   "backtest_trades.csv", "text/csv")

st.caption("Hinweis: Nur zu Informations- und Ausbildungszwecken. Keine Anlageberatung.")
