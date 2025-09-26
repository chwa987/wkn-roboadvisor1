# app.py
# Momentum-Screener mit erweiterten Filtern, Handlungsempfehlungen, Exposure-Übersicht
# und Backtest (wöchentlich; erster Handelstag der Woche).
#
# Erweiterungen, keine Dummies:
# - Analyse mit Momentum-Score (260/130), RS z, Volumen-Score, 52W-Drawdown, Volatilität
# - Filter: Min. ØVol (60T), Max. Drawdown (zum 52W-Hoch), Max. Volatilität, RS > Benchmark (130T)
# - Handlungsempfehlungen: Kaufen/Halten/Verkaufen (GD50/GD200 + Rank) + Exposure-Übersicht
# - Backtest: Wöchentliches Rebalancing am ersten Handelstag der Woche, Equal-Weight,
#             Transaktionskosten optional, Benchmark-Kurve, Logs.

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="Momentum-RoboAdvisor", page_icon="📈", layout="wide")

# ---------------------------- #
# Utils
# ---------------------------- #

@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_ohlc(ticker_list, start, end):
    """Download OHLCV für Ticker-Liste; gibt (price_df, volume_df) mit gemeinsamen Index zurück."""
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
        if closes is None or closes.dropna().empty:
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
    """Volumen-Multiplikator: aktuelles Vol / SMA(lookback). Caps (0.5 – 2.0)."""
    if vol_series is None or vol_series.dropna().empty:
        return np.nan
    cur = vol_series.dropna().iloc[-1]
    base = vol_series.rolling(lookback, min_periods=max(5, lookback//5)).mean().iloc[-1]
    if base is None or base == 0 or pd.isna(base) or pd.isna(cur):
        return np.nan
    return float(np.clip(cur / base, 0.5, 2.0))


def logp(x):
    if pd.isna(x):
        return np.nan
    return np.sign(x) * np.log1p(abs(x))


# ---------------------------- #
# Indikatoren + Score
# ---------------------------- #

def compute_indicators(price_df: pd.DataFrame, volume_df: pd.DataFrame, benchmark_df=None):
    """Berechnet Momentum-/Trend-/Risiko-Kennzahlen je Ticker (letzter Stand der übergebenen DFs)."""
    results = []

    # Cross-sectional RS (130T) im Universum
    mom130_universe = {t: pct_change_over_window(price_df[t], 130) for t in price_df.columns}
    mom130_series = pd.Series(mom130_universe).astype(float)
    mu, sigma = mom130_series.mean(), mom130_series.std(ddof=0)

    # Benchmark 130T-Return
    bm_return = None
    if benchmark_df is not None and not benchmark_df.empty:
        bm_return = pct_change_over_window(benchmark_df.iloc[:, 0], 130)

    for t in price_df.columns:
        s = price_df[t].dropna()
        v = volume_df[t].dropna() if (isinstance(volume_df, pd.DataFrame) and t in volume_df) else pd.Series(dtype=float)
        if s.empty or len(s) < 200:
            continue

        last = s.iloc[-1]
        sma50 = safe_sma(s, 50).iloc[-1]
        sma200 = safe_sma(s, 200).iloc[-1]

        mom260 = pct_change_over_window(s, 260)
        mom130 = pct_change_over_window(s, 130)

        rs_130 = mom130
        rs_z = zscore_last(rs_130, mu, sigma) if not np.isnan(rs_130) else np.nan

        vol_sc = volume_score(v, 60)
        avg_vol = v.rolling(60).mean().iloc[-1] if not v.empty else np.nan

        d50 = (last / sma50 - 1.0) * 100.0 if sma50 else np.nan
        d200 = (last / sma200 - 1.0) * 100.0 if sma200 else np.nan

        sig50 = "Über GD50" if last >= sma50 else "Unter GD50"
        sig200 = "Über GD200" if last >= sma200 else "Unter GD200"

        # 52W-Hoch & Drawdown
        high52 = s[-260:].max() if len(s) >= 260 else s.max()
        dd52 = (last / high52 - 1.0) * 100.0 if high52 else np.nan

        # Volatilität (annualisiert)
        vol = s.pct_change().std() * np.sqrt(252)

        # Relative Stärke vs. Benchmark (130T)
        rs_vs_bm = mom130 - bm_return if bm_return is not None else np.nan

        # Momentum-Score (40/30/20/10)
        score = (
            0.40 * logp(mom260) +
            0.30 * logp(mom130) +
            0.20 * (0 if np.isnan(rs_z) else rs_z) +
            0.10 * (0 if np.isnan(vol_sc) else (vol_sc - 1.0))
        )
        score = 0.0 if np.isnan(score) else float(score)

        results.append({
            "Ticker": t,
            "Kurs aktuell": round(last, 2),
            "MOM260 (%)": round(mom260, 2),
            "MOM130 (%)": round(mom130, 2),
            "RS (130T) (%)": round(rs_130, 2),
            "RS z-Score": round(rs_z, 2),
            "RS vs Benchmark (%)": round(rs_vs_bm, 2) if not np.isnan(rs_vs_bm) else np.nan,
            "Volumen-Score": round(vol_sc, 2) if not np.isnan(vol_sc) else np.nan,
            "Ø Volumen (60T)": round(avg_vol, 0) if not np.isnan(avg_vol) else np.nan,
            "Abstand GD50 (%)": round(d50, 2),
            "Abstand GD200 (%)": round(d200, 2),
            "GD50-Signal": sig50,
            "GD200-Signal": sig200,
            "52W-Drawdown (%)": round(dd52, 2),
            "Volatilität (ann.)": round(vol, 2) if not np.isnan(vol) else np.nan,
            "Momentum-Score": round(score, 3),
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df
    df = df.sort_values("Momentum-Score", ascending=False).reset_index(drop=True)
    df["Rank"] = np.arange(1, len(df) + 1)
    return df


def rec_row(row, in_port, top_n=10):
    """Kaufen/Halten/Verkaufen-Logik (GD50/GD200 + Rank)."""
    t = row["Ticker"]
    rank = row["Rank"]
    over50 = row["GD50-Signal"].startswith("Über")
    over200 = row["GD200-Signal"].startswith("Über")
    if t in in_port:
        if not over50:
            return "🔴 Verkaufen (unter GD50)"
        if rank <= top_n:
            return "🟡 Halten"
        return "🔴 Verkaufen (nicht mehr Top)"
    else:
        if rank <= top_n and over50 and over200:
            return "🟢 Kaufen"
        return "—"


def market_metrics(df: pd.DataFrame, top_n: int):
    """Einheitliche Kennzahlen für Übersicht/Exposure."""
    universe_n = int(len(df))
    if universe_n == 0:
        return {
            "universe_n": 0, "breadth_n": 0, "breadth_share": 0.0,
            "steps": 0, "exposure_share": 0.0, "eff_holdings": 0
        }
    above200 = df["GD200-Signal"].eq("Über GD200")
    breadth_n = int(above200.sum())
    breadth_share = breadth_n / universe_n
    steps = int(np.floor(breadth_share * 10 + 1e-12))  # 10%-Schritte, floor
    steps = min(max(steps, 0), 10)
    exposure_share = steps / 10.0
    eff_holdings = int(round(top_n * exposure_share))
    if exposure_share > 0 and eff_holdings == 0:
        eff_holdings = 1
    return {
        "universe_n": universe_n,
        "breadth_n": breadth_n,
        "breadth_share": breadth_share,
        "steps": steps,
        "exposure_share": exposure_share,
        "eff_holdings": eff_holdings,
    }


# ---------------------------- #
# Backtest (wöchentlich)
# ---------------------------- #

def weekly_first_trading_days(idx: pd.DatetimeIndex) -> list:
    """
    Liefert für jeden Kalender-Wochenblock (Mo-So) den ersten vorhandenen Handelstag.
    Das erfüllt: 'erster Tag der Woche, wenn Montag Feiertag war -> erster verfügbare Handelstag'.
    """
    s = pd.Series(1, index=idx)
    # Gruppiere in Wochen, die am Montag beginnen:
    grp = s.groupby(pd.Grouper(freq="W-MON"))
    # Für jede Woche: nimm die erste Index-Zeile (erster Handelstag in dieser Woche)
    firsts = []
    for _, g in grp:
        if not g.empty:
            firsts.append(g.index[0])
    return firsts


def run_backtest_weekly(prices: pd.DataFrame,
                        volumes: pd.DataFrame,
                        benchmark: pd.Series | None,
                        start_date,
                        end_date,
                        top_n=10,
                        min_volume=500000,
                        max_dd52=-30,
                        max_volatility=1.0,
                        apply_benchmark=True,
                        cost_bps=10.0,
                        slippage_bps=5.0):
    """
    Wöchentlicher Backtest:
      - Rebalance am ersten Handelstag jeder Woche
      - Auswahl basiert auf gefilterter Momentum-Tabelle per Stichtag
      - Equal-weight; Kosten = (Turnover * (cost+slippage))
    """
    idx = prices.index[(prices.index >= pd.to_datetime(start_date)) & (prices.index <= pd.to_datetime(end_date))]
    if len(idx) < 260:
        return pd.DataFrame(), pd.DataFrame()

    rebal_days = weekly_first_trading_days(idx)
    # nur Tage im Bereich behalten
    rebal_days = [d for d in rebal_days if d >= idx.min() and d <= idx.max()]
    if len(rebal_days) < 2:
        return pd.DataFrame(), pd.DataFrame()

    port_val = 1.0
    weights_prev = pd.Series(0.0, index=prices.columns)
    equity = []
    logs = []

    tc = (cost_bps + slippage_bps) / 10000.0

    for i in range(len(rebal_days)-1):
        asof = rebal_days[i]
        nxt  = rebal_days[i+1]

        # Slice bis asof
        p_slice = prices.loc[:asof]
        v_slice = volumes.loc[:asof] if isinstance(volumes, pd.DataFrame) else pd.DataFrame(index=p_slice.index)

        # Benchmark-Slice (optional)
        bm_slice = None
        if benchmark is not None:
            bm_slice = pd.DataFrame({"BM": benchmark.loc[:asof].dropna()})
            if bm_slice.empty:
                bm_slice = None

        # Indikatoren zum Stichtag
        snap = compute_indicators(p_slice, v_slice, benchmark_df=bm_slice)
        if snap.empty:
            equity.append((nxt, port_val))
            logs.append({"Date": asof, "NumHold": 0, "ExposureSteps": 0, "Turnover": 0.0,
                         "GrossRet": 0.0, "Cost": 0.0, "NetRet": 0.0, "PortVal": port_val})
            continue

        # Filter anwenden (identisch zur UI)
        # ØVolumen (60T), Drawdown, Volatilität, RS>BM
        filt = snap.copy()
        filt = filt[filt["Ø Volumen (60T)"] >= min_volume]
        filt = filt[filt["52W-Drawdown (%)"] >= max_dd52]
        filt = filt[filt["Volatilität (ann.)"] <= max_volatility]
        if apply_benchmark and "RS vs Benchmark (%)" in filt.columns:
            filt = filt[filt["RS vs Benchmark (%)"] > 0]
        filt = filt.sort_values("Momentum-Score", ascending=False).reset_index(drop=True)
        if filt.empty:
            equity.append((nxt, port_val))
            logs.append({"Date": asof, "NumHold": 0, "ExposureSteps": 0, "Turnover": 0.0,
                         "GrossRet": 0.0, "Cost": 0.0, "NetRet": 0.0, "PortVal": port_val})
            continue

        # Exposure (aus gefiltertem Universum)
        metrics = market_metrics(filt, top_n)
        eff_hold = metrics["eff_holdings"]

        sel = filt.head(eff_hold).copy() if eff_hold > 0 else filt.iloc[0:0]
        new_weights = pd.Series(0.0, index=prices.columns)
        if not sel.empty:
            w = 1.0 / len(sel)
            new_weights.loc[sel["Ticker"].values] = w

        # Renditen zwischen asof (ex-ante) und nxt (bis inkl. Vortag von nxt)
        rets = prices.loc[asof:nxt].pct_change().fillna(0)
        # Verwende Gewichte vom VORHERIGEN Rebalance für diese Periode
        gross_return = (rets.iloc[1:] * weights_prev).sum(axis=1).add(0).add(1).prod() - 1.0 if len(rets) > 1 else 0.0

        turnover = float((new_weights - weights_prev).abs().sum())
        cost = turnover * tc
        net_return = gross_return - cost
        port_val *= (1.0 + net_return)

        equity.append((nxt, port_val))
        logs.append({
            "Date": asof,
            "NumHold": int(len(sel)),
            "ExposureSteps": int(metrics["steps"]),
            "Turnover": float(turnover),
            "GrossRet": float(gross_return),
            "Cost": float(cost),
            "NetRet": float(net_return),
            "PortVal": float(port_val)
        })

        # Rebalance: neue Gewichte gelten für NÄCHSTE Woche
        weights_prev = new_weights.copy()

    eq_df = pd.DataFrame(equity, columns=["Date", "Equity"]).set_index("Date")
    logs_df = pd.DataFrame(logs)
    return eq_df, logs_df


# ---------------------------- #
# Sidebar
# ---------------------------- #

st.sidebar.header("⚙️ Einstellungen")
top_n = st.sidebar.number_input("Top-N (Kernpositionen)", min_value=3, max_value=50, value=10, step=1)
start_date = st.sidebar.date_input("Startdatum (Datenabruf)", value=datetime.today() - timedelta(days=900))
end_date   = st.sidebar.date_input("Enddatum", value=datetime.today())

st.sidebar.markdown("### Filter")
min_volume = st.sidebar.number_input("Min. Ø Volumen (60T)", min_value=0, value=5_000_000, step=100_000)
max_dd52 = st.sidebar.slider("Max. Drawdown zum 52W-Hoch (%)", -100, 0, -30, step=5)
max_volatility = st.sidebar.slider("Max. Volatilität (ann.)", 0.0, 2.0, 1.0, step=0.05)
apply_benchmark = st.sidebar.checkbox("Nur Aktien > Benchmark (130T)", value=True)
benchmark_ticker = st.sidebar.text_input("Benchmark-Ticker", "SPY")

st.sidebar.markdown("### Backtest-Kosten (bps)")
cost_bps = st.sidebar.number_input("Kommission (bps)", min_value=0.0, value=10.0, step=1.0)
slip_bps = st.sidebar.number_input("Slippage (bps)", min_value=0.0, value=5.0, step=1.0)

# ---------------------------- #
# Daten laden
# ---------------------------- #

st.title("📊 Momentum-Analyse (erweitert) – Live-Berechnung")

uploaded = st.file_uploader("CSV mit **Ticker** und optional **Name**", type=["csv"])
tickers_txt = st.text_input("Oder Ticker (kommagetrennt):", "AAPL, MSFT, TSLA, NVDA, META, AVGO, ORCL, COST, JPM, LLY")
portfolio_txt = st.text_input("(Optional) Aktuelle Portfolio-Ticker:", "")

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
            st.error("CSV benötigt mindestens die Spalte 'Ticker'.")
    except Exception as e:
        st.error(f"CSV konnte nicht gelesen werden: {e}")

tickers = [t.strip().upper() for t in tickers_txt.split(",") if t.strip()]
portfolio = set([t.strip().upper() for t in portfolio_txt.split(",") if t.strip()])

if not tickers:
    st.info("Bitte Ticker eingeben oder CSV laden.")
    st.stop()

with st.spinner("Lade Kursdaten …"):
    prices, volumes = fetch_ohlc(tickers, start_date, end_date)
    bm_prices, _ = fetch_ohlc([benchmark_ticker], start_date, end_date)

if prices.empty:
    st.warning("Keine Kursdaten geladen.")
    st.stop()

# ---------------------------- #
# Analyse & Filter
# ---------------------------- #

df = compute_indicators(prices, volumes, benchmark_df=bm_prices)
if df.empty:
    st.warning("Keine Kennzahlen berechnet (zu wenig Historie?).")
    st.stop()

df["Name"] = df["Ticker"].map(name_map).fillna(df["Ticker"])

# Filter anwenden
filtered = df.copy()
filtered = filtered[filtered["Ø Volumen (60T)"] >= min_volume]
filtered = filtered[filtered["52W-Drawdown (%)"] >= max_dd52]
filtered = filtered[filtered["Volatilität (ann.)"] <= max_volatility]
if apply_benchmark and "RS vs Benchmark (%)" in filtered.columns:
    filtered = filtered[filtered["RS vs Benchmark (%)"] > 0]
filtered = filtered.sort_values("Momentum-Score", ascending=False).reset_index(drop=True)
filtered["Rank"] = np.arange(1, len(filtered) + 1)

# Exposure-Metriken aus dem gefilterten Universum
metrics = market_metrics(filtered, top_n)

# ---------------------------- #
# Tabs
# ---------------------------- #

tab1, tab2, tab3 = st.tabs(["🔬 Analyse", "🧭 Handlungsempfehlungen", "📈 Backtest (wöchentlich)"])

with tab1:
    st.subheader("Analyse – Kennzahlen (gefiltert)")
    st.dataframe(filtered, use_container_width=True)

with tab2:
    st.subheader("Handlungsempfehlungen")
    # Übersicht oben
    st.markdown(
        f"**Gefiltertes Universum:** {metrics['universe_n']} Aktien  •  "
        f"**>GD200:** {metrics['breadth_n']}/{metrics['universe_n']} ({metrics['breadth_share']:.0%})  •  "
        f"**Exposure (10%-Stufen):** {metrics['steps']}  •  "
        f"**Geplante Holdings:** {metrics['eff_holdings']} (von Top-{top_n})"
    )

    rec_df = filtered.copy()
    rec_df["Handlung"] = rec_df.apply(lambda r: rec_row(r, portfolio, top_n=top_n), axis=1)
    cols = ["Rank", "Ticker", "Name", "Momentum-Score", "GD50-Signal", "GD200-Signal", "Handlung"]
    st.write(rec_df[cols].to_html(escape=False, index=False), unsafe_allow_html=True)

with tab3:
    st.subheader("Backtest – wöchentlich (erster Handelstag der Woche)")
    st.caption("Auswahl & Filter identisch zur Analyse; Equal-Weight; Kosten via Sidebar bps.")

    if st.button("▶️ Backtest starten"):
        with st.spinner("Berechne Backtest …"):
            bm_series = bm_prices.iloc[:, 0].copy() if not bm_prices.empty else None
            eq_df, logs_df = run_backtest_weekly(
                prices, volumes, bm_series, start_date, end_date,
                top_n=top_n,
                min_volume=min_volume,
                max_dd52=max_dd52,
                max_volatility=max_volatility,
                apply_benchmark=apply_benchmark,
                cost_bps=cost_bps,
                slippage_bps=slip_bps
            )
        if eq_df.empty:
            st.warning("Backtest lieferte keine Werte (zu wenig Daten oder zu strenge Filter?).")
        else:
            # Benchmark-Kurve
            fig, ax = plt.subplots(figsize=(9,4))
            ax.plot(eq_df.index, eq_df["Equity"], label="Strategie")
            if bm_prices is not None and not bm_prices.empty:
                bm_norm = bm_prices.iloc[:,0].loc[eq_df.index.min():eq_df.index.max()].dropna()
                if not bm_norm.empty:
                    bm_norm = (bm_norm / bm_norm.iloc[0])
                    ax.plot(bm_norm.index, bm_norm.values, label=f"Benchmark ({benchmark_ticker})", alpha=0.8)
            ax.set_title("Equity-Kurve (wöchentliches Rebalancing)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)

            st.markdown("**Rebalance-Log (pro Woche):**")
            st.dataframe(logs_df, use_container_width=True)
            st.download_button("📥 Logs (CSV)", logs_df.to_csv(index=False).encode("utf-8"), "weekly_backtest_logs.csv", "text/csv")
            st.download_button("📥 Equity (CSV)", eq_df.to_csv().encode("utf-8"), "weekly_backtest_equity.csv", "text/csv")

st.caption("Nur Informations- und Ausbildungszwecke. Keine Anlageempfehlung.")
