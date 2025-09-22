# -------------------------------------------
# ROBUSTER OHLCV-DOWNLOADER FÜR SEHR VIELE TICKER
# -------------------------------------------
@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_ohlc(ticker_list, start, end, batch_size: int = 150):
    """
    Lädt OHLCV für viele Ticker robust:
      - akzeptiert str "AAPL, MSFT" oder Liste
      - dedupliziert, säubert
      - lädt in Batches (Standard 150)
      - fallback auf Einzelticker-Download, wenn Multi-Download Daten auslässt
      - nimmt 'Adj Close' wenn vorhanden, sonst 'Close'
      - Volumen fehlt? -> leere Series; kein Crash
      - gibt preis_df, volumen_df zurück (Index vereinheitlicht)
    """
    # --- Helfer ---
    def _to_list(x):
        if isinstance(x, str):
            toks = [t.strip() for t in x.split(",")]
        else:
            toks = [str(t).strip() for t in list(x)]
        # Ticker-Formate wie BRK-B, RHM.DE, ^GSPC erlauben – nur leere raus
        return [t for t in dict.fromkeys(toks) if t]

    def _chunks(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    def _single_fallback(t):
        """Einzelticker-Fallback via yf.Ticker().history()"""
        try:
            h = yf.Ticker(t).history(start=start, end=end, auto_adjust=False)
        except Exception:
            return None, None
        if h is None or h.empty:
            return None, None
        closes = h["Adj Close"] if "Adj Close" in h.columns else h.get("Close")
        if closes is None or closes.empty:
            return None, None
        vols = h.get("Volume")
        closes = closes.rename(t).dropna()
        vols   = (vols.rename(t) if vols is not None else pd.Series(name=t, dtype=float))
        return closes, vols

    tickers = _to_list(ticker_list)
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()

    close_series_list, vol_series_list = [], []
    skipped = []

    for chunk in _chunks(tickers, batch_size):
        # Multi-Download für den Chunk
        try:
            data = yf.download(
                tickers=" ".join(chunk),
                start=start,
                end=end,
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
        except Exception as e:
            st.error(f"Download-Fehler für Batch {chunk[:5]}…: {e}")
            # kompletter Chunk notfalls per Einzelticker
            for t in chunk:
                c, v = _single_fallback(t)
                if c is None or c.empty:
                    skipped.append(t)
                else:
                    close_series_list.append(c)
                    vol_series_list.append(v if v is not None else pd.Series(name=t, dtype=float))
            continue

        if data is None or data.empty:
            # gar nichts gekommen -> Einzelticker-Fallback
            for t in chunk:
                c, v = _single_fallback(t)
                if c is None or c.empty:
                    skipped.append(t)
                else:
                    close_series_list.append(c)
                    vol_series_list.append(v if v is not None else pd.Series(name=t, dtype=float))
            continue

        # MultiIndex vs. SingleIndex sauber behandeln
        if isinstance(data.columns, pd.MultiIndex):
            present = set(data.columns.get_level_values(0))
            missing = [t for t in chunk if t not in present]
            skipped.extend(missing)

            for t in present:
                try:
                    df_t = data[t]
                except Exception:
                    df_t = None

                if df_t is None or df_t.empty:
                    # Fallback auf Einzelticker
                    c, v = _single_fallback(t)
                    if c is None or c.empty:
                        skipped.append(t)
                        continue
                    close_series_list.append(c)
                    vol_series_list.append(v if v is not None else pd.Series(name=t, dtype=float))
                    continue

                # Spalten sicher ziehen
                closes = df_t["Adj Close"] if "Adj Close" in df_t.columns else df_t.get("Close")
                if closes is None or closes.empty:
                    # nochmal Fallback
                    c, v = _single_fallback(t)
                    if c is None or c.empty:
                        skipped.append(t)
                        continue
                    close_series_list.append(c)
                    vol_series_list.append(v if v is not None else pd.Series(name=t, dtype=float))
                    continue

                vols = df_t.get("Volume")
                close_series_list.append(closes.rename(t).dropna())
                vol_series_list.append(vols.rename(t) if vols is not None else pd.Series(name=t, dtype=float))

        else:
            # Single-DataFrame (tritt auf, wenn nur 1 Ticker im Chunk)
            t = chunk[0]
            df_t = data
            closes = df_t["Adj Close"] if "Adj Close" in df_t.columns else df_t.get("Close")
            vols   = df_t.get("Volume")

            if closes is None or closes.empty:
                c, v = _single_fallback(t)
                if c is None or c.empty:
                    skipped.append(t)
                else:
                    close_series_list.append(c)
                    vol_series_list.append(v if v is not None else pd.Series(name=t, dtype=float))
            else:
                close_series_list.append(closes.rename(t).dropna())
                vol_series_list.append(vols.rename(t) if vols is not None else pd.Series(name=t, dtype=float))

    # Zusammenbauen & Index vereinheitlichen
    price = pd.concat(close_series_list, axis=1) if close_series_list else pd.DataFrame()
    price = price.sort_index()
    if not price.empty:
        price = price[~price.index.duplicated(keep="first")]  # doppelte Zeitstempel raus

    if vol_series_list:
        volume = pd.concat(vol_series_list, axis=1)
        volume = volume.reindex(price.index)
    else:
        volume = pd.DataFrame(index=price.index)

    if skipped:
        # nur kurze Liste anzeigen – Rest angedeutet
        msg = ", ".join(skipped[:15]) + (" …" if len(skipped) > 15 else "")
        st.warning(f"Keine/zu wenige Daten für {len(skipped)} Ticker übersprungen: {msg}")

    return price, volume
