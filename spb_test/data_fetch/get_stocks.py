"""Fetch OHLCV data for stocks using yfinance and save to CSV files.

API:
- `get_stocks_data(symbol, ...)` → `pandas.DataFrame` 
   with columns: `time`, `open`, `high`, `low`, `close`, `volume`, `symbol`.
- `main()` reads tickers from `stocks.csv` and writes `ohlc_{symbol}.csv` files.
"""

import yfinance as yf
import pandas as pd
import math


def get_stocks_data(symbol, interval="1d", period="1y", start=None, end=None):
    """Download OHLCV data for a single stock `symbol` via yfinance.

    If `start`/`end` are provided, `period` is ignored (as per yfinance rules).

    Args:
        symbol (str): Ticker, e.g. "AAPL".
        interval (str): Candle interval (e.g. "1d", "1h").
        period (str): Range when start/end are not set (e.g. "1y").
        start (str | None): Start date in YYYY-MM-DD.
        end (str | None): End date in YYYY-MM-DD.

    Returns:
        pandas.DataFrame: Columns ["time", "open", "high", "low", "close", "volume", "symbol"].
    """
    try:
        df = yf.download(
            tickers=symbol,
            interval=interval,
            period=None if start or end else period,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
        )
    except Exception as e:
        print(e)
        return pd.DataFrame()

    if df.empty:
        return df

    # yfinance sometimes returns a MultiIndex when multiple tickers used; normalize
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    # Standardize column names expected by downstream plotting code
    df = df.rename(
        columns={
            "Date": "time",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df["symbol"] = symbol
    return df[["time", "open", "high", "low", "close", "volume", "symbol"]]

def _to_float(value):
    """Convert value to float, returning None when not finite/convertible."""
    try:
        if value is None:
            return None
        x = float(value)
        if math.isfinite(x):
            return x
        return None
    except Exception:
        return None


def _get_fast_price(tkr: yf.Ticker):
    """Try multiple fields to get the latest price from fast_info/info."""
    price_fields = ("last_price", "lastPrice", "regularMarketPrice", "previousClose")
    fast = getattr(tkr, "fast_info", None)
    if isinstance(fast, dict):
        for key in price_fields:
            v = fast.get(key)
            fv = _to_float(v)
            if fv is not None:
                return fv
    info = getattr(tkr, "info", {}) or {}
    for key in price_fields:
        v = info.get(key)
        fv = _to_float(v)
        if fv is not None:
            return fv
    return None


def get_financial_ratios(symbol: str):
    """Return minimal set of financial ratios for a quick overview.

    Returned keys (when available):
    - price, pe_ttm, pb
    - roe, profit_margin
    - debt_to_equity, dividend_yield
    - market_cap
    """
    tkr = yf.Ticker(symbol)
    info = getattr(tkr, "info", {}) or {}
    fast = getattr(tkr, "fast_info", {}) or {}

    price = _get_fast_price(tkr)

    eps_ttm = _to_float(info.get("trailingEps"))
    book_value_ps = _to_float(info.get("bookValue"))

    pe_ttm = _to_float(info.get("trailingPE"))
    pb = _to_float(info.get("priceToBook"))

    if pe_ttm is None and price is not None and eps_ttm and eps_ttm != 0:
        pe_ttm = price / eps_ttm
    if pb is None and price is not None and book_value_ps and book_value_ps != 0:
        pb = price / book_value_ps

    roe = _to_float(info.get("returnOnEquity"))
    profit_margin = _to_float(info.get("profitMargins"))
    debt_to_equity = _to_float(info.get("debtToEquity"))

    dividend_yield = _to_float(info.get("dividendYield"))

    market_cap = _to_float(info.get("marketCap") or fast.get("market_cap"))

    return {
        "symbol": symbol,
        "price": price,
        "pe_ttm": pe_ttm,
        "pb": pb,
        "roe": roe,
        "profit_margin": profit_margin,
        "debt_to_equity": debt_to_equity,
        "dividend_yield": dividend_yield,
        "market_cap": market_cap,
    }