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

    price = info.get("regularMarketPrice")
    eps_ttm = info.get("trailingEps")
    book_value_ps = info.get("bookValue")
    pe_ttm = info.get("trailingPE")
    pb = info.get("priceToBook")
    roe = info.get("returnOnEquity")
    profit_margin = info.get("profitMargins")
    debt_to_equity = info.get("debtToEquity")
    dividend_yield = info.get("dividendYield")
    market_cap = info.get("marketCap")

    return {
        "symbol": symbol,
        "price": price,
        "eps_ttm": eps_ttm,
        "book_value_ps": book_value_ps,
        "pe_ttm": pe_ttm,
        "pb": pb,
        "roe": roe,
        "profit_margin": profit_margin,
        "debt_to_equity": debt_to_equity,
        "dividend_yield": dividend_yield,
        "market_cap": market_cap,
    }