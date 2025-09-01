"""Fetch OHLCV data for stocks using yfinance and save to CSV files.

API:
- `get_stocks_data(symbol, ...)` → `pandas.DataFrame` 
   with columns: `time`, `open`, `high`, `low`, `close`, `volume`, `symbol`.
- `main()` reads tickers from `stocks.csv` and writes `ohlc_{symbol}.csv` files.
"""

import yfinance as yf
import pandas as pd


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


def main():
    """Entry point for the CLI when used as a module or script."""
    instruments = pd.read_csv("stocks.csv", comment="#")

    for symbol in instruments["symbol"]:
        df = get_stocks_data(symbol)
        if not df.empty:
            # Save per-symbol OHLCV to CSV in the current directory
            df.to_csv(f"ohlc_{symbol}.csv", index=False)
            print(f"saved to ohlc_{symbol}.csv")
        else:
            print(f"empty data")


if __name__ == "__main__":
    main()
