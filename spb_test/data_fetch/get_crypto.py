"""Utilities to fetch OHLCV crypto data from Binance and save to CSV files.

This module exposes a small API:
- `get_crypto_data(symbol, interval, ...)` → `pandas.DataFrame`
   with columns: `open_time`, `open`, `high`, `low`, `close`, `volume`, `symbol`.
- `main()` reads symbols from `crypto.csv` and writes `ohlc_{symbol}.csv` files.
"""

import requests
import pandas as pd

BASE_URL = "https://api.binance.com/api/v3/klines"


def get_crypto_data(symbol, interval="1d", limit=1500, start_time=None, end_time=None):
    """Fetch OHLCV candles for a crypto `symbol` from Binance.

    Args:
        symbol (str): Trading pair, e.g. "BTCUSDT".
        interval (str): Candle interval supported by Binance (e.g. "1d", "1h").
        limit (int): Maximum number of candles to fetch (Binance allows up to 1500).
        start_time (int | None): Optional start time in milliseconds since epoch.
        end_time (int | None): Optional end time in milliseconds since epoch.

    Returns:
        pandas.DataFrame: Tidy frame with columns
            ["open_time", "open", "high", "low", "close", "volume", "symbol"].
            "open_time" is formatted as YYYY-MM-DD string.
    """
    # Build request parameters for Binance Klines endpoint
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    # Perform the HTTP request and raise for any non-2xx response
    r = requests.get(BASE_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    # Map the raw list response into a DataFrame with explicit column names
    df = pd.DataFrame(
        data,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )

    # Keep only the fields we chart/use
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    # Tag the symbol for downstream code
    df["symbol"] = symbol

    # Convert Binance millisecond timestamps to date string (daily resolution)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms").dt.strftime("%Y-%m-%d")

    return df



def main():
    """Entry point for the CLI when used as a module or script."""
    instruments = pd.read_csv("data/crypto.csv", comment="#")

    for symbol in instruments["symbol"]:
        try:
            df = get_crypto_data(symbol)
        except Exception as e:
            print(e)

        if not df.empty:
            # Save per-symbol OHLCV to CSV in the data directory
            df.to_csv(f"data/ohlc_{symbol}.csv", index=False)


if __name__ == "__main__":
    main()
