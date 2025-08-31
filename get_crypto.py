import requests
import pandas as pd

BASE_URL = "https://api.binance.com/api/v3/klines"


def get_crypto_data(symbol, interval="1d", limit=1500, start_time=None, end_time=None):
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    r = requests.get(BASE_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

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

    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df["symbol"] = symbol

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms").dt.strftime("%Y-%m-%d")

    return df



def main():
    instruments = pd.read_csv("crypto.csv", comment="#")

    for symbol in instruments["symbol"]:
        try:
            df = get_crypto_data(symbol)
        except Exception as e:
            print(e)

        if not df.empty:
            df.to_csv(f"ohlc_{symbol}.csv", index=False)


if __name__ == "__main__":
    main()
