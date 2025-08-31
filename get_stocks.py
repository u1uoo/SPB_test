import yfinance as yf
import pandas as pd


def get_stocks_data(symbol, interval="1d", period="1y", start=None, end=None):
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

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
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
    instruments = pd.read_csv("stocks.csv", comment="#")

    for symbol in instruments["symbol"]:
        df = get_stocks_data(symbol)
        if not df.empty:
            df.to_csv(f"ohlc_{symbol}.csv", index=False)
            print(f"saved to ohlc_{symbol}.csv")
        else:
            print(f"empty data")


if __name__ == "__main__":
    main()
