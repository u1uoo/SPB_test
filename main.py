import argparse
import os

from get_crypto import get_crypto_data
from get_stocks import get_stocks_data
from plot_data import load_csv as load_csv_plot, plot_indicators


def cmd_fetch(args):
    df = get_crypto_data(args.symbol, interval=args.timeframe)
    if df.empty:
        print("cannot fetch data")
        return 1
    out = f"ohlc_{args.symbol}.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    return 0


def cmd_plot(args):
    path = args.path or f"ohlc_{args.symbol}.csv"
    if not os.path.exists(path):
        print(f"file not found")
        return 1
    df = load_csv_plot(path)
    if df.empty:
        print("empty file")
        return 1
    symbol = args.symbol or (
        df["symbol"].iloc[0] if "symbol" in df.columns else "SYMBOL"
    )
    sma_periods = tuple(int(x) for x in args.sma.split(",")) if args.sma else (12, 26)
    ema_periods = tuple(int(x) for x in args.ema.split(",")) if args.ema else (12, 26)
    plot_indicators(df, symbol, sma_periods=sma_periods, ema_periods=ema_periods)
    return 0


def cmd_fetch_plot(args):
    fetch_rc = cmd_fetch(args)
    if fetch_rc != 0:
        return fetch_rc
    path = f"ohlc_{args.symbol}.csv"

    class P:
        pass

    p = P()
    p.path = path
    p.symbol = args.symbol
    p.sma = args.sma
    p.ema = args.ema
    return cmd_plot(p)


def build_parser():
    parser = argparse.ArgumentParser(description="")
    sub = parser.add_subparsers(dest="command")

    p_fetch = sub.add_parser("fetch", help="fetch data from binance")
    p_fetch.add_argument(
        "--symbol", type=str, default="BTCUSDT", help="symbol, e.g. BTCUSDT"
    )

    p_fetch.add_argument("--timeframe", type=str, default="1d", help="timeframe")
    p_fetch.set_defaults(func=cmd_fetch)

    p_plot = sub.add_parser("plot", help="Plot indicators from CSV")
    p_plot.add_argument(
        "--symbol", type=str, default=None, help="Symbol for titles/paths"
    )
    p_plot.add_argument(
        "--path", type=str, default=None, help="CSV path (default ohlc_{symbol}.csv)"
    )
    p_plot.add_argument(
        "--sma",
        type=str,
        default=None,
        help="SMA periods (e.g. 12,26)",
    )
    p_plot.add_argument(
        "--ema", type=str, default=None, help="EMA periods (e.g. 12,26)"
    )
    p_plot.set_defaults(func=cmd_plot)

    p_fp = sub.add_parser("fetch-plot", help="Fetch OHLC then plot")
    p_fp.add_argument(
        "--symbol", type=str, default="BTCUSDT", help="symbol, e.g. BTCUSDT"
    )
    p_fp.add_argument("--timeframe", type=str, default="1d", help="timeframe")
    p_fp.add_argument(
        "--sma",
        type=str,
        default=None,
        help="SMA periods (e.g. 12,26)",
    )
    p_fp.add_argument("--ema", type=str, default=None, help="EMA periods (e.g. 12,26)")
    p_fp.set_defaults(func=cmd_fetch_plot)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    main()
