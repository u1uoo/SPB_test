"""CLI for fetching OHLC data (crypto/stocks) and plotting indicators.

Subcommands:
- fetch-crypto / fetch-cryptos
- fetch-stock / fetch-stocks
- plot
- fetch-plot-crypto / fetch-plot-stock
"""

import argparse
import os
import sys
from types import SimpleNamespace

from spb_test.data_fetch.get_crypto import get_crypto_data
from spb_test.data_fetch.get_stocks import get_stocks_data, get_financial_ratios
from spb_test.plotting.plot_data import load_csv as load_csv_plot, plot_indicators
from spb_test.plotting.plotly_plot import plot_indicators_plotly


def eprint(msg):
    """Print to stderr (for errors and warnings)."""
    print(msg, file=sys.stderr)


def _parse_periods(opt, default=(12, 26)):
    """Parse comma-separated periods string into a tuple of ints.

    Accepts tuples/lists directly; validates positivity.
    """
    if not opt:
        return default
    try:
        periods = tuple(int(x.strip()) for x in opt.split(","))
        if not periods or any(p <= 0 for p in periods):
            raise ValueError
        return periods
    except Exception:
        eprint("invalid periods; example: 12,26")
        sys.exit(2)


def _ensure_dir(path):
    """Ensure parent directory for `path` exists."""
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _save_df(df, out):
    """Save DataFrame to CSV path, creating directories if needed."""
    _ensure_dir(out)
    df.to_csv(out, index=False)
    print(f"saved to {out}")


def _fetch_and_save(fetch_fn, symbol, outdir, **kwargs):
    """Generic fetch+save helper used by single-symbol commands."""
    try:
        df = fetch_fn(symbol, **kwargs)
    except Exception as exc:
        eprint(f"fetch error: {exc}")
        return 1
    if df.empty:
        eprint("cannot fetch data")
        return 1
    _save_df(df, os.path.join(outdir, f"ohlc_{symbol}.csv"))
    return 0


def _batch_fetch(list_path, fetch_fn, outdir, per_symbol_kwargs):
    """Batch read symbols from CSV and fetch+save each one.

    The `list_path` CSV must contain a `symbol` column.
    """
    import pandas as pd

    if not os.path.exists(list_path):
        eprint("file not found")
        return 1

    instruments = pd.read_csv(list_path, comment="#")
    if "symbol" not in instruments.columns:
        eprint("missing 'symbol' column")
        return 1

    total = 0
    for symbol in instruments["symbol"]:
        sym = str(symbol)
        try:
            df = fetch_fn(sym, **per_symbol_kwargs)
        except Exception as exc:
            eprint(f"{sym}: error: {exc}")
            continue
        if df.empty:
            eprint(f"{sym}: empty data")
            continue
        out = os.path.join(outdir, f"ohlc_{sym}.csv")
        _ensure_dir(out)
        df.to_csv(out, index=False)
        print(f"{sym}: saved {len(df)} rows to {out}")
        total += 1

    print(f"completed: {total} saved")
    return 0 if total > 0 else 1


def cmd_fetch_crypto(args):
    return _fetch_and_save(
        get_crypto_data,
        args.symbol,
        args.outdir,
        interval=args.interval,
    )


def cmd_fetch_crypto_batch(args):
    return _batch_fetch(
        args.list,
        get_crypto_data,
        args.outdir,
        {"interval": args.interval},
    )


def cmd_fetch_stock(args):
    return _fetch_and_save(
        get_stocks_data,
        args.symbol,
        args.outdir,
        interval=args.interval,
        period=args.period,
        start=args.start,
        end=args.end,
    )


def cmd_fetch_stocks_batch(args):
    return _batch_fetch(
        args.list,
        get_stocks_data,
        args.outdir,
        {
            "interval": args.interval,
            "period": args.period,
            "start": args.start,
            "end": args.end,
        },
    )


def cmd_plot(args):
    """Load CSV and render indicators with selected backend (mpl/plotly)."""
    if args.path:
        path = args.path
    elif args.symbol:
        path = os.path.join(args.outdir, f"ohlc_{args.symbol}.csv")
    else:
        eprint("either --path or --symbol is required for 'plot'")
        return 2

    if not os.path.exists(path):
        eprint("file not found")
        return 1

    df = load_csv_plot(path)
    if df.empty:
        eprint("empty file")
        return 1

    symbol = args.symbol or (df["symbol"].iloc[0] if "symbol" in df.columns else "SYMBOL")

    def _norm_periods(val, default=(12, 26)):
        """Normalize CLI option into a tuple of ints."""
        if val is None:
            return default
        if isinstance(val, (tuple, list)):
            return tuple(int(x) for x in val)
        return _parse_periods(val, default=default)

    sma_periods = _norm_periods(args.sma, default=(12, 26))
    ema_periods = _norm_periods(args.ema, default=(12, 26))
    backend = getattr(args, "backend", "mpl")
    if backend == "plotly":
        plot_indicators_plotly(
            df,
            symbol,
            sma_periods=sma_periods,
            ema_periods=ema_periods,
            is_crypto=getattr(args, "is_crypto", False),
        )
    else:
        plot_indicators(
            df,
            symbol,
            sma_periods=sma_periods,
            ema_periods=ema_periods,
            is_crypto=getattr(args, "is_crypto", False),
        )
    return 0


def cmd_fetch_plot_crypto(args):
    """Fetch crypto OHLC and immediately plot indicators."""
    if cmd_fetch_crypto(args) != 0:
        return 1
    p = SimpleNamespace(
        path=os.path.join(args.outdir, f"ohlc_{args.symbol}.csv"),
        symbol=args.symbol,
        sma=args.sma,
        ema=args.ema,
        outdir=args.outdir,
        is_crypto=True,
        backend=getattr(args, "backend", "mpl"),
    )
    return cmd_plot(p)


def cmd_fetch_plot_stock(args):
    """Fetch stock OHLC and immediately plot indicators."""
    if cmd_fetch_stock(args) != 0:
        return 1
    p = SimpleNamespace(
        path=os.path.join(args.outdir, f"ohlc_{args.symbol}.csv"),
        symbol=args.symbol,
        sma=args.sma,
        ema=args.ema,
        outdir=args.outdir,
        is_crypto=False,
        backend=getattr(args, "backend", "mpl"),
    )
    return cmd_plot(p)


def cmd_fundamentals(args):
    """Fetch financial ratios and save to CSV.

    Single symbol → fundamentals_{SYMBOL}.csv
    List of symbols → fundamentals.csv
    """
    import pandas as pd

    symbols = []
    if getattr(args, "symbol", None):
        symbols = [args.symbol]
    elif getattr(args, "list", None):
        if not os.path.exists(args.list):
            eprint("file not found")
            return 1
        instruments = pd.read_csv(args.list, comment="#")
        if "symbol" not in instruments.columns:
            eprint("missing 'symbol' column")
            return 1
        symbols = [str(s) for s in instruments["symbol"]]
    else:
        eprint("either --symbol or --list is required for 'fundamentals'")
        return 2

    rows = []
    for sym in symbols:
        try:
            ratios = get_financial_ratios(sym)
        except Exception as exc:
            eprint(f"{sym}: error: {exc}")
            continue
        rows.append(ratios)

    if not rows:
        eprint("no data")
        return 1

    import pandas as pd
    df = pd.DataFrame(rows)
    fname = f"fundamentals_{symbols[0]}.csv" if len(symbols) == 1 else "fundamentals.csv"
    out = os.path.join(args.outdir, fname)
    _ensure_dir(out)
    df.to_csv(out, index=False)
    print(f"saved to {out}")
    return 0

def build_parser():
    """CLI argument parser."""
    parser = argparse.ArgumentParser(description="Fetch OHLC data and plot indicators.")
    sub = parser.add_subparsers(dest="command")

    def add_common_out_args(p):
        """Add common output arguments to a subparser."""
        p.add_argument("--outdir", type=str, default=".", help="directory for outputs")


    p_fetch = sub.add_parser("fetch-crypto", help="fetch crypto OHLC from Binance")
    p_fetch.add_argument("--symbol", type=str, default="BTCUSDT", help="symbol, e.g. BTCUSDT")
    p_fetch.add_argument("--interval", type=str, default="1d", help="interval, e.g. 1d, 1h")
    add_common_out_args(p_fetch)
    p_fetch.set_defaults(func=cmd_fetch_crypto)


    p_fetch_crypto_batch = sub.add_parser("fetch-cryptos", help="batch fetch cryptos from CSV")
    p_fetch_crypto_batch.add_argument("--list", type=str, default="crypto.csv", help="csv with column 'symbol'")
    p_fetch_crypto_batch.add_argument("--interval", type=str, default="1d", help="interval, e.g. 1d, 1h")
    add_common_out_args(p_fetch_crypto_batch)
    p_fetch_crypto_batch.set_defaults(func=cmd_fetch_crypto_batch)


    p_fetch_stock = sub.add_parser("fetch-stock", help="fetch single stock OHLC via yfinance")
    p_fetch_stock.add_argument("--symbol", type=str, required=True, help="stock ticker, e.g. AAPL")
    p_fetch_stock.add_argument("--interval", type=str, default="1d", help="interval, e.g. 1d, 1h")
    p_fetch_stock.add_argument("--period", type=str, default="1y", help="period when start/end not provided")
    p_fetch_stock.add_argument("--start", type=str, default=None, help="start date YYYY-MM-DD")
    p_fetch_stock.add_argument("--end", type=str, default=None, help="end date YYYY-MM-DD")
    add_common_out_args(p_fetch_stock)
    p_fetch_stock.set_defaults(func=cmd_fetch_stock)


    p_fetch_stocks = sub.add_parser("fetch-stocks", help="batch fetch stocks from CSV")
    p_fetch_stocks.add_argument("--list", type=str, default="stocks.csv", help="csv with column 'symbol'")
    p_fetch_stocks.add_argument("--interval", type=str, default="1d", help="interval, e.g. 1d, 1h")
    p_fetch_stocks.add_argument("--period", type=str, default="1y", help="period when start/end not provided")
    p_fetch_stocks.add_argument("--start", type=str, default=None, help="start date YYYY-MM-DD")
    p_fetch_stocks.add_argument("--end", type=str, default=None, help="end date YYYY-MM-DD")
    add_common_out_args(p_fetch_stocks)
    p_fetch_stocks.set_defaults(func=cmd_fetch_stocks_batch)


    p_plot = sub.add_parser("plot", help="Plot indicators from CSV")
    p_plot.add_argument("--symbol", type=str, default=None, help="Symbol for titles/paths")
    p_plot.add_argument("--path", type=str, default=None, help="CSV path (default ohlc_{symbol}.csv)")
    p_plot.add_argument("--sma", type=str, default=None, help="SMA periods (e.g. 12,26)")
    p_plot.add_argument("--ema", type=str, default=None, help="EMA periods (e.g. 12,26)")
    p_plot.add_argument("--backend", type=str, choices=["mpl", "plotly"], default="mpl", help="Plotting backend")
    add_common_out_args(p_plot)
    p_plot.set_defaults(func=cmd_plot)


    p_fp = sub.add_parser("fetch-plot-crypto", help="Fetch crypto OHLC then plot")
    p_fp.add_argument("--symbol", type=str, default="BTCUSDT", help="symbol, e.g. BTCUSDT")
    p_fp.add_argument("--interval", type=str, default="1d", help="interval, e.g. 1d, 1h")
    p_fp.add_argument("--sma", type=str, default=(12, 26), help="SMA periods (e.g. 12,26)")
    p_fp.add_argument("--ema", type=str, default=(12, 26), help="EMA periods (e.g. 12,26)")
    p_fp.add_argument("--backend", type=str, choices=["mpl", "plotly"], default="mpl", help="Plotting backend")
    add_common_out_args(p_fp)
    p_fp.set_defaults(func=cmd_fetch_plot_crypto)


    p_fp_stock = sub.add_parser("fetch-plot-stock", help="Fetch stock OHLC then plot")
    p_fp_stock.add_argument("--symbol", type=str, required=True, help="stock ticker, e.g. AAPL")
    p_fp_stock.add_argument("--interval", type=str, default="1d", help="interval, e.g. 1d, 1h")
    p_fp_stock.add_argument("--period", type=str, default="1y", help="period when start/end not provided")
    p_fp_stock.add_argument("--start", type=str, default=None, help="start date YYYY-MM-DD")
    p_fp_stock.add_argument("--end", type=str, default=None, help="end date YYYY-MM-DD")
    p_fp_stock.add_argument("--sma", type=str, default=(12, 26), help="SMA periods (e.g. 12,26)")
    p_fp_stock.add_argument("--ema", type=str, default=(12, 26), help="EMA periods (e.g. 12,26)")
    p_fp_stock.add_argument("--backend", type=str, choices=["mpl", "plotly"], default="mpl", help="Plotting backend")
    add_common_out_args(p_fp_stock)
    p_fp_stock.set_defaults(func=cmd_fetch_plot_stock)

    p_fund = sub.add_parser("fundamentals", help="Fetch financial ratios and save CSV")
    p_fund.add_argument("--symbol", type=str, default=None, help="stock ticker, e.g. AAPL")
    p_fund.add_argument("--list", type=str, default=None, help="csv with column 'symbol'")
    add_common_out_args(p_fund)
    p_fund.set_defaults(func=cmd_fundamentals)

    return parser


def main(argv=None):
    """Entry point for the CLI when used as a module or script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
