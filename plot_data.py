import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.widgets import CheckButtons

from metrics import sma, ema, macd, rsi


def load_csv(path):
    df = pd.read_csv(path)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df.set_index("open_time")
    return df


def plot_indicators(df, symbol, sma_periods=(20, 50), ema_periods=(20, 50)):
    opens = df["open"].tolist()
    highs = df["high"].tolist()
    lows = df["low"].tolist()
    closes = df["close"].tolist()

    sma_series = {p: sma(closes, p) for p in sma_periods}
    ema_series = {p: ema(closes, p) for p in ema_periods}
    macd_line, signal_line, hist = macd(closes)
    rsi_series = rsi(closes)

    x_dt = df.index
    x = mdates.date2num(x_dt.to_pydatetime())

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.subplots_adjust(right=0.82)

    if len(x) > 1:
        dx = np.median(np.diff(x))
    else:
        dx = 1.0
    width = dx * 0.6

    for xi, o, h, l, c in zip(x, opens, highs, lows, closes):
        if pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(c):
            continue
        color = "green" if c >= o else "red"
        ax1.vlines(xi, l, h, color=color, linewidth=1)
        lower = min(o, c)
        height = abs(c - o)
        if height == 0:
            height = max(1e-9, 0.0)
        ax1.add_patch(
            plt.Rectangle(
                (xi - width / 2.0, lower),
                width,
                height,
                facecolor=color,
                edgecolor=color,
                linewidth=1,
            )
        )
    sma_lines = {}
    for p in sma_periods:
        line = ax1.plot(x, sma_series[p], label=f"SMA {p}")[0]
        sma_lines[p] = line
    ema_lines = {}
    for p in ema_periods:
        line = ax1.plot(x, ema_series[p], label=f"EMA {p}")[0]
        ema_lines[p] = line
    ax1.set_title(f"{symbol}")
    ax1.legend(loc="upper left")

    macd_line_artist = ax2.plot(x, macd_line, label="MACD", color="blue")[0]
    signal_line_artist = ax2.plot(x, signal_line, label="Signal", color="orange")[0]
    hist_bars = ax2.bar(
        x, [h if h is not None else 0 for h in hist], label="Hist", color="gray"
    )
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_title("MACD")
    ax2.legend(loc="upper left")

    rsi_line_artist = ax3.plot(x, rsi_series, label="RSI", color="purple")[0]
    ax3.axhline(30, color="red", linewidth=0.8, linestyle="--")
    ax3.axhline(70, color="green", linewidth=0.8, linestyle="--")
    ax3.set_ylim(0, 100)
    ax3.set_title("RSI")
    ax3.legend(loc="upper left")

    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)

    for ax in [ax1, ax2, ax3]:
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    fig.autofmt_xdate()

    state = {"press": None, "xlim": None, "ylim": None, "ax": None}

    def zoom_1d(center, lim, scale):
        span = lim[1] - lim[0]
        new_span = span / scale
        left = center - (center - lim[0]) / scale
        right = left + new_span
        return left, right

    def on_scroll(event):
        ax = event.inaxes
        if ax is None:
            return
        scale = 1.2 if event.button == "up" else (1.0 / 1.2)

        if event.xdata is not None:
            xlim = ax1.get_xlim()
            new_xlim = zoom_1d(event.xdata, xlim, scale)
            ax1.set_xlim(new_xlim)

        if event.key == "shift" and event.ydata is not None:
            ylim = ax.get_ylim()
            new_ylim = zoom_1d(event.ydata, ylim, scale)
            ax.set_ylim(new_ylim)
        fig.canvas.draw_idle()

    def on_press(event):
        if event.button != 1 or event.inaxes is None:
            return
        state["press"] = (event.xdata, event.ydata)
        state["xlim"] = ax1.get_xlim()
        state["ylim"] = event.inaxes.get_ylim()
        state["ax"] = event.inaxes

    def on_motion(event):
        if state["press"] is None or event.inaxes is None or event.xdata is None:
            return
        x0, y0 = state["press"]

        dx = event.xdata - x0
        xlim0 = state["xlim"]
        ax1.set_xlim(xlim0[0] - dx, xlim0[1] - dx)

        if event.key == "shift" and event.ydata is not None:
            dy = event.ydata - y0
            ylim0 = state["ylim"]
            state["ax"].set_ylim(ylim0[0] - dy, ylim0[1] - dy)
        fig.canvas.draw_idle()

    def on_release(event):
        state["press"] = None
        state["xlim"] = None
        state["ylim"] = None
        state["ax"] = None

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    labels = []
    artists = []
    for p, line in sma_lines.items():
        labels.append(f"SMA {p}")
        artists.append(line)
    for p, line in ema_lines.items():
        labels.append(f"EMA {p}")
        artists.append(line)
    labels += ["MACD", "Signal", "Hist", "RSI"]
    artists += [macd_line_artist, signal_line_artist, hist_bars, rsi_line_artist]

    check_ax = fig.add_axes([0.84, 0.15, 0.14, 0.3])
    checks = CheckButtons(check_ax, labels, [True] * len(labels))

    label_to_artist = {lab: art for lab, art in zip(labels, artists)}

    def on_check(label):
        art = label_to_artist[label]
        if hasattr(art, "get_visible"):
            art.set_visible(not art.get_visible())
        else:

            vis = not art.patches[0].get_visible() if len(art.patches) else True
            for patch in art.patches:
                patch.set_visible(vis)
        fig.canvas.draw_idle()

    checks.on_clicked(on_check)

    plt.tight_layout()
    plt.show()


def main():
    path = "ohlc_BTCUSDT.csv"
    df = load_csv(path)
    if df.empty:
        print("no data")
        return
    symbol = df["symbol"].iloc[0] if "symbol" in df.columns else "SYMBOL"
    plot_indicators(df, symbol)


if __name__ == "__main__":
    main()
