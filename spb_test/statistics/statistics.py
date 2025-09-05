import numpy as np
import pandas as pd
import os


def compute_simple_returns(close_values):
    returns = close_values.astype(float).diff()
    return returns.dropna().reset_index(drop=True)


def autocorrelation(values, lag = 1):
    if lag <= 0 or lag >= len(values):
        return float("nan")

    x = values.iloc[lag:].to_numpy()
    y = values.iloc[:-lag].to_numpy()

    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")

    corr_matrix = np.corrcoef(x, y)
    return float(corr_matrix[0, 1])


def acf(values, max_lag):
    results = {}
    valid_max_lag = max(1, min(max_lag, len(values) - 1))
    for lag in range(1, valid_max_lag + 1):
        results[lag] = autocorrelation(values, lag)
    return results


def load_ohlc_csv(csv_path):
    df = pd.read_csv(csv_path)

    for name in ("open_time", "time"):
        if name in df.columns:
            try:
                df[name] = pd.to_datetime(df[name], errors="coerce")
                df = df.sort_values(name, kind="stable")
            except Exception:
                pass
            break
    return df.reset_index(drop=True)


def analyze_file(csv_path, max_lag):
    df = load_ohlc_csv(csv_path)
    returns = compute_simple_returns(df["close"])
    values = acf(returns, max_lag=max_lag)

    if "symbol" in df.columns and not df["symbol"].isna().all():
        try:
            label = str(df["symbol"].dropna().iloc[0])
        except Exception:
            label = os.path.splitext(os.path.basename(str(csv_path)))[0]
    else:
        label = os.path.splitext(os.path.basename(str(csv_path)))[0]

    return label, values


def analyze_file_single_lag(csv_path, lag):
    df = load_ohlc_csv(csv_path)
    returns = compute_simple_returns(df["close"])
    value = autocorrelation(returns, lag)

    if "symbol" in df.columns and not df["symbol"].isna().all():
        try:
            label = str(df["symbol"].dropna().iloc[0])
        except Exception:
            label = os.path.splitext(os.path.basename(str(csv_path)))[0]
    else:
        label = os.path.splitext(os.path.basename(str(csv_path)))[0]

    return label, value


def format_acf(acf_values):
    lines = ["lag,autocorr"]
    for lag, value in acf_values.items():
        if pd.notna(value):
            lines.append(f"{lag},{value:.6f}")
        else:
            lines.append(f"{lag},nan")
    return "\n".join(lines)
