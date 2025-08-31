def sma(values, period):
    result = []
    running_sum = 0.0
    for i, value in enumerate(values):
        running_sum += value
        if i >= period:
            running_sum -= values[i - period]
        if i >= period - 1:
            result.append(running_sum / period)
        else:
            result.append(None)
    return result


def ema(values, period):
    result = []
    multiplier = 2.0 / (period + 1)
    running_sum = 0.0
    prev = None
    for i, value in enumerate(values):
        running_sum += value
        if i < period - 1:
            result.append(None)
            continue
        if i == period - 1:
            prev = running_sum / period
            result.append(prev)
            continue
        current = (value - prev) * multiplier + prev
        prev = current
        result.append(current)
    return result


def macd(values, fast=12, slow=26, signal=9):
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)

    macd_line = []
    for i in range(len(values)):
        a = ema_fast[i]
        b = ema_slow[i]
        if a is None or b is None:
            macd_line.append(None)
        else:
            macd_line.append(a - b)

    first_idx = None
    for i, v in enumerate(macd_line):
        if v is not None:
            first_idx = i
            break

    if first_idx is None:
        signal_line = [None] * len(values)
    else:
        macd_slice = macd_line[first_idx:]
        signal_slice = ema(macd_slice, signal)
        signal_line = [None] * first_idx + signal_slice

    histogram = []
    for i in range(len(values)):
        m = macd_line[i]
        s = signal_line[i]
        if m is None or s is None:
            histogram.append(None)
        else:
            histogram.append(m - s)

    return macd_line, signal_line, histogram


def rsi(values, period=14):
    result = []
    if not values:
        return result

    result.append(None)
    prev = values[0]
    gains = []
    losses = []

    avg_gain = None
    avg_loss = None

    for i in range(1, len(values)):
        change = values[i] - prev
        prev = values[i]

        gain = change if change > 0 else 0.0
        loss = -change if change < 0 else 0.0
        gains.append(gain)
        losses.append(loss)

        if i < period:
            result.append(None)
            continue

        if i == period:
            avg_gain = sum(gains[:period]) / period
            avg_loss = sum(losses[:period]) / period
        else:
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

        if avg_loss == 0:
            rsi_value = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_value = 100.0 - (100.0 / (1.0 + rs))

        result.append(rsi_value)

    return result
