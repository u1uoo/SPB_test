"""Plotly-based interactive plotting of OHLC data and indicators.

Provides a function `plot_indicators_plotly` that mirrors the Matplotlib version
"""

import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from spb_test.indicators.metrics import sma, ema, macd, rsi


def plot_indicators_plotly(df: pd.DataFrame, symbol: str, sma_periods=(20, 50), ema_periods=(20, 50), is_crypto: bool = False) -> None:
	"""Render OHLC with SMA/EMA, MACD, RSI using Plotly.

	Accepts a DataFrame with either a DatetimeIndex already set, or columns
	`open_time` / `time` which are converted to the index.
	"""
	if not isinstance(df.index, (pd.DatetimeIndex, pd.core.indexes.datetimes.DatetimeIndex)):
		if "open_time" in df.columns:
			df = df.copy()
			df["open_time"] = pd.to_datetime(df["open_time"], errors="coerce")
			df = df.set_index("open_time")
		elif "time" in df.columns:
			df = df.copy()
			df["time"] = pd.to_datetime(df["time"], errors="coerce")
			df = df.set_index("time")

	opens = df["open"].tolist()
	highs = df["high"].tolist()
	lows = df["low"].tolist()
	closes = df["close"].tolist()
	volumes = df["volume"].tolist()
	x = df.index

	sma_series = {p: sma(closes, p) for p in sma_periods}
	ema_series = {p: ema(closes, p) for p in ema_periods}
	macd_line, signal_line, hist = macd(closes)
	rsi_series = rsi(closes)

	fig = make_subplots(
		rows=3,
		cols=1,
		shared_xaxes=True,
		vertical_spacing=0.03,
		row_heights=[0.6, 0.22, 0.18],
		specs=[[{"secondary_y": True}], [ {"secondary_y": False}], [ {"secondary_y": False}]],
	)

	fig.add_trace(
		go.Candlestick(x=x, open=opens, high=highs, low=lows, close=closes, name="Price"),
		row=1, col=1, secondary_y=False,
	)
	for p in sma_periods:
		fig.add_trace(
			go.Scatter(x=x, y=sma_series[p], mode="lines", name=f"SMA {p}"),
			row=1, col=1, secondary_y=False,
		)
	for p in ema_periods:
		fig.add_trace(
			go.Scatter(x=x, y=ema_series[p], mode="lines", name=f"EMA {p}"),
			row=1, col=1, secondary_y=False,
		)

	vol_colors = ["green" if (c is not None and o is not None and not pd.isna(c) and not pd.isna(o) and c >= o) else "red" for o, c in zip(opens, closes)]
	fig.add_trace(
		go.Bar(x=x, y=[0 if pd.isna(v) else v for v in volumes], marker_color=vol_colors, opacity=0.3, name="Volume", showlegend=True),
		row=1, col=1, secondary_y=True,
	)
	fig.update_yaxes(showticklabels=False, row=1, col=1, secondary_y=True)

	fig.add_trace(go.Bar(x=x, y=[0 if h is None else h for h in hist], name="Hist", marker_color="gray"), row=2, col=1)
	fig.add_trace(go.Scatter(x=x, y=macd_line, mode="lines", name="MACD"), row=2, col=1)
	fig.add_trace(go.Scatter(x=x, y=signal_line, mode="lines", name="Signal"), row=2, col=1)

	fig.add_trace(go.Scatter(x=x, y=rsi_series, mode="lines", name="RSI"), row=3, col=1)
	fig.add_hline(y=30, line_color="red", line_width=1, line_dash="dash", row=3, col=1)
	fig.add_hline(y=70, line_color="green", line_width=1, line_dash="dash", row=3, col=1)
	fig.update_yaxes(range=[0, 100], row=3, col=1)

	def _compute_profile_shapes(x0=None, x1=None, bins=40, target_frac=0.15):
		"""Compute rectangles/lines representing Volume Profile + Value Area.
		"""
		if x0 is not None and x1 is not None:
			mask = (x >= pd.to_datetime(x0)) & (x <= pd.to_datetime(x1))
			prices = np.array(df.loc[mask, "close"], dtype=float)
			vols = np.array([0 if pd.isna(v) else v for v in df.loc[mask, "volume"]], dtype=float)
		else:
			prices = np.array(closes, dtype=float)
			vols = np.array([0 if pd.isna(v) else v for v in volumes], dtype=float)
		if prices.size == 0 or vols.size == 0 or not np.isfinite(prices).any():
			return []
		pmin = np.nanmin(prices)
		pmax = np.nanmax(prices)
		if not np.isfinite(pmin) or not np.isfinite(pmax) or pmax <= pmin:
			return []
		edges = np.linspace(pmin, pmax, bins + 1)
		hist, edges = np.histogram(prices, bins=edges, weights=vols)
		max_bin = float(np.max(hist)) if np.any(hist) else 0.0
		if max_bin <= 0:
			return []
		shapes = []

		# compute Value Area (70%) boundaries using POC expansion
		def _compute_value_area_bounds(hist_arr, edges_arr, coverage=0.7):
			"""POC expansion algorithm to get VAL/VAH for a given coverage."""
			total = float(np.sum(hist_arr))
			if total <= 0:
				return None
			poc = int(np.argmax(hist_arr))
			low = poc
			high = poc
			covered = float(hist_arr[poc])
			left = poc - 1
			right = poc + 1
			while covered / total < coverage:
				left_val = hist_arr[left] if left >= 0 else -1.0
				right_val = hist_arr[right] if right < hist_arr.shape[0] else -1.0
				if left_val < 0 and right_val < 0:
					break
				if right_val > left_val:
					covered += right_val
					high = right
					right += 1
				else:
					covered += left_val
					low = left
					left -= 1
			return float(edges_arr[low]), float(edges_arr[high + 1])

		va_bounds = _compute_value_area_bounds(hist, edges, coverage=0.7)
		for v, y0, y1 in zip(hist, edges[:-1], edges[1:]):
			if v <= 0:
				continue
			width_frac = (v / max_bin) * target_frac
			shapes.append(dict(type="rect", xref="x domain", yref="y", x0=1.0 - width_frac, x1=1.0, y0=float(y0), y1=float(y1), line=dict(width=0), fillcolor="rgba(31,119,180,0.22)", layer="below"))

		# add horizontal lines for VAL/VAH
		if va_bounds is not None:
			val, vah = va_bounds
			for y in (val, vah):
				shapes.append(dict(type="line", xref="x domain", yref="y", x0=1.0 - target_frac, x1=1.0, y0=float(y), y1=float(y), line=dict(width=1.2, dash="dash", color="rgba(31,119,180,0.9)")))
		return shapes

	fig.update_layout(shapes=tuple(_compute_profile_shapes()))

	fig.update_layout(
		title=symbol,
		height=900,
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
		margin=dict(l=60, r=20, t=60, b=40),
		xaxis_rangeslider_visible=False,
	)

	if not is_crypto:
		for r in (1, 2, 3):
			fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], row=r, col=1)

	try:
		figw = go.FigureWidget(fig)
		def _on_relayout(data):
			x0 = data.get("xaxis.range[0]")
			x1 = data.get("xaxis.range[1]")
			if x0 is None or x1 is None:
				return
			figw.layout.shapes = tuple(_compute_profile_shapes(x0, x1))
		figw.on_relayout(_on_relayout)
		figw
	except Exception:
		fig.show()
