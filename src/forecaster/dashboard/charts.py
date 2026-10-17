"""Plotly figure builders for the dashboard.

Palette: the validated default categorical slots (light / dark steps), the blue<->red diverging
pair for up/down candles, chrome tokens for ink/grid. Every chart uses one y-axis per panel
(stacked panels share x, never two y-scales on one plot). Light-mode slots 3-4 are below 3:1
contrast, so every chart ships with a legend and a data-table view in the app.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass(frozen=True)
class Theme:
    s1: str   # blue   - primary series (price, model, strategy)
    s2: str   # orange - comparison series (benchmark, SMA 20)
    s3: str   # aqua   - SMA 50
    s4: str   # yellow - SMA 200
    up: str   # diverging pole: up
    down: str  # diverging pole: down
    band: str  # light blue fill for forecast / Bollinger ranges
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str


LIGHT = Theme(s1="#2a78d6", s2="#eb6834", s3="#1baf7a", s4="#eda100", up="#2a78d6", down="#e34948",
              band="rgba(42,120,214,0.16)", ink="#0b0b0b", ink2="#52514e", muted="#898781",
              grid="#e1e0d9", axis="#c3c2b7")
DARK = Theme(s1="#3987e5", s2="#d95926", s3="#199e70", s4="#c98500", up="#3987e5", down="#e66767",
             band="rgba(57,135,229,0.22)", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
             grid="#2c2c2a", axis="#383835")
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


# Status palette (reserved: colour MEANS good/bad here, always paired with a text label).
STATUS_GOOD, STATUS_CRITICAL = "#0ca30c", "#d03b3b"


def theme(dark: bool) -> Theme:
    return DARK if dark else LIGHT


def _style(fig: go.Figure, t: Theme, height: int, hovermode: str = "x unified") -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36, b=8), hovermode=hovermode,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=t.ink2, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=t.ink2)),
        hoverlabel=dict(font=dict(family=FONT)),
    )
    fig.update_xaxes(showgrid=False, linecolor=t.axis, tickfont=dict(color=t.muted), zeroline=False)
    fig.update_yaxes(gridcolor=t.grid, gridwidth=1, linecolor=t.axis, tickfont=dict(color=t.muted), zeroline=False)
    return fig


def candlestick(df: pd.DataFrame, symbol: str, dark: bool = False) -> go.Figure:
    """Price (candles, SMAs, Bollinger band) / volume / RSI as three stacked panels."""
    t = theme(dark)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.62, 0.16, 0.22])
    x = df.index
    fig.add_trace(go.Scatter(x=x, y=df["bb_upper"], line=dict(width=0), hoverinfo="skip",
                             showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=df["bb_lower"], line=dict(width=0), fill="tonexty", fillcolor=t.band,
                             name="Bollinger (20, 2σ)", hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Candlestick(
        x=x, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name=symbol,
        increasing=dict(line=dict(color=t.up, width=1), fillcolor=t.up),
        decreasing=dict(line=dict(color=t.down, width=1), fillcolor=t.down)), 1, 1)
    for col, color, label in (("sma_20", t.s2, "SMA 20"), ("sma_50", t.s3, "SMA 50"), ("sma_200", t.s4, "SMA 200")):
        fig.add_trace(go.Scatter(x=x, y=df[col], name=label, line=dict(color=color, width=2)), 1, 1)
    up = df["close"] >= df["open"]
    fig.add_trace(go.Bar(x=x, y=df["volume"], name="Volume", marker_color=np.where(up, t.up, t.down),
                         marker_line_width=0, opacity=0.6, showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=df["rsi_14"], name="RSI 14", line=dict(color=t.s1, width=2),
                             showlegend=False), 3, 1)
    for level in (30, 70):
        fig.add_hline(y=level, line=dict(color=t.axis, width=1), row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], tickvals=[30, 50, 70], row=3, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, title=dict(text=f"{symbol} daily", font=dict(color=t.ink)))
    return _style(fig, t, 720)


def forecast_fan(history: pd.Series, forecasts: list[dict], dark: bool = False, lookback: int = 120) -> go.Figure:
    """Recent closes, then for each horizon the 80% range and median as a projection."""
    t = theme(dark)
    fig = go.Figure()
    h = history.iloc[-lookback:]
    fig.add_trace(go.Scatter(x=h.index, y=h.values, name="Close", line=dict(color=t.s1, width=2),
                             hovertemplate="%{y:,.4f}<extra>Close</extra>"))
    if forecasts:
        f0 = forecasts[0]
        start = pd.Timestamp(f0["as_of_time"])
        ref = f0["reference_price"]
        xs = [start] + [pd.Timestamp(f["target_time"]) for f in forecasts]
        lo = [ref] + [f["price_q10"] for f in forecasts]
        hi = [ref] + [f["price_q90"] for f in forecasts]
        md = [ref] + [f["price_q50"] for f in forecasts]
        fig.add_trace(go.Scatter(x=xs, y=hi, line=dict(width=0), showlegend=False,
                                 hovertemplate="%{y:,.4f}<extra>90th pct</extra>"))
        fig.add_trace(go.Scatter(x=xs, y=lo, line=dict(width=0), fill="tonexty", fillcolor=t.band,
                                 name="80% range (estimate)", hovertemplate="%{y:,.4f}<extra>10th pct</extra>"))
        # Dashed = projection, which is exactly what this line is.
        fig.add_trace(go.Scatter(x=xs, y=md, name="Median (estimate)", mode="lines+markers",
                                 line=dict(color=t.s1, width=2, dash="dash"), marker=dict(size=8),
                                 hovertemplate="%{y:,.4f}<extra>Median</extra>"))
        for f in forecasts:
            fig.add_annotation(x=pd.Timestamp(f["target_time"]), y=f["price_q90"], yshift=12, showarrow=False,
                               text=f"+{f['horizon_days']}d · P(up) {f['prob_up']:.0%}", font=dict(color=t.ink2))
    fig.update_yaxes(title_text="Price")
    fig.update_layout(title=dict(text="Forecast range (not financial advice)", font=dict(color=t.ink)))
    return _style(fig, t, 420)


def reliability_chart(table: pd.DataFrame, dark: bool = False) -> go.Figure:
    """Calibration: observed up-frequency vs predicted probability. On the diagonal = calibrated."""
    t = theme(dark)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                             line=dict(color=t.axis, width=1), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=table["mean_pred"], y=table["frac_up"], mode="lines+markers", name="Model",
        line=dict(color=t.s1, width=2), marker=dict(size=9, color=t.s1, line=dict(width=2, color="rgba(0,0,0,0)")),
        customdata=table["count"],
        hovertemplate="predicted %{x:.0%}<br>observed %{y:.0%}<br>n = %{customdata}<extra></extra>"))
    fig.update_xaxes(title_text="Predicted P(up)", range=[0, 1], tickformat=".0%")
    fig.update_yaxes(title_text="Observed share up", range=[0, 1], tickformat=".0%")
    fig.update_layout(title=dict(text="Calibration (out-of-sample)", font=dict(color=t.ink)))
    return _style(fig, t, 380, hovermode="closest")


def coverage_chart(coverage: pd.Series, dark: bool = False) -> go.Figure:
    t = theme(dark)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=coverage.index, y=coverage.values, name="Rolling 90-day coverage",
                             line=dict(color=t.s1, width=2), hovertemplate="%{y:.0%}<extra>coverage</extra>"))
    fig.add_hline(y=0.8, line=dict(color=t.axis, width=1), annotation_text="nominal 80%",
                  annotation_font_color=t.muted)
    fig.update_yaxes(title_text="Share inside 80% range", range=[0, 1], tickformat=".0%")
    fig.update_layout(title=dict(text="Price-range coverage over time", font=dict(color=t.ink)), showlegend=False)
    return _style(fig, t, 320)


def equity_chart(curve: pd.DataFrame, title: str, dark: bool = False) -> go.Figure:
    """Strategy vs buy-and-hold equity (top) and their drawdowns (bottom): two panels, one axis each."""
    t = theme(dark)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.7, 0.3])
    for col, color, label in (("strategy", t.s1, "Strategy (after costs)"), ("benchmark", t.s2, "Buy & hold")):
        eq = curve[col]
        dd = eq / eq.cummax() - 1
        fig.add_trace(go.Scatter(x=curve.index, y=eq, name=label, line=dict(color=color, width=2),
                                 hovertemplate="%{y:.3f}×<extra>" + label + "</extra>"), 1, 1)
        fig.add_trace(go.Scatter(x=curve.index, y=dd, name=label, showlegend=False,
                                 line=dict(color=color, width=2),
                                 hovertemplate="%{y:.1%}<extra>" + label + " drawdown</extra>"), 2, 1)
    fig.update_yaxes(title_text="Growth of 1", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_layout(title=dict(text=title, font=dict(color=t.ink)))
    return _style(fig, t, 460)


def signal_chart(closes: pd.Series, sig, dark: bool = False, lookback: int = 60) -> go.Figure:
    """Recent closes, then the plan's levels drawn across the signal's validity window."""
    t = theme(dark)
    fig = go.Figure()
    h = closes.iloc[-lookback:]
    fig.add_trace(go.Scatter(x=h.index, y=h.values, name="Close", line=dict(color=t.s1, width=2),
                             hovertemplate="%{y:,.4f}<extra>Close</extra>"))
    start, end = pd.Timestamp(sig.as_of_time), pd.Timestamp(sig.valid_until)
    fig.add_vrect(x0=start, x1=end, fillcolor=t.band, line_width=0, layer="below",
                  annotation_text=f"valid {sig.horizon_days}d", annotation_position="top left",
                  annotation_font_color=t.muted)
    if sig.entry is not None:
        fig.add_hrect(y0=sig.entry_low, y1=sig.entry_high, x0=0.0, x1=1.0, fillcolor=t.band, line_width=0,
                      layer="below")
        for name, level, color, dash in (("TP2", sig.tp2, STATUS_GOOD, "dot"), ("TP1", sig.tp1, STATUS_GOOD, "solid"),
                                         ("Entry", sig.entry, t.ink2, "solid"),
                                         ("Stop", sig.stop_loss, STATUS_CRITICAL, "solid")):
            pct = level / sig.reference_price - 1
            fig.add_trace(go.Scatter(
                x=[start, end], y=[level, level], mode="lines", name=name, showlegend=True,
                line=dict(color=color, width=2, dash=dash),
                hovertemplate=f"{name} %{{y:,.4f}} ({pct:+.1%})<extra></extra>"))
            fig.add_annotation(x=end, y=level, xanchor="left", showarrow=False, font=dict(color=t.ink2),
                               text=f" {name} {level:,.4f} ({pct:+.1%})")
    side = {"LONG": "▲ LONG", "SHORT": "▼ SHORT"}.get(sig.side, "■ NO TRADE")
    fig.update_yaxes(title_text="Price")
    fig.update_layout(title=dict(text=f"{side} · {sig.horizon_days}d plan (estimate, not advice)",
                                 font=dict(color=t.ink)), margin=dict(r=150))
    return _style(fig, t, 420)


def r_curve(history: pd.DataFrame, dark: bool = False) -> go.Figure:
    """Cumulative R of historical out-of-sample signals, one point per signal."""
    t = theme(dark)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["as_of_time"], y=history["r"].cumsum(), name="Cumulative R",
                             line=dict(color=t.s1, width=2),
                             customdata=history[["side", "outcome", "r"]].to_numpy(),
                             hovertemplate="%{customdata[0]} → %{customdata[1]} (%{customdata[2]:+.2f}R)"
                                           "<br>cumulative %{y:+.1f}R<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=t.axis, width=1))
    fig.update_yaxes(title_text="Cumulative R (after costs)")
    fig.update_layout(title=dict(text="Historical signals with these rules (out-of-sample)",
                                 font=dict(color=t.ink)), showlegend=False)
    return _style(fig, t, 320)
