"""Streamlit dashboard. Run with `forecaster dashboard` (or `streamlit run` on this file).

Reads the database directly; read-only apart from storing the live forecasts it displays.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from forecaster import DISCLAIMER
from forecaster.config import get_settings
from forecaster.dashboard import charts
from forecaster.dashboard import data as dd
from forecaster.db.session import get_sessionmaker
from forecaster.signals import (
    LONG,
    SHORT,
    atr_by_close_time,
    make_signal,
    rules_from_settings,
    signal_history,
    summarize_history,
)

st.set_page_config(page_title="Crypto Forecaster", page_icon="📈", layout="wide")
s = get_settings()
# Static cloud deployment over a committed snapshot (deploy/streamlit_app.py): never write to it.
SNAPSHOT = bool(os.environ.get("FORECASTER_SNAPSHOT"))


@st.cache_resource
def _factory():
    return get_sessionmaker()


def _session():
    return _factory()()


@st.cache_data(ttl=300, show_spinner=False)
def load_candles(symbol: str, days: int) -> pd.DataFrame:
    with _session() as session:
        return dd.candles_with_indicators(session, symbol, days)


@st.cache_data(ttl=300, show_spinner=False)
def load_daily(symbol: str) -> pd.DataFrame:
    """Full daily history (ATR and signal replay need more than the chart window)."""
    with _session() as session:
        return dd.candles_with_indicators(session, symbol, days=100_000)


@st.cache_data(ttl=300, show_spinner="Computing forecast…")
def load_forecasts(symbol: str, model: str):
    with _session() as session:
        return dd.current_forecasts(session, s, symbol, model, store=not SNAPSHOT)


@st.cache_data(ttl=300, show_spinner=False)
def load_models(symbol: str) -> pd.DataFrame:
    with _session() as session:
        return dd.model_table(session, symbol)


@st.cache_data(ttl=300, show_spinner=False)
def load_oos(symbol: str, horizon: int, model: str) -> pd.DataFrame:
    with _session() as session:
        return dd.oos_predictions(session, symbol, horizon, model)


@st.cache_data(ttl=300, show_spinner=False)
def load_backtests(symbol: str) -> list[dict]:
    with _session() as session:
        out = dd.backtests(session, symbol)
    # ORM rows are not cache-serialisable; keep plain fields only.
    for b in out:
        r = b.pop("row")
        b.update(strategy=r.strategy, horizon=r.horizon_days, period_start=r.period_start,
                 period_end=r.period_end, fee_bps=r.fee_bps, slippage_bps=r.slippage_bps,
                 beats=r.beats_benchmark)
    return out


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def status(verdict: str, good: bool, bad: bool) -> None:
    """Status colours never stand alone: Streamlit pairs them with an icon and the text label."""
    if good:
        st.success(verdict, icon="✅")
    elif bad:
        st.error(verdict, icon="❌")
    else:
        st.warning(verdict, icon="⚠️")


def pct(v, digits: int = 1) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:.{digits}%}"


def num(v, fmt: str = ".2f") -> str:
    return "n/a" if v is None or pd.isna(v) else format(v, fmt)


# --- header and filters -----------------------------------------------------------------------
st.title("Crypto Forecaster")
st.warning(f"**Estimates only.** {DISCLAIMER} This tool is read-only and never places orders.", icon="⚠️")

f1, f2, f3, f4 = st.columns([2, 2, 1, 1])
symbol = f1.selectbox("Symbol", s.target_symbols)
model = f2.selectbox("Model", s.train_models, index=s.train_models.index(s.serving_model)
                     if s.serving_model in s.train_models else 0)
horizon = f3.selectbox("Horizon (performance / backtest)", s.forecast_horizons)
days = f4.selectbox("Chart range", [90, 180, 365, 730], index=2, format_func=lambda d: f"{d} days")
dark = is_dark()

try:
    candles = load_candles(symbol, days)
except Exception as exc:  # database not configured / reachable
    st.error(f"Cannot read market data: {exc}. Check DB_* settings and run `forecaster db init`.")
    st.stop()
if candles.empty:
    st.info(f"No candles stored for {symbol}. Run `forecaster data backfill`.")
    st.stop()
if SNAPSHOT:
    last_close = candles.index.max() + pd.Timedelta(days=1)
    st.info(f"Static snapshot: market data and models as of **{last_close:%Y-%m-%d %H:%M} UTC**. "
            "It is refreshed manually, so forecasts are not live.", icon="🗂️")

tab_fc, tab_sig, tab_mkt, tab_perf, tab_bt = st.tabs(
    ["Forecast", "Signal", "Market", "Model performance", "Backtest"])

# --- forecast ---------------------------------------------------------------------------------
with tab_fc:
    forecasts, reason = load_forecasts(symbol, model)
    if not forecasts:
        st.info(f"No forecast available: {reason}")
    else:
        if any(f["stale"] for f in forecasts):
            hint = "the owner refreshes this snapshot periodically" if SNAPSHOT else "run `forecaster data update`"
            st.warning(f"Market data is more than 2 days old; {hint}.", icon="⏱️")
        cols = st.columns(len(forecasts))
        for col, f in zip(cols, forecasts, strict=True):
            with col:
                st.metric(f"P(up) in {f['horizon_days']} days", f"{f['prob_up']:.0%}")
                st.markdown(
                    f"**80% range:** {f['price_q10']:,.4f} – {f['price_q90']:,.4f}  \n"
                    f"Median {f['price_q50']:,.4f} · from {f['reference_price']:,.4f} "
                    f"on {f['as_of_time']:%Y-%m-%d}  \n"
                    f"Backtest coverage of this range: {pct(f['interval_coverage_backtest'], 0)} (nominal 80%)  \n"
                    f"<span style='opacity:.7'>{f['model']} {f['model_version']}</span>",
                    unsafe_allow_html=True)
        closes = candles["close"].copy()
        closes.index = closes.index + pd.Timedelta(days=1)  # plot at close time, like the forecast
        st.plotly_chart(charts.forecast_fan(closes, forecasts, dark), width="stretch")
        with st.expander("Forecast data table"):
            st.dataframe(pd.DataFrame(forecasts).drop(columns=["disclaimer"]), width="stretch")
        st.caption("A probability near 50% means the model has no meaningful directional view.")

# --- signal -----------------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner="Replaying historical signals…")
def load_signal_history(symbol: str, horizon: int, model: str, target: str) -> pd.DataFrame:
    return signal_history(load_oos(symbol, horizon, model), load_daily(symbol), horizon,
                          rules_from_settings(s), target)


def fmt_level(v: float, ref: float) -> str:
    return f"{v:,.4f} ({v / ref - 1:+.1%})"


with tab_sig:
    st.warning("**Research output, not financial advice.** These levels are generated mechanically from the "
               "model's forecast range. Out-of-sample, the models have not shown a statistically significant "
               "edge (see *Model performance* and *Backtest*). Do not risk money you cannot afford to lose.",
               icon="⚠️")
    forecasts, reason = load_forecasts(symbol, model)
    if not forecasts:
        st.info(f"No signal available: {reason}")
    else:
        rules = rules_from_settings(s)
        daily = load_daily(symbol)
        atr_now = atr_by_close_time(daily)
        verdicts = load_models(symbol)
        sigs = []
        for col, f in zip(st.columns(len(forecasts)), forecasts, strict=True):
            sig = make_signal(f, float(atr_now.get(pd.Timestamp(f["as_of_time"]), float("nan"))), rules)
            sigs.append(sig)
            with col, st.container(border=True):
                icon = {LONG: "▲", SHORT: "▼"}.get(sig.side, "■")
                st.markdown(f"### {icon} {sig.side} · {sig.horizon_days} days")
                st.caption(sig.reason)
                if sig.entry is not None:
                    ref = sig.reference_price
                    st.dataframe(pd.DataFrame([
                        {"level": "Entry (last close)", "price": f"{sig.entry:,.4f}"},
                        {"level": "Entry zone", "price": f"{sig.entry_low:,.4f} – {sig.entry_high:,.4f}"},
                        {"level": "Stop loss", "price": fmt_level(sig.stop_loss, ref)},
                        {"level": "TP1 (median)", "price": f"{fmt_level(sig.tp1, ref)} · R:R {sig.rr_tp1:.2f}"},
                        {"level": "TP2 (80% edge)", "price": f"{fmt_level(sig.tp2, ref)} · R:R {sig.rr_tp2:.2f}"},
                    ]), hide_index=True, width="stretch")
                st.markdown(f"**P(up)** {sig.prob_up:.0%} · strength **{sig.strength}** · "
                            f"ATR(14) {sig.atr:,.4f}  \n**Valid until** {sig.valid_until:%Y-%m-%d %H:%M} UTC")
                v = verdicts[(verdicts["model"] == model) & (verdicts["horizon"] == sig.horizon_days)]
                if not v.empty:
                    st.caption(f"Model walk-forward verdict: {v.iloc[0]['verdict']}")
                st.markdown("**Invalidation**\n" + "\n".join(f"- {x}" for x in sig.invalidation))

        sel = next((x for x in sigs if x.horizon_days == horizon), sigs[0])
        closes = daily["close"].copy()
        closes.index = closes.index + pd.Timedelta(days=1)
        st.plotly_chart(charts.signal_chart(closes, sel, dark), width="stretch", key="signal-chart")

        st.subheader(f"How these rules did historically · {sel.horizon_days}d · {model}")
        st.caption("Same rules replayed on walk-forward out-of-sample predictions: enter at the close, manage "
                   f"over the next {sel.horizon_days} daily candles, stop assumed first if a candle touches both, "
                   f"{rules.cost_bps:g} bps per side costs. Signals overlap in time, so they are not independent.")
        hist = {t: load_signal_history(symbol, sel.horizon_days, model, t) for t in ("tp1", "tp2")}
        if hist["tp1"].empty:
            st.info("No historical signals with these rules for this model and horizon.")
        else:
            table = pd.DataFrame([{"target": t.upper(), **summarize_history(h)} for t, h in hist.items()])
            st.dataframe(table.style.format({"win_rate": "{:.0%}", "target_hit": "{:.0%}", "stopped": "{:.0%}",
                                             "expired": "{:.0%}", "avg_r": "{:+.2f}", "total_r": "{:+.1f}"}),
                         hide_index=True, width="stretch")
            best = max(table["avg_r"])
            if best <= 0:
                st.error("Historically these signals lost money on average after costs (average R ≤ 0).", icon="❌")
            else:
                st.warning("Positive average R historically, but signals overlap and this is not a significance "
                           "test; see the Backtest tab's bootstrap p-value.", icon="⚠️")
            st.plotly_chart(charts.r_curve(hist["tp1"], dark), width="stretch", key="signal-r-curve")
            with st.expander("Historical signal table (TP1)"):
                st.dataframe(hist["tp1"].iloc[::-1], width="stretch", hide_index=True)

# --- market -----------------------------------------------------------------------------------
with tab_mkt:
    st.plotly_chart(charts.candlestick(candles, symbol, dark), width="stretch")
    with st.expander("Candle data table"):
        st.dataframe(candles.iloc[::-1], width="stretch")

# --- model performance ------------------------------------------------------------------------
with tab_perf:
    table = load_models(symbol)
    if table.empty:
        st.info("No trained models yet. Run `forecaster train`.")
    else:
        st.subheader("Walk-forward results (out-of-sample)")
        st.caption("Brier skill > 0 means better probabilities than the climatology baseline. "
                   "Coverage should be close to 80%.")
        st.dataframe(table.drop(columns=["run_id"]).style.format(
            {"accuracy": "{:.3f}", "roc_auc": "{:.3f}", "brier": "{:.4f}", "brier_skill": "{:+.3f}",
             "mae_q50": "{:.4f}", "coverage_80": "{:.0%}", "always_up_acc": "{:.3f}"}, na_rep="n/a"),
            width="stretch", hide_index=True)
        sel = table[(table["model"] == model) & (table["horizon"] == horizon)]
        if not sel.empty:
            v = sel.iloc[0]["verdict"]
            status(f"{model}, {horizon}d: {v}", good=v == "beats baseline",
                   bad=v.startswith("does NOT"))
        oos = load_oos(symbol, horizon, model)
        if oos.empty:
            st.info(f"No out-of-sample predictions for {model} at {horizon}d.")
        else:
            rel = dd.reliability(oos)
            cov = dd.rolling_coverage(oos)
            c1, c2 = st.columns(2)
            c1.plotly_chart(charts.reliability_chart(rel, dark), width="stretch")
            c2.plotly_chart(charts.coverage_chart(cov.dropna(), dark), width="stretch")
            with st.expander("Calibration and coverage data tables"):
                st.dataframe(rel, width="stretch", hide_index=True)
                st.dataframe(oos.iloc[::-1], width="stretch")

# --- backtest ---------------------------------------------------------------------------------
with tab_bt:
    results = [b for b in load_backtests(symbol) if b["horizon"] == horizon]
    if not results:
        st.info(f"No backtests for {symbol} at {horizon}d. Run `forecaster backtest`.")
    for b in results:
        st.subheader(f"{b['strategy']} · {b['horizon']}d signals")
        sig = b["beats"] and b["p_value"] is not None and b["p_value"] < 0.05
        status(b["verdict"], good=sig, bad=not b["beats"])
        ss, bs = b["strategy_stats"], b["benchmark_stats"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total return", pct(ss.get("total_return")),
                  f"B&H {pct(bs.get('total_return'))}", delta_color="off")
        m2.metric("Sharpe", num(ss.get("sharpe")), f"B&H {num(bs.get('sharpe'))}", delta_color="off")
        m3.metric("Max drawdown", pct(ss.get("max_drawdown")),
                  f"B&H {pct(bs.get('max_drawdown'))}", delta_color="off")
        m4.metric("Trades · win rate", f"{ss.get('n_trades', 0)} · {pct(ss.get('win_rate'), 0)}",
                  f"exposure {pct(ss.get('exposure'), 0)}", delta_color="off")
        st.caption(f"{b['period_start']:%Y-%m-%d} → {b['period_end']:%Y-%m-%d} · costs {b['fee_bps']:g} bps fee + "
                   f"{b['slippage_bps']:g} bps slippage per side · bootstrap p = {num(b['p_value'])}")
        if not b["curve"].empty:
            st.plotly_chart(charts.equity_chart(b["curve"], f"{b['strategy']}: equity vs buy & hold", dark),
                            width="stretch", key=f"equity-{b['strategy']}-{b['horizon']}")
            with st.expander("Equity curve data table"):
                st.dataframe(b["curve"].iloc[::-1], width="stretch")

st.divider()
st.caption(f"{DISCLAIMER} Past performance, including backtests, does not predict future results.")
