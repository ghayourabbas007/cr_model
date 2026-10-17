"""Streamlit dashboard. Run with `forecaster dashboard` (or `streamlit run` on this file).

Reads the database directly; read-only apart from storing the live forecasts it displays.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from forecaster import DISCLAIMER
from forecaster.config import get_settings
from forecaster.dashboard import charts
from forecaster.dashboard import data as dd
from forecaster.db.session import get_sessionmaker

st.set_page_config(page_title="Crypto Forecaster", page_icon="📈", layout="wide")
s = get_settings()


@st.cache_resource
def _factory():
    return get_sessionmaker()


def _session():
    return _factory()()


@st.cache_data(ttl=300, show_spinner=False)
def load_candles(symbol: str, days: int) -> pd.DataFrame:
    with _session() as session:
        return dd.candles_with_indicators(session, symbol, days)


@st.cache_data(ttl=300, show_spinner="Computing forecast…")
def load_forecasts(symbol: str, model: str):
    with _session() as session:
        return dd.current_forecasts(session, s, symbol, model)


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

tab_fc, tab_mkt, tab_perf, tab_bt = st.tabs(["Forecast", "Market", "Model performance", "Backtest"])

# --- forecast ---------------------------------------------------------------------------------
with tab_fc:
    forecasts, reason = load_forecasts(symbol, model)
    if not forecasts:
        st.info(f"No forecast available: {reason}")
    else:
        if any(f["stale"] for f in forecasts):
            st.warning("Market data is more than 2 days old; run `forecaster data update`.", icon="⏱️")
        cols = st.columns(len(forecasts))
        for col, f in zip(cols, forecasts):
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
        st.plotly_chart(charts.forecast_fan(closes, forecasts, dark), use_container_width=True)
        with st.expander("Forecast data table"):
            st.dataframe(pd.DataFrame(forecasts).drop(columns=["disclaimer"]), use_container_width=True)
        st.caption("A probability near 50% means the model has no meaningful directional view.")

# --- market -----------------------------------------------------------------------------------
with tab_mkt:
    st.plotly_chart(charts.candlestick(candles, symbol, dark), use_container_width=True)
    with st.expander("Candle data table"):
        st.dataframe(candles.iloc[::-1], use_container_width=True)

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
            use_container_width=True, hide_index=True)
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
            c1.plotly_chart(charts.reliability_chart(rel, dark), use_container_width=True)
            c2.plotly_chart(charts.coverage_chart(cov.dropna(), dark), use_container_width=True)
            with st.expander("Calibration and coverage data tables"):
                st.dataframe(rel, use_container_width=True, hide_index=True)
                st.dataframe(oos.iloc[::-1], use_container_width=True)

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
            st.plotly_chart(charts.equity_chart(b["curve"], "Equity vs buy & hold", dark), use_container_width=True)
            with st.expander("Equity curve data table"):
                st.dataframe(b["curve"].iloc[::-1], use_container_width=True)

st.divider()
st.caption(f"{DISCLAIMER} Past performance, including backtests, does not predict future results.")
