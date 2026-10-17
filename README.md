# Crypto Forecaster

Probabilistic 7- and 14-day price forecasts for Binance spot markets, starting with **SOLUSDT**.
More symbols are added through configuration (`SYMBOLS`, `TARGET_SYMBOLS`).

> **Disclaimer:** Forecasts are statistical estimates with large uncertainty. They are not
> financial advice. This system is **read-only**: it uses public Binance market-data endpoints
> only and never places orders.

## Status

| Phase | Scope | State |
|------:|-------|-------|
| 1 | Scaffold, config, logging, MySQL schema, Alembic | done |
| 2 | Data pipeline (backfill, incremental updater, upserts, funding/OI) | done |
| 3 | Feature engineering + leakage checks | done |
| 4 | Modeling (baselines, LightGBM/XGBoost, LSTM, walk-forward) | done |
| 5 | Backtesting | done |
| 6 | FastAPI | done |
| 7 | Streamlit dashboard | done |
| 8 | Quality: tests, lint, CI, live contract tests, docs | done |

## Layout

```
├── alembic.ini
├── migrations/               # Alembic env + versioned migrations
├── scripts/create_database.sql
├── src/forecaster/
│   ├── config.py             # pydantic-settings, all config from env / .env
│   ├── logging_setup.py      # JSON or console structured logging
│   ├── cli.py                # `forecaster` command
│   ├── db/                   # SQLAlchemy models, session, repository helpers
│   ├── data/                 # Binance REST client, dumps, backfill, updater, scheduler
│   ├── features/             # indicators, feature/target builder, leakage guard, storage
│   ├── modeling/             # baselines, GBM, LSTM, walk-forward, metrics, registry, train/predict
│   ├── backtest/             # vectorised engine, performance stats, bootstrap, runner
│   ├── api/                  # FastAPI app (read-only GET endpoints), response schemas
│   └── dashboard/            # Streamlit app, data queries, plotly charts
├── artifacts/models/         # trained model files (git-ignored)
└── tests/
```

## Setup

Requirements: Python 3.11 or 3.12 (torch, LightGBM and XGBoost wheels lag new Python releases),
MySQL 8.

```bash
# with uv (recommended; it can also download Python for you)
uv venv --python 3.12 .venv
uv pip install -r requirements.txt -e . --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu      # CPU-only torch, much smaller

# or with plain pip
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt -e .

cp .env.example .env            # then edit DB_* values
mysql -u root -p < scripts/create_database.sql
forecaster db init              # runs migrations + inserts configured symbols
```

## Quickstart: the whole pipeline

```bash
forecaster db init                            # schema + symbols
forecaster data backfill                      # about 6 min for SOL/BTC/ETH, 1d + 4h, from 2020-08
forecaster data context                       # funding rate + open interest
forecaster features check-leakage -s SOLUSDT  # must print OK
forecaster train                              # walk-forward eval + versioned models (LSTM is the slow part)
forecaster backtest                           # strategy vs buy-and-hold, after costs
forecaster predict                            # today's 7/14-day forecast
forecaster serve        # API  -> http://127.0.0.1:8000/docs
forecaster dashboard    # UI   -> http://127.0.0.1:8501
```

Keep it current with `forecaster data schedule` (or schedule `forecaster data update` and
`forecaster data context`), and retrain periodically, e.g. weekly, with `forecaster train`.

## Results on real data (SOLUSDT, run 2026-10-17)

Real Binance data from 2020-08 (2,258 daily rows, 0 gaps). The walk-forward out-of-sample period
runs from about 2021-09 to 2026-10. The leakage check passed at 9 cutoffs. Timings on a laptop
CPU: backfill 6 min, `train` 7.4 min (all 5 models × 2 horizons), backtest + predict 20 s.

| Horizon | Model | AUC | Brier skill vs climatology | 80% coverage | Backtest Sharpe (B&H) | Bootstrap p |
|---|---|---|---|---|---|---|
| 7d | climatology | 0.465 | baseline | 0.91 | 0.25 (0.68) | 0.83 |
| 7d | lgbm | 0.545 | -0.037 | 0.69 | 0.79 (0.68) | 0.39 |
| 7d | xgb | 0.549 | -0.031 | 0.77 | 0.83 (0.68) | 0.36 |
| 7d | lstm | 0.485 | -0.019 | 0.84 | 0.64 (0.68) | 0.56 |
| 14d | lgbm | 0.544 | -0.071 | 0.62 | 0.93 (0.69) | 0.30 |
| 14d | xgb | 0.547 | -0.057 | 0.74 | 0.97 (0.69) | 0.28 |
| 14d | lstm | 0.515 | -0.018 | 0.87 | 1.35 (0.69) | 0.09 |

**What this says:**

- **No model beats the baseline on probability quality.** Every Brier skill score is negative: the
  probabilities are worse than simply using the historical up-rate. The tree models rank a little
  better than chance (AUC about 0.545) but are over-confident. The verdict is "mixed" at best.
- **The tree models' 80% ranges are too narrow** (62–77% coverage), while climatology's are too wide
  (90%). Treat the price range as rougher than its label says.
- **Backtests look better than they are.** Several strategies beat buy-and-hold on Sharpe, mainly by
  being out of the market for most of the 2022 crash (buy-and-hold drawdown -96%). None is
  statistically significant (all p > 0.05). Five models × 2 horizons is also a multiple-comparison
  setting, which makes one lucky result likely.
- Bottom line: this is a sound, leakage-checked research framework, **not** a demonstrated
  trading edge.

## Data pipeline (phase 2)

```bash
forecaster data backfill                      # all symbols x intervals from BACKFILL_START
forecaster data backfill -s SOLUSDT -i 1d     # one symbol / interval
forecaster data backfill --source api         # skip data.binance.vision dumps
forecaster data update                        # incremental: new closed candles + recent gap repair
forecaster data context                       # funding rate + open interest (USD-M futures)
forecaster data gaps                          # list missing candle ranges
forecaster data schedule                      # long-running updater (cron, UTC)
```

- **Backfill (`auto`)**: monthly zips from data.binance.vision (SHA-256 verified) for complete
  months, REST API for the current month, then API refetch of any gaps.
- **Only closed candles are stored**; the forming candle is never written.
- **Idempotent**: all writes are upserts on `(symbol_id, interval, open_time)`; re-running is safe.
- **Rate limits**: exponential backoff with jitter on network errors and 5xx; `Retry-After` honoured
  on 429/418; pauses before `X-MBX-USED-WEIGHT-1M` reaches `BINANCE_WEIGHT_SOFT_LIMIT`.
- **Scheduling**: `forecaster data schedule` runs at `UPDATE_CRON_MINUTE` past each hour in
  `UPDATE_CRON_HOURS` (UTC). Alternatively run `forecaster data update` from Windows Task Scheduler
  or cron.


## Features (phase 3)

```bash
forecaster features build                     # build + store features for TARGET_SYMBOLS
forecaster features check-leakage -s SOLUSDT  # truncation test on real stored data
```

Feature set `v1` (daily, one row per candle):

| Group | Features |
|-------|----------|
| Returns | log returns over 1, 3, 7, 14, 30, 60 days |
| Volatility | rolling std of daily returns (7/14/30), ATR(14)/close, daily range, 7-day realized vol from 4h candles |
| Trend | close vs SMA 7/20/50/200 and EMA 12/26, MACD line/signal/histogram (scaled by price) |
| Oscillators | RSI(14), Bollinger %B and bandwidth (20, 2σ) |
| Volume | volume and trade count vs 20-day average, 5/20 volume trend, taker-buy ratio |
| Drawdown | distance below the 30- and 90-day high |
| Cross-asset | BTC and ETH returns (1/7/30), 30-day correlation and beta with the target, relative 7-day strength |
| Futures context | latest funding rate and 7-day mean funding; 7-day open-interest change (off by default) |

Targets (separate frame, never mixed into features): `fwd_logret_{7,14}` = log(close(t+h) / close(t)),
`fwd_up_{7,14}`, and `target_end_time_{h}`, the moment the target becomes known.

**Leakage controls**

- Each row carries `as_of_time` = close of its last candle. Every joined input (other symbols, 4h
  candles, funding, open interest) is aligned with a backward as-of join on its *publish* time.
  Open interest is treated as known only after its daily bucket ends.
- Only causal operations: trailing rolling windows and EWMs, no centred windows, no full-sample
  scaling. Any normalisation for models is fitted inside each walk-forward training fold (phase 4).
- Missing candles stay missing: the calendar is reindexed and those rows are blanked, never
  interpolated.
- `check_no_lookahead` rebuilds all features from inputs truncated at several cutoffs (including
  mid-candle) and requires every already-knowable value to be identical to the full-history build.
  Tests prove the check catches planted leaks (next-day price, centred window, full-sample mean).
- `make_dataset` keeps only rows whose target is realized by `now`.

## Modeling (phase 4)

```bash
forecaster train                              # all TARGET_SYMBOLS x FORECAST_HORIZONS x TRAIN_MODELS
forecaster train -s SOLUSDT --horizon 7 --models lgbm
forecaster models list                        # versions, metrics, * = active
forecaster models activate 42                 # roll back / switch version
forecaster predict                            # live forecast from the active SERVING_MODEL
```

Every model outputs `prob_up` and the 10th/50th/90th percentiles of the h-day forward log
return; the API converts them to a price range (`price = close × exp(q)`).

| Model | Description |
|-------|-------------|
| `climatology` | Baseline: training-window up-rate and empirical return quantiles. Always trained. |
| `momentum` | Baseline: the same statistics conditioned on the sign of the trailing 7/14-day return. |
| `lgbm` / `xgb` | Gradient-boosted direction classifier (Platt-calibrated on the most recent, purged 20% of the training window) + one quantile regressor per quantile. Conservative, regularised defaults. |
| `lstm` | PyTorch LSTM over the last 30 days of standardised features; direction head (BCE) + non-crossing quantile head (pinball loss), early stopping on a purged validation tail. |

**Validation.** Walk-forward with an expanding (or rolling, `WF_MAX_TRAIN_DAYS`) window: test
blocks of `WF_TEST_DAYS` after `WF_MIN_TRAIN_DAYS`. A training row is used only if its target was
realized before the test block starts, which purges the h-1 rows whose targets overlap it. A
fresh model is fitted per fold. No random splits anywhere.

**Metrics** (out-of-sample, all folds pooled): accuracy, precision, recall, ROC-AUC, Brier score,
log loss, a reliability table, MAE of the median, pinball loss per quantile, and 80% interval
coverage (with below/above-band rates). Each model is compared with `climatology`: Brier skill
score, pinball skill and accuracy delta. Each gets a verdict: *beats baseline*, *mixed* or
*does NOT beat baseline*. The "always up" accuracy (= buy-and-hold view) is printed too.

**Versioning.** `forecaster train` is the retrain command. Each run creates a new version
(UTC timestamp) per model, saves `artifacts/models/{symbol}/{model}/h{h}/{version}/model.joblib` +
`meta.json`, records a `model_runs` row with params and walk-forward metrics, stores the
out-of-sample predictions (`predictions.kind = 'oos'`) for backtests and calibration charts, and
activates the new version. `forecaster predict` stores `kind = 'live'` forecasts and later
back-fills their `realized_return`.

## Backtesting (phase 5)

```bash
forecaster backtest                           # all TARGET_SYMBOLS x horizons x TRAIN_MODELS
forecaster backtest -s SOLUSDT --horizon 7 --models lgbm,momentum
```

- **Signals** are the walk-forward out-of-sample predictions stored at training time
  (`predictions.kind = 'oos'`), so each was made by a model that never saw that period.
- **Rule** (fixed in advance, not tuned): long when P(up) >= `BT_ENTER_LONG` (0.55), flat when it
  falls below `BT_EXIT_LONG` (0.50). Optional shorts and a "median must agree" filter.
- **Timing**: the position decided at the close of day t earns the return from close t to close t+1.
  A test proves a strategy using same-day information cannot profit from look-ahead.
- **Costs**: `FEE_BPS` + `SLIPPAGE_BPS` per unit of position change (a long-to-short flip pays
  twice). Positions are closed at the end. Buy-and-hold pays the same entry and exit costs.
- **Reported**: total return, CAGR, volatility, Sharpe, Sortino, max drawdown, exposure, number of
  trades, win rate and average trade, all against buy-and-hold over the identical period.
- **Verdict**: `beats_benchmark` = higher Sharpe than buy-and-hold after costs. A paired circular
  block bootstrap (blocks of 20 days) gives a p-value for the Sharpe difference. A higher Sharpe
  that is not significant is labelled as such, and the command says so explicitly when no model
  shows a significant edge.
- Results, the full equity curve and all statistics are stored in `backtest_results`.

A custom engine (about 100 lines of pandas) is used instead of vectorbt or Backtrader. Every step
is visible and covered by hand-computed tests, and vectorbt's numba dependency pins NumPy versions.

## REST API (phase 6)

```bash
forecaster serve                              # http://127.0.0.1:8000, interactive docs at /docs
# or: uvicorn forecaster.api.main:app
```

| Endpoint | Returns |
|----------|---------|
| `GET /forecast/{symbol}?model=&horizons=7,14` | Per horizon: `prob_up`, `prob_down`, `direction`, 10/50/90% log-return quantiles, the 80% `price_range` (low / median / high), the model version, its walk-forward verdict, backtest interval coverage, and a `stale` flag |
| `GET /backtest/{symbol}?horizon=&include_curve=false` | Latest backtest per strategy: returns, Sharpe, drawdown, win rate, trades, the buy-and-hold figures, `beats_benchmark`, bootstrap p-value, verdict, and optionally the equity curve |
| `GET /models?symbol=&all=false` | Active (or all) model versions with walk-forward metrics and verdict vs climatology |
| `GET /health` | Database status, latest candle and staleness per target symbol, number of active models |

Every forecast and backtest response includes `disclaimer`; forecasts also set `is_estimate: true`.
Only `GET` exists (a test asserts this). The API binds to `127.0.0.1` and has no auth because it
is read-only; put it behind a reverse proxy with auth before exposing it. Forecasts are computed
on demand from the active model, stored as `kind = 'live'`, and cached until a new daily candle
closes or a new model is activated. `404` = not a configured target, `503` = no data or no
trained model yet.

## Dashboard (phase 7)

```bash
forecaster dashboard                          # http://127.0.0.1:8501
```

One filter row (symbol, model, horizon, chart range) scopes four tabs:

- **Forecast**: a P(up) tile and the 80% price range per horizon, the backtest coverage of that
  range, the model version, and a fan chart of recent closes with the projected range and
  median (dashed, because it is a projection).
- **Market**: daily candlesticks with SMA 20/50/200 and Bollinger bands; volume and RSI(14) in
  separate panels below.
- **Model performance**: the walk-forward metrics table for active models, the verdict vs
  climatology, a calibration (reliability) chart, and rolling 90-day coverage of the 80% range.
- **Backtest**: the verdict (significant, not significant, or does not beat), strategy vs
  buy-and-hold tiles, and the equity and drawdown curves.

A disclaimer banner sits at the top and the bottom. Every chart has a legend and a data-table
expander. Colours come from a CVD-validated palette with separate light and dark steps, and up/down
uses a blue/red pair. No chart uses two y-axes. Forecasts shown are stored as `live` predictions.

## Database schema

| Table | Purpose | Key |
|-------|---------|-----|
| `symbols` | Configured markets; `is_target` marks forecast targets vs. context symbols | `symbol` |
| `klines` | OHLCV + trades per symbol/interval | `(symbol_id, interval, open_time)` unique, used for upserts |
| `market_context` | Funding rate, open interest per symbol (`ts` = publish time) | `(symbol_id, metric, ts)` |
| `features` | Versioned feature vectors (JSON), stamped with `as_of_time` = when knowable | `(symbol_id, interval, as_of_time, feature_set_version)` |
| `model_runs` | Model versions, params, walk-forward metrics, artifact path, active flag | `(model_name, symbol_id, horizon_days, version)` |
| `predictions` | prob_up + q10/q50/q90 log-return & price; `kind` = `live` or `oos`; `realized_return` filled later | `(model_run_id, as_of_time, horizon_days, kind)` |
| `backtest_results` | Strategy vs. buy-and-hold metrics, explicit `beats_benchmark` flag, equity curve | — |

All timestamps are naive UTC.

## Tests and quality

```bash
pytest                                        # about 120 tests, no network, about 2 minutes
ruff check src tests migrations               # lint
FORECASTER_LIVE_TESTS=1 pytest -m live        # contract tests against real Binance public endpoints
FORECASTER_TEST_MYSQL_URL=mysql+pymysql://user:pw@127.0.0.1:3306/empty_test_db pytest tests/test_mysql.py
```

| Area | What is tested |
|------|----------------|
| Data pipeline | kline parsing (ms and µs epochs), validation, gap detection, retries/backoff/Retry-After, weight throttling, pagination, dump checksums, idempotent upserts, gap repair, incremental updates, context ingestion |
| Leakage | causality of every indicator; truncation test of the full feature set (incl. mid-candle cutoffs and gaps); planted-leak canaries the check must catch; purge correctness of walk-forward splits; a spy model proving folds never see test rows; LSTM windows unaffected by future data |
| Modeling | metrics, baselines, real-signal detection vs. a pure-noise guard (no fake skill), quantile ordering, calibration, artifact round-trips, versioning/activation, end-to-end train + forecast |
| Backtest | next-day timing (a look-ahead strategy cannot profit), costs, flips, gaps, bootstrap, verdicts |
| API / dashboard | response shapes, disclaimers, caching, error codes, GET-only; chart structure in both themes (no dual axes); full Streamlit app smoke run |
| Schema | real Alembic migrations on SQLite, migration/ORM column parity, MySQL upsert SQL compilation; optional MySQL round trip |

CI (`.github/workflows/ci.yml`) runs lint and tests on Python 3.11 and 3.12 against a MySQL 8
service container, which enables the MySQL integration test. The live contract job is
informational, because Binance blocks some cloud regions, including GitHub's US runners.

## Security

- No API keys are needed. If `BINANCE_API_KEY` / `BINANCE_API_SECRET` are ever set, they are read
  from the environment only, held as `SecretStr`, and masked in logs.
- `.env` is git-ignored; only `.env.example` is committed.

## Known limitations

- **Genuine gaps**: Binance has had exchange-wide outages; such gaps cannot be refetched. They are
  logged and listed by `forecaster data gaps`, never synthesised. Features must handle them.
- **Open interest history** is only served for about the last 30 days, so it accumulates from the
  first run onward and is too short for model training at first.
- **BTC dominance** is not available from Binance. The market-wide factor is BTC (and ETH) returns
  from their own klines.
- Funding/OI come from USD-M perpetuals; spot symbols without a perpetual are skipped.
- Dump timestamps switched from milliseconds to microseconds in 2025; both are handled.
- EWM-based indicators (EMA, RSI, ATR, MACD) carry their state across a missing candle; only the
  missing row itself is blanked. Rolling-window features are NaN for any window touching a gap.
- Long-window features (SMA 200) need 200 days of history, so the first ~200 rows per symbol are
  incomplete. Tree models handle NaN natively; the LSTM replaces NaN with the training mean (0 after
  standardisation).
- **Expect weak skill.** Liquid crypto prices are close to a random walk at 7–14 days. A small
  positive Brier skill score is a good outcome; the system reports when a model does not beat
  climatology instead of hiding it. Treat the 80% range as the more useful output.
- Overlapping h-day targets make out-of-sample rows highly autocorrelated, so the effective sample
  size is far smaller than the row count; metric differences of a few thousandths are noise.
- Artifacts are pickles (joblib). Only load files this system wrote.
- The backtest evaluates **daily** signals from models trained for 7/14-day targets. Positions are
  re-evaluated every day rather than held for the full horizon.
- Fills are assumed at the daily close with a flat slippage estimate. No funding costs for shorts,
  no market impact, no partial fills.
- The OOS period starts after `WF_MIN_TRAIN_DAYS`, so the backtest covers less than the full
  history. Results depend heavily on the market regime in that window (crypto bull and bear runs
  dominate buy-and-hold).

