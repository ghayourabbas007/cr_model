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
| 5 | Backtesting | not started |
| 6 | FastAPI | not started |
| 7 | Streamlit dashboard | not started |
| 8 | Test hardening, docs | ongoing |

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
│   ├── backtest/             # phase 5
│   ├── api/                  # phase 6
│   └── dashboard/            # phase 7
├── artifacts/models/         # trained model files (git-ignored)
└── tests/
```

## Setup

Requirements: Python 3.11+, MySQL 8.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
pip install -e .

cp .env.example .env            # then edit DB_* values
mysql -u root -p < scripts/create_database.sql
forecaster db init              # runs migrations + inserts configured symbols
```

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

## Database schema

| Table | Purpose | Key |
|-------|---------|-----|
| `symbols` | Configured markets; `is_target` marks forecast targets vs. context symbols | `symbol` |
| `klines` | OHLCV + trades per symbol/interval | `(symbol_id, interval, open_time)` unique, used for upserts |
| `market_context` | Funding rate, open interest per symbol (`ts` = publish time) | `(symbol_id, metric, ts)` |
| `features` | Versioned feature vectors (JSON), stamped with `as_of_time` = when knowable | `(symbol_id, interval, as_of_time, feature_set_version)` |
| `model_runs` | Model versions, params, walk-forward metrics, artifact path, active flag | `(model_name, symbol_id, horizon_days, version)` |
| `predictions` | prob_up + q10/q50/q90 log-return & price; `kind` = live | oos; `realized_return` filled later | `(model_run_id, as_of_time, horizon_days, kind)` |
| `backtest_results` | Strategy vs. buy-and-hold metrics, explicit `beats_benchmark` flag, equity curve | — |

All timestamps are naive UTC.

## Tests

```bash
pytest
```

Schema tests run the real Alembic migrations against a temporary SQLite database and check that
the migration and the ORM models have identical columns.

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

