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
| 3 | Feature engineering + leakage checks | not started |
| 4 | Modeling (baselines, LightGBM/XGBoost, LSTM, walk-forward) | not started |
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
│   ├── features/             # phase 3
│   ├── modeling/             # phase 4
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


## Database schema

| Table | Purpose | Key |
|-------|---------|-----|
| `symbols` | Configured markets; `is_target` marks forecast targets vs. context symbols | `symbol` |
| `klines` | OHLCV + trades per symbol/interval | `(symbol_id, interval, open_time)` unique, used for upserts |
| `market_context` | Funding rate, open interest per symbol (`ts` = publish time) | `(symbol_id, metric, ts)` |
| `features` | Versioned feature vectors (JSON), stamped with `as_of_time` = when knowable | `(symbol_id, interval, as_of_time, feature_set_version)` |
| `model_runs` | Model versions, params, walk-forward metrics, artifact path, active flag | `(model_name, symbol_id, horizon_days, version)` |
| `predictions` | prob_up + q10/q50/q90 return & price per horizon; `realized_return` filled later | `(model_run_id, as_of_time, horizon_days)` |
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

