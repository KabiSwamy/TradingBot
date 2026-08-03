# TradingBot — Mean-Reversion ETF Swing Bot

RSI(2) mean reversion on liquid ETFs: buy violent short-term dips inside
long-term uptrends, sell the bounce. Daily bars, next-open fills, costs always on.

**Read [`CLAUDE.md`](CLAUDE.md) first — it is the project's constitution.** Every
rule in it is binding on both humans and agents working in this repo.

Status: **Phase 1 (backtest harness)**. No broker integration, no live execution,
no API keys. See the phase gates in `CLAUDE.md`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python -m pytest                                   # all core tests, no network needed
python -m backtest.run --start 2010-01-01          # full backtest + report
```

(Use `python -m pytest` rather than bare `pytest` unless you are inside the
virtualenv — a `pytest` elsewhere on `PATH` may belong to a different interpreter
and will not see this project's dependencies.)

Every run appends a row to `results/experiments.csv` (rule 4) and writes an
equity curve to `results/`.

### Data sources

`--source yfinance` (default) fetches daily split/dividend-adjusted bars and
caches them as parquet under `data/cache/`.

`--source synthetic` generates deterministic seeded bars instead. This exists so
the engine and the full test suite are runnable in environments with no market-data
access — **its numbers are not real and must never be read as a research result.**
Synthetic runs are labelled in the report header and carry `source=synthetic` in
`results/experiments.csv`.

## Layout

    config/       settings.yaml — universe, thresholds, risk limits, costs
    data/         bar sources, parquet cache, validation
    strategies/   pure signal functions — DataFrame in, signals out (rule 9)
    backtest/     engine, cost model, metrics, report, CLI
    tests/        look-ahead, fill timing, indicators, costs, kill switch
    results/      experiments.csv, equity curves, decision logs (gitignored)

`strategies/` imports nothing from `backtest/`. Both the backtester and (in
Phase 3) the live job call the identical strategy function — that is what makes
paper results comparable to backtests.

---

Educational material for a personal research project. Not financial advice.
