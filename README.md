# TradingBot — Mean-Reversion ETF Swing Bot

RSI(2) mean reversion on liquid ETFs: buy violent short-term dips inside
long-term uptrends, sell the bounce. Daily bars, next-open fills, costs always on.

**Read [`CLAUDE.md`](CLAUDE.md) first — it is the project's constitution.** Every
rule in it is binding on both humans and agents working in this repo.

Status: **Phase 1 (backtest harness) complete.** No broker integration, no live
execution, no API keys. See the phase gates in `CLAUDE.md`.

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

Use `python -m pytest` rather than bare `pytest` unless you are inside the
virtualenv — a `pytest` elsewhere on `PATH` may belong to a different interpreter
and will not see this project's dependencies.

Every run prints the report, writes an equity curve and a JSON-lines decision log
to `results/`, and appends a row to `results/experiments.csv` (rule 4).

Useful flags:

| flag | effect |
|---|---|
| `--source yfinance\|synthetic` | where bars come from (default `yfinance`) |
| `--refresh` | re-fetch bars, ignoring the cache |
| `--entry-rsi / --exit-rsi / --time-stop / --capital` | override settings for one run |
| `--out-dir / --cache-dir` | redirect artifacts and cache |
| `--no-log` | skip the experiments.csv row (smoke tests only) |

### Data sources

`--source yfinance` fetches daily split/dividend-adjusted bars (`auto_adjust=True`)
and caches them as parquet under `data/cache/`. Refresh re-downloads full history
rather than appending — adjusted prices are rescaled retroactively by every
dividend, so appending would splice two price scales together and fabricate a
one-day return at the seam.

`--source synthetic` generates deterministic seeded bars instead. It exists so the
engine and the whole test suite run in environments with no market-data access —
**its numbers are not real and must never be read as a research result.** Synthetic
runs are banner-labelled in the report and on the plot, and carry `source=synthetic`
in `results/experiments.csv`.

> **Note for this environment:** Yahoo Finance is blocked by the egress policy here
> (`query1/query2.finance.yahoo.com` return 403 CONNECT), so `--source yfinance`
> cannot fetch data in this container. Run it on a machine with network access to
> produce real Phase 1 numbers; everything else is verifiable offline.

## Layout

    config/       settings.yaml + a loader that refuses rule-violating configs
    data/         bar sources, parquet cache, validation
    strategies/   pure signal functions — DataFrame in, signals out (rule 9)
    backtest/     engine, cost model, metrics, benchmark, report, CLI
    tests/        look-ahead, fill timing, indicators, costs, kill switch
    results/      experiments.csv, equity curves, decision logs (gitignored)

`strategies/` imports nothing from `backtest/` — `backtest/pipeline.py` joins them,
exactly as `jobs/nightly.py` will in Phase 3, so both call the identical strategy
function. That symmetry is what makes paper results comparable to backtests.

## What the tests actually guarantee

The five families named in `CLAUDE.md`, plus the guards that keep them honest:

- **Look-ahead** — mutate every bar after day *t* and rerun; every decision dated
  ≤ *t* must be byte-identical. Paired with a truncation variant (catches
  dependence on the *extent* of the series, which mutation can miss) and an
  assertion that later decisions *do* change, so the test cannot pass vacuously.
- **Fill timing** — signal-day close and next-day open differ sharply; fills must
  take the open, favourable or not.
- **Indicators** — Wilder RSI(2) against a hand-computed table, including the
  degenerate branches. Flat prices give 50, not 100 — the naive branch order
  returns 100 and invents exit signals from a series that never moved.
- **Costs** — a round trip on flat prices loses exactly 33.30 on 100k at 5bps.
- **Kill switch** — a −14.04% path must not fire; −15.03% must flatten and halt.
- **Purity** — the AST of every `strategies/` module is checked for forbidden
  imports and side effects, because a stated convention decays under deadline.
- **No network / no ledger pollution** — autouse fixtures fail any test that opens
  a socket or writes into the real `results/`.

---

Educational material for a personal research project. Not financial advice.
