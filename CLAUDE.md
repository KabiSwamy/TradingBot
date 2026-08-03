# Mean-Reversion ETF Swing Bot

## What this project is

A daily-bar swing trading system. First strategy: RSI(2) mean reversion on
liquid ETFs — buy violent short-term dips in long-term uptrends, sell the
bounce. One nightly job runs after US market close; orders execute at the
next session's open. Paper trading on Alpaca before any real money.

This is a research project with real financial risk. The default assumption
about any surprisingly good result is that it is a bug.

## Non-negotiable rules (enforce in every session, refuse work that violates them)

1. NO LOOK-AHEAD. A decision dated day t may use only data through day t's
   close, and is filled at day t+1's open — in backtests AND live. Unit
   tests must prove it (see Testing).
2. COSTS ALWAYS ON. Every backtest deducts commission + slippage
   (default: 0.05% per side, configurable). Never produce a zero-cost
   backtest, even "just to see."
3. PLATEAU, NOT PEAK. A parameter value qualifies only if its neighbors
   also perform well. Always pick the middle of a stable region, never the
   single best historical value. A spike surrounded by cliffs = curve-fit
   = rejected.
4. OUT-OF-SAMPLE DISCIPLINE. Tune on the train window (2010-01-01 to
   2019-12-31). Judge on the test window (2020-01-01 to present), which is
   touched only for final evaluation. Log EVERY backtest run to
   results/experiments.csv (timestamp, config, window, Sharpe, max DD,
   trade count) so we cannot quietly cherry-pick.
5. RISK LIVES IN THE EXECUTION LAYER, not the strategy layer: max 3
   concurrent positions, long-only, cash account, no margin, no shorting,
   and a 15% peak-to-trough kill switch on total equity that flattens
   everything, halts trading, and requires a manual restart flag.
6. TIME STOP. Every position is force-exited after 7 trading days,
   no exceptions.
7. PAPER BEFORE REAL. No live API keys enter this project until at least
   2 months of paper trading show fills within tolerance of the cost model.
   Going live is a human decision — never propose it proactively.
8. SECRETS: keys live in .env only. .env is in .gitignore from the first
   commit. Never print keys in logs or commit them.
9. STRATEGIES ARE PURE FUNCTIONS. DataFrame in → signals out. No I/O, no
   API calls, no randomness, no state inside strategy code. The exact same
   function runs in backtest and live — this is what makes paper results
   comparable to backtests.
10. LOG EVERY DECISION. Each nightly run appends JSON lines: date, prices
    used, indicator values, signals, orders placed, fills, and the reason
    for every action or deliberate inaction.
11. TEST-FIRST for anything that can lie: fill timing, cost application,
    RSI/SMA math, drawdown calculation, kill-switch triggering, position
    reconciliation. Never weaken a test to make it pass.
12. BENCHMARK HONESTY. Every backtest report includes buy-and-hold SPY on
    the same window with the same costs, and reports Sharpe, max drawdown,
    win rate, avg win/loss, and trade count — not just total return.

## Strategy spec (v1)

- Universe: SPY, QQQ, IWM, DIA, XLK, XLF, XLV, XLE, XLI, XLP
- Data: daily bars, adjusted for splits/dividends
- Regime filter: close > 200-day SMA (else no entries in that symbol)
- Entry signal: RSI(2) < 10 at the close; if more signals than free slots,
  take the lowest RSI first
- Exit signal: RSI(2) > 70 at the close, OR position age = 7 trading days
- Fills: next session's open after any signal
- Sizing (v1): equal weight — each position gets 1/3 of allocated capital
- All thresholds live in config/settings.yaml, never hardcoded

## Architecture

    config/       settings.yaml — universe, thresholds, risk limits, costs
    data/         fetch + local cache of daily bars (yfinance to start)
    strategies/   pure signal functions (rsi2_mean_reversion.py)
    backtest/     engine, cost model, metrics, report generation
    execution/    Alpaca client, order builder, risk checks, kill switch
    jobs/         nightly.py — the single scheduled entrypoint
    tests/
    results/      experiments.csv, decision logs, reports

Dependency rule: strategies/ imports nothing from backtest/ or execution/.
Both backtest/ and execution/ call the same strategy functions.

## Testing (the tests that matter most)

- Look-ahead test: run the engine on a dataset, then mutate all data AFTER
  day t and rerun — every decision dated <= t must be identical.
- Fill-timing test: synthetic data where signal-day close and next-day open
  differ sharply; assert fills happen at next-day open.
- Indicator tests: RSI(2) and SMA(200) against small hand-computed cases;
  document the RSI smoothing method (Wilder) and keep it identical everywhere.
- Cost test: a round trip on flat prices must lose exactly the modeled costs.
- Kill-switch test: simulated equity path crossing -15% from peak must
  flatten and halt.

## Phase gates (definition of done)

- Phase 1 (harness): backtest runs 2010→present with costs on; report shows
  equity curve vs SPY, Sharpe, max DD, win rate, trade count; all core
  tests pass.
- Phase 2 (validation): parameter sweeps show plateaus (entry RSI in
  {5,8,10,12,15}, exit RSI in {60,65,70,75,80}, time stop in {5,7,10});
  out-of-sample window evaluated ONCE; experiments.csv complete; a written
  verdict on whether v1 qualifies (OOS Sharpe > 0.5, beats SPY
  risk-adjusted OOS, plateau present, enough trades to mean something).
- Phase 3 (paper): nightly job runs on schedule against Alpaca paper,
  reconciles positions from the broker (never from local memory), logs
  every decision, and alerts on any error.
- Phase 4 (live): human decision, tiny capital, checklist in the ops
  rulebook. Claude never initiates this.

## Ops rulebook (written before launch; the human's contract with the system)

The core distinction: OUTCOMES vs ASSUMPTIONS.
- Outcomes the backtest said could happen → variance → do nothing.
  Examples: losing streaks within the historical envelope, drawdowns below
  the kill switch, trailing SPY over weeks or months.
- Assumptions reality is contradicting → breach → stop and fix.
  Examples: fills consistently worse than the cost model (>0.15% average
  gap), bad or missing data, duplicate/missed orders, job failures,
  positions that don't reconcile with the broker.

Intervention triggers (exhaustive — nothing else counts):
- Drawdown >= 15%: kill switch acts automatically; human reviews.
- Losing streak exceeding historical max + 2: pause entries, review.
- Any assumption breach above: pause, diagnose, fix, then resume.

Never tune parameters in response to live results. Any change goes through
the full pipeline: backtest → sweep → OOS → restart paper clock.

## Working style for Claude Code

- New phase = plan first, then implement in small, committed steps.
- Tests before implementation for engine logic.
- A backtest Sharpe above ~2.5 is treated as a probable bug and
  investigated before being believed.
- Never install new heavyweight frameworks without discussing; this
  project stays small and legible.
