# Phase 2 verdict — TEMPLATE (not a verdict)

> **This file is a template.** Copy it to `results/phase2_verdict.md` and fill it
> in from a real run. Do not fill it in from a `--source synthetic` run: synthetic
> bars are generated, and a verdict computed on them would be a fabricated
> research finding with the shape of a real one.
>
> From the kit: *"Be brutally honest — 'it does not qualify' is a valid and
> respectable outcome, and better than a flattering lie."*

## How to produce the inputs

```bash
# 1. Sweep the TRAIN window. Refuses to touch out-of-sample data.
python -m backtest.sweep --source yfinance

# 2. Read the plateau analysis it prints. Take the RECOMMENDED cell,
#    not the best cell. If they differ, that difference is the finding.

# 3. Evaluate out-of-sample ONCE, with the chosen parameters only.
python -m backtest.run --start 2020-01-01 --source yfinance \
    --entry-rsi <chosen> --exit-rsi <chosen> --time-stop <chosen>
```

Step 3 refuses a second out-of-sample evaluation under different parameters.
If it refuses, that is the rule working — not an obstacle to route around.

---

## 1. Plateau present? (rule 3)

| | best cell | recommended cell |
|---|---|---|
| entry RSI | | |
| exit RSI | | |
| time stop | | |
| train Sharpe | | |
| worst neighbour | | |
| robust score | | |

- Did best and recommended differ? →
- Was the best cell an isolated spike (> 1σ above its neighbourhood)? →
- Was the recommended cell on the grid edge (so possibly not a middle)? →
- **Is there a genuine plateau, or only isolated spikes?** →

A surface where good cells sit next to bad ones at every scale has no plateau,
and the correct conclusion is that the parameters are not identifiable — not
that you should take the highest number anyway.

## 2. Out-of-sample results (rule 4)

Evaluated on ______ (date), exactly once, with the parameters above.

| | train (2010–2019) | OOS (2020–) | buy & hold SPY (OOS) |
|---|---|---|---|
| Sharpe | | | |
| CAGR | | | |
| max drawdown | | | |
| win rate | | — | |
| trade count | | | — |
| exposure | | | — |

Degradation from train to OOS: ______

Some degradation is expected and healthy. Degradation to near zero, or a sign
flip, means the train result was fitting noise.

## 3. The gate (from CLAUDE.md)

| criterion | threshold | actual | pass? |
|---|---|---|---|
| OOS Sharpe | > 0.5 | | |
| Beats SPY risk-adjusted OOS | strategy Sharpe > benchmark Sharpe | | |
| Plateau present | stable region, not a spike | | |
| Enough trades to mean something | see note below | | |

On trade count: a handful of trades cannot distinguish skill from luck no matter
how good they look. Several hundred over a decade is a reasonable bar; state the
number and judge it rather than leaving it implicit.

## 4. Verdict

**Does v1 qualify for Phase 3 (paper trading)?**  YES / NO

Reasoning:

## 5. If NO — what would have to change

Note that per the ops rulebook, any change goes through the **full** pipeline
again: backtest → sweep → OOS → restart the paper clock. Re-running the OOS
window against a newly tuned parameter set is not a fix; it is the failure mode
rule 4 exists to prevent, and it will be refused.

## 6. Caveats that apply regardless of the verdict

- The universe is hindsight-selected: ten ETFs chosen knowing they exist and are
  liquid today. Results are optimistic by an amount this backtest cannot measure.
- RSI(2) mean reversion is widely published, so any edge has been mined since
  roughly 2009. A strong result deserves more suspicion than a weak one.
- A backtest Sharpe above ~2.5 is treated as a probable bug until explained.
- The backtest assumes fills at the open at the modelled cost. Phase 3's
  fill-quality log is what tests that assumption against reality.
