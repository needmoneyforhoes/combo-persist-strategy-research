# combo-persist-strategy-research

Offline backtest sweep for the `combo_AND_persist` cheap-entry strategy on Polymarket 5-minute crypto up/down markets (BTC/XRP).

A naive cheap buy (`ask <= e_cap`) does not beat breakeven in an efficient book: win-rate tracks entry price, so the bar is `2*entry`. The lever that lifts WR is a selector that requires several independent signals to agree at once and hold for X seconds before firing. These scripts search that selector space for a variant whose WR clears breakeven at usable coverage, and export the winner's per-market PnL column.

Both scripts are pure numpy backtests: no network, no funds, leak-free first-trigger scanning cd high to low.

## Scripts

| File | What it does |
|------|--------------|
| `newstrat_combo_AND_persist.py` | Engine plus full grid sweep. Precomputes per-market/per-side feature arrays once, evaluates every variant (signal set x `e_cap` x persistence x thresholds), picks the best breakeven-beater, prints buckets/coverage/WR, and writes the chosen variant's per-market PnL column to `$DATA_DIR/edge_pnl/combo_AND_persist.json`. |
| `combo_rerank.py` | Re-ranker over the same variant space. Imports the engine, re-scores with a stricter breakeven (`WR > 2*avg_entry`) and a robust mean (drop top-2 winners), keeps scalar metrics only. Prints top variants by raw mean, robust mean, and coverage. |

## Signals

Five booleans per side, ANDed 2-3 at a time with `ask <= e_cap` and a persistence floor:

- `dip`: recent price dip (mean-reversion entry)
- `ofi`: signed velocity confirms the turn
- `crowd` / `crowd_contra`: crowd aligned vs fade-the-crowd. Mutually exclusive, never paired in one set.
- `volc`: tight spread / vol-compression
- `slopefade`: 20s mid-slope no longer moving against the side

Sweep grids: `e_cap` in {0.10..0.25}, persistence in {1,2,3,5,8,12,20}s, per-signal thresholds. Viability floor is 6 fires.

## Usage

```bash
python3 newstrat_combo_AND_persist.py   # full sweep, writes chosen PnL column
python3 combo_rerank.py                 # re-rank with schema-breakeven + robust mean
```

## Data

Input panel `$DATA_DIR/market_panel.json` (`PANEL`) and output `$DATA_DIR/edge_pnl/combo_AND_persist.json` (`OUT_COL`) are hardcoded paths. The `data/` files are gitignored; fetch them from the private `polymarket-data` repo or repoint `PANEL` / `OUT_COL` at a local copy.

Requires Python 3.8+ and numpy.
