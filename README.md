# combo-persist-strategy-research

Research for the **`combo_AND_persist`** cheap-entry strategy — a dip / volatility-compression / slope-fade conjunction with a persistence gate — and its re-ranker, for Polymarket 5-minute crypto up/down markets (BTC/XRP).

## Why it exists
A naive cheap buy (`ask <= e_cap`) cannot beat breakeven in an efficient book: win-rate tracks the entry price (`WR ≈ entry`, so `2·entry` is the bar). The only lever that lifts WR is a **selector**: require several independent signals to AGREE simultaneously *and* hold for X seconds before committing the buy. This repo is the offline sweep that searches that selector space for a variant whose WR clears `2·avg_entry` at viable coverage, and exports the winning variant's per-market PnL column for downstream portfolio work.

## How it works
Two scripts, both pure offline numpy backtests (no network, no funds, leak-free first-trigger scanning cd high→low):

| File | Role |
|------|------|
| `newstrat_combo_AND_persist.py` | Core engine + full grid sweep. Precomputes per-market/per-side feature arrays once, then evaluates every variant = (signal set × `e_cap` × persistence × thresholds). Picks the best breakeven-beater, prints buckets/coverage/WR, and **writes the chosen variant's per-market PnL column** to `data/edge_pnl/combo_AND_persist.json`. Run as a module (`main()`). |
| `combo_rerank.py` | Lean re-ranker over the same variant space. Imports the engine, re-scores variants with a stricter schema breakeven (`WR > 2·avg_entry`, not `2·e_cap`) and a **robust mean** (drop the top-2 winners), keeping only scalar metrics per variant to stay memory-light. Prints top variants by raw mean, robust mean, and coverage. |

The 5 boolean signals (per side): `dip` (recent price dip / mean-reversion), `ofi` (signed velocity confirms the turn), `crowd` / `crowd_contra` (crowd aligned vs. fade-the-crowd — mutually exclusive, never paired), `volc` (tight spread / vol-compression), `slopefade` (20s mid-slope no longer moving against the side). A variant ANDs 2–3 of these with `ask <= e_cap` and a persistence floor.

## Requirements
- Python 3.8+
- `numpy` (only third-party dependency)
- Read access to the private **`polymarket-data`** repo (see Data)
- No wallet, key, ClobClient, or RPC needed — this is research only, it never places orders or touches funds.

## Usage
```bash
# Full grid sweep: evaluates all variants, writes the chosen per-market PnL column
python3 newstrat_combo_AND_persist.py

# Re-rank the same space with schema-breakeven + robust (drop-top-2) mean
python3 combo_rerank.py
```

## Data
Both scripts read the input panel from a hardcoded path:

- **Input:** `…/polymarket-bot/data/market_panel.json` (`PANEL`) — the per-market tick panel.
- **Output:** `…/polymarket-bot/data/edge_pnl/combo_AND_persist.json` (`OUT_COL`) — chosen variant's per-market PnL column.

These `data/` files are **not** committed (see `.gitignore`); fetch/symlink them from the private `polymarket-data` repo, or repoint `PANEL` / `OUT_COL` at your local copy.

> Private research software. No warranty; trades/handles real funds at your own risk.
