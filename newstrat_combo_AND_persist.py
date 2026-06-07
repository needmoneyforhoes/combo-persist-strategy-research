#!/usr/bin/env python3
"""
newstrat_combo_AND_persist.py — COMBINATION (AND) + PERSISTENCE cheap-buy strategy.

FAMILY: combo_AND_persist

IDEA
----
A naive cheap buy (ask <= e_cap) does NOT beat breakeven (efficient book: WR ~= e).
To clear 2*e_cap we must ADD win-rate via a SELECTOR. The highest-WR lever is the
INTERSECTION of multiple independent signals, all required to AGREE simultaneously,
AND to PERSIST for X seconds before we commit a cheap buy.

For a candidate side (UP or DN), 5 boolean signals are computed from the cached tick:

  dip      : mean-reversion entry — the side's price recently DIPPED.
             UP: d10_up <= -thr_dip ;  DN: d10_dn <= -thr_dip
  ofi      : order-flow/momentum confirms the side is turning in its favor NOW.
             UP: vel >= thr_ofi ;  DN: vel <= -thr_ofi   (vel = signed UP-velocity)
  crowd    : the crowd is aligned with the side.
             UP: crowd == 1 ;  DN: crowd == 0
  volc     : volatility-compression — the side's book is tight (low spread).
             UP: spread_up <= thr_volc ;  DN: spread_dn <= thr_volc
  slopefade: 20s mid-slope is not still moving AGAINST the side (slope-fade/revert).
             UP: mid_chg_20s >= -thr_slope ;  DN: mid_chg_20s <= thr_slope

A variant picks K of {2,3} signals that must ALL hold, requires the candidate side's
ask <= e_cap, and requires the WHOLE conjunction (signals + ask<=e_cap) to hold
continuously for >= X seconds (persistence) before firing.

FIRST-TRIGGER, LEAK-FREE: scan cd HIGH->LOW. A decision at cd=K uses only the tick at
cd>=K. Persistence measured in cd-seconds: the conjunction must have held on every
observed tick from cd_anchor down to cd_now with (cd_anchor - cd_now) >= X. Fire on the
FIRST tick where persistence is satisfied. Winner used ONLY at settle.

PnL per $1 stake: win -> (1/entry - 1) ; loss -> -1. Mean over ALL markets (unfired=$0).

WS-SAFE: every signal is O(1) from the current cached tick; persistence is a single
cd-anchor per (side, signalset) -> a trivial ring-buffer worker computation.

FAST ENGINE: per-market numpy arrays are precomputed once; each variant evaluates with
vectorized boolean ops + a tight first-trigger persistence scan.
"""

import json
import os
import itertools
from collections import Counter
import numpy as np

PANEL = "/home/polybot/polymarket-bot/data/market_panel.json"
OUT_COL = "/home/polybot/polymarket-bot/data/edge_pnl/combo_AND_persist.json"
CD_FILL_FLOOR = 15

SIGNALS = ["dip", "ofi", "crowd", "crowd_contra", "volc", "slopefade"]
# NOTE: "crowd" = crowd ALIGNED with the cheap side (rare, since cheap=>crowd usually
#        against it). "crowd_contra" = crowd AGAINST the cheap side = classic
#        fade-the-crowd dip-reversal (the math-correct interpretation for cheap entries).
# We do NOT put crowd and crowd_contra in the SAME signal set (mutually exclusive);
# such pairs are skipped in the sweep.

E_CAPS = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
PERSISTS = [1, 2, 3, 5, 8, 12, 20]
THR_DIP = [0.02, 0.04, 0.06]
THR_OFI = [0.03, 0.06, 0.10]
THR_VOLC = [0.02, 0.03, 0.05]
THR_SLOPE = [0.05, 0.10, 0.20]

MIN_FIRES = 6  # quant viability floor


def load_arrays():
    """Per-market, per-side precomputed numpy arrays (cd-ordered high->low, cd>floor).
    Returns list of dicts with side-keyed feature arrays for vectorized evaluation."""
    with open(PANEL) as f:
        markets = json.load(f)
    out = []
    for m in markets:
        ticks = [t for t in m["ticks"] if t["cd"] > CD_FILL_FLOOR]
        nt = len(ticks)
        if nt == 0:
            out.append(None)
            continue
        cd = np.array([t["cd"] for t in ticks], dtype=np.int32)

        def col(name):
            return np.array([(np.nan if t.get(name) is None else t.get(name))
                             for t in ticks], dtype=np.float64)

        up_ask = col("up_ask")
        dn_ask = col("dn_ask")
        d10_up = col("d10_up")
        d10_dn = col("d10_dn")
        vel = col("vel")
        crowd = col("crowd")
        spread_up = col("spread_up")
        spread_dn = col("spread_dn")
        mid_chg = col("mid_chg_20s")
        out.append({
            "slug": m["slug"], "winner": m["winner"], "cd": cd, "nt": nt,
            "up_ask": up_ask, "dn_ask": dn_ask,
            "d10_up": d10_up, "d10_dn": d10_dn, "vel": vel, "crowd": crowd,
            "spread_up": spread_up, "spread_dn": spread_dn, "mid_chg": mid_chg,
        })
    return markets, out


def side_signal_masks(M, side, thr):
    """Vectorized boolean arrays for each base signal, for one side. NaN => False."""
    if side == "UP":
        d10 = M["d10_up"]; spr = M["spread_up"]; ask = M["up_ask"]
        ofi = M["vel"] >= thr["ofi"]
        crowd = M["crowd"] == 1
        crowd_contra = M["crowd"] == 0
        slope = M["mid_chg"] >= -thr["slope"]
    else:
        d10 = M["d10_dn"]; spr = M["spread_dn"]; ask = M["dn_ask"]
        ofi = M["vel"] <= -thr["ofi"]
        crowd = M["crowd"] == 0
        crowd_contra = M["crowd"] == 1
        slope = M["mid_chg"] <= thr["slope"]
    dip = d10 <= -thr["dip"]
    volc = spr <= thr["volc"]
    # NaN comparisons yield False already; force NaN->False explicitly for safety
    dip = np.where(np.isnan(d10), False, dip)
    ofi = np.where(np.isnan(M["vel"]), False, ofi)
    crowd = np.where(np.isnan(M["crowd"]), False, crowd)
    crowd_contra = np.where(np.isnan(M["crowd"]), False, crowd_contra)
    volc = np.where(np.isnan(spr), False, volc)
    slope = np.where(np.isnan(M["mid_chg"]), False, slope)
    return {"dip": dip, "ofi": ofi, "crowd": crowd, "crowd_contra": crowd_contra,
            "volc": volc, "slopefade": slope}, ask


def _qualify_index(conj, cd, persist):
    """Vectorized: for boolean conj (cd descending), return the earliest index i where
    conj has held continuously since an anchor with (cd[anchor]-cd[i]) >= persist.
    Anchor = cd at the start (highest cd) of the current contiguous True run.
    Returns index or -1."""
    if not conj.any():
        return -1
    # run-start anchor cd: at each True position, the cd where this True-run began.
    # Build by carrying forward cd at positions where prev was False (run start).
    n = conj.shape[0]
    # is this position a run-start? True now and (first OR prev False)
    prev = np.empty(n, dtype=bool)
    prev[0] = False
    prev[1:] = conj[:-1]
    run_start = conj & (~prev)
    # anchor_cd: for each index, cd at the most recent run_start at or before it (when True)
    anchor_cd = np.where(run_start, cd, 0).astype(np.int64)
    # forward-fill anchor over the True run; reset to 0 on False
    # Do it with a cumulative max within runs: since cd is descending, the run-start cd
    # is the MAX cd in the run. Use running max that resets on False.
    out_anchor = np.zeros(n, dtype=np.int64)
    cur = 0
    for_fill = anchor_cd
    # cheap forward fill (one pass) — but vectorize via accumulation trick:
    # mark group ids by cumulative sum of run_start, take groupwise first cd.
    gid = np.cumsum(run_start)  # group id increments at each run start
    # for True positions, anchor = cd at first index of that group
    # first index cd per group == run_start cd. Map via gid.
    # Build lookup: max group id
    if run_start.any():
        # cd at run starts indexed by gid value
        starts_cd = cd[run_start]  # ordered by gid 1..G
        # gid for True positions is >=1; anchor = starts_cd[gid-1]
        held = np.zeros(n, dtype=np.int64)
        mask_true = conj
        g_idx = gid[mask_true] - 1
        anchor_vals = starts_cd[g_idx]
        held_true = anchor_vals - cd[mask_true]
        qualify = np.zeros(n, dtype=bool)
        qt = held_true >= persist
        qualify[mask_true] = qt
        idxs = np.nonzero(qualify)[0]
        if idxs.size:
            return int(idxs[0])
    return -1


def first_fire_market(M, sigset, e_cap, persist, thr):
    """Return (cd, side, ask) of first persistence-satisfied fire, or None.
    Fully vectorized conjunction + persistence per side; pick the EARLIER cd; if both
    sides qualify at the same cd, pick the cheaper ask."""
    cd = M["cd"]
    best = None  # (cd, side, ask, idx)
    for side in ("UP", "DN"):
        masks, ask = side_signal_masks(M, side, thr)
        conj = (ask <= e_cap)
        conj = np.where(np.isnan(ask), False, conj)
        for s in sigset:
            conj = conj & masks[s]
        conj = conj.astype(bool)
        i = _qualify_index(conj, cd, persist)
        if i < 0:
            continue
        ci = int(cd[i]); a = float(ask[i])
        if best is None:
            best = (ci, side, a, i)
        else:
            # earlier fire = higher cd wins; same cd -> cheaper ask
            if ci > best[0] or (ci == best[0] and a < best[2]):
                best = (ci, side, a, i)
    if best is None:
        return None
    return best[0], best[1], best[2]


def pnl(entry, won):
    return (1.0 / entry) - 1.0 if won else -1.0


def cd_bucket(cd):
    return (">240" if cd > 240 else "240-180" if cd > 180 else
            "180-120" if cd > 120 else "120-60" if cd > 60 else "60-15")


def eval_variant(arrs, sigset, e_cap, persist, thr, n):
    fires = wins = 0
    sum_pnl = 0.0
    sum_e = 0.0
    per_market = []
    bf = Counter(); bw = Counter()
    for M in arrs:
        if M is None:
            per_market.append((None, 0.0))
            continue
        f = first_fire_market(M, sigset, e_cap, persist, thr)
        if f is None:
            per_market.append((M["slug"], 0.0))
            continue
        cd, side, ask = f
        won = (side == M["winner"])
        p = pnl(ask, won)
        fires += 1
        sum_e += ask
        if won:
            wins += 1
        sum_pnl += p
        per_market.append((M["slug"], p))
        b = cd_bucket(cd)
        bf[b] += 1
        if won:
            bw[b] += 1
    return {
        "fires": fires, "wins": wins, "sum_pnl": sum_pnl, "sum_e": sum_e,
        "per_market": per_market, "bf": dict(bf), "bw": dict(bw),
    }


def main():
    markets, arrs = load_arrays()
    n = len(markets)
    wd = Counter(m["winner"] for m in markets)
    print(f"Panel: {PANEL}")
    print(f"Markets: {n}  (UP={wd.get('UP',0)} DN={wd.get('DN',0)})  window entry->cd{CD_FILL_FLOOR}")
    print("FAMILY: combo_AND_persist\n")

    sigsets = []
    for k in (2, 3):
        for c in itertools.combinations(SIGNALS, k):
            if "crowd" in c and "crowd_contra" in c:
                continue  # mutually exclusive
            sigsets.append(c)
    print(f"Signal sets (2- and 3-combos, crowd/contra excl): {len(sigsets)}")

    results = []
    total = 0
    for sigset in sigsets:
        dip_grid = THR_DIP if "dip" in sigset else [THR_DIP[0]]
        ofi_grid = THR_OFI if "ofi" in sigset else [THR_OFI[0]]
        volc_grid = THR_VOLC if "volc" in sigset else [THR_VOLC[0]]
        slope_grid = THR_SLOPE if "slopefade" in sigset else [THR_SLOPE[0]]
        for e_cap in E_CAPS:
            for persist in PERSISTS:
                for td in dip_grid:
                    for to in ofi_grid:
                        for tv in volc_grid:
                            for ts in slope_grid:
                                thr = {"dip": td, "ofi": to, "volc": tv, "slope": ts}
                                r = eval_variant(arrs, sigset, e_cap, persist, thr, n)
                                total += 1
                                if r["fires"] == 0:
                                    continue
                                wr = 100.0 * r["wins"] / r["fires"]
                                be = 100.0 * 2.0 * e_cap
                                cov = 100.0 * r["fires"] / n
                                mean_fire = r["sum_pnl"] / r["fires"]
                                mean_mkt = r["sum_pnl"] / n
                                avg_e = r["sum_e"] / r["fires"]
                                results.append({
                                    "sigset": sigset, "e_cap": e_cap, "persist": persist,
                                    "thr": thr, "fires": r["fires"], "wins": r["wins"],
                                    "wr": wr, "be": be, "cov": cov, "mean_fire": mean_fire,
                                    "mean_mkt": mean_mkt, "avg_e": avg_e,
                                    "beats": wr > be, "per_market": r["per_market"],
                                    "bf": r["bf"], "bw": r["bw"],
                                })
    print(f"Total variants evaluated: {total}")
    print(f"Variants with >=1 fire: {len(results)}")
    beating = [r for r in results if r["beats"]]
    print(f"Variants BEATING breakeven (WR>2e): {len(beating)}")
    cand = [r for r in beating if r["fires"] >= MIN_FIRES]
    print(f"Breakeven-beaters with fires>={MIN_FIRES}: {len(cand)}\n")

    def show(r, label):
        used = {k: v for k, v in r["thr"].items()
                if (k == "dip" and "dip" in r["sigset"]) or
                   (k == "ofi" and "ofi" in r["sigset"]) or
                   (k == "volc" and "volc" in r["sigset"]) or
                   (k == "slope" and "slopefade" in r["sigset"])}
        print(f"=== {label} ===")
        print(f"  sigset={r['sigset']} e_cap={r['e_cap']} persist={r['persist']}s thr={used}")
        print(f"  fires={r['fires']} cov={r['cov']:.1f}% WR={r['wr']:.1f}% be(2e)={r['be']:.1f}% "
              f"beats={r['beats']} avg_e={r['avg_e']:.3f}")
        print(f"  mean/fire={r['mean_fire']:+.4f} mean/mkt={r['mean_mkt']:+.4f}")
        print(f"  buckets fire={r['bf']} win={r['bw']}\n")

    best_mean = best_cov = None
    if cand:
        best_mean = max(cand, key=lambda r: r["mean_mkt"])
        best_cov = max(cand, key=lambda r: r["cov"])
        show(best_mean, "BEST per-market mean (beats be, fires>=6)")
        show(best_cov, "HIGHEST coverage (beats be, fires>=6)")
        # top 10 by mean_mkt
        print("Top 10 breakeven-beaters by mean/mkt:")
        for r in sorted(cand, key=lambda r: -r["mean_mkt"])[:10]:
            print(f"  mean/mkt={r['mean_mkt']:+.4f} cov={r['cov']:.1f}% WR={r['wr']:.1f}% "
                  f"be={r['be']:.1f}% f={r['fires']} e={r['e_cap']} X={r['persist']} "
                  f"sig={r['sigset']}")
        print()
    else:
        print("NO breakeven-beating variant with fires>=6.\n")

    if results:
        best_any = max(results, key=lambda r: r["mean_mkt"])
        show(best_any, "BEST per-market mean (ANY fires)")
        nonbeat = [r for r in results if not r["beats"] and r["fires"] >= 20]
        if nonbeat:
            closest = max(nonbeat, key=lambda r: r["wr"] - r["be"])
            print(f"Closest non-beater (fires>=20): WR-be={closest['wr']-closest['be']:+.1f}pts "
                  f"WR={closest['wr']:.1f} be={closest['be']:.1f} sig={closest['sigset']} "
                  f"e={closest['e_cap']} X={closest['persist']} cov={closest['cov']:.1f}%\n")

    chosen = best_mean if best_mean is not None else (
        max(results, key=lambda r: r["mean_mkt"]) if results else None)
    if chosen is not None:
        os.makedirs(os.path.dirname(OUT_COL), exist_ok=True)
        col = {slug: p for slug, p in chosen["per_market"] if slug is not None}
        with open(OUT_COL, "w") as f:
            json.dump(col, f)
        mm = sum(col.values()) / n
        print(f"Wrote chosen variant per-market column -> {OUT_COL} ({len(col)} mkts, mean/mkt={mm:+.4f})")
        used = {k: v for k, v in chosen["thr"].items()
                if (k == "dip" and "dip" in chosen["sigset"]) or
                   (k == "ofi" and "ofi" in chosen["sigset"]) or
                   (k == "volc" and "volc" in chosen["sigset"]) or
                   (k == "slope" and "slopefade" in chosen["sigset"])}
        print("\n=== SUMMARY (chosen) ===")
        print(json.dumps({
            "family": "combo_AND_persist",
            "signals": list(chosen["sigset"]),
            "logic": "ALL signals + ask<=e_cap, persisted X sec, first-trigger high->low",
            "e_cap": chosen["e_cap"], "persist_sec": chosen["persist"], "thresholds": used,
            "coverage_pct": round(chosen["cov"], 2), "wr": round(chosen["wr"], 2),
            "breakeven_wr": round(chosen["be"], 2), "beats_breakeven": chosen["beats"],
            "per_fire_mean": round(chosen["mean_fire"], 4),
            "per_market_mean": round(chosen["mean_mkt"], 4),
            "n_fires": chosen["fires"], "avg_entry": round(chosen["avg_e"], 4),
        }, indent=2))


if __name__ == "__main__":
    main()
