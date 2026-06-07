#!/usr/bin/env python3
"""Lean re-rank: schema breakeven (WR > 2*avg_entry) + robust mean (drop top-2 winners).
Stores ONLY scalar metrics per variant (no per_market lists) to stay memory-light."""
import itertools, sys
import numpy as np
import newstrat_combo_AND_persist as S

markets, arrs = S.load_arrays()
n = len(markets)
SIGNALS = S.SIGNALS
sigsets = []
for k in (2, 3):
    for c in itertools.combinations(SIGNALS, k):
        if "crowd" in c and "crowd_contra" in c:
            continue
        sigsets.append(c)

rows = []
truebe = []
done = 0
for sigset in sigsets:
    dip_g = S.THR_DIP if "dip" in sigset else [S.THR_DIP[0]]
    ofi_g = S.THR_OFI if "ofi" in sigset else [S.THR_OFI[0]]
    volc_g = S.THR_VOLC if "volc" in sigset else [S.THR_VOLC[0]]
    slope_g = S.THR_SLOPE if "slopefade" in sigset else [S.THR_SLOPE[0]]
    for e in S.E_CAPS:
        for X in S.PERSISTS:
            for td in dip_g:
                for to in ofi_g:
                    for tv in volc_g:
                        for ts in slope_g:
                            thr = {"dip": td, "ofi": to, "volc": tv, "slope": ts}
                            r = S.eval_variant(arrs, sigset, e, X, thr, n)
                            done += 1
                            if r["fires"] < 6:
                                continue
                            wr = 100.0 * r["wins"] / r["fires"]
                            avg_e = r["sum_e"] / r["fires"]
                            true_be = 100.0 * 2.0 * avg_e
                            pm = np.fromiter((p for sl, p in r["per_market"] if sl is not None),
                                             dtype=np.float64, count=n)
                            fired = pm[pm != 0]
                            top2 = float(np.sort(fired)[::-1][:2].sum()) if fired.size >= 2 else float(fired.max())
                            robust_mean = (pm.sum() - top2) / n
                            row = (pm.sum() / n, robust_mean, 100.0 * r["fires"] / n, wr, true_be,
                                   r["fires"], avg_e, e, X, sigset,
                                   {k: v for k, v in thr.items() if k in sigset})
                            rows.append(row)
                            if wr > true_be:
                                truebe.append(row)
    sys.stdout.write(f"...progress sigset={sigset} total_done={done}\n"); sys.stdout.flush()

def fmt(r):
    mm, rob, cov, wr, tbe, f, ae, e, X, sig, thr = r
    return (f"mean/mkt={mm:+.4f} robust={rob:+.4f} cov={cov:.1f}% WR={wr:.1f}% trueBE={tbe:.1f}% "
            f"f={f} avg_e={ae:.3f} e_cap={e} X={X} sig={sig} thr={thr}")

print(f"\nVariants fires>=6: {len(rows)}")
print(f"Variants with WR > 2*avg_entry (schema breakeven), fires>=6: {len(truebe)}")
if truebe:
    print("\nTop by mean/mkt among WR>2*avg_entry:")
    for r in sorted(truebe, key=lambda r: -r[0])[:15]:
        print("  " + fmt(r))
    print("\nHighest COVERAGE among WR>2*avg_entry:")
    for r in sorted(truebe, key=lambda r: -r[2])[:8]:
        print("  " + fmt(r))
    print("\nMost ROBUST among WR>2*avg_entry (drop top-2 winners):")
    for r in sorted(truebe, key=lambda r: -r[1])[:8]:
        print("  " + fmt(r))

print("\nTop 12 by ROBUST mean/mkt (drop top-2 winners) fires>=6:")
for r in sorted(rows, key=lambda r: -r[1])[:12]:
    print("  " + fmt(r))

print("\nTop 12 by raw mean/mkt fires>=6:")
for r in sorted(rows, key=lambda r: -r[0])[:12]:
    print("  " + fmt(r))

# closest non-beater with high coverage (>=70%)
nb = [r for r in rows if r[3] <= r[4] and r[2] >= 70]
if nb:
    print("\nClosest-to-breakeven among HIGH-COVERAGE (cov>=70%) non-beaters:")
    for r in sorted(nb, key=lambda r: (r[3] - r[4]))[-8:]:
        print(f"  gap(WR-trueBE)={r[3]-r[4]:+.1f}pts " + fmt(r))
