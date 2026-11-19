#!/usr/bin/env python3
"""
power_sim.py - how many runs do we need?

The main test is a one-sample t-test across runs on d = change(test) - change(control), one number per run
(log10 power). With few runs the test can only detect a big effect. This script shows, for 2-8 runs,
the smallest true effect that the test finds in 80% of experiments (two-sided, alpha 0.05).

It needs one number: how much d varies from run to run when there is NO effect (sd_d).
We do not know that until we have runs of our own. Two ways to get a first guess:
  1. from Peter's pilot windows (pilot_windows.csv from pilot_compare.py): the variation of a 24 h-vs-24 h
     change between two sensors. Rough: the pilot has no neighbour event and one apple per sensor.
  2. a grid of assumed values (0.05 ... 0.40).
Replace the guess with the real sd_d (the "sd" in runs_summary / report) after the first real runs.

  python power_sim.py [--pilot pilot_windows.csv] [--band 5-20]
"""
import argparse
import itertools
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--pilot", default=None)
ap.add_argument("--band", default="5-20")
ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--power", type=float, default=0.80)
a = ap.parse_args()


_rng = np.random.default_rng(1)
_Z = {}


def min_effect(n, sd, alpha=a.alpha, power=a.power, reps=40000):
    """Smallest true mean difference (in units of sd_d times sd) found with the wanted power.
    Simulation: draw n run-level differences, run the one-sample t-test, repeat. Same random draws for every
    effect size, so the power curve is smooth and a bisection finds the effect."""
    z = _Z.setdefault(n, _rng.standard_normal((reps, n)))
    tc = stats.t.ppf(1 - alpha / 2, n - 1)

    def pw(lam):
        x = z + lam
        t = x.mean(1) / (x.std(1, ddof=1) / np.sqrt(n))
        return float(np.mean(np.abs(t) > tc))
    lo, hi = 0.0, 200.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pw(mid) < power else (lo, mid)
    return hi * sd


sds = {}
if a.pilot:
    import pandas as pd
    W = pd.read_csv(a.pilot)
    W = W[W.flag == "ok"]
    col = "p_" + a.band
    deltas = {}
    for sen, g in W.groupby("sensor"):
        g = g.sort_values("t_epoch")
        t = (g.t_epoch.to_numpy() - g.t_epoch.min()) / 3600
        y = np.log10(g[col].where(g[col] > 0)).to_numpy()
        ds = []
        for t0 in np.arange(0, t.max() - 48, 1.0):
            b = y[(t >= t0) & (t < t0 + 24)]
            af = y[(t >= t0 + 24) & (t < t0 + 48)]
            ds.append(np.nanmean(af) - np.nanmean(b) if np.isfinite(b).sum() > 100 and np.isfinite(af).sum() > 100 else np.nan)
        deltas[sen] = np.array(ds)
    sd_pairs = []
    for s1, s2 in itertools.combinations(deltas, 2):
        d = deltas[s1] - deltas[s2]
        sd_pairs.append(np.nanstd(d[np.isfinite(d)], ddof=1))
    sds["pilot guess (median over sensor pairs)"] = float(np.median(sd_pairs))
    print(f"Pilot band {a.band} Hz: SD of the difference between two sensors' 24 h-vs-24 h change = "
          f"{', '.join(f'{x:.3f}' for x in sd_pairs)}  (pairs, log10)\n")
for v in (0.05, 0.10, 0.20, 0.30, 0.40):
    sds[f"assumed sd_d = {v:.2f}"] = v

ns = list(range(2, 9))
print(f"Smallest effect found with {a.power:.0%} power, two-sided alpha {a.alpha}: log10 units (and the power ratio 10^x)")
print(f"{'':42s}" + "".join(f"{f'{n} runs':>14s}" for n in ns))
for lab, sd in sds.items():
    row = [min_effect(n, sd) for n in ns]
    print(f"{lab:42s}" + "".join(f"{x:6.2f} (x{10 ** x:4.1f})".rjust(14) for x in row))
e3 = min_effect(3, 0.10)
print(f"\nReading it: with 3 runs and sd_d = 0.10, only a true change of about {e3:.2f} log10 (x{10 ** e3:.1f} in power) or more is "
      "found reliably. Smaller real effects will often be missed: 'not significant' then means 'not detected', not 'no effect'.")
print("2 runs is almost useless for a t-test (only 1 degree of freedom). The pilot-based sd is probably too optimistic: "
      "it has no apple-to-apple variation.")
print("Pilot reference: healthy vs damaged apples differed by about 0.55 log10 (x3.6) in 5-20 Hz and 0.88 (x7.6) in 2-5 Hz.")
