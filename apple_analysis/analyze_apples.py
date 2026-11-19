#!/usr/bin/env python3
"""
analyze_apples.py - analysis for the COINs Team 2 apple experiment
===================================================================

Question: does the signal of a HEALTHY apple change when a ROTTEN apple is
added next to it, compared with adding a healthy apple (control)?

Design (per run): three channels recorded in parallel, each in its own bag.
  dummy    two electrode pads stuck together, sham "addition"
  control  healthy apple alone, then a healthy neighbour is added
  test     healthy apple alone, then a rotten neighbour is added
Phase 1 = before the neighbour is added, phase 2 = after (48 h + 48 h).

What this script does
  1. Reads the 5-minute WAV chunks written by the Biolingo server.
  2. Puts every channel on a common 5-minute time grid.
  3. For each 5-minute bin computes band power (Welch PSD) and drops bins
     that are saturated (signal pinned at the amplifier rails) or have gaps.
  4. Subtracts the dummy (log10 band power of the apple minus log10 band
     power of the dummy in the same bin).
  5. Per run: change = mean(after) - mean(before), for control and test.
     Effect of the rotten neighbour = change(test) - change(control).
  6. Across runs: PRIMARY test = one-sample t-test of that difference
     against 0 (two-sided, alpha 0.05). The run is the unit, not the 5-min
     bin, because neighbouring bins are strongly autocorrelated.
  7. Robustness checks and exploratory band (2-5 Hz), plus a regression with
     sensor type as a control variable (Peter's advice).
  8. An exploratory sweep over other frequency bands (slow bands 0.01-0.5 Hz, then 0.5-45 Hz,
     and 55-95 Hz for 400 Hz data), corrected for looking at many bands at once (Holm).

The PRIMARY measure and test are fixed below. Decide them BEFORE recording
and do not change them after looking at data. Everything else is exploratory.

Input
  --data    folder with the server's files (<device>/<device>_<UTC>_<epoch>.wav)
  --design  CSV, one row per run x device (see design_example.csv)
  --events  optional CSV with a column "time" (UTC): bag openings, touches.
            Bins within --exclude-events-min of an event are dropped.

Output (in --out)
  windows.csv, runs_summary.csv, sweep.csv, dropped.csv, report.txt, run_<id>.png,
  spectrogram_run_<id>.png, summary.png, sweep.png
"""
import argparse
import re
import sys
import wave
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.ndimage import uniform_filter1d

# ----- PRIMARY measure and test: fix before recording ----------------------
PRIMARY_BAND = (5.0, 20.0)   # Hz, band from Peter's pilot. Peter says band choice is trial and error:
                             # pick it from the test run / pilot files, then LOCK it (fill in the next line).
PRIMARY_LOCKED_ON = "not locked yet"   # e.g. "2026-12-03, chosen from test-run files, before real runs"
EXTRA_BAND = (2.0, 5.0)      # Hz, exploratory (best band in the pilot)
ALPHA = 0.05
# Exploratory frequency sweep (Peter's pilot bands, then 55-95 Hz for 400 Hz data).
# The 50 Hz mains line and its harmonics are cut out of every band (see MAINS_EXCL_HZ).
# Slow bands (Peter: "slow bands are sometimes more interesting"; he mentioned one change every ~44 s = 0.023 Hz).
# Anything with its upper edge at or below SLOW_MAX_HZ goes through the slow path (long Welch segments, no
# high-pass), see bin_features(). 0.01-0.03 Hz is a 33-100 s rhythm and contains 0.023 Hz.
SLOW_MAX_HZ = 0.5
SLOW_BANDS = [(0.01, 0.03), (0.03, 0.1), (0.1, 0.5)]
FAST_BANDS = [(0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 45.0), (55.0, 95.0)]
SWEEP_BANDS = SLOW_BANDS + FAST_BANDS
MAINS_EXCL_HZ = 1.5          # drop +/- this many Hz around 50, 100, 150 ... Hz
# ---------------------------------------------------------------------------

FILE_RE = re.compile(r"^(?P<device>.+)_(?P<stamp>\d{8}T\d{6}Z)_(?P<epoch>\d+)\.wav$")
RAIL_HI = 32767     # int16 clamp: 3300 mV x10 does not fit, so a railed-high reading is 32767
RAIL_LO = 16        # <= 1.6 mV x10: within 2 ADC counts of 0


def band_name(b):
    return f"{b[0]:g}-{b[1]:g}"


ALL_BANDS = list(dict.fromkeys([PRIMARY_BAND, EXTRA_BAND] + SWEEP_BANDS))


def is_slow(b):
    return b[1] <= SLOW_MAX_HZ


# ------------------------------------------------------------------ reading
def find_files(data_dir):
    """device -> sorted list of (start_epoch, path)."""
    out = {}
    for p in Path(data_dir).rglob("*.wav"):
        m = FILE_RE.match(p.name)
        if not m:
            continue
        out.setdefault(m["device"], []).append((float(m["epoch"]), p))
    return {d: sorted(v) for d, v in out.items()}


def read_wav(path):
    """Returns (sample_rate, int16 array). Units: mV x 10."""
    with wave.open(str(path), "rb") as w:
        fs = w.getframerate()
        ch = w.getnchannels()
        if w.getsampwidth() != 2:
            raise ValueError(f"{path.name}: expected 16-bit WAV")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    if ch > 1:
        x = x[::ch]
    return fs, x


# ---------------------------------------------------------------- features
def bin_features(t0, arr, fs, a):
    """Band powers of one 5-minute bin. arr is float32 with NaN for missing samples."""
    obs = ~np.isnan(arr)
    cov = float(obs.mean())
    out = {"t_epoch": t0, "fs": fs, "coverage": cov, "sat_frac": np.nan,
           "std_mv": np.nan, "flag": "ok"}
    for b in ALL_BANDS:
        out["p_" + band_name(b)] = np.nan
    if cov < a.min_coverage or obs.sum() < 10 * fs:
        out["flag"] = "coverage"
        return out
    v = arr[obs]
    sat = float(np.mean((v >= RAIL_HI) | (v <= RAIL_LO)))
    out["sat_frac"] = sat
    if sat > a.max_sat:
        out["flag"] = "saturated"
        return out
    idx = np.arange(len(arr))
    x = np.interp(idx, idx[obs], arr[obs]) / 10.0          # mV
    x = x - x.mean()
    x_slow = x.copy()                                       # slow path: no filtering at all (see below)
    if a.box_notch and fs >= 200 and round(fs) % 50 == 0:   # optional: moving average over one 50 Hz period
        x = uniform_filter1d(x, int(round(fs / 50)), mode="nearest")
    if a.highpass > 0:
        sos = signal.butter(4, a.highpass, "highpass", fs=fs, output="sos")
        x = signal.sosfiltfilt(sos, x)
    nper = int(round(a.segment_s * fs))
    f, p = signal.welch(x, fs=fs, window="hann", nperseg=nper,
                        noverlap=nper // 2, detrend="constant")
    df = float(f[1] - f[0])
    near_mains = np.zeros(len(f), bool)                     # cut the mains line and harmonics out of every band
    for k in range(1, int(f[-1] // 50) + 1):
        near_mains |= np.abs(f - 50.0 * k) <= MAINS_EXCL_HZ
    for b in ALL_BANDS:
        lo, hi = b
        if is_slow(b):
            continue                                        # handled by the slow path below
        if hi <= fs / 2:
            m = (f >= lo) & (f <= hi) & ~near_mains
            if m.sum() >= 2:
                out["p_" + band_name(b)] = float(p[m].sum() * df)   # mV^2
    # slow bands: one long Welch segment length (default 150 s -> 3 half-overlapping segments in a 5-min bin,
    # frequency steps of ~0.007 Hz), linear detrend instead of a high-pass. A high-pass at 0.5 Hz would delete
    # these bands, and a filter near 0.01 Hz would ring at the edges of a 5-minute bin.
    if any(is_slow(b) for b in ALL_BANDS):
        nslow = int(round(a.slow_segment_s * fs))
        if nslow <= len(x_slow):
            fs_, ps = signal.welch(x_slow, fs=fs, window="hann", nperseg=nslow,
                                   noverlap=nslow // 2, detrend="linear")
            dfs = float(fs_[1] - fs_[0])
            for b in ALL_BANDS:
                if is_slow(b):
                    m = (fs_ >= b[0]) & (fs_ <= b[1])
                    if m.sum() >= 2:
                        out["p_" + band_name(b)] = float(ps[m].sum() * dfs)
    out["std_mv"] = float(x.std())
    out["_psd"] = p.astype(np.float32)      # kept for the spectrograms
    out["_f"] = f.astype(np.float32)
    return out


def load_device(files, a):
    """All 5-minute bins of one device, aligned to a common epoch grid."""
    rows, buf, spectra, hold = [], {}, {}, {}
    L = lambda fs: int(round(a.bin_seconds * fs))

    def finalize(b):
        arr, fs = buf.pop(b)
        row = bin_features(b * a.bin_seconds, arr, fs, a)
        psd, fr = row.pop("_psd", None), row.pop("_f", None)
        if psd is not None:
            spectra[row["t_epoch"]] = psd
            hold.setdefault("f", fr)
        rows.append(row)

    for epoch, path in files:
        fs, x = read_wav(path)
        n = len(x)
        b0 = int(np.floor(epoch / a.bin_seconds))
        for b in [k for k in buf if k < b0]:   # nothing later can reach these bins
            finalize(b)
        i = 0
        while i < n:
            t = epoch + i / fs
            b = int(np.floor(t / a.bin_seconds))
            j = min(n, int(np.ceil(((b + 1) * a.bin_seconds - epoch) * fs)))
            j = max(j, i + 1)
            off = int(round((t - b * a.bin_seconds) * fs))
            if b not in buf:
                buf[b] = [np.full(L(fs), np.nan, np.float32), fs]
            arr, fs_b = buf[b]
            if fs_b == fs:
                seg = x[i:j]
                end = min(len(arr), off + len(seg))
                if end > off:
                    arr[off:end] = seg[: end - off]
            i = j
    for b in sorted(buf):
        finalize(b)
    return pd.DataFrame(rows), spectra, hold.get("f")


# ------------------------------------------------------------------- design
def to_epoch(s):
    ts = pd.to_datetime(s, utc=True)
    return (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()


def read_design(path):
    d = pd.read_csv(path)
    need = {"run", "device", "condition", "sensor_type", "start_utc", "neighbour_utc"}
    missing = need - set(d.columns)
    if missing:
        sys.exit(f"design file is missing columns: {sorted(missing)}")
    d["condition"] = d["condition"].str.strip().str.lower()
    bad = set(d["condition"]) - {"dummy", "control", "test"}
    if bad:
        sys.exit(f"condition must be dummy, control or test; got {sorted(bad)}")
    d["start"] = to_epoch(d["start_utc"])
    d["nb"] = to_epoch(d["neighbour_utc"])
    if "end_utc" in d.columns and d["end_utc"].notna().any():
        d["end"] = to_epoch(d["end_utc"])
    else:
        d["end"] = np.nan
    d["end"] = d["end"].fillna(d["nb"] + (d["nb"] - d["start"]))   # default: as long after as before
    return d


# ---------------------------------------------------------------- statistics
def run_level(vals):
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    n = len(v)
    r = {"n": n, "mean": np.nan, "sd": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
         "t": np.nan, "p": np.nan, "dz": np.nan}
    if n == 0:
        return r
    r["mean"] = float(v.mean())
    if n >= 2:
        sd = float(v.std(ddof=1))
        se = sd / np.sqrt(n)
        tc = stats.t.ppf(0.975, n - 1)
        r.update(sd=sd, ci_lo=r["mean"] - tc * se, ci_hi=r["mean"] + tc * se)
        if sd > 0:
            t, p = stats.ttest_1samp(v, 0.0)
            r.update(t=float(t), p=float(p), dz=r["mean"] / sd)
    return r


def phase_stats(g, col):
    """Mean before/after and change for one run x channel."""
    b = g.loc[g.phase == "before", col].dropna()
    af = g.loc[g.phase == "after", col].dropna()
    r = {"n_before": len(b), "n_after": len(af), "delta": np.nan,
         "delta_wins": np.nan, "delta_h1": np.nan, "delta_h2": np.nan}
    if len(b) < 12 or len(af) < 12:      # need at least an hour of bins on each side
        return r
    r["delta"] = af.mean() - b.mean()
    lo, hi = pd.concat([b, af]).quantile([0.01, 0.99])
    r["delta_wins"] = af.clip(lo, hi).mean() - b.clip(lo, hi).mean()
    half = len(af) // 2
    r["delta_h1"] = af.iloc[:half].mean() - b.mean()
    r["delta_h2"] = af.iloc[half:].mean() - b.mean()
    return r


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (keeps NaN as NaN)."""
    p = np.asarray(pvals, float)
    adj = np.full(len(p), np.nan)
    ok = np.where(~np.isnan(p))[0]
    order = ok[np.argsort(p[ok])]
    m = len(order)
    run_max = 0.0
    for rank, i in enumerate(order):
        run_max = max(run_max, (m - rank) * p[i])
        adj[i] = min(1.0, run_max)
    return adj


def ols(X, y, names):
    n, k = X.shape
    if n <= k:
        return None
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = float(res @ res) / (n - k)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), n - k)
    return pd.DataFrame({"coef": beta, "se": se, "t": t, "p": p}, index=names).assign(df=n - k)


# ------------------------------------------------------------- spectrograms
def plot_spectrogram(run, drows, spectra, freqs, a, out, plt):
    """Every frequency over time, one panel per channel. White = dropped or missing."""
    order = {"dummy": 0, "control": 1, "test": 2}
    drows = sorted(drows, key=lambda r: order.get(r["condition"], 9))
    mats = []
    for r in drows:
        dev = r["device"]
        f, sp = freqs.get(dev), spectra.get(dev, {})
        if f is None or not sp:
            mats.append(None)
            continue
        idx0 = int(np.floor(r["start"] / a.bin_seconds))
        n = int(np.floor((r["end"] - 1) / a.bin_seconds)) - idx0 + 1
        keep = f >= 0.5
        fk = f[keep]
        M = np.full((len(fk), n), np.nan)
        for t0, psd in sp.items():
            j = int(round(t0 / a.bin_seconds)) - idx0
            if 0 <= j < n:
                M[:, j] = 10 * np.log10(np.maximum(psd[keep], 1e-12))
        near = np.zeros(len(fk), bool)                 # hide the mains lines
        for k in range(1, int(fk[-1] // 50) + 1):
            near |= np.abs(fk - 50.0 * k) <= MAINS_EXCL_HZ
        M[near, :] = np.nan
        mats.append((M, fk, idx0, n))
    good = [m[0][np.isfinite(m[0])] for m in mats if m is not None]
    vals = np.concatenate(good) if good else np.array([])
    if vals.size == 0:
        return
    vmin, vmax = np.percentile(vals, [2, 98])
    fig, axes = plt.subplots(len(drows), 1, figsize=(10, 2.7 * len(drows)), sharex=True, squeeze=False)
    im = None
    for ax, m, r in zip(axes[:, 0], mats, drows):
        ax.set_title(f"{r['condition']} ({r['device']})", fontsize=9, loc="left")
        if m is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        M, fk, idx0, n = m
        xe = ((np.arange(n + 1) + idx0) * a.bin_seconds - r["start"]) / 3600
        d = fk[1] - fk[0]
        ye = np.append(fk - d / 2, fk[-1] + d / 2)
        im = ax.pcolormesh(xe, ye, M, vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_yscale("log")
        ax.set_ylim(0.5, min(fk[-1], 100))
        ax.set_ylabel("Hz")
        ax.axvline((r["nb"] - r["start"]) / 3600, color="white", ls="--", lw=1)
    axes[-1, 0].set_xlabel("hours since start (dashed line = neighbour added)")
    fig.suptitle(f"Run {run}: every frequency over time (white = dropped or missing)", fontsize=10)
    if im is not None:
        fig.colorbar(im, ax=axes[:, 0].tolist(), label="power (dB)", pad=0.01)
    fig.savefig(out / f"spectrogram_run_{run}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--events", default=None)
    ap.add_argument("--bin-seconds", type=int, default=300)
    ap.add_argument("--segment-s", type=float, default=4.0, help="Welch segment length (s)")
    ap.add_argument("--highpass", type=float, default=0.5,
                    help="high-pass Hz for the bands from 0.5 Hz up, 0 = off (slow bands are never high-passed)")
    ap.add_argument("--slow-segment-s", type=float, default=150.0,
                    help="Welch segment length (s) for the slow bands below 0.5 Hz; must be at most half the bin "
                         "so that at least 3 segments average. Lowest usable frequency is about 2 / this value.")
    ap.add_argument("--box-notch", dest="box_notch", action="store_true",
                    help="also smooth 400 Hz data with a moving average over one 50 Hz period (old behaviour). "
                         "Off by default: the mains line is cut out in the frequency domain instead.")
    ap.add_argument("--min-coverage", type=float, default=0.95)
    ap.add_argument("--max-sat", type=float, default=0.01, help="drop bins with more saturated samples")
    ap.add_argument("--skip-start-hours", type=float, default=2.0, help="settling time after start")
    ap.add_argument("--exclude-event-hours", type=float, default=1.0, help="around the neighbour being added")
    ap.add_argument("--exclude-events-min", type=float, default=15.0, help="around logged touches")
    a = ap.parse_args()
    if a.slow_segment_s > a.bin_seconds / 2:
        print(f"WARNING: --slow-segment-s {a.slow_segment_s:g} is more than half of --bin-seconds {a.bin_seconds}: "
              "slow-band estimates will average fewer than 3 segments and be very noisy.", file=sys.stderr)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    design = read_design(a.design)
    files = find_files(a.data)
    events = []
    if a.events:
        ev = pd.read_csv(a.events)
        events = list(to_epoch(ev["time"]))

    # 1. features per device
    cache, spectra, freqs = {}, {}, {}
    for dev in sorted(design["device"].unique()):
        if dev not in files:
            print(f"WARNING: no files found for device '{dev}'", file=sys.stderr)
            continue
        print(f"reading {dev}: {len(files[dev])} files")
        cache[dev], spectra[dev], freqs[dev] = load_device(files[dev], a)

    # 2. windows table per run
    parts = []
    for _, r in design.iterrows():
        if r["device"] not in cache:
            continue
        f = cache[r["device"]]
        f = f[(f.t_epoch >= r["start"]) & (f.t_epoch + a.bin_seconds <= r["end"])].copy()
        f["run"], f["device"] = r["run"], r["device"]
        f["condition"], f["sensor_type"] = r["condition"], r["sensor_type"]
        f["hours"] = (f.t_epoch - r["start"]) / 3600
        ex = a.exclude_event_hours * 3600
        t_end = f.t_epoch + a.bin_seconds
        ph = np.full(len(f), "excluded", dtype=object)
        ph[((f.t_epoch >= r["start"] + a.skip_start_hours * 3600) & (t_end <= r["nb"] - ex)).to_numpy()] = "before"
        ph[((f.t_epoch >= r["nb"] + ex) & (t_end <= r["end"])).to_numpy()] = "after"
        for e in events:                       # logged bag openings or touches
            near = (f.t_epoch < e + a.exclude_events_min * 60) & (t_end > e - a.exclude_events_min * 60)
            ph[near.to_numpy()] = "excluded"
        f["phase"] = ph
        f.loc[f.flag != "ok", "phase"] = "excluded"
        parts.append(f)
    if not parts:
        sys.exit("no data matched the design file")
    W = pd.concat(parts, ignore_index=True)
    names = [band_name(b) for b in ALL_BANDS]
    for n in names:
        W["log_" + n] = np.log10(W["p_" + n].where(W["p_" + n] > 0))

    # 3. subtract the dummy bin by bin
    dcols = {"log_" + n: "d_" + n for n in names}
    dummy = W[W.condition == "dummy"][["run", "t_epoch"] + list(dcols)].rename(columns=dcols)
    A = W[W.condition != "dummy"].merge(dummy, on=["run", "t_epoch"], how="left")
    for n in names:
        A["y_" + n] = A["log_" + n] - A["d_" + n]
    W = W.merge(A[["run", "device", "t_epoch"] + ["y_" + n for n in names]],
                on=["run", "device", "t_epoch"], how="left")
    for key, b in (("main", PRIMARY_BAND), ("extra", EXTRA_BAND)):   # short names used below
        W["log_" + key] = W["log_" + band_name(b)]
        W["y_" + key] = W["y_" + band_name(b)]
    W.to_csv(out / "windows.csv", index=False)

    # data used and dropped, per run and channel
    drop_rows = []
    for _, r in design.iterrows():
        g = W[(W.run == r["run"]) & (W.device == r["device"])]
        ok = g[g.flag == "ok"]
        exp = int((r["end"] - r["start"]) // a.bin_seconds)
        drop_rows.append({
            "run": r["run"], "channel": r["condition"], "device": r["device"],
            "expected_bins": exp, "never_recorded": max(exp - len(g), 0),
            "dropped_gaps": int((g.flag == "coverage").sum()),
            "dropped_saturated": int((g.flag == "saturated").sum()),
            "left_out_by_design": int((ok.phase == "excluded").sum()),
            "used_before": int((ok.phase == "before").sum()),
            "used_after": int((ok.phase == "after").sum())})
    D = pd.DataFrame(drop_rows)
    exp_ = D.expected_bins.replace(0, np.nan)
    D["share_clean"] = ((D.expected_bins - D.never_recorded - D.dropped_gaps - D.dropped_saturated) / exp_).round(2)
    D["share_used"] = ((D.used_before + D.used_after) / exp_).round(2)
    D.to_csv(out / "dropped.csv", index=False)

    # 4. per run summary
    rows = []
    for run, g in W.groupby("run", sort=True):
        rec = {"run": run}
        for cond in ("control", "test"):
            gc = g[g.condition == cond]
            rec[cond + "_device"] = gc["device"].iloc[0] if len(gc) else ""
            rec[cond + "_sensor"] = gc["sensor_type"].iloc[0] if len(gc) else ""
        for band in ("main", "extra"):
            res = {}
            for cond in ("control", "test"):
                gc = g[g.condition == cond].sort_values("t_epoch")
                res[cond] = phase_stats(gc, "y_" + band)
                rec[f"{cond}_delta_{band}"] = res[cond]["delta"]
                rec[f"{cond}_n_{band}"] = res[cond]["n_before"] + res[cond]["n_after"]
                rec[f"{cond}_raw_delta_{band}"] = phase_stats(gc, "log_" + band)["delta"]
            gd = g[g.condition == "dummy"].sort_values("t_epoch")
            rec[f"dummy_delta_{band}"] = phase_stats(gd, "log_" + band)["delta"]
            for k in ("delta", "delta_wins", "delta_h1", "delta_h2"):
                rec[f"did_{k}_{band}" if k != "delta" else f"did_{band}"] = res["test"][k] - res["control"][k]
            rec[f"did_raw_{band}"] = rec[f"test_raw_delta_{band}"] - rec[f"control_raw_delta_{band}"]
        for b in SWEEP_BANDS:
            n = band_name(b)
            rc = phase_stats(g[g.condition == "control"].sort_values("t_epoch"), "y_" + n)["delta"]
            rt = phase_stats(g[g.condition == "test"].sort_values("t_epoch"), "y_" + n)["delta"]
            rec["sweep_control_" + n], rec["sweep_test_" + n], rec["sweep_did_" + n] = rc, rt, rt - rc
        rows.append(rec)
    S = pd.DataFrame(rows)
    S.to_csv(out / "runs_summary.csv", index=False)

    # 5. report
    lines = []
    P = lines.append
    P("COINs Team 2 apple experiment: analysis report")
    P("=" * 60)
    P(f"Runs analysed: {len(S)}   bin = {a.bin_seconds}s   Welch segment = {a.segment_s}s   high-pass = {a.highpass} Hz"
      f"   slow-band segment = {a.slow_segment_s:g}s")
    P("")
    P("DATA USED AND DROPPED (5-minute bins, per run and channel; also in dropped.csv)")
    P("  never_recorded = no file covered the bin; dropped_gaps = more than "
      f"{(1 - a.min_coverage) * 100:g}% of samples missing; dropped_saturated = more than "
      f"{a.max_sat * 100:g}% of readings at the rails;")
    P("  left_out_by_design = skipped on purpose (settling time, around the neighbour being added, near logged touches).")
    P(D.to_string(index=False))
    P("  share_clean = recorded and passed the quality checks; share_used = actually used in the statistics.")
    P("  A channel with a share_clean under about 0.9 deserves a look before you trust its numbers.")
    P("")
    P("All effects are on log10 band power, apple minus dummy. A value of 0.30 means "
      "about 2x the power; -0.30 about half.")
    P("")
    prim = run_level(S["did_main"])
    P(f"PRIMARY (confirmatory): band {PRIMARY_BAND[0]:g}-{PRIMARY_BAND[1]:g} Hz   [locked: {PRIMARY_LOCKED_ON}]")
    if PRIMARY_LOCKED_ON == "not locked yet":
        P("  WARNING: the primary band is not locked. A band picked after seeing the real runs makes this p-value meaningless.")
    P("  effect of a rotten neighbour = change(test) - change(control), per run")
    for _, r in S.iterrows():
        P(f"  run {r['run']}:  control {r['control_delta_main']:+.3f}   test {r['test_delta_main']:+.3f}"
          f"   difference {r['did_main']:+.3f}   (control sensor {r['control_sensor']}, test sensor {r['test_sensor']})")
    P(f"  runs n = {prim['n']}   mean difference = {prim['mean']:+.3f} (x{10 ** prim['mean']:.2f} in power)")
    if prim["n"] >= 2:
        P(f"  95% CI [{prim['ci_lo']:+.3f}, {prim['ci_hi']:+.3f}]   t = {prim['t']:.2f}   "
          f"p = {prim['p']:.3f} (two-sided, alpha {ALPHA})   effect size d = {prim['dz']:.2f}")
        verdict = "significant" if prim["p"] < ALPHA else "not significant"
        P(f"  Result: {verdict} at alpha {ALPHA}.")
    else:
        P("  Not enough runs for a test (need at least 2).")
    P(f"  Note: with n = {prim['n']} runs this test has little power and the t-test relies on "
      "normality. Report the per-run values and the effect size, not only p.")
    P("")
    P("ROBUSTNESS (same primary band)")
    for label, col in (("with extreme values clipped (1st-99th pct)", "did_delta_wins_main"),
                       ("first half of the after phase only", "did_delta_h1_main"),
                       ("second half of the after phase only", "did_delta_h2_main"),
                       ("WITHOUT subtracting the dummy", "did_raw_main")):
        s = run_level(S[col])
        P(f"  {label}: mean {s['mean']:+.3f}, runs with positive sign "
          f"{int((S[col] > 0).sum())}/{int(S[col].notna().sum())}")
    d = run_level(S["dummy_delta_main"])
    P(f"  dummy channel change (should be near 0; checks the event itself): mean {d['mean']:+.3f}")
    c = run_level(S["control_delta_main"])
    P(f"  control change (adding a healthy neighbour): mean {c['mean']:+.3f}")
    P("")
    ex = run_level(S["did_extra"])
    P(f"EXPLORATORY: band {EXTRA_BAND[0]:g}-{EXTRA_BAND[1]:g} Hz   mean difference {ex['mean']:+.3f}"
      + (f", 95% CI [{ex['ci_lo']:+.3f}, {ex['ci_hi']:+.3f}], p = {ex['p']:.3f}" if ex["n"] >= 2 else ""))
    P("  Exploratory results are reported as 'interesting, needs follow-up', not as findings.")
    P("")
    # frequency sweep (exploratory)
    sw = []
    for b in SWEEP_BANDS:
        n = band_name(b)
        col = "sweep_did_" + n
        s = run_level(S[col])
        sw.append({"band_hz": n, "runs": s["n"], "mean_diff": s["mean"], "power_ratio": 10 ** s["mean"] if s["n"] else np.nan,
                   "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"], "p": s["p"],
                   "runs_positive": int((S[col] > 0).sum())})
    SW = pd.DataFrame(sw)
    SW["p_holm"] = holm(SW["p"])
    SW.to_csv(out / "sweep.csv", index=False)
    P("EXPLORATORY FREQUENCY SWEEP (all bands use the same steps; mains lines at 50/100/150 Hz are cut out)")
    P("  mean difference = change(test) - change(control), log10 power; p_holm corrects for looking at "
      f"{int(SW['p'].notna().sum())} bands at once.")
    P(SW.assign(mean_diff=SW.mean_diff.round(3), power_ratio=SW.power_ratio.round(2), ci_lo=SW.ci_lo.round(3),
                ci_hi=SW.ci_hi.round(3), p=SW.p.round(3), p_holm=SW.p_holm.round(3))
      .to_string(index=False))
    P("  Bands above 45 Hz exist only for 400 Hz data. Treat anything here as 'interesting, needs follow-up'.")
    P("  Bands below 0.5 Hz (slow bands) use long segments and no high-pass. Check two things before believing them: (1) the")
    P("  dummy channel (it should show no change there), (2) temperature and humidity, which drift slowly too and are the")
    P("  most likely false explanation for a slow-band change. Also check that the board itself passes these frequencies.")
    P("")
    # sensor type as a control variable
    recs = []
    for _, r in S.iterrows():
        for cond in ("control", "test"):
            recs.append({"run": r["run"], "is_test": int(cond == "test"),
                         "sensor": r[cond + "_sensor"], "delta": r[cond + "_delta_main"]})
    R = pd.DataFrame(recs).dropna()
    P("SENSOR TYPE AS A CONTROL VARIABLE (apple channels, change in log10 power)")
    if len(R) and R["sensor"].nunique() > 0:
        types = sorted(R["sensor"].unique())
        X = np.column_stack([np.ones(len(R)), R["is_test"].to_numpy()] +
                            [(R["sensor"] == t).astype(float).to_numpy() for t in types[1:]])
        names = ["intercept", "test vs control"] + [f"sensor {t} vs {types[0]}" for t in types[1:]]
        fit = ols(X, R["delta"].to_numpy(), names)
        if fit is None:
            P(f"  Too few data points ({len(R)}) for this many terms. Needs more runs.")
        else:
            P(fit.round(3).to_string())
            P("  With few runs these estimates are very uncertain. If the sensors were rotated across "
              "conditions, 'test vs control' is the effect after allowing for sensor type.")
    else:
        P("  not available")
    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # 6. figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping figures")
        return
    for run, g in W.groupby("run", sort=True):
        fig, ax = plt.subplots(figsize=(9, 4))
        row = design[design.run == run].iloc[0]
        for cond, colr in (("control", "#2E75B6"), ("test", "#ED7D31")):
            gc = g[(g.condition == cond) & (g.flag == "ok")].sort_values("hours")
            if gc.empty:
                continue
            ax.plot(gc["hours"], gc["y_main"], ".", ms=2.5, color=colr, alpha=0.35)
            ax.plot(gc["hours"], gc["y_main"].rolling(12, min_periods=6, center=True).median(),
                    "-", color=colr, lw=2, label=f"{cond} (1 h median)")
        ax.axvline((row["nb"] - row["start"]) / 3600, color="k", ls="--", lw=1, label="neighbour added")
        ax.set_xlabel("hours since start")
        ax.set_ylabel("log10 band power, apple minus dummy")
        ax.set_title(f"Run {run}: {PRIMARY_BAND[0]:g}-{PRIMARY_BAND[1]:g} Hz")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"run_{run}.png", dpi=130)
        plt.close(fig)
    for run in sorted(design["run"].unique()):
        plot_spectrogram(run, [r for _, r in design[design.run == run].iterrows()],
                         spectra, freqs, a, out, plt)
    fig, ax = plt.subplots(figsize=(9, 4))
    ok = SW[SW["runs"] > 0]
    xpos = np.arange(len(ok))
    err = np.vstack([ok["mean_diff"] - ok["ci_lo"], ok["ci_hi"] - ok["mean_diff"]]).clip(min=0)
    ax.bar(xpos, ok["mean_diff"], color="#9DC3E6", yerr=err if len(ok) and ok["ci_lo"].notna().all() else None, capsize=3)
    for i, (_, r) in enumerate(ok.iterrows()):
        vals = S["sweep_did_" + r["band_hz"]].dropna()
        ax.plot([i] * len(vals), vals, "o", color="#1F3864", ms=4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(list(ok["band_hz"]))
    ax.set_xlabel("frequency band (Hz)")
    ax.set_ylabel("test - control change (log10 power)")
    ax.set_title("Exploratory sweep: bars = mean over runs, dots = single runs")
    fig.tight_layout()
    fig.savefig(out / "sweep.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = {"control": 0, "test": 1, "test - control": 2}
    for _, r in S.iterrows():
        ax.plot([0, 1, 2], [r["control_delta_main"], r["test_delta_main"], r["did_main"]],
                "o-", alpha=0.7, label=f"run {r['run']}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(list(xs))
    ax.set_ylabel("change in log10 band power (after - before)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
