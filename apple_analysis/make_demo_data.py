#!/usr/bin/env python3
"""
make_demo_data.py - makes FAKE data in the same format as the Biolingo server,
so you can try analyze_apples.py before the real runs.

  python make_demo_data.py --out demo_data --runs 3 --hours 3 --fs 100
  python analyze_apples.py --data demo_data --design demo_data/design.csv --out demo_results

The fake apples get a real effect on purpose: in the test channel the signal gets
a chosen factor larger (default 1.5x in amplitude, about 2.25x in power) after the neighbour is added. If the analysis
reports a difference near +0.35 (log10), it works. The data also contains a missing
chunk, a saturated chunk, a 1-second timing jitter and 50 Hz hum (at 400 Hz).
"""
import argparse
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import signal

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="demo_data")
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--hours", type=float, default=3.0, help="hours per phase (before / after)")
ap.add_argument("--fs", type=int, default=100)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--effect", type=float, default=1.5, help="amplitude factor in the test channel after the neighbour is added (1.0 = no effect)")
ap.add_argument("--effect-band", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                help="put the effect ONLY in this band (Hz) instead of all frequencies")
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

out = Path(a.out)
BASE = 1_793_000_000                                   # an arbitrary start time (Nov 2026)
devices = ["biolingo-demo-A1", "biolingo-demo-A2", "biolingo-demo-B3"]
sensor = {"biolingo-demo-A1": "biolingo", "biolingo-demo-A2": "biolingo", "biolingo-demo-B3": "breadboard"}
gain = {"biolingo": 1.0, "breadboard": 1.3}             # different sensors read differently
conds = ["dummy", "control", "test"]
rows = []


def write_chunk(dev, epoch, mv):
    v = np.clip(np.round(mv * 10), -32768, 32767).astype("<i2")
    d = out / dev
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with wave.open(str(d / f"{dev}_{stamp}_{int(epoch)}.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(a.fs)
        w.writeframes(v.tobytes())


for r in range(a.runs):
    start = BASE + r * 86400
    nb = start + a.hours * 3600
    end = nb + a.hours * 3600
    for k, dev in enumerate(devices):
        cond = conds[(k - r) % 3]                       # rotate sensors across conditions
        rows.append((r + 1, dev, cond, sensor[dev],
                     datetime.fromtimestamp(start, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     datetime.fromtimestamp(nb, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
        t = start
        n_chunk = int((end - start) // 300)
        for c in range(n_chunk):
            after = t >= nb
            sigma = 0.4 if cond == "dummy" else 1.5     # mV
            extra_s = 0.0
            if cond == "test" and after:
                if a.effect_band:
                    lo, hi = a.effect_band
                    extra_s = sigma * gain[sensor[dev]] * np.sqrt(a.effect ** 2 - 1)
                else:
                    sigma *= a.effect
            sigma *= gain[sensor[dev]]
            n = 300 * a.fs
            mv = 1650 + rng.normal(0, sigma, n) + 3 * np.sin(2 * np.pi * np.arange(n) / (a.fs * 600))
            if extra_s > 0:                                # extra power only inside the chosen band
                sos = signal.butter(4, [a.effect_band[0], a.effect_band[1]], "bandpass", fs=a.fs, output="sos")
                mv += signal.sosfiltfilt(sos, rng.normal(0, 1, n)) * extra_s
            if a.fs >= 200:
                mv += 2 * np.sin(2 * np.pi * 50 * np.arange(n) / a.fs)   # mains hum
            if c == 7:                                   # one saturated chunk
                mv[:] = 3300
            if c == 11:                                  # one missing chunk
                t += 300
                continue
            write_chunk(dev, t, mv)
            t += 300 + (1 if c % 5 == 0 else 0)         # start times drift by a second now and then

import csv
with open(out / "design.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "device", "condition", "sensor_type", "start_utc", "neighbour_utc"])
    w.writerows(rows)
print(f"wrote demo data to {out}/ and {out}/design.csv")
