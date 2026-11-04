# Apple experiment: analysis code

Question: does a healthy apple's signal change when a rotten apple is added next to it,
compared with adding a healthy apple? Three channels run in parallel (dummy, control, test),
48 h before and 48 h after the neighbour is added.

## Files
- `analyze_apples.py`: the analysis.
- `make_demo_data.py`: makes fake data in the server's format, to try the analysis first.
- `design_example.csv`: template for the design file (example values only).

## Set up
```
pip install numpy scipy pandas matplotlib
```

## Try it with fake data
```
python make_demo_data.py --out demo_data --runs 3 --hours 6 --fs 100
python analyze_apples.py --data demo_data --design demo_data/design.csv --out demo_results
```
The fake test channel gets a real effect of 1.5x amplitude (about 2.25x power, 0.35 in log10).
The report should show a difference near +0.35. Use `--effect 1.0` for no effect, or
`--effect-band 10 20` to put the effect in one frequency band only and see the sweep pick it out.
(Use at least 6 hours per phase for the demo: the script skips 2 h after the start and 1 h around the event.)

## Use it on the real data
1. Copy the server's data folder (`<device>/<device>_<UTC>_<epoch>.wav`) to your computer.
2. Copy `design_example.csv` to `design.csv` and fill in one row per run and device:
   - `condition` is `dummy`, `control` or `test`
   - `sensor_type` is `biolingo` or `breadboard`
   - times are UTC. Cologne winter time is UTC+1, so 15:00 in Cologne is `14:00:00Z`.
     `end_utc` is optional (default: as long after as before).
3. Optional: `events.csv` with a column `time` (UTC) listing every bag opening or touch.
4. Run:
```
python analyze_apples.py --data path/to/data --design design.csv --events events.csv --out results
```

## What you get
- `report.txt`: the primary test, robustness checks, the exploratory band, the frequency sweep and the sensor-type regression
- `sweep.csv` and `sweep.png`: the exploratory frequency sweep (below)
- `runs_summary.csv`: one row per run
- `windows.csv`: every 5-minute bin (flags show dropped bins)
- `run_<id>.png` and `summary.png`: plots

## How to read it
- Effects are on log10 band power, apple minus dummy. 0.30 is about 2x the power.
- **Primary (confirmatory) test:** change(test) minus change(control), per run, then a one-sample
  t-test across runs. The band (5-20 Hz) and the test are fixed at the top of the script.
  Change them before recording, never after.
- Everything else (2-5 Hz band, the frequency sweep, robustness, regression) is exploratory.
- **Frequency sweep:** the boards record raw at 400 Hz, so the files hold much more than 5-20 Hz.
  The sweep repeats the same steps for 0.5-1, 1-2, 2-5, 5-10, 10-20, 20-45 and (400 Hz data only)
  55-95 Hz. `p_holm` corrects for looking at many bands, so a lucky band does not look significant.
  Edit `SWEEP_BANDS` at the top of the script to change the list. The sweep never replaces the primary test.
- **Mains hum:** the 50 Hz line and its harmonics (50, 100, 150 Hz, plus or minus 1.5 Hz) are cut out of every
  band in the frequency domain. The old option of also smoothing 400 Hz data with a moving average is off by
  default; turn it on with `--box-notch`.
- **The run is the unit, not the 5-minute bin.** Neighbouring bins are strongly correlated, so
  treating them as independent would give false confidence.
- **With 3 runs the test has little power.** Report the per-run values and the effect size,
  not only the p-value.

## Things to check on the first real data
- Look at `windows.csv` for the `flag` column. Many `saturated` bins mean a bad electrode setup.
- Check the plots: the dummy-corrected signal should be steady before the event.
- The dummy's own change (in the report) should be near 0.
- Old files (before 23 Oct 2026) are 100 Hz and already filtered on the board. The script
  reads the sample rate from each file's header, so bands above 45 Hz are skipped for those files.
- The code was tested only on the fake data above (an effect in all bands, an effect in 10-20 Hz only,
  and no effect), not on real recordings.
