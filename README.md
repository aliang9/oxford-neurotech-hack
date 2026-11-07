# oxford-neurotech-hack

Live EEG and motion from a g.tec **BCI Core-8** (8 ch, 250 Hz, Bluetooth LE) on macOS, via
g.tec's [g.Pype](https://gpype.gtec.at/) SDK, plus a cue program and an offline decoder for
task experiments.

```
tasks.py ──UDP markers──┐
BCICore ────────────────┴─> Router ─┬─> HDF5 file (raw EEG + markers)        acquire.py --record
   │  [--oscar artifact removal]    └─> 50 Hz notch ─> 1–40 Hz bandpass ─┬─> live scope
   │                                                                     └─> LSL "core8_eeg"
   └─ accelerometer / gyroscope ─────────────────────────────────────────────> LSL "core8_motion"

calibrate.py / tasks.py check:  LSL "core8_eeg" ─> per-channel amplitude, drift, muscle ─> pass/fail
decode.py:  HDF5 ─> zero-phase filters ─> epochs per task ─> band power ─> LDA + cross-validation
receive.py: any LSL stream ─> per-channel mean/std every second (+ optional CSV)
dance/:     LSL "core8_raw" (unfiltered) ─> block CSVs ─> window features ─> random forest ─> live moves
```

| file | what it does |
|---|---|
| `acquire.py` | headset → filters → scope + LSL; records raw EEG + task markers with `--record` |
| `motion.py` | `BCICore` subclass that also publishes the accelerometer and gyroscope over LSL |
| `calibrate.py` | live per-channel contact check plus an eyes-closed alpha test; writes a pass/fail report |
| `quality.py` | the signal-quality metrics and thresholds used by `calibrate.py` and `tasks.py` |
| `tasks.py` | full-screen task cues, randomised trials, sends markers to `acquire.py`; refuses to start on bad channels |
| `decode.py` | offline decoding of the recorded tasks, with chance level and a permutation test |
| `receive.py` | standalone pylsl reader, e.g. for checking streams or saving them to CSV |
| `dance/` | dance-move classifier: record, train, decode live with a dashboard (see below) |

## Setup

Python 3.10–3.14 (tested with Homebrew Python 3.13, gpype 4.1.0).

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
git clone https://github.com/gtec-medical-engineering/gpype ../gpype   # examples, for reference
```

### Bluetooth permission (macOS)

macOS grants Bluetooth to the **app that launches Python**, not to Python itself.

1. Open **System Settings → Privacy & Security → Bluetooth**.
2. Turn on the app you run the scripts from (Terminal, iTerm, VS Code, PyCharm…).
   If it isn't listed, run `acquire.py` once and macOS should ask. Otherwise add it with **+**.
3. **Quit and reopen that app.** The permission only applies to a fresh process.

Without this, the BLE scan quietly returns nothing and it looks like the headset is off.

## Running an experiment session

Switch on the headset. g.Pype finds it by BLE scan, so you do not pair it in System Settings.

```bash
# Terminal 1: acquire + record. Leave it running for the whole session.
.venv/bin/python acquire.py --record data/p01_s1.h5          # add --oscar for artifact removal

# Terminal 2: fix the electrodes until every channel is green, then do the alpha check
.venv/bin/python calibrate.py

# Then run the session. It re-checks the signal for 5 s and won't start on a red channel.
.venv/bin/python tasks.py --tasks rest_open rest_closed math --trials 15 --log data/p01_s1_tasks.csv

# When tasks.py says "Done", close the scope window to finish the file. Then:
.venv/bin/python decode.py data/p01_s1_*.h5
```

`--record` adds a date and time to the file name, e.g. `data/p01_s1_20261107_150102_123.h5`.

- **Tasks** (`tasks.py --tasks …`): `rest_open`, `rest_closed`, `left_hand`, `right_hand` and
  `feet` (motor imagery), `math` (serial subtraction), `words` (word generation). The marker
  codes are fixed in `TASKS` in `tasks.py`. Add your own there.
- **Trial structure:** fixation cross for 2–3 s, then the cue for `--duration` s (default 5),
  with a beep at the start and end of each cue. There's a break every 20 trials. Press Space
  to start or continue, Esc to abort. The trial order is randomised and balanced, and written
  to the `--log` CSV.
- **Decoding:** `decode.py` epochs each task from 0.5 to 5 s after the cue and drops epochs
  over 150 µV peak-to-peak. Before rejecting, it prints each channel's median epoch
  peak-to-peak and drops channels over 2× the median (`--bad-factor`, or name them with
  `--exclude Fz C4`). `--tasks math rest_open` decodes a subset of the tasks.
  It takes log theta, alpha, beta and gamma power per channel
  (32 features) and runs a shrinkage-LDA classifier with 5-fold CV. It prints accuracy,
  chance level, a permutation p-value, the confusion matrix and the most discriminative
  features. Pass several files to pool sessions.

### Choosing tasks for 8 channels (Fz C3 Cz C4 Pz PO7 POz PO8)

- **Start with `rest_open` vs `rest_closed`.** Closing the eyes raises alpha (8–13 Hz) at
  PO7/POz/PO8, which almost always decodes. If this fails, fix the setup before trying
  harder tasks.
- **`math` / `words` vs rest** is usually decodable from frontal theta and changes in
  posterior alpha.
- **`left_hand` vs `right_hand` imagery** shows up as mu/beta desynchronisation at C3/C4.
  It is weak without training. Expect near chance in a first session.
- **Watch for confounds.** A decoder will happily learn from eye movements, blinks, jaw
  tension or head motion if they differ between tasks, e.g. eyes moving while reading the
  math cue. If the top features are Fz or gamma, or the motion stream differs between
  tasks, suspect artifacts. Keep cues short and have the participant fixate and stay still.
- **Aim for ≥ 30 clean trials per class.** With 15 per class, accuracies swing a lot, so
  rely on the permutation p-value.

## Dance-move classifier (`dance/`)

This classifies 7 moves (spin, arm pumps, bounce, stand still, the robot, the floss, head bob)
from movement artifacts and EMG, the opposite of what the rest of the repo tries to do. It uses
the unfiltered `core8_raw` stream, because the 1–40 Hz filter on `core8_eeg` removes both slow
head motion and EMG. The motion stream is not used.

| file | what it does |
|---|---|
| `dance/config.py` | moves, window sizes and online settings. **Add or remove a move here.** |
| `dance/stream.py` | sources: live LSL or replay of a recorded CSV (same interface), and labels from markers |
| `dance/record.py` | full-screen cues for one block (chained sequences, or single trials); writes `data/dance/<subject>_b<block>.csv` |
| `dance/features.py` | per-window features, shared by training and live |
| `dance/train.py` | leave-one-block-out evaluation against a majority baseline, then saves the model |
| `dance/live.py` | decoding (Space start/stop by default, or continuous) with the dashboard |

```bash
# Terminal 1, whole session (the HDF5 is a backup; dance/ records its own CSVs)
.venv/bin/python acquire.py --record data/dance/p01.h5
# Terminal 2: contact check while sitting still, then one block (~4 min); repeat for more blocks
.venv/bin/python calibrate.py --no-alpha
.venv/bin/python -m dance.record --subject p01 --block 1
# Train (after 2 blocks already, to check the moves are separable at all)
.venv/bin/python -m dance.train data/dance/p01_b*.csv --out models/p01.joblib
# Decode live: Space to start, Space to stop -> top move
.venv/bin/python -m dance.live --model models/p01.joblib
# Continuous decoding, live or replaying a block the model has not seen
.venv/bin/python -m dance.live --model models/p01.joblib --mode continuous
.venv/bin/python -m dance.train data/dance/p01_b*.csv --holdout 5 --out models/p01_no5.joblib
.venv/bin/python -m dance.live --model models/p01_no5.joblib --mode continuous --replay data/dance/p01_b5.csv
```

**Recording, `--protocol sequence` (the default).** Moves are chained the way they are danced
live: a sequence is 7 moves back to back (`--seq-len`, default: one per move), each lasting a
random 3–6 s (`--segment`), with a beep and a new marker at every switch and no rest in
between. During the last second of each move the screen shows "then: NEXT MOVE" so the dancer
flows straight into it. Sequences are separated by a 10 s break (`--break`). Each block has
every move 5 times (`--reps`), never twice in a row, so 5 sequences, about 3.5 min and 30
transitions per block. Stand still is one of the moves, so the data also has move → still →
move. The dancer keeps repeating each move until the cue changes, never just once. Compared
with the discrete protocol, this gives the classifier real move-to-move transitions, variable
durations (so it can't learn trial timing), and a replay that looks like live use.

The recording window shows the cues on the right and the electrodes on the left, like
`acquire.py`'s scope: the last 10 s of the 1–40 Hz `core8_eeg` stream, ±50 µV per channel
(`--scale`), with a dashed line labelled with the move at every marker. A flat line there is a
dead electrode. `--no-scope` hides it.

**Recording, `--protocol discrete`.** The original trials: rest (2–3 s), "Next: MOVE" (1 s),
GO (beep, 4 s of dancing, `--duration`), STOP (beep). Blocks of both kinds can be trained on
together.

In both protocols the marker codes (13–20, end of a run = 99) travel over UDP into
`acquire.py`, so they sit in channel 9 of the same stream as the EEG. A move code ends the
current segment and starts the next one. The CSV is `time, Fz..PO8, marker`, with the time on
gpype's clock. At the end, the recorder prints the sample rate, the segments it found per move
(against the number cued, so lost markers show up) and how many move-to-move transitions it
covered.

**Changing moves.** Codes are fixed once recorded. Step-touch (11), rock step (12) and
shoulder shimmy (14) were retired, as they were confused with spin and with the arm moves:
blocks recorded with them still load, and their trials are skipped. Never reuse a retired
code.

**Features** (per 1.5 s window, every 0.25 s): only each channel's mean is removed, nothing
else is filtered. Per channel: log RMS, log RMS of the first difference, and log band power at
0.5–3 Hz (motion), 3–8, 8–13, 13–30, 30–45 and 55–95 Hz (EMG, skipping 50 Hz mains). Plus the
28 channel-pair correlations. The first 0.5 s after each cue counts as no move (reaction
time). By default (`--purity 1`) training uses only windows that lie fully inside one segment.
`--purity 0.67` also keeps windows that straddle a transition, labelled with the move that
fills at least 2/3 of them, which is what the live decoder sees during transitions. On the
two discrete blocks this made no clear difference (there, the edge windows are move + standing
around a cue, not move → move); compare the two with the replay scores below once you have
sequence blocks.

**Leaving out an electrode.** `dance.train ... --exclude POz` (several allowed, any case) drops
every feature that uses that electrode: its 8 per-channel features and its 7 correlations. The
saved model remembers this, and `dance.live` drops the same features, so nothing else changes.

**Evaluation.** Leave-one-block-out, so overlapping windows from one block are never in both
train and test. It prints accuracy and macro-F1 for the random forest and the majority
baseline, the confusion matrix, a per-class report and the top features.

**Online, `--mode continuous`.** It runs the classifier every 0.25 s on the last 1.5 s and averages the probabilities
(`SMOOTH_ALPHA`). The decoded move changes only after another move has been ≥ `THRESHOLD`
for `MIN_DUR_S`. Each change is printed with its LSL timestamp. In replay it also scores the
decoded sequence against the cues:
- *step accuracy*: decoded == cued, every 0.25 s
- *segments detected*: the cued move was decoded at some point before the next cue
- *latency*: from the cue to the first correct decode (median and 90th percentile)
- *sequence accuracy*: 1 − edit distance / length between the decoded and cued order of
  moves, ignoring stand still. This is the score for "did we get the dance right".

`--no-gui` prints only the transitions, and `--speed 0` replays as fast as possible.

`SMOOTH_ALPHA`, `THRESHOLD` and `MIN_DUR_S` trade latency for stability. On the two discrete
blocks (train on one, replay the other) the defaults gave about 2.7 s median latency and
0.58 sequence accuracy. Switching on every step (`MIN_DUR_S = 0`, `THRESHOLD = 0.35`) cut
latency to about 1.6 s and nearly doubled step accuracy, but the extra false switches made sequence
accuracy negative. So with the current defaults, hold a move for at least ~3 s when dancing
live. Retune these on held-out sequence blocks. The dashboard shows the raw EEG, per-channel quality
(meaningful only when still), the current move with its confidence, the probabilities, and
the decoded and cued moves over time.

**Online, `--mode discrete` (the default).** Press Space to start, dance, then press Space
again to stop. The captured stretch is cut into the same 1.5 s windows as in training and their
probabilities are averaged. The dashboard shows the top move and all the probabilities. The
terminal prints the top three with the capture's LSL start and stop times. A capture needs
at least 1.5 s. This mode needs the dashboard, so `--no-gui` works only with
`--mode continuous`.

**Check on the device before collecting data.** None of this has been run on the headset.
- `receive.py --stream-name core8_raw` should show 9 channels at about 250 samples/s. This
  checks that a second gpype `LSLSender` on the unfiltered Router output works.
- Shake your head and check the raw units, DC offset and saturation in the mean/std output.
- Record a short block (`--reps 1`). The summary should show about 250 samples/s, which
  rules out Bluetooth dropouts while dancing, and one segment per move, which shows that
  codes above 9 pass through `UDPReceiver`.
- The LSL timestamps are gpype's `time.monotonic()`. Don't mix them with
  `pylsl.local_clock()`.

**Without the headset:** `acquire.py --simulate`, then `dance.record --subject sim --block 1
--reps 2 --skip-check` (twice, blocks 1 and 2), then `dance.train`. It should score near
chance, which shows the whole chain runs.

## Calibration: getting every electrode recording properly

The Core-8 can't measure impedance (neither the driver nor gpype exposes it), so
`calibrate.py` judges each channel from its live signal on the `core8_eeg` stream.
Run it while `acquire.py` is streaming.

**Stage 1: contact.** A table refreshed every second shows, per channel:

| metric | meaning | good | bad |
|---|---|---|---|
| ampl | median over 1 s chunks of the 1–40 Hz std (robust to the odd blink) | ≤ 25 µV | > 40 µV, or < 0.5 µV (flat) |
| 1-4Hz | slow drift: poor contact | | |
| 20-40 | muscle | ≤ 6 µV | > 10 µV |

- Every failing channel gets a hint: *flat* (no contact), *slow drift* (part the hair, re-wet,
  press the electrode on) or *muscle* (relax jaw, forehead and neck).
- The stage passes once every channel is green for 10 s in a row (`--hold`).
- The participant sits still, eyes open.
- The thresholds come from session p01_s1. The good channels sat at 9–12 µV; Fz, Cz and C4
  had poor contact at 28–57 µV, almost all of it in 1–4 Hz. Run on that session, this check
  would have flagged all three before the experiment started.

**Stage 2: alpha check.** 15 s eyes open, then 15 s eyes closed, cued by beeps.
- Closing the eyes must at least multiply 8–13 Hz power at Pz/PO7/POz/PO8 by 1.3. Session
  p01_s1 gave 3–5×, and every trial was above 2×.
- This shows the electrodes pick up brain signal, not just low noise.
- Skip it with `--no-alpha`.

The result goes to `data/calibration_<time>.json`, and the exit code is 0 on a pass.
Use `--ignore CH` to leave out an electrode you can't fix (the decoder then needs
`--exclude CH` too).

**The gate in `tasks.py`.**
- Before the first trial, it checks 5 s of signal with the same metrics. It refuses to start
  if any channel is red (yellow is allowed, with a warning).
- At every break, it re-checks and lists channels that have degraded, so you can re-wet
  them before continuing.
- `--ignore CH` and `--skip-check` override it.

## Artifacts: what's built in

- **OSCAR** (`acquire.py --oscar`) is g.tec's real-time artifact removal, built into
  g.Pype's amplifier sources (`BCICore(enable_oscar=True)`). Its key is bundled with gpype.
  - g.Tec doesn't document the algorithm.
  - In my synthetic test (`OscarLive` directly, 8 ch at 250 Hz), it cut RMS error ~13× for
    150 µV frontal blinks, ~5× for broadband muscle bursts and ~14× for electrode pops,
    while keeping 10 Hz alpha.
  - It adds 83 samples (0.33 s) of delay. gpype's changelog says events are held back by
    that delay too, so markers should stay aligned. I couldn't check this without the
    headset: blink on cue once and look at the recording.
  - Whether to use it: record with and without OSCAR on the same task and compare decoding.
    Removing "artifacts" can also remove real signal, and it can hide the confounds above.
- **Filtering** (always on): 48–52 Hz bandstop for UK mains, and a 1–40 Hz bandpass. The
  bandpass removes slow drift (sweat, movement) and most EMG above 40 Hz.
- **Epoch rejection** in `decode.py` (`--reject`, µV peak-to-peak): the simplest method from
  the Bitbrain artifact overview. It drops contaminated trials instead of trying to repair them.
- **Not built in**: ICA, ASR and regression against EOG. With 8 channels and no EOG
  electrode, ICA is of limited use. MNE-Python (`pip install mne`) has these if needed.
  gpype's example `s4e9_into_mne.py` shows the hand-off.

## Accelerometer and gyroscope

gpype's `BCICore` keeps only the EEG columns of each sample. The driver also delivers ACC, GYR,
battery and counter columns, which `BCICore` discards. `motion.py` subclasses it and
publishes any ACC/GYR columns as a second LSL stream, **`core8_motion`** (type `Motion`,
250 Hz, labels `AccX..AccZ, GyrX..GyrZ`):

```bash
.venv/bin/python receive.py --stream-name core8_motion --save data/p01_s1_motion.csv
```

- **Channel map:** at connect, `acquire.py` prints the device's channel map. If it says the
  headset reports no ACC/GYR channels, there's nothing to publish.
- **Units:** whatever the driver delivers. For the Unicorn, gpype documents g and deg/s.
  This isn't verified for the Core-8; check by tilting the headset (gravity ≈ 1 g).
- **Fragility:** it hooks two private gpype methods (`_resolve_channels`, `_assemble`), so
  `requirements.txt` pins gpype to 4.1.x.

## LSL streams and timestamps

- `core8_eeg`: 9 × float64 at 250 Hz, filtered, µV. Channels: `Fz C3 Cz C4 Pz PO7 POz PO8`,
  plus `Ch09`, the task marker (0, or the task code at cue onset).
- `core8_motion`: see above.

**Clock caveat.** gpype stamps LSL samples with `time.monotonic()`, not with pylsl's
`local_clock()`. On the test Mac the two differed by ~19 hours. The two streams above share
gpype's clock, so they line up with each other. A third-party LSL stream that uses default
timestamps will not line up with them, e.g. markers from PsychoPy recorded in LabRecorder.
That's why task markers go **into** the pipeline over UDP instead of as a separate LSL
marker stream.

Reading from your own code:

```python
from pylsl import StreamInlet, resolve_byprop
inlet = StreamInlet(resolve_byprop("name", "core8_eeg", timeout=10)[0])
samples, timestamps = inlet.pull_chunk()
```

## Testing without the headset

`acquire.py --simulate` swaps the headset for gpype's signal generator, with the same channel
names, markers and LSL streams. It has no motion stream. A session recorded this way should
decode at chance, which is a useful null check.

## Troubleshooting

**Headset not discovered**
- Check the Bluetooth permission above, and that you restarted the terminal/IDE afterwards.
- Bluetooth is on, the headset is charged and on, and within a few metres.
- Nothing else is connected to it (g.tec's own app, another laptop). A BLE device accepts one connection.
- Several g.tec headsets in range? Pass `--serial` (printed on the device).
- gpype's log is in `~/Library/Application Support/gtec/gPype`. Connection errors there name the cause.

**Flat or noisy channels**
- Flat (~0 µV): the electrode isn't touching the scalp, or the headset isn't streaming. Re-seat it.
- Big drifts or saturation: poor contact. Part the hair and rewet or regel the electrodes. Wait a minute for them to settle.
- Strong 50 Hz noise despite the notch: move away from chargers and mains cables, and run the laptop on battery.
- Bursts of large noise: blinks show at Fz, jaw clenching and muscle tension everywhere. Relax and sit still.
- The scope's amplitude control rescales the view. Real EEG is roughly ±10–50 µV.

**No markers in the recording**
- Start `acquire.py` before `tasks.py`, and keep it running until `tasks.py` finishes.
- The ports must match: `--marker-port` on `acquire.py` and `--port` on `tasks.py` (default 5005).
  Ports below 1024 need root on macOS.
- Markers appear as coloured lines in the scope at each cue.

**LSL stream not found (`receive.py`)**
- Start `acquire.py` first and wait for the scope to show data.
- The names must match: `--stream-name` must be the same in both scripts.
- On a different machine: same network, and the firewall must allow LSL (UDP 16571 for
  discovery, TCP/UDP 16572–16604 for data).
- Increase the search time: `receive.py --timeout 30`.

## Licence note

g.Pype is under the g.tec Non-Commercial License. Recordings are stamped `NON-COMMERCIAL USE`.
This is fine for research and hackathons, but check before using it commercially.
