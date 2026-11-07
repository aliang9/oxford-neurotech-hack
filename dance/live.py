"""Decode dance moves, live or from a recorded block, with a dashboard.

    python -m dance.live --model models/p01.joblib                       # Space start / stop
    python -m dance.live --model models/p01.joblib --mode continuous     # sliding window
    python -m dance.live --model models/p01.joblib --mode continuous --replay data/dance/p01_b5.csv
    python -m dance.live --model models/p01.joblib --mode continuous --replay ... --speed 0 --no-gui

--mode discrete (default): press Space to start capturing, Space again to
stop. The captured stretch is cut into the same windows as in training,
their probabilities are averaged, and the top move is shown and printed.

--mode continuous: every 0.25 s of new samples, features of the
last 1.5 s -> class probabilities -> moving average. The decoded move
switches only when another move has probability >= THRESHOLD for MIN_DUR_S
in a row. Each switch is printed with its LSL timestamp.
"""

import argparse
import time
from collections import deque

import joblib
import numpy as np

from . import config as cfg
from .features import extract, windows
from .stream import LSLSource, ReplaySource, sample_labels, segments


class Decoder:
    """Turns a stream of samples into a stable sequence of moves."""

    def __init__(self, bundle):
        self.model = bundle["model"]
        self.classes = list(self.model.classes_)
        self.buffer = deque(maxlen=cfg.WIN)
        self.n = 0              # samples seen so far
        self.t0 = None          # timestamp of the first sample
        self.probs = None       # smoothed class probabilities
        self.move, self.conf = None, 0.0
        self.candidate, self.candidate_since = None, None
        self.history = []       # (sample count, time, move, confidence) per step
        self.transitions = []   # (time, move, confidence) per switch

    def push(self, samples, times):
        for row, t in zip(samples, times):
            if self.t0 is None:
                self.t0 = t
            self.buffer.append(row[:cfg.N_EEG])
            self.n += 1
            # same window positions as features.windows() on a recorded file
            if self.n >= cfg.WIN and (self.n - cfg.WIN) % cfg.STEP == 0:
                self.step(t)

    def step(self, t):
        p = self.model.predict_proba(extract(np.array(self.buffer))[None])[0]
        a = cfg.SMOOTH_ALPHA
        self.probs = p if self.probs is None else a * p + (1 - a) * self.probs

        best = int(np.argmax(self.probs))
        winner = self.classes[best] if self.probs[best] >= cfg.THRESHOLD else self.move
        if winner == self.move:
            self.candidate = None
        elif winner != self.candidate:
            self.candidate, self.candidate_since = winner, t
        elif t - self.candidate_since >= cfg.MIN_DUR_S:
            self.move, self.candidate = winner, None
            self.transitions.append((t, winner, float(self.probs[best])))
        self.conf = float(self.probs[self.classes.index(self.move)]) if self.move else 0.0
        self.history.append((self.n, t, self.move, self.conf))


def print_new_transitions(decoder, printed):
    for t, move, conf in decoder.transitions[printed:]:
        print(f"{t:12.3f}  (+{t - decoder.t0:7.2f} s)  -> {move:12s} p={conf:.2f}", flush=True)
    return len(decoder.transitions)


def runs(decoder):
    """Decoded sequence as [start, end, move] segments, seconds since the first sample."""
    segments = []
    for _, t, move, _ in decoder.history:
        t = t - decoder.t0
        if segments and segments[-1][2] == move:
            segments[-1][1] = t + cfg.STEP_S
        else:
            segments.append([t, t + cfg.STEP_S, move])
    return [s for s in segments if s[2] is not None]


def classify_segment(model, samples):
    """Moves ranked by probability for a user-marked stretch of samples (None if too short)."""
    features = [extract(w) for _, w in windows(samples)]
    if not features:
        return None
    probs = model.predict_proba(np.array(features)).mean(axis=0)
    return [(model.classes_[i], float(probs[i])) for i in np.argsort(probs)[::-1]]


def edit_distance(a, b):
    """Levenshtein distance between two sequences."""
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def collapse(names):
    """Drop rest and merge repeats: the order of the dance moves only."""
    out = []
    for n in names:
        if n != "rest" and (not out or out[-1] != n):
            out.append(n)
    return out


def replay_summary(decoder, source):
    """How well the decoded sequence matches the recorded cues."""
    markers = source.data[:, cfg.MARKER_CH]
    codes, _ = sample_labels(markers, cfg.ONSET_SKIP_S)
    scored = [(cfg.MOVES[codes[n - 1]][0], move) for n, _, move, _ in decoder.history
              if codes[n - 1] and move]
    if not scored:
        return print("\nNo move was decoded during the cued segments: nothing to score.")
    accuracy = np.mean([true == move for true, move in scored])

    # Per cued segment: was its move decoded before the next cue, and how late?
    cued = [(s, e, cfg.MOVES[c][0]) for s, e, c in segments(markers)]
    steps = np.array([n - 1 for n, _, _, _ in decoder.history])
    moves = [m for _, _, m, _ in decoder.history]
    lags = []
    for k, (s, e, name) in enumerate(cued):
        stop = cued[k + 1][0] if k + 1 < len(cued) else len(markers)
        hit = [i for i in np.flatnonzero((steps >= s) & (steps < stop)) if moves[i] == name]
        lags.append((steps[hit[0]] - s) / cfg.FS if hit else None)
    found = [x for x in lags if x is not None]

    true_seq = collapse(name for _, _, name in cued)
    decoded_seq = collapse(move for _, move, _ in decoder.transitions)
    seq_acc = 1 - edit_distance(true_seq, decoded_seq) / max(len(true_seq), 1)

    print(f"\nReplay ({len(scored)} labelled steps, {len(cued)} cued segments):")
    print(f"  step accuracy      {accuracy:.2f}   (decoded move == cued move, every 0.25 s)")
    print(f"  segments detected  {len(found) / len(cued):.2f}   (cued move decoded before the next cue)")
    if found:
        print(f"  latency            {np.median(found):.2f} s median, "
              f"{np.percentile(found, 90):.2f} s 90th pct   (cue -> decoded)")
    print(f"  sequence accuracy  {seq_acc:.2f}   (1 - edit distance / length, rest ignored; "
          f"{len(decoded_seq)} decoded vs {len(true_seq)} cued moves)")
    print("If this was a training block, the score is optimistic: use --holdout in dance.train.")


def run_headless(source, decoder):
    printed = 0
    try:
        while not source.done:
            samples, times = source.pull(timeout=0.1)
            decoder.push(samples, times)
            printed = print_new_transitions(decoder, printed)
            if isinstance(source, ReplaySource) and source.speed > 0:
                time.sleep(0.02)
    except KeyboardInterrupt:
        pass


def run_dashboard(source, decoder, scale, discrete):
    import pyqtgraph as pg
    from PySide6 import QtCore, QtGui, QtWidgets
    from scipy.signal import butter, sosfiltfilt

    import quality

    app = QtWidgets.QApplication([])
    win = QtWidgets.QWidget()
    win.setWindowTitle("Dance decoder (" + ("discrete" if discrete else "continuous") + ")")
    grid = QtWidgets.QGridLayout(win)
    colors = {c: pg.intColor(i, hues=len(decoder.classes)) for i, c in enumerate(decoder.classes)}

    # Incoming EEG, last 5 s, one trace per channel
    display = deque(maxlen=5 * cfg.FS)
    eeg = pg.PlotWidget(title="Raw EEG, last 5 s (mean removed)")
    eeg.getAxis("left").setTicks([[(-i * scale, ch) for i, ch in enumerate(cfg.CHANNELS)]])
    eeg.setYRange(-(cfg.N_EEG - 0.5) * scale, 0.5 * scale)
    # one neutral colour: hues here would look like the move colours used below
    curves = [eeg.plot(pen=pg.mkPen("#ddd", width=1)) for _ in range(cfg.N_EEG)]
    grid.addWidget(eeg, 0, 0, 3, 1)

    # Signal quality per channel (only meaningful while standing still)
    quality_row = QtWidgets.QHBoxLayout()
    quality_row.addWidget(QtWidgets.QLabel("Quality (when still):"))
    chips = [QtWidgets.QLabel(ch) for ch in cfg.CHANNELS]
    for chip in chips:
        chip.setAlignment(QtCore.Qt.AlignCenter)
        quality_row.addWidget(chip)
    grid.addLayout(quality_row, 3, 0)
    sos = butter(4, [1, 40], btype="bandpass", fs=cfg.FS, output="sos")

    # Current move, confidence and per-class probabilities
    move_label = QtWidgets.QLabel("...")
    move_label.setFont(QtGui.QFont("Helvetica", 44, QtGui.QFont.Bold))
    move_label.setAlignment(QtCore.Qt.AlignCenter)
    grid.addWidget(move_label, 0, 1)
    conf_bar = QtWidgets.QProgressBar(format="confidence %p%")
    grid.addWidget(conf_bar, 1, 1)
    prob_box = QtWidgets.QVBoxLayout()
    prob_bars = {}
    for c in decoder.classes:
        bar = QtWidgets.QProgressBar(format=f"{c}  %p%")
        bar.setStyleSheet(f"QProgressBar::chunk {{ background: {colors[c].name()}; }}")
        prob_box.addWidget(bar)
        prob_bars[c] = bar
    grid.addLayout(prob_box, 2, 1)

    # Discrete mode: status of the Space start / stop capture
    capture_label = QtWidgets.QLabel("Space: start capturing a move")
    capture_label.setFont(QtGui.QFont("Helvetica", 16))
    capture_label.setWordWrap(True)
    if discrete:
        grid.addWidget(capture_label, 3, 1)

    # Continuous mode: decoded sequence (and the cued one, when replaying), last 60 s
    timeline = pg.PlotWidget(title="Moves over time (s): decoded, and cued in replay")
    timeline.getAxis("left").setTicks([[(0, "decoded"), (1, "cued")]])
    timeline.setYRange(-0.6, 1.6)
    decoded_bars = pg.BarGraphItem(x0=[], width=[], y=[], height=0.8)
    timeline.addItem(decoded_bars)
    if not discrete:
        grid.addWidget(timeline, 4, 0, 1, 2)
        grid.setRowStretch(4, 2)
    grid.setColumnStretch(0, 3)
    grid.setColumnStretch(1, 1)
    grid.setRowStretch(0, 1)
    grid.setRowStretch(2, 2)

    if isinstance(source, ReplaySource) and not discrete:  # draw the cued moves once
        t = source.times - source.times[0]
        x0, width, brushes = [], [], []
        for s, e, code in segments(source.data[:, cfg.MARKER_CH]):  # back to back in sequences
            x0.append(t[s])
            width.append(t[e - 1] - t[s])
            brushes.append(colors.get(cfg.MOVES[code][0], pg.mkColor("#666")))
        timeline.addItem(pg.BarGraphItem(x0=x0, width=width, y=[1] * len(x0), height=0.8,
                                         brushes=brushes))

    state = {"printed": 0, "last_quality": 0.0, "summarised": False,
             "capture": None, "capture_times": []}

    def toggle_capture():
        if state["capture"] is None:  # start
            state["capture"], state["capture_times"] = [], []
            capture_label.setText("● Capturing... Space to stop")
            capture_label.setStyleSheet("color: #c22;")
            return
        # stop: classify what was captured
        samples = np.vstack(state["capture"]) if state["capture"] else np.empty((0, cfg.N_EEG + 1))
        times = np.concatenate(state["capture_times"]) if state["capture_times"] else []
        state["capture"] = None
        capture_label.setStyleSheet("")
        ranked = classify_segment(decoder.model, samples)
        if ranked is None:
            capture_label.setText(f"Too short: need at least {cfg.WIN_S:g} s. "
                                  "Space: start capturing a move")
            return
        (top, p), others = ranked[0], ranked[1:3]
        move_label.setText(top)
        conf_bar.setValue(int(100 * p))
        for c, q in ranked:
            prob_bars[c].setValue(int(100 * q))
        capture_label.setText(f"Captured {len(samples) / cfg.FS:.1f} s. Space: capture again")
        print(f"CAPTURE {times[0]:.3f}-{times[-1]:.3f} ({len(samples) / cfg.FS:.1f} s): "
              f"{top} p={p:.2f}  (" + ", ".join(f"{c} {q:.2f}" for c, q in others) + ")",
              flush=True)

    if discrete:
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), win, activated=toggle_capture)

    def tick():
        samples, times = source.pull()
        if len(times) and state["capture"] is not None:
            state["capture"].append(samples)
            state["capture_times"].append(times)
            n = sum(len(s) for s in state["capture"])
            capture_label.setText(f"● Capturing {n / cfg.FS:.1f} s... Space to stop")
        if len(times):
            display.extend(samples[:, :cfg.N_EEG])
            if not discrete:
                decoder.push(samples, times)
                state["printed"] = print_new_transitions(decoder, state["printed"])

        if len(display) > 1:
            x = np.array(display)
            x -= x.mean(axis=0)
            secs = (np.arange(len(x)) - len(x)) / cfg.FS
            for i, curve in enumerate(curves):
                curve.setData(secs, np.clip(x[:, i], -scale / 2, scale / 2) - i * scale)

        if len(display) >= 2 * cfg.FS and time.monotonic() - state["last_quality"] > 1.0:
            state["last_quality"] = time.monotonic()
            filtered = sosfiltfilt(sos, np.array(display), axis=0)
            for chip, row in zip(chips, quality.assess(filtered, cfg.FS, cfg.CHANNELS)):
                color = {"good": "#2a2", "warn": "#c80", "bad": "#c22"}[row["status"]]
                chip.setStyleSheet(f"background: {color}; color: white; padding: 2px;")
                chip.setToolTip(f"{row['amplitude']:.0f} uV  {row['hint']}")

        if decoder.probs is not None:
            move_label.setText(decoder.move or "...")
            conf_bar.setValue(int(100 * decoder.conf))
            for c, p in zip(decoder.classes, decoder.probs):
                prob_bars[c].setValue(int(100 * p))

        if decoder.history:
            segments = runs(decoder)
            decoded_bars.setOpts(x0=[s for s, _, _ in segments],
                                 width=[e - s for s, e, _ in segments],
                                 y=[0] * len(segments),
                                 brushes=[colors[m] for _, _, m in segments])
            now = decoder.history[-1][1] - decoder.t0
            timeline.setXRange(max(0, now - 60), max(60, now), padding=0)

        if source.done and not discrete and not state["summarised"]:
            state["summarised"] = True
            replay_summary(decoder, source)

    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(50)
    win.resize(1400, 900)
    win.show()
    app.exec()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--replay", metavar="CSV", help="replay a recorded block instead of LSL")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed, 0 = instant")
    ap.add_argument("--mode", choices=["discrete", "continuous"], default="discrete",
                    help="discrete: Space to start/stop a capture (default); "
                         "continuous: sliding-window decoding")
    ap.add_argument("--stream", default=cfg.RAW_STREAM)
    ap.add_argument("--no-gui", action="store_true", help="only print transitions (continuous)")
    ap.add_argument("--scale", type=float, default=200.0, help="EEG plot spacing, uV")
    args = ap.parse_args(argv)
    discrete = args.mode == "discrete"
    if discrete and args.no_gui:
        ap.error("--no-gui only works with --mode continuous (discrete needs Space in the dashboard)")

    decoder = Decoder(joblib.load(args.model))
    source = ReplaySource(args.replay, args.speed) if args.replay else LSLSource(args.stream)
    print(f"Classes: {', '.join(decoder.classes)}.")
    print("Space to start / stop a capture." if discrete else "Transitions (LSL time, since start):")
    if args.no_gui:
        run_headless(source, decoder)
        if args.replay:
            replay_summary(decoder, source)
    else:
        run_dashboard(source, decoder, args.scale, discrete)


if __name__ == "__main__":
    main()
