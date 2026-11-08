"""Decode dance moves, live or from a recorded block, with a dashboard.

    python -m dance.live --model models/p01.joblib                       # Space start / stop
    python -m dance.live --model models/p01.joblib --mode continuous     # sliding window
    python -m dance.live --model models/p01.joblib --mode continuous --replay data/dance/p01_b5.csv
    python -m dance.live --model models/p01.joblib --mode continuous --replay ... --speed 0 --no-gui
    python -m dance.live --model models/p01.joblib --demo                # camera, avatar, knobs

--mode discrete (default): press Space to start capturing, Space again to
stop. The captured stretch is cut into the same windows as in training,
their probabilities are averaged, and the top move is shown and printed.

--mode continuous: every 0.25 s of new samples, features of the
last 1.5 s -> class probabilities -> moving average. The decoded move
switches only when another move has probability >= THRESHOLD for MIN_DUR_S
in a row; the defaults (no averaging, no threshold, 0 s) switch to the top
move of every step at once. Each switch is printed with its LSL timestamp.
The dashboard shows the decoded move's avatar from figures/ (config.MOVE_GIFS).

--demo: continuous decoding laid out for an audience. A mirrored live view of
the computer's camera replaces the EEG traces (top left), and dials for
smoothing, threshold, hold time and update rate (bottom left) change the
decoder while it runs.
"""

import argparse
import os
import time
from collections import deque

import joblib
import numpy as np

from . import config as cfg
from .features import extract, keep_mask, windows
from .stream import LSLSource, ReplaySource, sample_labels, segments


class Decoder:
    """Turns a stream of samples into a stable sequence of moves."""

    def __init__(self, bundle):
        self.model = bundle["model"]
        self.mask = keep_mask(bundle.get("exclude", []))  # electrodes left out in training
        self.classes = list(self.model.classes_)
        self.buffer = deque(maxlen=cfg.WIN)
        self.n = 0              # samples seen so far
        self.t0 = None          # timestamp of the first sample
        self.probs = None       # smoothed class probabilities
        self.move, self.conf = None, 0.0
        self.candidate, self.candidate_since = None, None
        # sensitivity and latency: config defaults, changed live by the --demo knobs
        self.alpha, self.threshold, self.min_dur = cfg.SMOOTH_ALPHA, cfg.THRESHOLD, cfg.MIN_DUR_S
        self.step_n = cfg.STEP  # samples between classifications
        self.last_step = None   # sample count at the last classification
        self.history = []       # (sample count, time, move, confidence, step in s) per step
        self.transitions = []   # (time, move, confidence) per switch

    def push(self, samples, times):
        for row, t in zip(samples, times):
            if self.t0 is None:
                self.t0 = t
            self.buffer.append(row[:cfg.N_EEG])
            self.n += 1
            # with the default step, the same window positions as features.windows()
            if self.n >= cfg.WIN and (self.last_step is None or
                                      self.n - self.last_step >= self.step_n):
                self.last_step = self.n
                self.step(t)

    def step(self, t):
        p = self.model.predict_proba(extract(np.array(self.buffer))[self.mask][None])[0]
        a = self.alpha
        self.probs = p if self.probs is None else a * p + (1 - a) * self.probs

        best = int(np.argmax(self.probs))
        winner = self.classes[best] if self.probs[best] >= self.threshold else self.move
        if winner == self.move:
            self.candidate = None
        else:
            if winner != self.candidate:
                self.candidate, self.candidate_since = winner, t
            if t - self.candidate_since >= self.min_dur:  # 0: switch at once
                self.move, self.candidate = winner, None
                self.transitions.append((t, winner, float(self.probs[best])))
        self.conf = float(self.probs[self.classes.index(self.move)]) if self.move else 0.0
        self.history.append((self.n, t, self.move, self.conf, self.step_n / cfg.FS))


def print_new_transitions(decoder, printed):
    for t, move, conf in decoder.transitions[printed:]:
        print(f"{t:12.3f}  (+{t - decoder.t0:7.2f} s)  -> {move:12s} p={conf:.2f}", flush=True)
    return len(decoder.transitions)


def runs(decoder):
    """Decoded sequence as [start, end, move] segments, seconds since the first sample."""
    segments = []
    for _, t, move, _, step in decoder.history:
        t = t - decoder.t0
        if segments and segments[-1][2] == move:
            segments[-1][1] = t + step
        else:
            segments.append([t, t + step, move])
    return [s for s in segments if s[2] is not None]


def classify_segment(model, samples, mask):
    """Moves ranked by probability for a user-marked stretch of samples (None if too short)."""
    features = [extract(w)[mask] for _, w in windows(samples)]
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
    scored = [(cfg.MOVES[codes[n - 1]][0], move) for n, _, move, *_ in decoder.history
              if codes[n - 1] and move]
    if not scored:
        return print("\nNo move was decoded during the cued segments: nothing to score.")
    accuracy = np.mean([true == move for true, move in scored])

    # Per cued segment: was its move decoded before the next cue, and how late?
    cued = [(s, e, cfg.MOVES[c][0]) for s, e, c in segments(markers)]
    steps = np.array([n - 1 for n, *_ in decoder.history])
    moves = [m for _, _, m, *_ in decoder.history]
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


def start_camera(app, index):
    """Live, mirrored view of a computer camera in a QLabel. Returns the objects to keep alive."""
    from PySide6 import QtCore, QtGui, QtMultimedia, QtWidgets

    label = QtWidgets.QLabel("Starting the camera...")
    label.setAlignment(QtCore.Qt.AlignCenter)
    label.setMinimumSize(1, 1)
    label.setStyleSheet("background: black; color: white; font-size: 18px;")
    keep = {"label": label}

    def on_frame(frame):
        image = frame.toImage()
        if image.isNull():
            return
        image = image.flipped(QtCore.Qt.Horizontal)  # like a mirror
        label.setPixmap(QtGui.QPixmap.fromImage(image).scaled(
            label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def start():
        devices = QtMultimedia.QMediaDevices.videoInputs()
        if not devices:
            return label.setText("No camera found")
        device = devices[index] if index < len(devices) else devices[0]
        print(f"Camera: {device.description()} "
              f"(of {', '.join(d.description() for d in devices)}; pick with --camera N)")
        cam = QtMultimedia.QCamera(device)
        session = QtMultimedia.QMediaCaptureSession()
        sink = QtMultimedia.QVideoSink()
        session.setCamera(cam)
        session.setVideoSink(sink)
        sink.videoFrameChanged.connect(on_frame)
        cam.errorOccurred.connect(lambda _err, msg: label.setText(f"Camera error: {msg}"))
        keep.update(cam=cam, session=session, sink=sink)
        cam.start()

    def denied():
        label.setText("No camera access.\nAllow your terminal app in System Settings >\n"
                      "Privacy & Security > Camera, then restart it.")

    permission = QtCore.QCameraPermission()
    status = app.checkPermission(permission)
    if status == QtCore.Qt.PermissionStatus.Granted:
        start()
    elif status == QtCore.Qt.PermissionStatus.Denied:
        denied()
    else:
        # Qt can't ask from a plain python binary (no Info.plist), so ask AVFoundation directly:
        # macOS then prompts once, for the terminal app. Poll until it is answered.
        label.setText("Asking for camera access:\nallow your terminal app in the macOS prompt.")
        try:
            import AVFoundation
            AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
                AVFoundation.AVMediaTypeVideo, lambda granted: None)
        except ImportError:
            pass

        def poll():
            status = app.checkPermission(permission)
            if status != QtCore.Qt.PermissionStatus.Undetermined:
                timer.stop()
                start() if status == QtCore.Qt.PermissionStatus.Granted else denied()
        timer = QtCore.QTimer(label, interval=500, timeout=poll)
        timer.start()
        keep["timer"] = timer
    return keep


def make_knobs(decoder):
    """Dials for the decoder's sensitivity and latency settings, applied immediately."""
    from PySide6 import QtCore, QtGui, QtWidgets

    def set_alpha(v): decoder.alpha = 1 - v / 100
    def set_threshold(v): decoder.threshold = v / 100
    def set_hold(v): decoder.min_dur = v / 100
    def set_step(v): decoder.step_n = v

    # title, range, current value, setter, value text, tooltip
    specs = [
        ("Smoothing", 0, 95, round(100 * (1 - decoder.alpha)), set_alpha, lambda v: f"{v}%",
         "Averages the probabilities over recent steps. Steadier, but slower to follow a new move."),
        ("Threshold", 0, 100, round(100 * decoder.threshold), set_threshold, lambda v: f"{v}%",
         "Probability a move needs before it can be shown. Higher: fewer false switches, "
         "and the last move stays up when unsure."),
        ("Hold", 0, 300, round(100 * decoder.min_dur), set_hold, lambda v: f"{v / 100:.2f} s",
         "How long a new move must stay on top before it is shown. Removes flicker, adds this "
         "much delay."),
        ("Update every", round(0.1 * cfg.FS), cfg.FS, decoder.step_n, set_step,
         lambda v: f"{v / cfg.FS:.2f} s",
         "How often the classifier runs on the last 1.5 s. Shorter: less delay, more CPU."),
    ]
    box = QtWidgets.QGroupBox("Decoder knobs (window fixed at 1.5 s)")
    layout = QtWidgets.QGridLayout(box)
    dials = []
    for col, (title, lo, hi, value, setter, text, tip) in enumerate(specs):
        name = QtWidgets.QLabel(title, alignment=QtCore.Qt.AlignCenter)
        name.setFont(QtGui.QFont("Helvetica", 14, QtGui.QFont.Bold))
        shown = QtWidgets.QLabel(text(value), alignment=QtCore.Qt.AlignCenter)
        shown.setFont(QtGui.QFont("Helvetica", 16))
        dial = QtWidgets.QDial(minimum=lo, maximum=hi, notchesVisible=True)
        dial.setFixedSize(130, 130)

        def changed(v, setter=setter, shown=shown, text=text):
            setter(v)
            shown.setText(text(v))
        dial.valueChanged.connect(changed)
        dial.setValue(value)
        for w in (name, shown, dial):
            w.setToolTip(tip)
        layout.addWidget(name, 0, col)
        layout.addWidget(dial, 1, col)
        layout.addWidget(shown, 2, col)
        dials.append(dial)

    presets = QtWidgets.QHBoxLayout()
    for label, values in [("Instant", (0, 0, 0, cfg.STEP)), ("Steady", (60, 50, 75, cfg.STEP))]:
        button = QtWidgets.QPushButton(label)
        button.clicked.connect(lambda _=False, values=values: [d.setValue(v) for d, v in
                                                                zip(dials, values)])
        presets.addWidget(button)
    layout.addLayout(presets, 3, 0, 1, len(specs))
    layout.setRowStretch(1, 1)
    return box


def run_dashboard(source, decoder, scale, discrete, demo=False, camera=0):
    import pyqtgraph as pg
    from PySide6 import QtCore, QtGui, QtWidgets
    from scipy.signal import butter, sosfiltfilt

    import quality

    app = QtWidgets.QApplication([])
    win = QtWidgets.QWidget()
    win.setWindowTitle("Dance decoder (" + ("demo" if demo else "discrete" if discrete
                                            else "continuous") + ")")
    grid = QtWidgets.QGridLayout(win)
    colors = {c: pg.intColor(i, hues=len(decoder.classes)) for i, c in enumerate(decoder.classes)}

    # Incoming EEG, last 5 s, one trace per channel
    display = deque(maxlen=5 * cfg.FS)
    eeg = pg.PlotWidget(title="Raw EEG, last 5 s (mean removed)")
    eeg.getAxis("left").setTicks([[(-i * scale, ch) for i, ch in enumerate(cfg.CHANNELS)]])
    eeg.setYRange(-(cfg.N_EEG - 0.5) * scale, 0.5 * scale)
    # one neutral colour: hues here would look like the move colours used below
    curves = [eeg.plot(pen=pg.mkPen("#ddd", width=1)) for _ in range(cfg.N_EEG)]
    if not demo:  # the demo shows the camera instead
        grid.addWidget(eeg, 0, 0, 3, 1)

    # Signal quality per channel (only meaningful while standing still)
    quality_row = QtWidgets.QHBoxLayout()
    quality_row.addWidget(QtWidgets.QLabel("Quality (when still):"))
    chips = [QtWidgets.QLabel(ch) for ch in cfg.CHANNELS]
    for chip in chips:
        chip.setAlignment(QtCore.Qt.AlignCenter)
        quality_row.addWidget(chip)
    if not demo:
        grid.addLayout(quality_row, 3, 0)
    sos = butter(4, [1, 40], btype="bandpass", fs=cfg.FS, output="sos")

    # Current move as its avatar (or its name, if it has none), confidence, probabilities
    move_label = QtWidgets.QLabel("...")
    move_label.setFont(QtGui.QFont("Helvetica", 44, QtGui.QFont.Bold))
    move_label.setAlignment(QtCore.Qt.AlignCenter)
    move_label.setMinimumSize(1, 1)  # let the layout shrink it; the GIF is scaled to fit
    grid.addWidget(move_label, 0, 1)
    movies = {}
    for c in decoder.classes:
        path = os.path.join(cfg.FIGURES_DIR, cfg.MOVE_GIFS.get(c, ""))
        movie = QtGui.QMovie(path)  # a still image plays as a one-frame movie
        if os.path.isfile(path) and movie.isValid():
            movies[c] = (movie, QtGui.QImageReader(path).size())
        else:
            print(f"No avatar for {c} ({path}): showing its name.")

    def show_move(name):
        if name == state.get("shown"):
            return
        state["shown"] = name
        if move_label.movie():
            move_label.movie().stop()
        if name not in movies:
            move_label.setMovie(None)
            move_label.setText(name or "...")
            return
        movie, size = movies[name]
        size = size.scaled(move_label.size(), QtCore.Qt.KeepAspectRatio)
        movie.setScaledSize(size)
        move_label.setMovie(movie)
        movie.start()
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
    if demo:
        grid.addWidget(timeline, 3, 1)
        grid.setRowStretch(3, 2)
    elif not discrete:
        grid.addWidget(timeline, 4, 0, 1, 2)
        grid.setRowStretch(4, 2)
    grid.setColumnStretch(0, 3)
    grid.setColumnStretch(1, 2 if demo else 1)
    grid.setRowStretch(0, 3)  # the avatar
    grid.setRowStretch(2, 2)

    if demo:  # camera top left, knobs bottom left
        state_demo = {"camera": start_camera(app, camera)}
        left = QtWidgets.QVBoxLayout()
        left.addWidget(state_demo["camera"]["label"], stretch=1)
        left.addWidget(make_knobs(decoder))
        grid.addLayout(left, 0, 0, 4, 1)

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
        ranked = classify_segment(decoder.model, samples, decoder.mask)
        if ranked is None:
            capture_label.setText(f"Too short: need at least {cfg.WIN_S:g} s. "
                                  "Space: start capturing a move")
            return
        (top, p), others = ranked[0], ranked[1:3]
        show_move(top)
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

        if len(display) > 1 and not demo:
            x = np.array(display)
            x -= x.mean(axis=0)
            secs = (np.arange(len(x)) - len(x)) / cfg.FS
            for i, curve in enumerate(curves):
                curve.setData(secs, np.clip(x[:, i], -scale / 2, scale / 2) - i * scale)

        if not demo and len(display) >= 2 * cfg.FS and time.monotonic() - state["last_quality"] > 1.0:
            state["last_quality"] = time.monotonic()
            filtered = sosfiltfilt(sos, np.array(display), axis=0)
            for chip, row in zip(chips, quality.assess(filtered, cfg.FS, cfg.CHANNELS)):
                color = {"good": "#2a2", "warn": "#c80", "bad": "#c22"}[row["status"]]
                chip.setStyleSheet(f"background: {color}; color: white; padding: 2px;")
                chip.setToolTip(f"{row['amplitude']:.0f} uV  {row['hint']}")

        if decoder.probs is not None:
            show_move(decoder.move)
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
    ap.add_argument("--demo", action="store_true",
                    help="continuous decoding for an audience: camera instead of the EEG traces, "
                         "and knobs for smoothing, threshold, hold time and update rate")
    ap.add_argument("--camera", type=int, default=0, help="which camera --demo shows (default 0)")
    args = ap.parse_args(argv)
    if args.demo:
        if args.no_gui:
            ap.error("--demo needs the dashboard: drop --no-gui")
        args.mode = "continuous"
    discrete = args.mode == "discrete"
    if discrete and args.no_gui:
        ap.error("--no-gui only works with --mode continuous (discrete needs Space in the dashboard)")

    bundle = joblib.load(args.model)
    decoder = Decoder(bundle)
    if bundle.get("exclude"):
        print(f"Model trained without {', '.join(bundle['exclude'])}.")
    source = ReplaySource(args.replay, args.speed) if args.replay else LSLSource(args.stream)
    print(f"Classes: {', '.join(decoder.classes)}.")
    print("Space to start / stop a capture." if discrete else "Transitions (LSL time, since start):")
    if args.no_gui:
        run_headless(source, decoder)
        if args.replay:
            replay_summary(decoder, source)
    else:
        run_dashboard(source, decoder, args.scale, discrete, args.demo, args.camera)


if __name__ == "__main__":
    main()
