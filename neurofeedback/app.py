"""Tandem: posterior-alpha neurofeedback, with a calm participant view and a debugging dashboard.

    python -m neurofeedback                                  # live, from acquire.py's core8_eeg
    python -m neurofeedback --ui debug                       # the debugging dashboard instead
    python -m neurofeedback --ui both                        # both windows (e.g. two screens)
    python -m neurofeedback --source sim                     # simulated EEG, no headset
    python -m neurofeedback --source sim --speed 5           # simulated, 5x real time
    python -m neurofeedback --source csv --csv data/dance/p01_b1.csv

The score reflects sustained modulation of an EEG feature (posterior alpha power
relative to an eyes-open resting baseline). It is not a validated measure of meditation depth.
"""

import argparse
import time

import numpy as np

from . import config as cfg
from .engine import (CALIBRATING, FINISHED, INDEFINITE, PAUSED, POOR, READY, RECORDING, TIMED,
                     Engine, Worker)
from .history import SessionStore
from .sources import make_source

STATE_COLORS = {"waiting for data": "#888", "needs baseline": "#c90", CALIBRATING: "#38f",
                READY: "#aaa", RECORDING: "#2b2", PAUSED: "#c90", FINISHED: "#aaa", POOR: "#e33"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", choices=["lsl", "csv", "sim"], default="lsl")
    ap.add_argument("--stream-name", default=cfg.LSL_STREAM, help="LSL stream (default: %(default)s)")
    ap.add_argument("--csv", help="CSV to replay (header row; time and marker columns optional)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="replay/simulation speed, x real time (csv: 0 = as fast as possible)")
    ap.add_argument("--channels", default=",".join(cfg.CHANNELS),
                    help="comma-separated electrodes to average (default: %(default)s; "
                         "Oz falls back to POz on the Core-8)")
    ap.add_argument("--threshold", type=float, default=cfg.THRESHOLD, help="reward threshold, z")
    ap.add_argument("--baseline-s", type=float, default=cfg.BASELINE_S, help="baseline length, s")
    ap.add_argument("--tau", type=float, default=cfg.EMA_TAU_S, help="smoothing time constant, s")
    ap.add_argument("--no-artifacts", action="store_true", help="simulator: no artifact bursts")
    ap.add_argument("--ui", choices=["user", "debug", "both"], default="user",
                    help="user: calm participant view with sound (default); debug: the plots")
    ap.add_argument("--mute", action="store_true", help="user view: start with sound off")
    ap.add_argument("--no-auto-start", action="store_true",
                    help="user view: don't start the session by itself after calibration")
    ap.add_argument("--participant", default="anonymous",
                    help="name for the session history and personal bests (default: %(default)s)")
    ap.add_argument("--history", default=cfg.HISTORY_FILE, help="session history CSV (default: %(default)s)")
    ap.add_argument("--minutes", type=float, default=None,
                    help="preselect a timed session of this many minutes (default: indefinite)")
    return ap.parse_args(argv)


def build_debug_window(engine):
    import pyqtgraph as pg
    from PySide6 import QtCore, QtGui, QtWidgets

    win = QtWidgets.QWidget()
    win.setWindowTitle("Tandem (debug)")
    win.resize(1100, 800)
    layout = QtWidgets.QVBoxLayout(win)

    # --- status + readouts
    top = QtWidgets.QHBoxLayout()
    state = QtWidgets.QLabel("…")
    state.setFont(QtGui.QFont("Helvetica", 22, QtGui.QFont.Bold))
    state.setMinimumWidth(330)
    top.addWidget(state)
    readouts = {}
    for key, title in [("z", "alpha z (smoothed)"), ("rate", "reward rate"),
                       ("total", "total reward"), ("fb", "session (left)"), ("t", "plot time")]:
        box = QtWidgets.QVBoxLayout()
        cap = QtWidgets.QLabel(title)
        cap.setStyleSheet("color: #888")
        val = QtWidgets.QLabel("–")
        val.setFont(QtGui.QFont("Helvetica", 22 if key == "total" else 18, QtGui.QFont.Bold))
        box.addWidget(cap)
        box.addWidget(val)
        top.addLayout(box)
        readouts[key] = val
    layout.addLayout(top)

    # --- plots
    reward_plot = pg.PlotWidget(title="Cumulative reward")
    reward_plot.setLabel("bottom", "time (s)")
    reward_plot.showGrid(x=True, y=True, alpha=0.2)
    reward_curve = reward_plot.plot(pen=pg.mkPen("#2b2", width=3))
    reward_plot.setYRange(0, 1)
    best_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("#e90", width=2, style=QtCore.Qt.DashLine),
                                label="personal best", labelOpts=dict(position=0.03, color="#e90"))
    best_line.hide()
    reward_plot.addItem(best_line)

    z_plot = pg.PlotWidget(title="Alpha z-score (smoothed; raw in grey)")
    z_plot.setLabel("bottom", "time (s)")
    z_plot.showGrid(x=True, y=True, alpha=0.2)
    z_plot.setXLink(reward_plot)
    z_raw = z_plot.plot(pen=pg.mkPen("#666", width=1), connect="finite")
    z_curve = z_plot.plot(pen=pg.mkPen("#38f", width=2), connect="finite")
    z_plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#555")))
    thr_line = pg.InfiniteLine(pos=engine.reward.threshold, angle=0,
                               pen=pg.mkPen("#e90", width=2, style=QtCore.Qt.DashLine),
                               label="threshold", labelOpts=dict(position=0.03, color="#e90"))
    z_plot.addItem(thr_line)
    layout.addWidget(reward_plot, 3)
    layout.addWidget(z_plot, 2)

    # --- controls
    row = QtWidgets.QHBoxLayout()
    b_cal = QtWidgets.QPushButton("Start baseline calibration")
    mode = QtWidgets.QComboBox()
    mode.addItems(["Indefinite", "Timed"])
    minutes = QtWidgets.QSpinBox(minimum=1, maximum=180, value=int(engine.minutes), suffix=" min")
    b_fb = QtWidgets.QPushButton("Start session")
    b_stop = QtWidgets.QPushButton("End session")
    b_reset = QtWidgets.QPushButton("Reset reward")
    b_export = QtWidgets.QPushButton("Export CSV…")
    thr = QtWidgets.QDoubleSpinBox(minimum=-3, maximum=5, singleStep=0.1, decimals=2,
                                   value=engine.reward.threshold, prefix="threshold z = ")
    for w in (b_cal, mode, minutes, b_fb, b_stop, b_reset, b_export, thr):
        row.addWidget(w)
    layout.addLayout(row)

    # one checkbox per electrode: untick to exclude it; red while it is being auto-rejected
    ch_row = QtWidgets.QHBoxLayout()
    ch_row.addWidget(QtWidgets.QLabel("Electrodes:"))
    boxes = []
    for i, name in enumerate(engine.channels):
        box = QtWidgets.QCheckBox(name)
        box.setChecked(True)
        box.toggled.connect(lambda on, i=i: engine.set_included(i, on))
        ch_row.addWidget(box)
        boxes.append(box)
    ch_row.addStretch()
    layout.addLayout(ch_row)

    message = QtWidgets.QLabel()
    message.setWordWrap(True)
    layout.addWidget(message)
    note = QtWidgets.QLabel(
        "Each electrode has its own baseline; z is averaged over the clean, ticked ones. "
        "Baseline: 60 s eyes open, resting. The score tracks sustained posterior-alpha modulation relative to baseline, "
        "not a validated measure of meditation depth.")
    note.setWordWrap(True)
    note.setStyleSheet("color: #888")
    layout.addWidget(note)

    def on_threshold(v):
        engine.set_threshold(v)
        thr_line.setValue(v)

    def on_export():
        default = time.strftime("data/neurofeedback_%Y%m%d_%H%M%S.csv")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(win, "Export time series", default, "CSV (*.csv)")
        if path:
            n = engine.export_csv(path)
            engine.message = f"Exported {n} rows to {path}"

    b_cal.clicked.connect(engine.start_calibration)
    def on_mode(*_):
        engine.set_session(TIMED if mode.currentText() == "Timed" else INDEFINITE, minutes.value())

    mode.setCurrentText("Timed" if engine.mode == TIMED else "Indefinite")
    mode.currentTextChanged.connect(on_mode)
    minutes.valueChanged.connect(on_mode)
    b_fb.clicked.connect(engine.toggle_feedback)
    b_stop.clicked.connect(engine.stop_session)
    b_reset.clicked.connect(engine.reset)
    b_export.clicked.connect(on_export)
    thr.valueChanged.connect(on_threshold)

    def fmt(v, spec):
        return "–" if v is None or not np.isfinite(v) else format(v, spec)

    def refresh():
        s = engine.snapshot()
        last = s["last"]
        shown = POOR if last and not last["valid"] else s["state"]
        text = shown
        if s["calib_progress"]:
            done, total = s["calib_progress"]
            text = f"calibrating {done:.0f}/{total:.0f} s"
            if shown == POOR:
                text += " (poor signal)"
        elif shown == POOR and s["state"] == RECORDING:
            text = "poor signal (paused)"
        if s["done"]:
            text += " · stream ended"
        state.setText(text)
        state.setStyleSheet(f"color: {STATE_COLORS.get(shown, '#aaa')}")
        if shown == POOR:
            state.setToolTip(last["reason"])

        readouts["z"].setText(fmt(last.get("z_smooth"), "+.2f"))
        readouts["rate"].setText(fmt(last.get("reward_rate"), ".2f") + " /s")
        readouts["total"].setText(fmt(s["total"], ".1f"))
        sess = s["session"]
        if sess:
            secs = sess["elapsed"] if sess["mode"] == INDEFINITE else sess["minutes"] * 60 - sess["elapsed"]
            readouts["fb"].setText(time.strftime("%H:%M:%S", time.gmtime(max(0, secs))))
        else:
            readouts["fb"].setText("–")
        readouts["t"].setText(time.strftime("%M:%S", time.gmtime(last.get("time_s", 0) or 0)))

        msg = s["message"]
        if shown == POOR:
            msg = f"Poor signal, reward paused: {last['reason']}" + (f"  |  {msg}" if msg else "")
        elif last.get("reason"):
            msg = (f"Using {last['channels_used']}; dropped {last['reason']}"
                   + (f"  |  {msg}" if msg else ""))
        message.setText(msg)

        base = s["baseline"]
        for i, box in enumerate(boxes):
            tip = "no baseline" if base is not None and not base.available[i] else ""
            problem = last.get("problems", [""] * len(boxes))[i] if last else ""
            box.setStyleSheet("color: #e33" if problem or tip else "")
            box.setToolTip(problem or tip)

        has_baseline = s["baseline"] is not None
        in_session = s["state"] in (RECORDING, PAUSED)
        b_cal.setEnabled(s["state"] in ("needs baseline", READY, FINISHED))
        b_fb.setEnabled(has_baseline and s["state"] in (READY, RECORDING, PAUSED, FINISHED))
        b_fb.setText({RECORDING: "Pause", PAUSED: "Resume"}.get(s["state"], "Start session"))
        b_stop.setEnabled(in_session)
        b_reset.setEnabled(not in_session)
        mode.setEnabled(not in_session)
        minutes.setEnabled(not in_session and mode.currentText() == "Timed")
        best = sess["best_before"] if sess else None
        best_line.setVisible(best is not None)
        if best is not None:
            best_line.setValue(best)

        t = s["time_s"]
        reward_curve.setData(t, s["cumulative_reward"])
        reward_plot.setYRange(0, max(1.0, s["total"] * 1.1, (best or 0) * 1.1), padding=0)
        z_raw.setData(t, s["z_alpha"])
        z_curve.setData(t, s["z_smooth"])

    timer = QtCore.QTimer(win)
    timer.timeout.connect(refresh)
    timer.start(int(cfg.STEP_S * 1000))
    refresh()
    return win


def main(argv=None):
    args = parse_args(argv)
    source = make_source(args)
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    engine = Engine(source.fs, source.labels, channels, threshold=args.threshold,
                    baseline_s=args.baseline_s, tau_s=args.tau,
                    store=SessionStore(args.history, args.participant))
    if args.minutes:
        engine.set_session(TIMED, args.minutes)
    print(f"Source: {args.source}, {source.fs:g} Hz, channels {source.labels}")
    print(f"Alpha from: {engine.channels}")
    for w in engine.warnings:
        print("Warning:", w)

    from PySide6 import QtWidgets

    app = QtWidgets.QApplication([])
    worker = Worker(source, engine)
    worker.start()
    windows = []
    if args.ui in ("debug", "both"):
        windows.append(build_debug_window(engine))
    if args.ui in ("user", "both"):
        from .calm import CalmWindow
        from .sounds import SoundBoard

        windows.append(CalmWindow(engine, SoundBoard(muted=args.mute),
                                  auto_start=not args.no_auto_start))
    for w in windows:
        w.show()
    if hasattr(source, "alpha_gain"):
        # simulator: alpha rises while a session runs, as when the eyes close after an eyes-open baseline
        from PySide6 import QtCore

        sim_timer = QtCore.QTimer()
        sim_timer.timeout.connect(lambda: setattr(source, "alpha_gain",
                                                  3.0 if engine.state == RECORDING else 1.0))
        sim_timer.start(250)
    try:
        app.exec()
    finally:
        engine.stop_session()  # closing the window mid-session still saves it
        worker.stop()


if __name__ == "__main__":
    main()
