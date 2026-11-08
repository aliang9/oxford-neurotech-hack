"""Step through a held-out night and watch the HMM and HSMM forecasts evolve.

    python -m sleep.dashboard                       # first test night
    python -m sleep.dashboard --recording SC4XXX    # a specific night (any recording)
    python -m sleep.dashboard --screenshot figures/sleep_dashboard.png   # render and exit

Keys: Left/Right step 30 s, Shift+Left/Right step 5 min, Space play/pause.
Click the hypnogram to jump to that time.

Panels: the night's hypnogram with the forecast origin and its 30 min window; the stage
distribution over the next 30 min from the HMM and from the HSMM (stacked, with the actual
stages as a strip underneath); forecast uncertainty (normalised entropy) for both; and the
held-out metrics by horizon (models/sleep/metrics.json) with a reliability diagram.
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from . import config as cfg
from . import data
from .evaluate import stacked_states
from .models import forecast, forecast_bins, from_dict
from .train import MODEL_NAMES

H = 30 * cfg.EPOCHS_PER_MIN               # forecast window: 30 min in epochs

# Ink and surface (light), from the reference data-viz palette.
SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                         "#e1e0d9", "#c3c2b7")
# Stages: categorical slots 1-5, assigned in stacking order (W top ... N3 bottom).
STAGE_COLOR = {cfg.W: "#2a78d6", cfg.REM: "#eb6834", cfg.N1: "#1baf7a",
               cfg.N2: "#eda100", cfg.N3: "#e87ba4"}
STACK = [cfg.N3, cfg.N2, cfg.N1, cfg.REM, cfg.W]      # bottom -> top
LEVEL = {s: i for i, s in enumerate(STACK)}          # hypnogram height
# Models: hue = family, dash = stationary, solid = time-dependent.
MODEL_STYLE = {
    "persistence": ("#898781", QtCore.Qt.PenStyle.DotLine, "Persistence"),
    "hmm": ("#6250d6", QtCore.Qt.PenStyle.DashLine, "HMM"),
    "hmm_time": ("#6250d6", QtCore.Qt.PenStyle.SolidLine, "HMM, time-dep."),
    "hsmm": ("#008300", QtCore.Qt.PenStyle.DashLine, "HSMM"),
    "hsmm_time": ("#008300", QtCore.Qt.PenStyle.SolidLine, "HSMM, time-dep."),
}
METRICS = {"log_loss": "Log loss (nats)", "brier": "Brier score", "accuracy": "Accuracy",
           "ece": "Calibration error (ECE)"}


def pen(color, width=2, style=QtCore.Qt.PenStyle.SolidLine):
    p = pg.mkPen(color, width=width)
    p.setStyle(style)
    p.setCosmetic(True)
    return p


def style_plot(p, title=None, left=None, bottom=None):
    p.setMenuEnabled(False)
    p.hideButtons()
    for ax in ("left", "bottom"):
        a = p.getAxis(ax)
        a.setPen(pg.mkPen(AXIS))
        a.setTextPen(pg.mkPen(MUTED))
    p.showGrid(x=False, y=True, alpha=0.25)
    if title:
        p.setTitle(f"<span style='color:{INK};font-size:11pt'>{title}</span>")
    if left:
        p.setLabel("left", left, color=INK2)
    if bottom:
        p.setLabel("bottom", bottom, color=INK2)


def entropy(P):
    P = np.clip(P, 1e-12, 1)
    return -(P * np.log(P)).sum(-1) / np.log(cfg.K)


class Dashboard(QtWidgets.QWidget):
    def __init__(self, recs, models, metrics, start=None):
        super().__init__()
        self.recs, self.models, self.metrics = recs, models, metrics
        self.cache = {}
        self.setWindowTitle("Sleep-stage forecasts")
        self.setStyleSheet(f"background:{SURFACE}; color:{INK};")
        self.resize(1400, 900)

        # controls
        self.rec_box = QtWidgets.QComboBox()
        self.rec_box.addItems([f"{r.rec_id}  (subject {r.subject}, night {r.night})"
                               for r in recs])
        self.pair_box = QtWidgets.QComboBox()
        self.pair_box.addItems(["time-dependent models", "stationary models"])
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.play_btn = QtWidgets.QPushButton("Play")
        self.info = QtWidgets.QLabel()
        self.hover = QtWidgets.QLabel(" ")
        self.hover.setStyleSheet(f"color:{INK2};")
        for w in (self.info,):
            w.setStyleSheet(f"color:{INK}; font-weight:600;")
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Night"))
        top.addWidget(self.rec_box)
        top.addWidget(self.pair_box)
        top.addWidget(self.play_btn)
        top.addWidget(self.slider, 1)

        legend = QtWidgets.QHBoxLayout()
        for s in STACK[::-1]:
            sw = QtWidgets.QLabel()
            sw.setFixedSize(14, 14)
            sw.setStyleSheet(f"background:{STAGE_COLOR[s]}; border-radius:3px;")
            legend.addWidget(sw)
            legend.addWidget(QtWidgets.QLabel(cfg.STAGES[s]))
            legend.addSpacing(10)
        legend.addStretch(1)
        legend.addWidget(self.info)

        # plots
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(SURFACE)
        self.p_hyp = self.glw.addPlot(row=0, col=0, colspan=3)
        self.p_hmm = self.glw.addPlot(row=1, col=0)
        self.p_hsmm = self.glw.addPlot(row=1, col=1)
        self.p_unc = self.glw.addPlot(row=1, col=2)
        self.p_agg = self.glw.addPlot(row=2, col=0, colspan=2)
        self.p_rel = self.glw.addPlot(row=2, col=2)
        for p in (self.p_hyp, self.p_hmm, self.p_hsmm, self.p_unc, self.p_agg, self.p_rel):
            p.getAxis("left").setWidth(64)
        lay = self.glw.ci.layout
        for r, f in enumerate([2, 3, 3]):
            lay.setRowStretchFactor(r, f)

        style_plot(self.p_hyp, "Hypnogram (click to set the forecast origin)",
                   bottom="hours since sleep onset")
        self.p_hyp.getAxis("left").setTicks([[(LEVEL[s], cfg.STAGES[s]) for s in STACK]])
        self.p_hyp.setYRange(-0.6, 4.6, padding=0)
        self.p_hyp.showGrid(x=True, y=False, alpha=0.2)
        self.p_hyp.setMouseEnabled(x=False, y=False)
        self.p_hyp.scene().sigMouseClicked.connect(self.on_click)
        for p, name in ((self.p_hmm, "HMM"), (self.p_hsmm, "HSMM")):
            style_plot(p, f"{name}: next 30 min", left="P(stage)", bottom="minutes ahead")
            p.setYRange(-0.16, 1.0, padding=0)
            p.setXRange(0, 30, padding=0.01)
            p.setMouseEnabled(x=False, y=False)
            p.getAxis("left").setTicks([[(-0.09, "actual")] +
                                        [(v, f"{v:g}") for v in (0, 0.25, 0.5, 0.75, 1)]])
        style_plot(self.p_unc, "Uncertainty (normalised entropy)", bottom="minutes ahead")
        self.p_unc.setYRange(0, 1, padding=0.02)
        self.p_unc.setXRange(0, 30, padding=0.01)
        self.p_unc.setMouseEnabled(x=False, y=False)
        self.unc_legend = self.p_unc.addLegend(offset=(-5, -5), labelTextColor=INK2)

        self.metric_box = QtWidgets.QComboBox()
        self.metric_box.addItems(list(METRICS.values()))
        self.rel_box = QtWidgets.QComboBox()
        self.rel_box.addItems([f"{h:g} min" for h in metrics["horizons_min"]] if metrics
                              else [])
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(QtWidgets.QLabel("Held-out metric"))
        bottom.addWidget(self.metric_box)
        bottom.addSpacing(20)
        bottom.addWidget(QtWidgets.QLabel("Reliability at"))
        bottom.addWidget(self.rel_box)
        bottom.addSpacing(20)
        bottom.addWidget(self.hover, 1)

        root = QtWidgets.QVBoxLayout(self)
        root.addLayout(top)
        root.addLayout(legend)
        root.addWidget(self.glw, 1)
        root.addLayout(bottom)

        self.timer = QtCore.QTimer(self, interval=80, timeout=lambda: self.step(1, wrap=False))
        self.rec_box.currentIndexChanged.connect(self.set_recording)
        self.pair_box.currentIndexChanged.connect(lambda _: self.set_recording(self.ri))
        self.slider.valueChanged.connect(self.update_forecast)
        self.play_btn.clicked.connect(self.toggle_play)
        self.metric_box.currentIndexChanged.connect(self.draw_aggregate)
        self.rel_box.currentIndexChanged.connect(self.draw_reliability)
        self.mouse_proxy = [pg.SignalProxy(p.scene().sigMouseMoved, rateLimit=30,
                                           slot=lambda ev, p=p: self.on_hover(ev[0], p))
                            for p in (self.p_hmm,)]
        self.ri = 0
        self.draw_aggregate()
        self.draw_reliability()
        idx = 0 if start is None else next(i for i, r in enumerate(recs) if r.rec_id == start)
        self.rec_box.setCurrentIndex(idx)
        self.set_recording(idx)

    # --- data -------------------------------------------------------------------------
    def pair(self):
        return ("hmm_time", "hsmm_time") if self.pair_box.currentIndex() == 0 else ("hmm", "hsmm")

    def states(self, name):
        key = (self.ri, name)
        if key not in self.cache:
            self.cache[key] = stacked_states(self.models[name], self.recs[self.ri])
        return self.cache[key]

    def forecast_curves(self, name, t):
        """(H + 1, K): the filtered belief at the origin, then each 30 s step ahead."""
        m = self.models[name]
        rec = self.recs[self.ri]
        state = self.states(name)[t:t + 1]
        bins = forecast_bins([(t - rec.onset) / cfg.EPOCHS_PER_MIN], H, m.time_dependent)
        fc = forecast(m, state, bins, list(range(1, H + 1)))
        return np.vstack([m.marginal(state)] + [fc[k] for k in range(1, H + 1)])

    # --- drawing ----------------------------------------------------------------------
    def set_recording(self, i):
        self.ri = i
        rec = self.recs[i]
        st = rec.stages.astype(int)
        self.hours = (np.arange(len(st)) - rec.onset) / 120.0
        p = self.p_hyp
        p.clear()
        w = 1 / 120.0
        starts = np.flatnonzero(np.diff(st, prepend=-2) != 0)      # one bar per run
        ends = np.append(starts[1:], len(st))
        for s in STACK:
            sel = st[starts] == s
            if sel.any():
                p.addItem(pg.BarGraphItem(x0=self.hours[starts[sel]],
                                          width=(ends[sel] - starts[sel]) * w, y=LEVEL[s],
                                          height=0.55, brush=STAGE_COLOR[s], pen=None))
        lv = np.array([LEVEL[s] if s >= 0 else np.nan for s in st], float)
        xs = np.repeat(self.hours, 2)[1:]
        ys = np.repeat(lv, 2)[:-1]
        p.plot(np.append(xs, self.hours[-1] + w), np.append(ys, ys[-1]),
               pen=pen(INK, 1), connect="finite")
        self.window = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(11, 11, 11, 22),
                                          pen=pg.mkPen(None))
        self.origin_line = pg.InfiniteLine(angle=90, pen=pen(INK, 2))
        p.addItem(self.window)
        p.addItem(self.origin_line)
        p.setXRange(self.hours[0], self.hours[-1] + w, padding=0.01)
        self.slider.blockSignals(True)
        self.slider.setRange(rec.onset, len(st) - 1)
        self.slider.setValue(min(rec.onset + 4 * 120, len(st) - 1))  # 4 h after onset
        self.slider.blockSignals(False)
        self.update_forecast()

    def update_forecast(self, *_):
        t = self.slider.value()
        rec = self.recs[self.ri]
        st = rec.stages.astype(int)
        h0 = self.hours[t]
        self.origin_line.setValue(h0)
        self.window.setRegion((h0, h0 + 0.5))
        x = np.arange(H + 1) / cfg.EPOCHS_PER_MIN
        future = st[t:t + H + 1]
        self.curves = {}
        self.p_unc.clear()
        self.unc_legend.clear()
        for plot, name in zip((self.p_hmm, self.p_hsmm), self.pair()):
            P = self.forecast_curves(name, t)
            self.curves[name] = P
            plot.clear()
            cum = np.zeros(H + 1)
            prev = plot.plot(x, cum, pen=pen(SURFACE, 1))   # FillBetween needs a drawn curve
            for s in STACK:
                cum = cum + P[:, s]
                cur = plot.plot(x, cum, pen=pen(SURFACE, 1))
                plot.addItem(pg.FillBetweenItem(prev, cur, brush=STAGE_COLOR[s]))
                prev = cur
            ok = future >= 0
            for s in STACK:
                idx = np.flatnonzero(ok & (future == s))
                if len(idx):
                    plot.addItem(pg.BarGraphItem(x0=x[idx], width=0.5, y=-0.09, height=0.08,
                                                 brush=STAGE_COLOR[s], pen=None))
            ll = -np.log(np.clip(P[np.arange(len(future))[ok], future[ok]], 1e-12, 1))
            color, style, label = MODEL_STYLE[name]
            plot.setTitle(f"<span style='color:{INK};font-size:11pt'>{label}: next 30 min"
                          f"</span>  <span style='color:{INK2};font-size:9pt'>"
                          f"mean log loss on this window {ll.mean():.2f}</span>")
            self.p_unc.plot(x, entropy(P), pen=pen(color, 2, style), name=label)
        cur = cfg.STAGES[st[t]] if st[t] >= 0 else "unscored"
        run = 1
        while t - run >= 0 and st[t - run] == st[t]:
            run += 1
        self.info.setText(f"origin {h0:+.2f} h after onset   current {cur} "
                          f"for {run / 2:g} min")

    def draw_aggregate(self, *_):
        p = self.p_agg
        p.clear()
        if not self.metrics:
            style_plot(p, "No metrics.json: run python -m sleep.evaluate")
            return
        key = list(METRICS)[self.metric_box.currentIndex()]
        style_plot(p, f"Held-out {METRICS[key].lower()} by horizon  "
                      f"<span style='color:{INK2};font-size:9pt'>"
                      f"({self.metrics['description']}; lower is better"
                      f"{', except accuracy' if key == 'accuracy' else ''})</span>",
                   bottom="horizon (minutes)")
        hs = np.array(self.metrics["horizons_min"])
        if not hasattr(self, "agg_legend"):
            self.agg_legend = p.addLegend(offset=(10, 5), labelTextColor=INK2,
                                          colCount=5)
        self.agg_legend.clear()
        for name in MODEL_NAMES:
            m = self.metrics["models"].get(name)
            if not m:
                continue
            vals = [m[str(int(h * cfg.EPOCHS_PER_MIN))][key] for h in hs]
            color, style, label = MODEL_STYLE[name]
            p.plot(hs, vals, pen=pen(color, 2, style), symbol="o", symbolSize=8,
                   symbolBrush=color, symbolPen=pg.mkPen(SURFACE, width=2), name=label)
            if key == "log_loss":   # subject-bootstrap 95% CI
                ci = np.array([m[str(int(h * cfg.EPOCHS_PER_MIN))]["log_loss_ci"]
                               for h in hs])
                p.addItem(pg.ErrorBarItem(x=hs, y=np.array(vals), top=ci[:, 1] - vals,
                                          bottom=vals - ci[:, 0], beam=0.3,
                                          pen=pen(color, 1)))
        p.getAxis("bottom").setTicks([[(h, f"{h:g}") for h in hs]])
        p.setXRange(hs[0] - 1.5, hs[-1] + 1.5, padding=0)

    def draw_reliability(self, *_):
        p = self.p_rel
        p.clear()
        if not self.metrics:
            return
        h = self.metrics["horizons_min"][self.rel_box.currentIndex()]
        style_plot(p, f"Reliability, {h:g} min ahead (lines as left)", left="observed frequency",
                   bottom="predicted probability")
        p.plot([0, 1], [0, 1], pen=pen(AXIS, 1, QtCore.Qt.PenStyle.DashLine))
        for name in ("hmm_time", "hsmm_time", "hmm", "hsmm"):
            r = self.metrics["models"][name][str(int(h * cfg.EPOCHS_PER_MIN))]["reliability"]
            cnt = np.array(r["count"])
            ok = cnt >= 30
            color, style, label = MODEL_STYLE[name]
            p.plot(np.array(r["mean_pred"])[ok], np.array(r["frac"])[ok],
                   pen=pen(color, 2, style), symbol="o", symbolSize=8, symbolBrush=color,
                   symbolPen=pg.mkPen(SURFACE, width=2))
        p.setRange(xRange=(0, 1), yRange=(0, 1), padding=0.03)

    # --- interaction ------------------------------------------------------------------
    def on_hover(self, pos, plot):
        for p in (self.p_hmm, self.p_hsmm):
            if p.sceneBoundingRect().contains(pos):
                xm = p.vb.mapSceneToView(pos).x()
                k = int(np.clip(round(xm * cfg.EPOCHS_PER_MIN), 0, H))
                parts = []
                for name in self.pair():
                    P = self.curves[name][k]
                    parts.append(MODEL_STYLE[name][2] + ": " + "  ".join(
                        f"{cfg.STAGES[s]} {P[s]:.2f}" for s in STACK[::-1]))
                self.hover.setText(f"+{k / 2:g} min   " + "   |   ".join(parts))
                return

    def on_click(self, ev):
        if self.p_hyp.sceneBoundingRect().contains(ev.scenePos()):
            hx = self.p_hyp.vb.mapSceneToView(ev.scenePos()).x()
            t = int(round(hx * 120 + self.recs[self.ri].onset))
            self.slider.setValue(int(np.clip(t, self.slider.minimum(), self.slider.maximum())))

    def step(self, n, wrap=True):
        v = self.slider.value() + n
        if v > self.slider.maximum():
            if not wrap:
                return self.toggle_play()
            v = self.slider.maximum()
        self.slider.setValue(max(v, self.slider.minimum()))

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("Play")
        else:
            self.timer.start()
            self.play_btn.setText("Pause")

    def keyPressEvent(self, ev):
        n = 10 if ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier else 1
        if ev.key() == QtCore.Qt.Key.Key_Right:
            self.step(n)
        elif ev.key() == QtCore.Qt.Key.Key_Left:
            self.step(-n)
        elif ev.key() == QtCore.Qt.Key.Key_Space:
            self.toggle_play()
        else:
            super().keyPressEvent(ev)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default=os.path.join(cfg.MODELS_DIR, "models.joblib"))
    ap.add_argument("--metrics", default=None, help="default: metrics.json next to the models")
    ap.add_argument("--recording", default=None, help="e.g. SC4XXX; default: first test night")
    ap.add_argument("--all", action="store_true", help="list every night, not only test ones")
    ap.add_argument("--origin-hours", type=float, default=None,
                    help="initial origin, hours after sleep onset")
    ap.add_argument("--screenshot", default=None, help="save a PNG of the dashboard and exit")
    args = ap.parse_args(argv)

    blob = joblib.load(args.models)
    models = {n: from_dict(blob["models"][n]) for n in MODEL_NAMES}
    recs = data.load()
    if not args.all and not args.recording:
        recs = [r for r in recs if r.subject in set(blob["test_subjects"])]
    if args.recording:
        recs = [r for r in recs if r.rec_id == args.recording] or sys.exit(
            f"no recording {args.recording}")
        if recs[0].subject not in set(blob["test_subjects"]):
            print(f"note: {args.recording} was in the training set of these models")
    mpath = args.metrics or os.path.join(os.path.dirname(args.models), "metrics.json")
    metrics = json.load(open(mpath)) if os.path.exists(mpath) else None

    pg.setConfigOptions(antialias=True, background=SURFACE, foreground=INK2)
    app = pg.mkQApp("Sleep-stage forecasts")
    app.setFont(QtGui.QFont(app.font().family(), 11))
    win = Dashboard(recs, models, metrics)
    if args.origin_hours is not None:
        win.slider.setValue(recs[0].onset + int(round(args.origin_hours * 120)))
    if args.screenshot:
        win.show()
        app.processEvents()
        os.makedirs(os.path.dirname(args.screenshot) or ".", exist_ok=True)
        win.grab().save(args.screenshot)
        print(f"wrote {args.screenshot}")
        return
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
