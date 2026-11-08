"""Smart alarm: replay a held-out night and watch the Bayesian alarm decide when to wake.

    python -m smartalarm                       # first held-out night, simulated EEG stager
    python -m smartalarm --night SC4XXX --speed 600 --input labels
    python -m smartalarm --screenshot out.png --at 06:10   # render headless and exit

Left: the phone (clock, countdown, current stage probabilities, recommendation, wake-up
window, start / pause / reset). Right: the replay lab (night, input, policy, speed, jump,
ground truth) with three charts:
  the night so far   probabilistic hypnogram: shade = the model's posterior per stage;
                     the lighter block after "now" is the forecast; the dashed ink line is
                     the true hypnogram (hidden from the model; dotted after "now")
  next 60 minutes    the HSMM forecast, stacked, with the true stages as a strip underneath
  cost of waking     expected cost of waking at each moment of the window, if decided now,
                     and the policy's two numbers: wake now vs keep sleeping and re-check
Keys: Space start/pause, R reset, J jump to 30 min before the window.
"""

import argparse
import datetime as dt
import math
import os
import sys

import joblib
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen

from sleep import data
from sleep.models import from_dict
from . import config as cfg
from .policy import cost_vector, open_loop_costs
from .session import Session
from .sources import ReplaySource, stager_prior

# --- look: warm paper, ink outlines, pastel blue (the sleep-tracking mockup) -------------
PAPER, CARD, INK, INK2, MUTED, LINE = ("#FBF7EE", "#FFFFFF", "#1F1F1F", "#4A4843",
                                       "#8A8781", "#E6E1D6")
BLUE, BLUE_TINT, BLUE_PALE, NAVY = "#83A9DB", "#DCE7F6", "#EEF3FA", "#2F4B7C"
BLUE_DARK_TXT = "#4F78B3"
STAR, GREEN, ORANGE, CORAL = "#F4C542", "#4CAF50", "#E88A49", "#E8735A"
# Stages: deeper sleep = deeper blue; wake = sunny yellow; REM = coral
STAGE_COLOR = {"W": "#F4C542", "N1": "#B9D0EE", "N2": "#83A9DB", "N3": "#2F4B7C",
               "REM": "#E8735A"}
ROWS = ["W", "REM", "N1", "N2", "N3"]                     # hypnogram, top to bottom
STACK = ["N3", "N2", "N1", "REM", "W"]                    # stacked area, bottom to top
SI = {s: i for i, s in enumerate(cfg.STAGES)}
SPEEDS = [("Real time", 1), ("60×", 60), ("300×", 300), ("900×", 900), ("3000×", 3000)]
FORECAST_MIN = 60


def font(px, bold=False, rounded=True):
    f = QFont()
    f.setFamilies(["Arial Rounded MT Bold", "Nunito", "Avenir Next", "Helvetica Neue"]
                  if rounded else ["Avenir Next", "Helvetica Neue", "Segoe UI", "Arial"])
    f.setPixelSize(int(px))
    f.setWeight(QFont.Bold if bold else QFont.Normal)
    return f


def mix(c1, c2, t):
    a, b = QColor(c1), QColor(c2)
    return np.array([a.red() + (b.red() - a.red()) * t, a.green() + (b.green() - a.green()) * t,
                     a.blue() + (b.blue() - a.blue()) * t])


def pen(color, w=2.0, style=Qt.SolidLine):
    p = pg.mkPen(color, width=w)
    p.setStyle(style)
    p.setCapStyle(Qt.RoundCap)
    p.setCosmetic(True)
    return p


class Card(QtWidgets.QFrame):
    """White rounded card with an ink outline."""

    def __init__(self, radius=18, fill=CARD, border=INK, width=2):
        super().__init__()
        self.setObjectName("card")
        self.setStyleSheet(f"#card {{ background:{fill}; border:{width}px solid {border};"
                           f" border-radius:{radius}px; }}")
        self.lay = QtWidgets.QVBoxLayout(self)
        self.lay.setContentsMargins(16, 12, 16, 12)
        self.lay.setSpacing(6)


def label(text="", px=14, bold=False, color=INK, rounded=True, align=None, wrap=False):
    w = QtWidgets.QLabel(text)
    w.setFont(font(px, bold, rounded))
    w.setStyleSheet(f"color:{color}; background:transparent; border:none;")
    if align is not None:
        w.setAlignment(align)
    w.setWordWrap(wrap)
    return w


# --- the illustration --------------------------------------------------------------------

class Scene(QtWidgets.QWidget):
    """Sleeping blob under a blanket on a cloud; stars, moon and floating Zs. When the alarm
    fires: eyes open, the sun comes up, a little shake."""

    def __init__(self):
        super().__init__()
        self.t = 0.0
        self.awake = 0.0          # 0 asleep .. 1 awake (eased)
        self.awake_target = 0.0
        self.ring = 0.0           # alarm shake, decays
        self.setMinimumHeight(190)

    def tick(self, dt_s):
        self.t += dt_s
        self.awake += (self.awake_target - self.awake) * min(1, dt_s / 0.6)
        self.ring = max(0.0, self.ring - dt_s / 2.5)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        s = min(w / 340, h / 200)
        p.translate(w / 2, h / 2 + 10 * s)
        p.scale(s, s)
        ink = QPen(QColor(INK), 2.4)
        ink.setCapStyle(Qt.RoundCap)
        ink.setJoinStyle(Qt.RoundJoin)
        a = self.awake

        # stars twinkle, fade at dawn
        for i, (x, y, r) in enumerate([(-125, -62, 9), (-150, -10, 6), (70, -70, 7),
                                       (140, -20, 5), (-95, -88, 5)]):
            tw = 0.75 + 0.25 * math.sin(self.t * 1.7 + i * 1.3)
            self.star(p, x, y, r * tw, 1 - a, ink)
        # moon -> sun
        if a < 0.99:
            p.setOpacity(1 - a)
            p.setPen(ink)
            p.setBrush(QColor(STAR))
            moon = QPainterPath()
            moon.addEllipse(QPointF(118, -82), 16, 16)
            cut = QPainterPath()
            cut.addEllipse(QPointF(127, -88), 14, 14)
            p.drawPath(moon.subtracted(cut))
            p.setOpacity(1)
        if a > 0.01:
            p.setOpacity(a)
            p.setPen(ink)
            p.setBrush(QColor(STAR))
            p.drawEllipse(QPointF(118, -78), 15, 15)
            for k in range(8):
                th = k * math.pi / 4 + self.t * 0.4
                p.drawLine(QPointF(118 + 21 * math.cos(th), -78 + 21 * math.sin(th)),
                           QPointF(118 + 28 * math.cos(th), -78 + 28 * math.sin(th)))
            p.setOpacity(1)

        shake = 4 * self.ring * math.sin(self.t * 40)
        p.translate(shake, 0)
        breath = math.sin(self.t * 2 * math.pi / 5) * (1 - a)
        # cloud
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(BLUE_TINT))
        for cx, cy, r in [(-85, 62, 30), (-40, 52, 40), (15, 50, 44), (70, 56, 36), (110, 66, 24),
                          (-120, 72, 18)]:
            p.drawEllipse(QPointF(cx, cy), r, r * 0.62)
        p.drawRoundedRect(QRectF(-135, 55, 260, 30), 15, 15)
        # head (rises when awake)
        hx, hy = -55, 8 - 18 * a
        p.setPen(ink)
        p.setBrush(QColor(CARD))
        p.drawEllipse(QPointF(hx, hy), 34, 32)
        # face
        if a < 0.5:
            for ex in (-14, 6):
                p.drawArc(QRectF(hx + ex - 6, hy - 6, 12, 10), 200 * 16, 140 * 16)
            p.drawArc(QRectF(hx - 8, hy + 6, 12, 8), 200 * 16, 140 * 16)
        else:
            p.setBrush(QColor(INK))
            for ex in (-10, 10):
                p.drawEllipse(QPointF(hx + ex - 2, hy - 3), 3.2, 4.2)
            p.setBrush(Qt.NoBrush)
            p.drawArc(QRectF(hx - 10, hy + 2, 18, 12), 200 * 16, 140 * 16)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 180, 160, 120))
        p.drawEllipse(QPointF(hx - 22, hy + 10), 6, 4)
        p.drawEllipse(QPointF(hx + 16, hy + 10), 6, 4)
        # blanket
        p.setPen(ink)
        p.setBrush(QColor(BLUE))
        bl = QPainterPath()
        top = 18 - 3 * breath
        bl.moveTo(-100, 58)
        bl.cubicTo(-100, top + 10, -60, top - 4, -20, top + 2)
        bl.cubicTo(30, top - 18, 95, top - 8, 105, 40)
        bl.cubicTo(110, 52, 100, 60, 90, 60)
        bl.lineTo(-90, 60)
        bl.cubicTo(-98, 60, -100, 58, -100, 58)
        p.drawPath(bl)
        p.setPen(QPen(QColor(CARD), 2.0))
        p.drawLine(QPointF(-28, top + 12), QPointF(-14, top + 30))
        p.drawLine(QPointF(30, top + 2), QPointF(42, top + 22))
        # Zzz
        p.setPen(QColor(INK))
        for k in range(3):
            ph = (self.t * 0.35 + k / 3) % 1
            p.setOpacity((1 - a) * math.sin(ph * math.pi))
            p.setFont(font(14 + 6 * k, True))
            p.drawText(QPointF(-18 + 16 * k + 10 * ph, -34 - 16 * k - 20 * ph), "z")
        p.setOpacity(1)

    @staticmethod
    def star(p, x, y, r, alpha, ink):
        if alpha <= 0.01:
            return
        p.setOpacity(alpha)
        path = QPainterPath()
        for k in range(8):
            rr = r if k % 2 == 0 else r * 0.38
            th = k * math.pi / 4 - math.pi / 2
            pt = QPointF(x + rr * math.cos(th), y + rr * math.sin(th))
            path.moveTo(pt) if k == 0 else path.lineTo(pt)
        path.closeSubpath()
        p.setPen(QPen(QColor(INK), 1.8))
        p.setBrush(QColor(STAR))
        p.drawPath(path)
        p.setOpacity(1)


class StageBars(QtWidgets.QWidget):
    """Five rounded bars: the current posterior P(stage)."""

    def __init__(self):
        super().__init__()
        self.p = np.full(5, 0.2)
        self.setMinimumHeight(118)

    def set(self, p):
        self.p = np.asarray(p, float)
        self.update()

    def paintEvent(self, _):
        q = QPainter(self)
        q.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        rowh = self.height() / 5
        for i, s in enumerate(ROWS):
            v = float(self.p[SI[s]])
            y = i * rowh + rowh * 0.18
            bh = rowh * 0.64
            q.setPen(QColor(INK2))
            q.setFont(font(13, True))
            q.drawText(QRectF(0, y - 2, 40, bh + 4), Qt.AlignVCenter, s)
            x0, x1 = 44, w - 46
            q.setPen(Qt.NoPen)
            q.setBrush(QColor(BLUE_PALE))
            q.drawRoundedRect(QRectF(x0, y, x1 - x0, bh), bh / 2, bh / 2)
            if v > 0.005:
                q.setBrush(QColor(STAGE_COLOR[s]))
                q.drawRoundedRect(QRectF(x0, y, max(bh, (x1 - x0) * v), bh), bh / 2, bh / 2)
            q.setPen(QColor(INK))
            q.setFont(font(13))
            q.drawText(QRectF(x1 + 4, y - 2, 42, bh + 4), Qt.AlignVCenter | Qt.AlignRight,
                       f"{v:.0%}")


# --- charts --------------------------------------------------------------------------------

class ClockAxis(pg.AxisItem):
    """Epoch index -> HH:MM on the replay clock."""

    def __init__(self, win, **kw):
        super().__init__(orientation="bottom", **kw)
        self.win = win

    def tickStrings(self, values, scale, spacing):
        return [self.win.session.clock(int(v)).strftime("%H:%M") if self.win.session else ""
                for v in values]

    def tickValues(self, minVal, maxVal, size):
        span = maxVal - minVal
        step = 120 if span > 400 else 20 if span > 100 else 10
        if not self.win.session:
            return []
        start_sec = self.win.session.start_clock.minute * 60 + self.win.session.start_clock.second
        off = (-(start_sec // 30)) % step                     # align ticks to clock minutes
        first = math.ceil((minVal - off) / step) * step + off
        return [(step, list(np.arange(first, maxVal + 1, step)))]


def chart(win, title, height):
    card = Card(radius=16, border=LINE, width=2)
    card.lay.addWidget(label(title, 15, True))
    pw = pg.PlotWidget(axisItems={"bottom": ClockAxis(win)})
    pw.setBackground(CARD)
    pw.setMinimumHeight(height)
    pi = pw.getPlotItem()
    pi.setMenuEnabled(False)
    pi.hideButtons()
    pi.setMouseEnabled(x=False, y=False)
    for ax in ("left", "bottom"):
        a = pi.getAxis(ax)
        a.setPen(pg.mkPen(LINE))
        a.setTextPen(pg.mkPen(MUTED))
        a.setTickFont(font(11, rounded=False))
    pi.getAxis("left").setWidth(44)
    card.lay.addWidget(pw)
    return card, pi


# --- main window ---------------------------------------------------------------------------

class AlarmApp(QtWidgets.QWidget):
    def __init__(self, model, nights, prior, args):
        super().__init__()
        self.model, self.nights, self.prior = model, nights, prior
        self.stage_cost = dict(cfg.STAGE_COST)
        self.sleep_value = cfg.SLEEP_VALUE_PER_MIN
        self.costs = cost_vector(self.stage_cost)
        self.session = None
        self.running = False
        self.acc = 0.0
        self.last_draw = 0.0
        self.speed = args.speed
        self.compare = {}
        try:
            from neurofeedback.sounds import SoundBoard
            self.sounds = SoundBoard(muted=args.mute)
        except Exception:  # noqa: BLE001 - sound is optional
            self.sounds = None

        self.setWindowTitle("Smart Alarm")
        self.setStyleSheet(f"background:{PAPER}; color:{INK};")
        self.resize(1440, 900)
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(20)
        root.addWidget(self.build_phone(), 0)
        root.addLayout(self.build_lab(), 1)

        self.start_btn.setFocus()
        self.timer = QtCore.QTimer(self, interval=50, timeout=self.tick)
        self.timer.start()
        self.elapsed = QtCore.QElapsedTimer()
        self.elapsed.start()

        start = 0
        if args.night:
            start = next((i for i, r in enumerate(nights) if r.rec_id == args.night), 0)
        self.night_box.setCurrentIndex(start)
        self.load_night()

    # --- layout -----------------------------------------------------------------------
    def build_phone(self):
        phone = Card(radius=38, width=3)
        phone.setFixedWidth(400)
        L = phone.lay
        L.setContentsMargins(22, 16, 22, 18)
        L.setSpacing(10)

        top = QtWidgets.QHBoxLayout()
        self.pill = label("SLEEP", 14, True, NAVY, align=Qt.AlignCenter)
        self.pill.setFixedHeight(30)
        self.pill.setMinimumWidth(120)
        top.addStretch(1)
        top.addWidget(self.pill)
        top.addStretch(1)
        L.addLayout(top)

        self.scene = Scene()
        L.addWidget(self.scene, 1)
        L.addWidget(label("Smart Alarm", 22, True, align=Qt.AlignCenter))
        self.clock_lbl = label("--:--:--", 46, True, align=Qt.AlignCenter)
        L.addWidget(self.clock_lbl)
        self.countdown = label("", 14, False, INK2, align=Qt.AlignCenter)
        L.addWidget(self.countdown)

        mon = Card(radius=16)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(label("EEG Sleep Monitoring", 15, True))
        row.addStretch(1)
        self.live_dot = label("●", 16, color=GREEN)
        self.live_lbl = label("Replay", 14)
        row.addWidget(self.live_dot)
        row.addWidget(self.live_lbl)
        mon.lay.addLayout(row)
        self.stage_lbl = label("Current stage: –", 16, True)
        mon.lay.addWidget(self.stage_lbl)
        self.bars = StageBars()
        mon.lay.addWidget(self.bars)
        L.addWidget(mon)

        rec = Card(radius=16, fill=BLUE_PALE)
        self.rec_title = label("Waiting", 20, True)
        self.rec_text = label("", 13, False, INK2, rounded=False, wrap=True)
        rec.lay.addWidget(self.rec_title)
        rec.lay.addWidget(self.rec_text)
        L.addWidget(rec)

        win = Card(radius=16)
        win.lay.addWidget(label("Wake-up window", 15, True))
        r = QtWidgets.QHBoxLayout()
        r.addWidget(label("earliest", 13, False, INK2))
        self.earliest_edit = QtWidgets.QTimeEdit()
        self.latest_edit = QtWidgets.QTimeEdit()
        for e in (self.earliest_edit, self.latest_edit):
            e.setDisplayFormat("HH:mm")
            e.setFont(font(15, True))
            e.setStyleSheet(f"QTimeEdit {{ background:{BLUE_PALE}; border:2px solid {INK};"
                            f" border-radius:10px; padding:2px 6px; color:{INK}; }}")
            e.editingFinished.connect(self.apply_window)
        r.addWidget(self.earliest_edit)
        r.addStretch(1)
        r.addWidget(label("latest", 13, False, INK2))
        r.addWidget(self.latest_edit)
        win.lay.addLayout(r)
        L.addWidget(win)

        btns = QtWidgets.QHBoxLayout()
        self.start_btn = self.button("Start", BLUE)
        self.reset_btn = self.button("Reset", CARD)
        self.start_btn.clicked.connect(self.toggle)
        self.reset_btn.clicked.connect(self.reset)
        btns.addWidget(self.start_btn, 2)
        btns.addWidget(self.reset_btn, 1)
        L.addLayout(btns)
        return phone

    def button(self, text, fill):
        b = QtWidgets.QPushButton(text)
        b.setFont(font(16, True))
        b.setCursor(Qt.PointingHandCursor)
        b.setMinimumHeight(42)
        b.setStyleSheet(f"QPushButton {{ background:{fill}; color:{INK}; border:2px solid {INK};"
                        f" border-radius:21px; padding:4px 14px; }}"
                        f"QPushButton:hover {{ background:{BLUE_TINT}; }}"
                        f"QPushButton:focus {{ border:3px solid {NAVY}; }}")
        return b

    def build_costs(self):
        """Editable cost of waking per stage and value of a minute of sleep."""
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(label("Cost of waking from", 13, True))
        spin_css = (f"QDoubleSpinBox {{ background:{CARD}; border:2px solid {INK};"
                    f" border-radius:10px; padding:2px 4px; color:{INK}; }}")
        self.cost_spins = {}
        for st in ["W", "N1", "N2", "REM", "N3"]:
            sw = QtWidgets.QLabel()
            sw.setFixedSize(12, 12)
            sw.setStyleSheet(f"background:{STAGE_COLOR[st]}; border:1.5px solid {INK};"
                             " border-radius:6px;")
            row.addSpacing(6)
            row.addWidget(sw)
            row.addWidget(label(st, 13, True))
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(0, 50)
            sp.setSingleStep(0.5)
            sp.setDecimals(1)
            sp.setValue(self.stage_cost[st])
            sp.setFont(font(13, rounded=False))
            sp.setStyleSheet(spin_css)
            sp.setFixedWidth(64)
            sp.valueChanged.connect(self.costs_edited)
            self.cost_spins[st] = sp
            row.addWidget(sp)
        row.addSpacing(14)
        row.addWidget(label("+ per minute of sleep given up", 13, True))
        self.sleep_spin = QtWidgets.QDoubleSpinBox()
        self.sleep_spin.setRange(0, 1)
        self.sleep_spin.setSingleStep(0.01)
        self.sleep_spin.setDecimals(2)
        self.sleep_spin.setValue(self.sleep_value)
        self.sleep_spin.setFont(font(13, rounded=False))
        self.sleep_spin.setStyleSheet(spin_css)
        self.sleep_spin.setFixedWidth(70)
        self.sleep_spin.valueChanged.connect(self.costs_edited)
        row.addWidget(self.sleep_spin)
        reset = self.button("Defaults", CARD)
        reset.setFont(font(13, True))
        reset.setMinimumHeight(30)
        reset.clicked.connect(self.default_costs)
        row.addSpacing(8)
        row.addWidget(reset)
        row.addStretch(1)
        # apply after the user stops typing / clicking, not on every intermediate value
        self.cost_timer = QtCore.QTimer(self, singleShot=True, interval=350,
                                        timeout=self.apply_costs)
        return row

    def costs_edited(self, *_):
        self.cost_timer.start()

    def default_costs(self):
        for st, sp in self.cost_spins.items():
            sp.blockSignals(True)
            sp.setValue(cfg.STAGE_COST[st])
            sp.blockSignals(False)
        self.sleep_spin.blockSignals(True)
        self.sleep_spin.setValue(cfg.SLEEP_VALUE_PER_MIN)
        self.sleep_spin.blockSignals(False)
        self.apply_costs()

    def apply_costs(self):
        """New costs: rebuild the session and catch up causally to the current epoch."""
        self.stage_cost = {st: sp.value() for st, sp in self.cost_spins.items()}
        self.sleep_value = self.sleep_spin.value()
        self.costs = cost_vector(self.stage_cost)
        has = self.session.inf.alpha is not None
        was_running = self.running
        self.make_session(keep_epoch=self.session.now + 1 if has else None)  # seek(n) replays n epochs
        if was_running and not self.session.fired:
            self.toggle()

    def build_lab(self):
        lab = QtWidgets.QVBoxLayout()
        lab.setSpacing(12)
        head = QtWidgets.QHBoxLayout()
        head.addWidget(label("Replay lab", 24, True))
        head.addSpacing(10)
        head.addWidget(label("held-out Sleep-EDF nights the model never trained on", 13,
                             False, MUTED, rounded=False))
        head.addStretch(1)
        lab.addLayout(head)

        ctl = QtWidgets.QHBoxLayout()
        ctl.setSpacing(8)
        combo_css = (f"QComboBox {{ background:{CARD}; border:2px solid {INK}; border-radius:10px;"
                     f" padding:3px 8px; color:{INK}; }}")
        self.night_box = QtWidgets.QComboBox()
        self.night_box.addItems([f"{r.rec_id} · subject {r.subject}, night {r.night}"
                                 for r in self.nights])
        self.input_box = QtWidgets.QComboBox()
        self.input_box.addItems(["Simulated EEG stager", "Perfect stage labels"])
        self.policy_box = QtWidgets.QComboBox()
        self.policy_box.addItems(["Bayesian alarm", "Reactive alarm", "Fixed alarm"])
        self.speed_box = QtWidgets.QComboBox()
        self.speed_box.addItems([n for n, _ in SPEEDS])
        self.speed_box.setCurrentIndex(min(range(len(SPEEDS)),
                                           key=lambda i: abs(SPEEDS[i][1] - self.speed)))
        for b, lbl in ((self.night_box, "Night"), (self.input_box, "Input"),
                       (self.policy_box, "Policy"), (self.speed_box, "Speed")):
            b.setFont(font(13, rounded=False))
            b.setStyleSheet(combo_css)
            ctl.addWidget(label(lbl, 13, True))
            ctl.addWidget(b)
        self.jump_btn = self.button("Jump to 30 min before window", CARD)
        self.jump_btn.setFont(font(13, True))
        self.jump_btn.setMinimumHeight(34)
        self.jump_btn.clicked.connect(self.jump_before_window)
        ctl.addWidget(self.jump_btn)
        self.truth_chk = QtWidgets.QCheckBox("Show true stages")
        self.truth_chk.setFont(font(13, rounded=False))
        self.truth_chk.setChecked(True)
        self.truth_chk.toggled.connect(lambda _: self.redraw(force=True))
        ctl.addWidget(self.truth_chk)
        ctl.addStretch(1)
        lab.addLayout(ctl)
        lab.addLayout(self.build_costs())

        self.night_box.currentIndexChanged.connect(lambda _: self.load_night())
        self.input_box.currentIndexChanged.connect(lambda _: self.reset())
        self.policy_box.currentIndexChanged.connect(lambda _: self.reset())
        self.speed_box.currentIndexChanged.connect(
            lambda i: setattr(self, "speed", SPEEDS[i][1]))

        c1, self.p_night = chart(self, "The night so far · what the model believes "
                                       "(shade = probability), and the truth it never sees", 210)
        c2, self.p_fc = chart(self, f"Forecast · next {FORECAST_MIN} minutes", 170)
        c3, self.p_cost = chart(self, "Cost of waking · lower is better", 170)
        c3.lay.insertWidget(1, label(
            "Blue line: expected cost if the alarm were fixed at that time, given what the "
            "model knows now  ·  dashed: expected cost of sleeping on and re-checking every "
            "30 s  ·  dot: cost of waking now", 12, False, MUTED, rounded=False, wrap=True))
        self.night_key = label("", 12, False, MUTED, rounded=False)
        c1.lay.insertWidget(1, self.night_key)
        lab.addWidget(c1, 4)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(c2, 1)
        row.addWidget(c3, 1)
        lab.addLayout(row, 3)

        self.p_night.getAxis("left").setTicks([[(4 - i, s) for i, s in enumerate(ROWS)]])
        self.p_night.setYRange(-0.6, 4.6, padding=0)
        self.p_night.scene().sigMouseClicked.connect(self.on_night_click)
        self.night_img = pg.ImageItem()
        self.p_night.addItem(self.night_img)
        self.truth_past = self.p_night.plot(pen=pen(INK, 1.6, Qt.DashLine), connect="finite")
        self.truth_future = self.p_night.plot(pen=pen(MUTED, 1.6, Qt.DotLine), connect="finite")
        self.window_band = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(244, 197, 66, 45),
                                               pen=pen(STAR, 1.5))
        self.p_night.addItem(self.window_band)
        self.now_line = pg.InfiniteLine(angle=90, pen=pen(INK, 2.5))
        self.p_night.addItem(self.now_line)
        self.alarm_line = pg.InfiniteLine(angle=90, pen=pen(CORAL, 3))
        self.p_night.addItem(self.alarm_line)
        self.fc_box = pg.PlotDataItem(pen=pen(INK2, 1.2, Qt.DashLine))
        self.p_night.addItem(self.fc_box)

        self.p_fc.setYRange(-0.16, 1.03, padding=0)
        self.p_fc.getAxis("left").setTicks([[(-0.09, "truth"), (0, "0"), (0.5, "0.5"),
                                             (1, "1")]])
        self.p_cost.setLabel("left", "cost", color=MUTED)

        foot = Card(radius=16, border=LINE)
        self.outcome_lbl = label("", 14, False, INK, rounded=False, wrap=True)
        foot.lay.addWidget(self.outcome_lbl)
        lab.addWidget(foot)
        return lab

    # --- session control --------------------------------------------------------------
    def rec(self):
        return self.nights[self.night_box.currentIndex()]

    def load_night(self):
        self.session = None
        self.make_session(default_window=True)

    def make_session(self, default_window=False, keep_epoch=None):
        r = self.rec()
        mode = "stager" if self.input_box.currentIndex() == 0 else "labels"
        pol = ["bayes", "reactive", "fixed"][self.policy_box.currentIndex()]
        if default_window or self.session is None:
            latest = r.onset + int(cfg.DEFAULT_LATEST_AFTER_ONSET_H * 120)
            latest = min(latest, len(r.stages) - 1)
            earliest = latest - cfg.DEFAULT_WINDOW_MIN * 2
        else:
            earliest, latest = self.session.earliest, self.session.latest
        self.src = ReplaySource(r, mode, self.prior, seed=r.subject * 10 + r.night)
        self.session = Session(self.src, self.model, earliest, latest, pol, self.stage_cost,
                               self.sleep_value)
        if default_window:
            latest = self.default_latest(r)
            self.session.latest = latest
            self.session.earliest = latest - cfg.DEFAULT_WINDOW_MIN * 2
        self.set_edits()
        self.compare = self.compare_policies()
        self.running = False
        self.start_btn.setText("Start")
        self.scene.awake_target = 0.0
        self.alarm_rung = False
        if keep_epoch:
            self.session.seek(keep_epoch)
            self.after_fire_check()
        self.redraw(force=True)

    def default_latest(self, r):
        """Demo default: the latest 5-minute clock time, 4-7.5 h after onset, whose 30 min
        window this sleeper spent asleep throughout (so the demo never starts with them
        already awake). Picked from the true stages; the model never sees it."""
        w = cfg.DEFAULT_WINDOW_MIN * 2
        top = self.epoch_of(self.round_time(self.session.clock(
            min(r.onset + int(cfg.DEFAULT_LATEST_AFTER_ONSET_H * 120), len(r.stages) - 1))))
        for latest in range(top, r.onset + 4 * 120, -10):        # 10 epochs = 5 min
            if latest < len(r.stages) and not np.any(r.stages[latest - w:latest + 1] == 0):
                return latest
        return top

    @staticmethod
    def round_time(t):
        return t.replace(minute=t.minute - t.minute % 5, second=0)

    def epoch_of(self, clock_dt):
        return int((clock_dt - self.session.start_clock).total_seconds() // cfg.EPOCH_S)

    def set_edits(self):
        for e, ep in ((self.earliest_edit, self.session.earliest),
                      (self.latest_edit, self.session.latest)):
            e.blockSignals(True)
            e.setTime(QtCore.QTime.fromString(self.session.clock(ep).strftime("%H:%M"), "HH:mm"))
            e.blockSignals(False)

    def apply_window(self):
        def to_epoch(edit):
            t = edit.time().toPython()
            c = dt.datetime.combine(self.session.start_clock.date(), t)
            if c < self.session.start_clock:
                c += dt.timedelta(days=1)
            return self.epoch_of(c)
        e, l = to_epoch(self.earliest_edit), to_epoch(self.latest_edit)
        if l <= e:
            l = e + 2
        self.session.earliest, self.session.latest = e, l
        self.set_edits()
        self.compare = self.compare_policies()
        self.redraw(force=True)

    def compare_policies(self):
        """Headless replays of the same night and noise with each policy (for the footer)."""
        out = {}
        mode = "stager" if self.input_box.currentIndex() == 0 else "labels"
        r = self.rec()
        for pol in ("fixed", "reactive", "bayes"):
            s = Session(ReplaySource(r, mode, self.prior, seed=r.subject * 10 + r.night),
                        self.model, self.session.earliest, self.session.latest, pol,
                        self.stage_cost, self.sleep_value, log=False)
            s.advance(10 ** 6)
            out[pol] = (s.fired, s.outcome())
        return out

    def toggle(self):
        if self.session.fired:
            return
        self.running = not self.running
        self.start_btn.setText("Pause" if self.running else "Start")
        self.redraw(force=True)

    def reset(self):
        self.make_session()

    def jump_before_window(self):
        self.session.seek(max(self.session.earliest - 60, 0))
        self.after_fire_check()
        self.redraw(force=True)

    def on_night_click(self, ev):
        if self.p_night.sceneBoundingRect().contains(ev.scenePos()):
            x = self.p_night.vb.mapSceneToView(ev.scenePos()).x()
            self.session.seek(int(np.clip(x, 0, self.src.n - 1)))
            self.after_fire_check()
            self.redraw(force=True)

    def keyPressEvent(self, ev):
        k = ev.key()
        if k == Qt.Key_Space:
            self.toggle()
        elif k == Qt.Key_R:
            self.reset()
        elif k == Qt.Key_J:
            self.jump_before_window()
        else:
            super().keyPressEvent(ev)

    # --- loop -------------------------------------------------------------------------
    def tick(self):
        dt_s = self.elapsed.restart() / 1000.0
        self.scene.tick(dt_s)
        if self.sounds:
            self.sounds.tick(dt_s)
        if self.running and not self.session.fired:
            self.acc += self.speed * dt_s / cfg.EPOCH_S
            n = int(self.acc)
            if n:
                self.acc -= n
                self.session.advance(n)
                self.after_fire_check()
                if self.src.finished and not self.session.fired:
                    self.running = False
        self.redraw()

    def after_fire_check(self):
        if self.session.fired and not self.alarm_rung:
            self.alarm_rung = True
            self.running = False
            self.start_btn.setText("Start")
            self.scene.awake_target = 1.0
            self.scene.ring = 1.0
            if self.sounds:
                self.sounds.cue_done()
        elif not self.session.fired:
            self.alarm_rung = False
            self.scene.awake_target = 0.0

    # --- drawing ----------------------------------------------------------------------
    def redraw(self, force=False):
        t_ms = QtCore.QTime.currentTime().msecsSinceStartOfDay()
        if not force and t_ms - self.last_draw < 200:
            return
        self.last_draw = t_ms
        s = self.session
        inf = s.inf
        t = s.now
        has = inf.alpha is not None

        # phone
        if s.fired:
            t = s.fired["epoch"]
        self.clock_lbl.setText(s.clock(t).strftime("%H:%M:%S"))
        mins = (s.latest - t) / 2
        e_clock, l_clock = s.clock(s.earliest).strftime("%H:%M"), s.clock(s.latest).strftime("%H:%M")
        self.countdown.setText(
            f"Alarm rang · {mins:g} min before the latest time ({l_clock})" if s.fired else
            f"Window {e_clock}–{l_clock} · latest in {int(mins // 60)} h {int(mins % 60):02d} min")
        post = inf.posterior() if has else np.full(5, 0.2)
        self.bars.set(post)
        top = int(np.argmax(post))
        names = {"W": "Awake", "N1": "Light sleep (N1)", "N2": "Light sleep (N2)",
                 "N3": "Deep sleep (N3)", "REM": "REM sleep"}
        self.stage_lbl.setText(f"Current stage: {names[cfg.STAGES[top]]}" if has
                               else "Current stage: –")
        missing = has and inf.marginals and s.source.obs[min(t, len(s.source.obs) - 1)] is None
        self.live_dot.setStyleSheet(f"color:{ORANGE if missing or s.error else GREEN};"
                                    "background:transparent;border:none;")
        self.live_lbl.setText("No signal" if missing else
                              ("Replay " + SPEEDS[self.speed_box.currentIndex()][0]
                               if self.running else "Paused"))

        if s.fired:
            state, fill = "ALARM", "#FBD9CF"
            self.rec_title.setText(f"Wake up! {s.fired['clock'][:5]}")
            self.rec_text.setText(
                "The latest wake-up time arrived, so the alarm fired regardless of stage."
                if s.fired["reason"] == "deadline" else
                f"Waking here was expected to cost {s.decision.get('stop', 0):.1f}, against "
                f"{s.decision.get('cont', 0):.1f} for sleeping on, so this was the best moment.")
        elif t >= s.earliest:
            state, fill = "WAKE WINDOW", "#FFF1C9"
            self.rec_title.setText("Keep sleeping" if not s.decision.get("wake") else "Wake now")
            self.rec_text.setText(s.text)
        else:
            state = "SLEEP" if self.running else ("PAUSED" if has else "READY")
            fill = BLUE_TINT
            self.rec_title.setText("Sleeping" if has else "Ready")
            self.rec_text.setText(s.text if has else
                                  "Press Start to replay the night. The model only sees each "
                                  "30 s epoch as it arrives.")
        self.pill.setText(state)
        self.pill.setStyleSheet(f"color:{INK}; background:{fill}; border:2px solid {INK};"
                                " border-radius:15px; padding:0 14px;")

        self.draw_night(t, has)
        self.draw_forecast(t, has)
        self.draw_cost(t, has)
        self.draw_outcome()

    def draw_night(self, t, has):
        s, inf = self.session, self.session.inf
        n = self.src.n
        x_end = max(n, s.latest + 20)
        img = np.zeros((x_end, 5, 4), np.uint8)
        img[..., :3] = mix(CARD, CARD, 0)
        img[..., 3] = 255
        rows_lvl = {s_: 4 - i for i, s_ in enumerate(ROWS)}
        if has:
            M = np.array(inf.marginals)                       # (t+1, K)
            fc = inf.forecast(FORECAST_MIN * 2)[1:]
            for s_ in cfg.STAGES:
                lv = rows_lvl[s_]
                col = np.array(mix(CARD, STAGE_COLOR[s_], 1.0))
                base = np.array(mix(CARD, CARD, 0))
                past = M[:, SI[s_]][:, None]
                img[:len(M), lv, :3] = (base + (col - base) * past).astype(np.uint8)
                k = min(len(fc), x_end - len(M))
                fut = 0.55 * fc[:k, SI[s_]][:, None]
                img[len(M):len(M) + k, lv, :3] = (base + (col - base) * fut).astype(np.uint8)
            k = min(FORECAST_MIN * 2, x_end - t - 1)
            self.fc_box.setData([t + 1, t + 1 + k, t + 1 + k, t + 1, t + 1],
                                [-0.5, -0.5, 4.5, 4.5, -0.5])
        else:
            self.fc_box.setData([], [])
        self.night_img.setImage(img, autoLevels=False, levels=(0, 255))
        self.night_img.setRect(QRectF(0, -0.5, x_end, 5))
        truth = self.src.truth
        lv = np.array([rows_lvl[cfg.STAGES[v]] if v >= 0 else np.nan for v in truth], float)
        xs = np.arange(len(lv) + 1)
        ys = np.append(lv, lv[-1])
        show = self.truth_chk.isChecked()
        cut = min(t + 1, len(lv))
        self.truth_past.setData(xs[:cut + 1], ys[:cut + 1], stepMode="left") if show and has \
            else self.truth_past.setData([], [])
        self.truth_future.setData(xs[cut:], ys[cut:], stepMode="left") if show \
            else self.truth_future.setData([], [])
        self.window_band.setRegion((s.earliest, s.latest))
        self.now_line.setValue(t if has else 0)
        self.alarm_line.setVisible(bool(s.fired))
        if s.fired:
            self.alarm_line.setValue(s.fired["epoch"])
        self.p_night.setXRange(0, x_end, padding=0.005)
        self.night_key.setText(
            "Solid shading: the model's belief so far  ·  pale shading in the dashed box: "
            "forecast  ·  yellow: wake-up window" + ("  ·  dashed line: true stages so far  ·  "
            "dotted line: true future, hidden from the model" if show else ""))

    def draw_forecast(self, t, has):
        p = self.p_fc
        p.clear()
        if not has:
            return
        H = FORECAST_MIN * 2
        P = self.session.inf.forecast(H)
        x = t + np.arange(H + 1)
        cum = np.zeros(H + 1)
        prev = p.plot(x, cum, pen=pen(CARD, 1))
        for s_ in STACK:
            cum = cum + P[:, SI[s_]]
            cur = p.plot(x, cum, pen=pen(CARD, 1.5))
            p.addItem(pg.FillBetweenItem(prev, cur, brush=STAGE_COLOR[s_]))
            prev = cur
        if self.truth_chk.isChecked():
            fut = self.src.truth[t:t + H + 1]
            for s_ in STACK:
                idx = np.flatnonzero(fut == SI[s_])
                if len(idx):
                    p.addItem(pg.BarGraphItem(x0=x[idx], width=1, y=-0.09, height=0.08,
                                              brush=STAGE_COLOR[s_], pen=None))
        s = self.session
        for ep in (s.earliest, s.latest):
            if t <= ep <= t + H:
                p.addItem(pg.InfiniteLine(ep, angle=90, pen=pen(STAR, 2, Qt.DashLine)))
        p.setXRange(t, t + H, padding=0.01)

    def draw_cost(self, t, has):
        p = self.p_cost
        p.clear()
        s = self.session
        if not has or t >= s.latest:
            return
        n = s.latest - t
        path = s.inf.forecast(n)
        us, exp = open_loop_costs(path, t, s.earliest, s.latest, self.costs, s.sleep_value)
        p.plot(us, exp, pen=pen(BLUE, 3))

        d = s.decision
        if "cont" in d and np.isfinite(d.get("cont", np.inf)):
            p.addItem(pg.InfiniteLine(d["cont"], angle=0, pen=pen(NAVY, 2, Qt.DashLine)))
            tx = pg.TextItem(f"keep sleeping and re-check: {d['cont']:.2f}", color=NAVY,
                             anchor=(1, 1))
            tx.setPos(s.latest, d["cont"])
            p.addItem(tx)
        if t >= s.earliest and "stop" in d:
            p.plot([t], [d["stop"]], pen=None, symbol="o", symbolSize=12,
                   symbolBrush=CORAL, symbolPen=pg.mkPen(INK, width=2))
            right = t > (s.earliest + s.latest) / 2
            tx = pg.TextItem(f"wake now: {d['stop']:.2f}", color=INK, anchor=(1 if right else 0, 0))
            tx.setPos(t, d["stop"])
            p.addItem(tx)
        if len(exp):
            hi = max(exp.max(), d.get("stop", 0), d.get("cont", 0) if np.isfinite(
                d.get("cont", np.inf)) else 0)
            p.setYRange(0, hi * 1.15 + 0.1, padding=0)
        p.setXRange(s.earliest - 2, s.latest + 2, padding=0)

    def draw_outcome(self):
        s = self.session
        parts = []
        names = {"fixed": "Fixed alarm", "reactive": "Reactive alarm", "bayes": "Bayesian alarm"}
        for pol, (fired, out) in self.compare.items():
            if fired and out:
                parts.append(f"{names[pol]}: {fired['clock'][:5]}, woke from "
                             f"{out['true_stage']}, cost {out['realised_cost']:.2f}")
        cmp_txt = " · ".join(parts)
        if s.fired:
            o = s.outcome()
            self.outcome_lbl.setText(
                f"<b>Alarm at {s.fired['clock']}</b> ({s.fired['minutes_before_latest']:g} min "
                f"before the latest time). True stage at that moment: <b>{o['true_stage']}</b>, "
                f"cost {o['realised_cost']:.2f}.<br>This night, same input: {cmp_txt}.")
        else:
            self.outcome_lbl.setText(
                "Costs: " + " · ".join(f"{st} {self.stage_cost[st]:g}" for st in
                                       ["W", "N1", "N2", "REM", "N3"]) +
                f", plus {self.session.sleep_value:g} per minute of sleep given up. "
                f"Heuristic proxies, not clinical measures.<br>Log: {cfg.SESSIONS_DIR}/")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--night", default=None, help="recording ID, e.g. SC4XXX (held-out only)")
    ap.add_argument("--speed", type=float, default=300, help="replay speed (x real time)")
    ap.add_argument("--input", choices=["stager", "labels"], default="stager")
    ap.add_argument("--mute", action="store_true")
    ap.add_argument("--screenshot", default=None, help="render to PNG and exit")
    ap.add_argument("--at", default=None, help="with --screenshot: replay up to HH:MM first")
    args = ap.parse_args(argv)

    blob = joblib.load(cfg.MODEL_FILE)
    model = from_dict(blob["models"][cfg.MODEL_NAME])
    recs = data.load()
    test_s = set(blob["test_subjects"])
    nights = [r for r in recs if r.subject in test_s]
    prior = stager_prior([r for r in recs if r.subject not in test_s])

    pg.setConfigOptions(antialias=True)
    app = pg.mkQApp("Smart Alarm")
    win = AlarmApp(model, nights, prior, args)
    if args.input == "labels":
        win.input_box.setCurrentIndex(1)
    if args.screenshot:
        win.sounds = None
        if args.at:
            t = dt.datetime.combine(win.session.start_clock.date(),
                                    dt.time.fromisoformat(args.at))
            if t < win.session.start_clock:
                t += dt.timedelta(days=1)
            win.session.seek(win.epoch_of(t))
            win.after_fire_check()
            win.scene.awake = win.scene.awake_target
        win.show()
        win.redraw(force=True)
        app.processEvents()
        win.redraw(force=True)
        app.processEvents()
        os.makedirs(os.path.dirname(args.screenshot) or ".", exist_ok=True)
        win.grab().save(args.screenshot)
        print(f"wrote {args.screenshot}")
        return
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
