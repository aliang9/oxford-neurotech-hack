"""The participant-facing dashboard: calm, minimal, and mostly heard rather than seen.

Look: the Tandem "MIND" style: warm off-white paper, ink line art, serif headlines, and
soft blue (#83A9DB) as the accent; orange (#E88A49) only flags a poor signal.

    a meditator           line-art figure that breathes slowly; eyes open for the eyes-open
                          calibration, closed during the session, smiling more as alpha
                          rises; a colour blob behind it shows the state (warm grey resting,
                          blue as alpha nears the threshold, pale blue calibrating, orange
                          poor signal);
                          an ink ripple goes out with every chime; a ring shows calibration
    points                the cumulative score, big, with a serif headline and a hint
    one curve             cumulative reward over the session, with the previous personal
                          best (for this session mode) as a dashed target line
    session setup         timed (minutes, with a countdown) or open-ended, chosen before starting
    one button            Begin calibration -> (auto) session -> Pause / Resume; End session
    sounds (sounds.py)    a soft chime per reward point, a deeper bell every 10 and on passing
                          the personal best, an ambient pad that swells as alpha nears the
                          threshold, cues around calibration and at the end of a session

Keys: Space start/pause, M mute, F full screen.
"""

import math
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen

from . import logo
from .engine import (CALIBRATING, FINISHED, INDEFINITE, NEEDS_BASELINE, PAUSED, READY, RECORDING,
                     TIMED, WAITING)

CHIME_EVERY = 1.0        # reward points per chime
MILESTONE_EVERY = 10.0   # reward points per milestone bell
MIN_CHIME_GAP_S = 0.7    # never chime faster than this
BREATH_S = 10.0          # breathing period: 6 breaths a minute

# Palette (Tandem design system, MIND): paper, ink, soft blue; orange only for warnings
PAPER, CARD, INK = "#FAF9F6", "#FFFFFF", "#1F1F1F"
MUTED, LINE = "#8A8781", "#E6E3DD"
BLUE, BLUE_DARK, BLUE_TINT, BLUE_PALE = "#83A9DB", "#5B84BF", "#DCE7F6", "#EEF3FA"
ORANGE, ORANGE_TINT = "#E88A49", "#FBE5D4"
SPLASH = {                # blob behind the figure
    "rest": "#ECE9E3",    # warm grey: resting / waiting / paused
    "earn": BLUE,         # blue: at / past the threshold
    "calib": BLUE_TINT,   # pale blue: calibrating
    "poor": "#F3B98F",    # soft orange: poor signal
    "done": "#A9C4E8",    # mid blue: session complete
}
TAG = {"rest": "#EFEDE8", "earn": BLUE_TINT, "calib": BLUE_PALE, "poor": ORANGE_TINT, "done": BLUE_TINT}
BLUSH = QColor("#F2B48A")


def font(px, weight=QFont.Normal, spacing=100, serif=False):
    f = QFont()
    f.setFamilies(["Charter", "Georgia", "Times New Roman"] if serif else
                  ["Inter", "Helvetica Neue", "Segoe UI", "Arial"])
    f.setPixelSize(int(px))
    f.setWeight(weight)
    f.setLetterSpacing(QFont.PercentageSpacing, spacing)
    return f


def smoothstep(e0, e1, x):
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def rgb(hex_color):
    c = QColor(hex_color)
    return np.array([c.red(), c.green(), c.blue()], float)


def qcolor(v, alpha=255):
    r, g, b = np.clip(v, 0, 255).astype(int)
    return QColor(r, g, b, int(alpha))


class Anim:
    """Animated values, eased toward their targets every frame."""

    def __init__(self):
        self.t = 0.0
        self.splash = rgb(SPLASH["rest"])
        self.splash_target = self.splash.copy()
        self.joy = 0.0             # 0..1: how close alpha is to / past the threshold (smile, blush)
        self.joy_target = 0.0
        self.depth = 0.0           # 0..1: how locked in the character looks (pose, eyes)
        self.depth_target = 0.0
        self.worried = False       # poor signal: eyes open, a sweat drop
        self.shown_total = 0.0
        self.total = 0.0
        self.progress = None       # calibration 0..1, or None
        self.ripples = []          # (start time, strength)
        self.points = np.empty((0, 2))
        self.best = None           # previous personal best: the target line
        self.planned_s = None      # timed session length: the curve's x range

    def step(self, dt):
        self.t += dt
        k = min(1.0, dt / 0.9)
        self.splash += (self.splash_target - self.splash) * k
        self.joy += (self.joy_target - self.joy) * k
        self.depth += (self.depth_target - self.depth) * min(1.0, dt / 1.6)  # slow, calm changes
        self.shown_total += (self.total - self.shown_total) * min(1.0, dt / 0.35)
        self.ripples = [(s, w) for s, w in self.ripples if self.t - s < 2.4]


class Figure(QtWidgets.QWidget):
    """The blob guy on a flat colour blob; his pose follows alpha (see paint_character)."""

    def __init__(self, anim):
        super().__init__()
        self.a = anim
        self.setMinimumSize(260, 260)

    def paintEvent(self, _):
        a = self.a
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        half = min(self.width(), self.height()) / 2
        R = half * 0.78
        c = QPointF(self.width() / 2, self.height() / 2)
        breath = 0.5 + 0.5 * math.sin(2 * math.pi * a.t / BREATH_S)
        ink = QColor(INK)
        lw = max(2.2, R * 0.022)

        # flat colour blob, slowly morphing, centred behind the character
        blob = QPainterPath()
        bc = c
        for i in range(121):
            th = 2 * math.pi * i / 120
            rr = R * 0.86 * (1 + 0.035 * math.sin(3 * th + 0.35 * a.t)
                             + 0.025 * math.sin(5 * th - 0.25 * a.t)) * (0.97 + 0.03 * breath)
            pt = bc + QPointF(rr * math.cos(th), rr * math.sin(th))
            blob.moveTo(pt) if i == 0 else blob.lineTo(pt)
        p.setPen(Qt.NoPen)
        halo = qcolor(a.splash, 70)  # a paler ring of the same colour, for a soft edge
        p.setBrush(halo)
        p.drawEllipse(bc, R * 0.98 * (0.97 + 0.03 * breath), R * 0.98 * (0.97 + 0.03 * breath))
        p.setBrush(qcolor(a.splash))
        p.drawPath(blob)

        # ink ripples, one per chime
        for start, strength in a.ripples:
            age = (a.t - start) / 2.4
            ripple = QColor(BLUE_DARK)
            ripple.setAlpha(int(100 * strength * (1 - age) ** 2))
            p.setPen(QPen(ripple, lw * 0.6))
            p.setBrush(Qt.NoBrush)
            rr = R * (0.9 + 0.22 * age)
            p.drawEllipse(bc, rr, rr)

        # calibration progress ring
        if a.progress is not None:
            ring = QRectF(c.x() - R * 1.0, c.y() - R * 1.0, R * 2.0, R * 2.0)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(LINE), lw))
            p.drawEllipse(ring)
            pen = QPen(QColor(BLUE), lw * 1.5)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.drawArc(ring, 90 * 16, -int(360 * 16 * a.progress))

        # ground placed so the standing figure (head to shadow) is centred vertically in the blob
        self.paint_character(p, c, R * 1.2, breath, ground=c.y() + R * 0.55)

    def paint_character(self, p, c, R, breath, ground):
        """The tandem blob. a.depth (0..1, eased) sets the pose:
        0 standing, eyes open -> ~0.4 sleepy eyes, sinking -> ~0.7 seated cross-legged, hands
        resting on his knees, eyes closed, smiling -> 1 fully locked in: floating a little, the
        shadow shrinking, blue ticks radiating around his head."""
        a = self.a
        m = a.depth
        sit = smoothstep(0.35, 0.75, m)
        lev = smoothstep(0.85, 1.0, m)
        shut = smoothstep(0.2, 0.6, m)
        smile = 0.35 + 0.65 * max(smoothstep(0.4, 0.9, m), a.joy)  # always a small smile, wider as he settles
        ink, white = QColor(INK), QColor(CARD)

        # body proportions measured from the reference drawing: an elongated egg ~1.5x as tall as
        # it is wide, a narrow dome on top, widest two-thirds of the way down
        cx = c.x()
        W = R * 0.50
        H = W * 1.50 * (1 + 0.012 * breath)
        lw = max(2.6, W * 0.05)
        pen = QPen(ink, lw)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        leg = W * 0.31
        lap_h = W * 0.32
        float_up = lev * R * (0.07 + 0.015 * math.sin(a.t * 0.8))
        bottom = ground - (leg * (1 - sit) + lap_h * 0.55 * sit) - float_up
        top = bottom - H
        lap_top = ground - lap_h - float_up

        def at(fx, fy):  # fx in body widths from the centre, fy in body heights from the top
            return QPointF(cx + fx * W, top + fy * H)

        def mix(a0, a1):
            return a0 + (a1 - a0) * sit

        # a solid dark shadow under him, shrinking as he floats
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(28, 28, 28))
        p.drawEllipse(QPointF(cx, ground), W * mix(0.50, 0.66) * (1 - 0.35 * lev), W * 0.045 * (1 - 0.3 * lev))

        # legs: two short stumps, folding away as he sits
        if sit < 0.98:
            p.setPen(pen)
            p.setBrush(white)
            lh = leg * (1 - sit) + lw * 2.5
            for sx in (-1, 1):
                p.drawRoundedRect(QRectF(cx + sx * W * 0.165 - W * 0.16, bottom - lw * 2.5, W * 0.32, lh),
                                  W * 0.09, W * 0.09)

        # arms: from hanging at his sides to bending down so the hands rest on his knees
        arm_w = W * 0.21
        for sx in (-1, 1):
            sh = at(sx * 0.36, mix(0.40, 0.42))
            hd = QPointF(cx + sx * W * mix(0.60, 0.56), mix(top + H * 0.74, lap_top + lap_h * 0.2))
            ctrl = QPointF((sh.x() + hd.x()) / 2 + sx * W * 0.14 * sit, (sh.y() + hd.y()) / 2)
            arm = QPainterPath(sh)
            arm.quadTo(ctrl, hd)
            for color, w in ((ink, arm_w), (white, arm_w - 2 * lw)):
                pn = QPen(color, w)
                pn.setCapStyle(Qt.RoundCap)
                p.setPen(pn)
                p.setBrush(Qt.NoBrush)
                p.drawPath(arm)

        # body
        right = [(0.20, 0.0), (0.34, 0.07), (0.36, 0.20), (0.40, 0.36), (0.50, 0.48), (0.50, 0.66),
                 (0.50, 0.92), (0.30, 1.0), (0.0, 1.0)]
        left = [(-x, y) for x, y in reversed(right[:-1])] + [(0.0, 0.0)]
        body = QPainterPath(at(0, 0))
        for pts in (right, left):
            for i in range(0, 9, 3):
                body.cubicTo(at(*pts[i]), at(*pts[i + 1]), at(*pts[i + 2]))
        body.closeSubpath()
        p.setPen(pen)
        p.setBrush(white)
        p.drawPath(body)

        # crossed legs in front when seated, with the hands resting on the knees
        if sit > 0.02:
            lap_w = W * (0.95 + 0.42 * sit)
            lap = QRectF(cx - lap_w / 2, lap_top + lap_h * (1 - sit), lap_w, lap_h * sit)
            p.setPen(pen)
            p.setBrush(white)
            p.drawRoundedRect(lap, lap.height() / 2, lap.height() / 2)
            if sit > 0.5:  # where one shin crosses over the other
                p.drawLine(QPointF(cx - lap_w * 0.05, lap.top() + lap.height() * 0.22),
                           QPointF(cx + lap_w * 0.04, lap.bottom() - lap.height() * 0.22))
                for sx in (-1, 1):
                    p.drawEllipse(QPointF(cx + sx * W * 0.56, lap_top + lap_h * 0.2), arm_w * 0.45, arm_w * 0.45)

        # face: two dot eyes -> sleepy -> closed arcs; a small smile, wider as he settles
        ey = top + H * 0.21
        er = W * 0.034
        face = QPen(ink, lw * 0.8)
        face.setCapStyle(Qt.RoundCap)
        for sx in (-1, 1):
            ex = cx + sx * W * 0.11
            if a.worried or shut < 0.5:
                ry = er if a.worried else max(lw * 0.35, er * (1 - 1.7 * shut))
                p.setPen(Qt.NoPen)
                p.setBrush(ink)
                p.drawEllipse(QPointF(ex, ey), er, ry)
            else:
                d = er * (0.5 + 0.9 * shut)
                lid = QPainterPath(QPointF(ex - er * 1.6, ey - d * 0.2))
                lid.quadTo(QPointF(ex, ey + d), QPointF(ex + er * 1.6, ey - d * 0.2))
                p.setPen(face)
                p.setBrush(Qt.NoBrush)
                p.drawPath(lid)
        my = ey + H * 0.055
        p.setPen(face)
        p.setBrush(Qt.NoBrush)
        if a.worried:  # a small flat mouth
            p.drawLine(QPointF(cx - W * 0.045, my + W * 0.02), QPointF(cx + W * 0.045, my + W * 0.02))
        else:
            mw = W * (0.05 + 0.03 * smile)
            mouth = QPainterPath(QPointF(cx - mw, my))
            mouth.quadTo(QPointF(cx, my + W * (0.03 + 0.04 * smile)), QPointF(cx + mw, my))
            p.drawPath(mouth)
        if a.worried:  # a little sweat drop
            drop = QPainterPath(at(0.46, 0.02))
            drop.quadTo(at(0.56, 0.13), at(0.46, 0.15))
            drop.quadTo(at(0.36, 0.13), at(0.46, 0.02))
            p.setPen(QPen(ink, lw * 0.6))
            p.setBrush(QColor(BLUE))
            p.drawPath(drop)

        # fully locked in: little blue ticks radiating around his head
        if lev > 0.02:
            tick = QColor(BLUE_DARK)
            tick.setAlpha(int(255 * lev))
            tp = QPen(tick, lw * 0.9)
            tp.setCapStyle(Qt.RoundCap)
            p.setPen(tp)
            hc = at(0, 0.22)
            for ang in (-160, -130, -90, -50, -20):
                th = math.radians(ang)
                r0, r1 = W * 0.62, W * (0.62 + 0.15 * lev)
                p.drawLine(QPointF(hc.x() + r0 * math.cos(th), hc.y() + r0 * math.sin(th)),
                           QPointF(hc.x() + r1 * math.cos(th), hc.y() + r1 * math.sin(th)))


class Logo(QtWidgets.QWidget):
    """The Tandem logo (logo.py): the two-circle mark and the handwritten wordmark."""

    def __init__(self, size=30):
        super().__init__()
        self.size_px = size
        fm = QtGui.QFontMetricsF(logo.word_font(size))
        self.mark_w = size * 0.34 * 3.15
        width = self.mark_w + size * 0.22 + fm.horizontalAdvance("tandem")
        self.top = size * 0.30                       # room for the motion ticks
        self.setFixedSize(int(width + 6), int(self.top + fm.ascent() + fm.descent() + 2))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        baseline = self.top + QtGui.QFontMetricsF(logo.word_font(self.size_px)).ascent()
        logo.paint_logo(p, 2, baseline, self.size_px)


class Curve(QtWidgets.QWidget):
    """Cumulative reward over the session, on a white card: blue line over a pale blue fill."""

    def __init__(self, anim):
        super().__init__()
        self.a = anim
        self.setFixedHeight(130)

    def paintEvent(self, _):
        pts = self.a.points
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        card = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        p.setPen(QPen(QColor(LINE), 1))
        p.setBrush(QColor(CARD))
        p.drawRoundedRect(card, 12, 12)
        p.setFont(font(12, QFont.DemiBold))
        p.setPen(QColor(INK))
        p.drawText(QRectF(16, 10, 200, 18), Qt.AlignLeft | Qt.AlignVCenter, "Points")
        p.setFont(font(12))
        p.setPen(QColor(MUTED))
        p.drawText(QRectF(62, 10, 200, 18), Qt.AlignLeft | Qt.AlignVCenter, "(this session)")
        m = 16
        top = 34
        W, H = self.width() - 2 * m, self.height() - top - m
        p.setPen(QPen(QColor(LINE), 1))
        p.drawLine(QPointF(m, top + H), QPointF(m + W, top + H))
        best = self.a.best
        ymax = max(pts[-1, 1] * 1.15 if len(pts) else 0, (best or 0) * 1.25, 5.0)
        if best is not None:  # the target: previous personal best
            yb = top + H * (1 - best / ymax)
            pen = QPen(QColor(MUTED), 1.2, Qt.DashLine)
            pen.setDashPattern([3, 4])
            p.setPen(pen)
            p.drawLine(QPointF(m, yb), QPointF(m + W, yb))
            p.setFont(font(11, QFont.DemiBold))
            p.setPen(QColor(MUTED))
            p.drawText(QRectF(m, yb - 17, W, 15), Qt.AlignLeft | Qt.AlignVCenter,
                       f"Personal best · {int(best)}")
        if len(pts) < 2:
            p.setFont(font(13))
            p.setPen(QColor(MUTED))
            p.drawText(QRectF(m, top, W, H), Qt.AlignCenter, "Your points over the session will draw here.")
            return
        keep = np.r_[True, np.any(np.diff(pts, axis=0) != 0, axis=1)]  # drop repeated points
        t, y = pts[keep, 0], pts[keep, 1]
        tmax = self.a.planned_s or max(t[-1], 60.0)
        xs = m + W * t / tmax
        ys = top + H * (1 - y / ymax)
        line = QPainterPath(QPointF(xs[0], ys[0]))
        for x_, y_ in zip(xs[1:], ys[1:]):
            line.lineTo(x_, y_)
        fill = QPainterPath(line)
        fill.lineTo(xs[-1], top + H)
        fill.lineTo(xs[0], top + H)
        fill.closeSubpath()
        fill.setFillRule(Qt.WindingFill)
        p.fillPath(fill, QColor(BLUE_PALE))
        pen = QPen(QColor(BLUE_DARK), 2.2)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)  # stroke only (the card's white brush would fill the curve's chord)
        p.drawPath(line)
        p.setBrush(QColor(CARD))
        p.setPen(QPen(QColor(BLUE_DARK), 2))
        p.drawEllipse(QPointF(xs[-1], ys[-1]), 4.5, 4.5)


PRIMARY_CSS = f"""
QPushButton {{ color: #FFFFFF; background: #2B2B2B; border: none; border-radius: 23px; padding: 0 30px; }}
QPushButton:hover {{ background: #444444; }}
QPushButton:disabled {{ color: {MUTED}; background: {LINE}; }}
"""
GHOST_CSS = f"""
QPushButton {{ color: {INK}; background: {CARD}; border: 1px solid {LINE}; border-radius: 20px;
  padding: 0 18px; }}
QPushButton:hover {{ background: {BLUE_PALE}; border-color: {BLUE_TINT}; }}
QPushButton:disabled {{ color: #C4BEB9; }}
"""
SEGMENT_CSS = f"""
QPushButton {{ color: {MUTED}; background: #F1EFEA; border: 1px solid {LINE}; border-radius: 17px;
  padding: 0 18px; }}
QPushButton:hover {{ color: {INK}; }}
QPushButton:checked {{ color: {INK}; background: {CARD}; border: 1.5px solid {BLUE}; }}
"""
SPIN_CSS = f"""
QSpinBox {{ color: {INK}; background: {CARD}; border: 1px solid {LINE}; border-radius: 17px;
  padding: 0 10px; }}
QSpinBox::up-button, QSpinBox::down-button {{ width: 0; border: none; }}
QSpinBox:disabled {{ color: #C4BEB9; background: #F1EFEA; }}
"""
PROGRESS_CSS = f"""
QProgressBar {{ background: #F1EFEA; border: 1px solid {LINE}; border-radius: 5px; }}
QProgressBar::chunk {{ background: {BLUE}; border-radius: 4px; }}
"""


def tag_css(kind):
    return (f"color: {INK}; background: {TAG[kind]}; border: none; "
            "border-radius: 12px; padding: 4px 12px;")


class CalmWindow(QtWidgets.QWidget):
    def __init__(self, engine, sounds, auto_start=True):
        super().__init__()
        logo.load_fonts()
        self.engine, self.sounds, self.auto_start = engine, sounds, auto_start
        self.a = Anim()
        self.prev_state = None
        self.chimed = 0
        self.milestones = 0
        self.last_chime = -1e9
        self.above_best = False
        self.snap = None
        self.shown_int = None

        self.setWindowTitle("Tandem")
        self.resize(880, 860)
        self.setMinimumSize(560, 700)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(36, 24, 36, 20)
        root.setSpacing(0)

        top = QtWidgets.QHBoxLayout()
        brand = Logo(40)
        self.clock = QtWidgets.QLabel("")
        self.clock.setFont(font(16, QFont.DemiBold))
        self.clock.setStyleSheet(f"color: {INK}")
        self.sound_btn = QtWidgets.QPushButton()
        self.sound_btn.setFixedHeight(32)
        self.sound_btn.setFont(font(13, QFont.Medium))
        self.sound_btn.setStyleSheet(GHOST_CSS.replace("20px", "16px"))  # radius <= height/2
        self.sound_btn.setCursor(Qt.PointingHandCursor)
        self.sound_btn.clicked.connect(self.toggle_mute)
        top.addWidget(brand)
        top.addStretch()
        top.addWidget(self.clock)
        top.addSpacing(14)
        top.addWidget(self.sound_btn)
        root.addLayout(top)

        self.figure = Figure(self.a)
        root.addWidget(self.figure, 5)

        self.headline = QtWidgets.QLabel("")
        self.headline.setAlignment(Qt.AlignCenter)
        self.headline.setFont(font(30, QFont.Bold, serif=True))
        self.headline.setStyleSheet(f"color: {INK}")
        root.addWidget(self.headline)
        root.addSpacing(6)

        self.points = QtWidgets.QLabel("0")
        self.points.setAlignment(Qt.AlignCenter)
        self.points.setTextFormat(Qt.RichText)
        self.points.setFont(font(40, QFont.DemiBold))
        self.points.setStyleSheet(f"color: {INK}")
        root.addWidget(self.points)
        root.addSpacing(8)

        tag_row = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel("")
        self.status.setFont(font(13, QFont.DemiBold))
        tag_row.addStretch()
        tag_row.addWidget(self.status)
        tag_row.addStretch()
        root.addLayout(tag_row)
        root.addSpacing(8)
        self.hint = QtWidgets.QLabel("")
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setWordWrap(True)
        self.hint.setFont(font(15))
        self.hint.setStyleSheet(f"color: {MUTED}")
        root.addWidget(self.hint)
        root.addSpacing(18)

        self.curve = Curve(self.a)
        root.addWidget(self.curve)
        root.addSpacing(10)
        self.progress = QtWidgets.QProgressBar(textVisible=False, maximum=1000)
        self.progress.setFixedHeight(10)
        self.progress.setStyleSheet(PROGRESS_CSS)
        root.addWidget(self.progress)
        root.addSpacing(16)

        # session setup: timed (minutes) or open-ended; shown only between sessions
        self.setup = QtWidgets.QWidget()
        srow = QtWidgets.QHBoxLayout(self.setup)
        srow.setContentsMargins(0, 0, 0, 0)
        srow.setSpacing(8)
        self.timed_btn = QtWidgets.QPushButton("Timed")
        self.open_btn = QtWidgets.QPushButton("Open-ended")
        group = QtWidgets.QButtonGroup(self)
        for b in (self.timed_btn, self.open_btn):
            b.setCheckable(True)
            b.setStyleSheet(SEGMENT_CSS)
            b.setFixedHeight(34)
            b.setFont(font(13, QFont.Medium))
            b.setCursor(Qt.PointingHandCursor)
            group.addButton(b)
        self.minutes = QtWidgets.QSpinBox(minimum=1, maximum=180, value=int(engine.minutes),
                                          suffix=" min")
        self.minutes.setStyleSheet(SPIN_CSS)
        self.minutes.setFixedSize(88, 34)
        self.minutes.setAlignment(Qt.AlignCenter)
        self.minutes.setFont(font(13, QFont.Medium))
        (self.timed_btn if engine.mode == TIMED else self.open_btn).setChecked(True)
        self.timed_btn.toggled.connect(self.on_mode)
        self.minutes.valueChanged.connect(self.on_mode)
        srow.addStretch()
        srow.addWidget(self.timed_btn)
        srow.addWidget(self.minutes)
        srow.addSpacing(6)
        srow.addWidget(self.open_btn)
        srow.addStretch()
        root.addWidget(self.setup)
        root.addSpacing(12)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(12)
        self.recal_btn = QtWidgets.QPushButton("Recalibrate")
        self.primary = QtWidgets.QPushButton("")
        self.end_btn = QtWidgets.QPushButton("End session")
        for b, css, h in ((self.recal_btn, GHOST_CSS, 40), (self.primary, PRIMARY_CSS, 46),
                          (self.end_btn, GHOST_CSS, 40)):
            b.setStyleSheet(css)
            b.setFixedHeight(h)
            b.setCursor(Qt.PointingHandCursor)
            b.setFont(font(15 if b is self.primary else 13, QFont.DemiBold if b is self.primary else QFont.Medium))
        self.primary.setMinimumWidth(210)
        self.primary.clicked.connect(self.on_primary)
        self.recal_btn.clicked.connect(self.start_calibration)
        self.end_btn.clicked.connect(self.engine.stop_session)
        row.addStretch()
        row.addWidget(self.recal_btn)
        row.addWidget(self.primary)
        row.addWidget(self.end_btn)
        row.addStretch()
        root.addLayout(row)
        root.addSpacing(14)

        foot = QtWidgets.QLabel("Feedback on posterior alpha activity relative to your eyes-open "
                                "resting baseline. Not a measure of meditation depth.")
        foot.setAlignment(Qt.AlignCenter)
        foot.setWordWrap(True)
        foot.setFont(font(11))
        foot.setStyleSheet("color: #B2ACA6")
        root.addWidget(foot)

        for key, fn in ((Qt.Key_Space, self.on_primary), (Qt.Key_M, self.toggle_mute),
                        (Qt.Key_F, self.toggle_fullscreen)):
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=fn)

        self.update_sound_btn()
        self.poll()
        self._last_frame = time.monotonic()
        self.frame_timer = QtCore.QTimer(self, timeout=self.frame)
        self.frame_timer.start(33)
        self.poll_timer = QtCore.QTimer(self, timeout=self.poll)
        self.poll_timer.start(250)

    # ---- actions ------------------------------------------------------------
    def on_primary(self):
        s = self.engine.snapshot()["state"]
        if s == NEEDS_BASELINE:
            self.start_calibration()
        elif s in (READY, FINISHED, RECORDING, PAUSED):
            self.engine.toggle_feedback()
            if s in (READY, FINISHED, PAUSED):
                self.sounds.cue_start()
        self.poll()

    def on_mode(self, *_):
        self.engine.set_session(TIMED if self.timed_btn.isChecked() else INDEFINITE,
                                self.minutes.value())

    def start_calibration(self):
        self.engine.start_calibration()
        self.sounds.cue_start()
        self.poll()

    def toggle_mute(self):
        self.sounds.set_muted(not self.sounds.muted)
        self.update_sound_btn()

    def update_sound_btn(self):
        self.sound_btn.setText("🎧  Sound off" if self.sounds.muted else "🎧  Sound on")

    def toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    # ---- data (4 Hz) -----------------------------------------------------------
    def poll(self):
        s = self.engine.snapshot()
        self.snap = s
        state, last = s["state"], s["last"]
        poor = bool(last) and not last["valid"]
        thr = s["threshold"]
        z = last.get("z_smooth", np.nan) if last else np.nan
        rewarding = state == RECORDING and not poor and np.isfinite(z) and z > thr

        if self.prev_state == CALIBRATING and state != CALIBRATING:
            if state == READY:
                self.sounds.cue_done()
                if self.auto_start:
                    self.engine.toggle_feedback()
                    state = RECORDING
        if self.prev_state in (RECORDING, PAUSED) and state == FINISHED:
            self.sounds.cue_done()  # the session is over (eyes may be closed)
            self.a.ripples.append((self.a.t, 1.0))
        self.prev_state = state

        sess = s["session"]
        self.a.best = sess["best_before"] if sess else None
        self.a.planned_s = sess["minutes"] * 60 if sess and sess["mode"] == TIMED else None

        # chimes and milestones
        total = s["total"]
        self.a.total = total
        k = int(total // CHIME_EVERY)
        if k < self.chimed:  # reset elsewhere (e.g. the debug window)
            self.chimed, self.milestones = k, int(total // MILESTONE_EVERY)
        if k > self.chimed:
            if self.a.t - self.last_chime >= MIN_CHIME_GAP_S:
                self.chimed = k
                self.last_chime = self.a.t
                self.sounds.chime()
                self.a.ripples.append((self.a.t, 0.6))
        m = int(total // MILESTONE_EVERY)
        if m > self.milestones:
            self.milestones = m
            self.sounds.milestone()
            self.a.ripples.append((self.a.t, 1.0))
        above = self.a.best is not None and total > self.a.best
        if above and not self.above_best and state == RECORDING:  # passing the personal best
            self.sounds.milestone()
            self.a.ripples.append((self.a.t, 1.0))
        self.above_best = above

        # the figure and the pad follow smoothed z around the threshold
        closeness = float(np.clip((z - (thr - 1.0)) / 1.5, 0, 1)) if np.isfinite(z) else 0.0
        active = state == RECORDING and not poor
        self.a.joy_target = closeness if active else (0.6 if state == FINISHED else 0.0)
        self.sounds.set_pad_target(0.25 + 0.75 * closeness if active else 0.0)
        # pose: z 0 -> standing, eyes open; z = threshold -> seated, eyes closed; z >= threshold+1
        # -> fully locked in. Only during a session with a clean signal.
        self.a.depth_target = float(np.clip(z / (thr + 1.0), 0, 1)) if active and np.isfinite(z) else 0.0
        self.a.worried = poor and state in (RECORDING, CALIBRATING)
        if poor and state in (RECORDING, CALIBRATING):
            splash = rgb(SPLASH["poor"])
        elif state == CALIBRATING:
            splash = rgb(SPLASH["calib"])
        elif state == FINISHED:
            splash = rgb(SPLASH["done"])
        elif active:  # grey -> yellow as alpha approaches the threshold
            splash = rgb(SPLASH["rest"]) + (rgb(SPLASH["earn"]) - rgb(SPLASH["rest"])) * closeness
        else:
            splash = rgb(SPLASH["rest"])
        self.a.splash_target = splash

        self.a.progress = None
        if s["calib_progress"]:
            done, dur = s["calib_progress"]
            self.a.progress = min(1.0, done / dur)
        # reward against active session time: pauses don't stretch it, a timed session fills the width
        pts = np.column_stack([s["session_s"], s["cumulative_reward"]])
        self.a.points = pts[np.isfinite(pts[:, 0])]

        self.set_text(state, poor, rewarding, s)

    def set_text(self, state, poor, rewarding, s):
        sess = s["session"]
        tag = "rest"
        if state == WAITING:
            title, hint, btn = "Connecting", "Waiting for the headset's signal.", "Connecting…"
        elif state == NEEDS_BASELINE:
            failed = s["message"].startswith("Baseline failed")
            title = "Let's try that again" if failed else "Welcome"
            hint = ("The signal was too noisy to calibrate. Get comfortable, relax your jaw, "
                    "and try again." if failed else
                    "Choose your session below. Calibration takes one minute with your eyes "
                    "open — the session then starts by itself.")
            btn = "Begin calibration"
            tag = "poor" if failed else "rest"
        elif state == CALIBRATING:
            done, dur = s["calib_progress"] or (0, 1)
            left = max(0, int(round(dur - done)))
            title = "A moment to settle."
            hint = ("Keep still and relax your jaw." if poor else
                    f"Eyes open, gaze resting on the figure. Breathe normally.  ·  {left // 60}:{left % 60:02d}")
            btn = "Calibrating…"
            tag = "poor" if poor else "calib"
        elif state == RECORDING:
            if poor:
                title, hint, tag = "Hold still", "The signal is unclear. Relax your jaw and keep still.", "poor"
            elif self.above_best:
                title, hint, tag = "Beyond your best", "Every chime now is new ground.", "earn"
            elif rewarding:
                title, hint, tag = "Flowing", "Each chime is a point. Stay with it.", "earn"
            else:
                title, hint = "Resting", "Let your attention soften. Nothing to force."
            btn = "Pause"
        elif state == PAUSED:
            title, hint, btn = "Paused", "The clock is stopped. Resume when you're ready.", "Resume"
        elif state == FINISHED:
            r = s["result"] or {}
            mins = int(r.get("active_duration_s", 0))
            title = "New personal best" if r.get("personal_best") else "Session complete"
            hint = f"{int(r.get('score', 0))} points in {mins // 60}:{mins % 60:02d}"
            if sess and sess["best_before"] is not None and not r.get("personal_best"):
                hint += f"  ·  your best: {int(sess['best_before'])}"
            btn = "Start new session"
            tag = "done"
        else:  # READY
            title = "Ready"
            hint = "Start the session, then close your eyes. Chimes mean you're earning."
            btn = "Start session"
        if s["done"]:
            hint = "The recording has ended."

        self.headline.setText(title)
        tag_text = {"rest": "Resting", "earn": "Above baseline", "calib": "Calibrating · eyes open",
                    "poor": "Poor signal · paused", "done": "Saved"}[tag]
        if state in (WAITING, NEEDS_BASELINE, READY, FINISHED, PAUSED) and tag in ("rest",):
            tag_text = {WAITING: "Connecting", NEEDS_BASELINE: "Not calibrated", READY: "Calibrated",
                        PAUSED: "Paused"}.get(state, "Resting")
        if state == NEEDS_BASELINE and tag == "poor":
            tag_text = "Calibration failed"
        self.status.setText(tag_text)
        self.status.setStyleSheet(tag_css(tag))
        self.hint.setText(hint)
        self.hint.setStyleSheet(f"color: {ORANGE if tag == 'poor' else MUTED}")
        timed = bool(sess) and sess["mode"] == TIMED and state in (RECORDING, PAUSED, FINISHED)
        self.progress.setVisible(timed)
        if timed:
            self.progress.setValue(int(1000 * min(1.0, sess["elapsed"] / (sess["minutes"] * 60))))
        self.primary.setText(btn)
        self.primary.setEnabled(state in (NEEDS_BASELINE, READY, RECORDING, PAUSED, FINISHED))
        self.recal_btn.setVisible(state in (READY, FINISHED))
        self.end_btn.setVisible(state in (RECORDING, PAUSED))
        self.setup.setVisible(state in (WAITING, NEEDS_BASELINE, READY, FINISHED))
        self.minutes.setEnabled(self.timed_btn.isChecked())
        clock = ""
        if sess and state in (RECORDING, PAUSED):
            secs = sess["elapsed"] if sess["mode"] == INDEFINITE else sess["minutes"] * 60 - sess["elapsed"]
            secs = int(math.ceil(max(0.0, secs)))
            clock = f"{secs // 60}:{secs % 60:02d}" + (" left" if sess["mode"] == TIMED else "")
        self.clock.setText(clock)

    # ---- animation (30 Hz) ------------------------------------------------------
    def frame(self):
        now = time.monotonic()
        dt, self._last_frame = now - self._last_frame, now
        self.a.step(dt)
        self.sounds.tick(dt)
        shown = int(self.a.shown_total + 1e-6)
        if shown != self.shown_int:
            self.shown_int = shown
            best = self.a.best
            extra = (f"<span style='font-size:20px; color:{MUTED}; font-weight:400'> / {int(best)} best</span>"
                     if best is not None else
                     f"<span style='font-size:20px; color:{MUTED}; font-weight:400'> points</span>")
            self.points.setText(f"{shown}{extra}")
        self.update()
        self.figure.update()
        self.curve.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(PAPER))
        # a soft wash across the top, tinted by the state (blue, or peach for poor signal)
        wash = QtGui.QRadialGradient(QPointF(self.width() / 2, -self.height() * 0.08),
                                     max(self.width(), 500) * 0.75)
        wash.setColorAt(0.0, qcolor(self.a.splash, 150))
        wash.setColorAt(0.55, qcolor(self.a.splash, 45))
        wash.setColorAt(1.0, qcolor(self.a.splash, 0))
        p.fillRect(self.rect(), wash)
