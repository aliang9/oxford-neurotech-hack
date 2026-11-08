"""The Tandem logo, drawn in the hand-drawn style of the illustrations.

Two overlapping circles (orange = MOVE, blue = MIND) with ink outlines that are only just
imperfect (a hint of the pen, not a wobble), a few motion ticks, and the word "tandem"
handwritten in Patrick Hand.
Deterministic (seeded), so it looks the same every time.

    paint_logo(painter, x, baseline_y, size, faces=False, bounce=True)  -> width drawn
"""

import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainterPath, QPen, QTransform

ORANGE, BLUE, INK = "#E88A49", "#83A9DB", "#2A2623"
FONT_FAMILY = "Patrick Hand"     # bundled in fonts/ (SIL OFL), loaded by load_fonts()
WORD_SCALE = 1.15                # Patrick Hand runs small; scale it to sit with the mark
FONT_DIR = Path(__file__).with_name("fonts")


def load_fonts():
    """Register the bundled fonts with Qt (call once, after the QApplication exists)."""
    from PySide6.QtGui import QFontDatabase

    for path in FONT_DIR.glob("*.ttf"):
        QFontDatabase.addApplicationFont(str(path))


def wobbly_circle(c, r, seed, overshoot=0.0, wobble=0.012, n=96):
    """A hand-drawn circle: radius drifts a little; with overshoot > 0 the end passes the start."""
    rng = np.random.default_rng(seed)
    k = rng.normal(0, 1, 4)
    ph = rng.uniform(0, 2 * np.pi, 4)
    start = rng.uniform(0, 2 * np.pi)
    total = 2 * np.pi * (1 + overshoot)
    path = QPainterPath()
    for i in range(n + 1):
        t = total * i / n
        th = start + t
        rr = r * (1 + wobble * (k[0] * math.sin(2 * th + ph[0]) + 0.6 * k[1] * math.sin(3 * th + ph[1])
                                + 0.3 * k[2] * math.sin(5 * th + ph[2])))
        rr *= 1 + 0.008 * k[3] * (t / total)         # the pen drifts out a hair as it goes round
        pt = QPointF(c.x() + rr * math.cos(th), c.y() + rr * math.sin(th))
        path.moveTo(pt) if i == 0 else path.lineTo(pt)
    if overshoot <= 0:
        path.closeSubpath()
    return path


def _pen(width):
    pen = QPen(QColor(INK), width)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    return pen


def paint_mark(p, x, baseline, size, faces=False):
    """The two-circle mark; returns its width."""
    r = size * 0.34
    cy = baseline - size * 0.36
    c1, c2 = QPointF(x + r, cy), QPointF(x + r * 2.15, cy)
    lw = max(1.2, size * 0.038)

    # fills, sitting neatly inside the outlines
    a = wobbly_circle(c1, r, seed=3, wobble=0.0)
    b = wobbly_circle(c2, r, seed=7, wobble=0.0)
    p.fillPath(a.subtracted(b), QColor(ORANGE))
    p.fillPath(b.subtracted(a), QColor(BLUE))

    # ink outlines, with just a hint of the pen where they close
    p.setBrush(Qt.NoBrush)
    p.setPen(_pen(lw))
    p.drawPath(wobbly_circle(c1, r, seed=11, overshoot=0.015))
    p.drawPath(wobbly_circle(c2, r, seed=13, overshoot=0.015))

    if faces:  # two little friends: closed happy eyes and a smile each
        fl = max(1.0, lw * 0.8)
        p.setPen(_pen(fl))
        for c, look in ((c1, -1), (c2, 1)):
            fx = c.x() + look * r * 0.32
            for ex in (-0.22, 0.22):
                e = QPainterPath(QPointF(fx + (ex - 0.09) * r, cy - r * 0.06))
                e.quadTo(QPointF(fx + ex * r, cy - r * 0.16), QPointF(fx + (ex + 0.09) * r, cy - r * 0.06))
                p.drawPath(e)
            m = QPainterPath(QPointF(fx - 0.14 * r, cy + r * 0.18))
            m.quadTo(QPointF(fx, cy + r * 0.34), QPointF(fx + 0.14 * r, cy + r * 0.18))
            p.drawPath(m)

    # motion ticks: orange over the left circle, blue over the right
    tl = max(1.1, lw * 0.85)
    for c, color, angles in ((c1, ORANGE, (-145, -120, -95)), (c2, BLUE, (-85, -60, -35))):
        pen = QPen(QColor(color), tl)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        for ang in angles:
            th = math.radians(ang)
            p.drawLine(QPointF(c.x() + r * 1.30 * math.cos(th), c.y() + r * 1.30 * math.sin(th)),
                       QPointF(c.x() + r * 1.50 * math.cos(th), c.y() + r * 1.50 * math.sin(th)))
    return r * 3.15


def word_font(size):
    f = QFont()
    f.setFamilies([FONT_FAMILY, "Arial Rounded MT Bold", "Helvetica Neue"])
    f.setPixelSize(int(size * WORD_SCALE))
    return f


def paint_word(p, x, baseline, size, bounce=False):
    """'tandem' in the handwritten font (optionally with a bounce); returns its width."""
    f = word_font(size)
    word = "tandem"
    lift = [0.0, -0.035, 0.01, -0.03, 0.02, -0.015] if bounce else [0.0] * 6
    tilt = [-3.0, 2.0, -1.5, 2.5, -2.0, 1.5] if bounce else [0.0] * 6
    from PySide6.QtGui import QFontMetricsF
    fm = QFontMetricsF(f)
    if not bounce:  # the whole word at once, so the font's kerning and hinting are kept
        p.setFont(f)
        p.setPen(QColor(INK))
        p.drawText(QPointF(x, baseline), word)
        return fm.horizontalAdvance(word)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(INK))
    for i, (ch, dy, rot) in enumerate(zip(word, lift, tilt)):
        # position from the advance of everything before it, which includes kerning
        left = x + fm.horizontalAdvance(word[:i])
        w = fm.horizontalAdvance(ch)
        path = QPainterPath()
        path.addText(QPointF(0, 0), f, ch)
        t = QTransform()
        t.translate(left + w / 2, baseline + dy * size)
        t.rotate(rot)
        t.translate(-w / 2, 0)
        p.drawPath(t.map(path))
    return fm.horizontalAdvance(word)


def paint_logo(p, x, baseline, size, faces=False, bounce=False):
    w = paint_mark(p, x, baseline, size, faces=faces)
    gap = size * 0.22
    return w + gap + paint_word(p, x + w + gap, baseline, size, bounce=bounce)
