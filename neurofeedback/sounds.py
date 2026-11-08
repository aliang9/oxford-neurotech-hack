"""Gentle feedback sounds, synthesised with numpy and played with Qt (no extra dependencies).

    chime      soft bell, one per reward point; notes wander over a D-major pentatonic scale
    milestone  a fuller, lower bell every MILESTONE_EVERY points
    cue_*      short two-note cues: calibration started / finished
    pad        a quiet looping drone whose volume follows how close alpha is to the threshold

Everything is written to WAV files in a temp folder at startup and played with QSoundEffect.
"""

import tempfile
import wave
from pathlib import Path

import numpy as np

SR = 44100
PENTATONIC = [293.66, 329.63, 369.99, 440.00, 493.88, 587.33, 659.25]  # D4 E4 F#4 A4 B4 D5 E5
CHIME_VOLUME = 0.22
PAD_MAX_VOLUME = 0.28


def _bell(f, dur=3.5, decay=1.1, brightness=1.0):
    """A soft struck bell / singing bowl: fast attack, long decay, a few slightly inharmonic partials."""
    t = np.arange(int(dur * SR)) / SR
    attack = 1 - np.exp(-t / 0.008)
    x = np.sin(2 * np.pi * f * t) * np.exp(-t / decay)
    x += 0.30 * brightness * np.sin(2 * np.pi * 2.01 * f * t) * np.exp(-t / (decay * 0.45))
    x += 0.10 * brightness * np.sin(2 * np.pi * 3.02 * f * t) * np.exp(-t / (decay * 0.2))
    x += 0.15 * np.sin(2 * np.pi * 0.5 * f * t) * np.exp(-t / (decay * 1.3))  # warm undertone
    return _finish(x * attack)


def _cue(freqs, gap=0.28):
    parts = [np.pad(_bell(f, dur=2.5, decay=0.8, brightness=0.6), (int(i * gap * SR), 0))
             for i, f in enumerate(freqs)]
    n = max(len(p) for p in parts)
    return _finish(sum(np.pad(p, (0, n - len(p))) for p in parts))


def _pad(dur=8.0):
    """Seamless loop: every frequency and the swell are whole multiples of 1/dur Hz."""
    t = np.arange(int(dur * SR)) / SR
    x = np.zeros_like(t)
    for i, f in enumerate([146.83, 220.00, 293.66, 369.99]):  # D3 A3 D4 F#4
        f = round(f * dur) / dur
        swell = 0.6 + 0.4 * np.sin(2 * np.pi * t / dur + i * np.pi / 2)
        detune = 2 / dur  # 0.25 Hz beating, a slow shimmer
        x += swell * (np.sin(2 * np.pi * f * t) + 0.6 * np.sin(2 * np.pi * (f + detune) * t)) / (1 + i * 0.6)
    return _finish(x, fade=False)


def _finish(x, peak=0.8, fade=True):
    x = x / (np.max(np.abs(x)) + 1e-9) * peak
    if fade:
        n = int(0.05 * SR)
        x[-n:] *= np.linspace(1, 0, n)
    return x


def _write(path, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype("<i2").tobytes())


class SoundBoard:
    """Plays the feedback sounds. Needs a running QApplication."""

    def __init__(self, muted=False):
        from PySide6.QtCore import QUrl
        from PySide6.QtMultimedia import QSoundEffect

        folder = Path(tempfile.mkdtemp(prefix="neurofeedback_sounds_"))

        def effect(name, samples, volume, copies=1):
            path = folder / f"{name}.wav"
            _write(path, samples)
            out = []
            for _ in range(copies):  # several copies so a note can ring on while it repeats
                e = QSoundEffect()
                e.setSource(QUrl.fromLocalFile(str(path)))
                e.setVolume(volume)
                out.append(e)
            return out

        self.notes = [effect(f"note{i}", _bell(f), CHIME_VOLUME, copies=2)
                      for i, f in enumerate(PENTATONIC)]
        self.milestone_fx = effect("milestone", _bell(146.83, dur=6, decay=2.2, brightness=0.8), 0.3)[0]
        self.start_fx = effect("cue_start", _cue([440.00, 293.66]), 0.25)[0]
        self.done_fx = effect("cue_done", _cue([293.66, 369.99, 440.00]), 0.28)[0]
        self.pad = effect("pad", _pad(), 0.0)[0]
        self.pad.setLoopCount(QSoundEffect.Loop.Infinite.value)
        self.pad_level = 0.0
        self.pad_target = 0.0
        self.note = 2
        self.turn = 0
        self.rng = np.random.default_rng()
        self.muted = muted

    def set_muted(self, muted):
        self.muted = muted
        if muted:
            self.pad.stop()
            self.pad_level = 0.0
            self.pad.setVolume(0.0)

    def chime(self):
        """Next note of a gentle random walk over the scale (like wind chimes)."""
        if self.muted:
            return
        self.note = int(np.clip(self.note + self.rng.choice([-2, -1, 1, 1, 2]), 0, len(self.notes) - 1))
        self.turn ^= 1
        self.notes[self.note][self.turn].play()

    def milestone(self):
        if not self.muted:
            self.milestone_fx.play()

    def cue_start(self):
        if not self.muted:
            self.start_fx.play()

    def cue_done(self):
        if not self.muted:
            self.done_fx.play()

    def set_pad_target(self, level):
        """0..1; the pad glides toward it in tick()."""
        self.pad_target = float(np.clip(level, 0, 1))

    def tick(self, dt):
        """Call often (e.g. 30 Hz): glides the pad volume, ~1.5 s time constant."""
        if self.muted:
            return
        self.pad_level += (self.pad_target - self.pad_level) * min(1.0, dt / 1.5)
        if self.pad_level > 0.005 and not self.pad.isPlaying():
            self.pad.play()
        elif self.pad_level <= 0.005 and self.pad_target == 0 and self.pad.isPlaying():
            self.pad.stop()
        self.pad.setVolume(self.pad_level * PAD_MAX_VOLUME)
