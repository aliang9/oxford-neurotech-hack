"""Where samples come from: the live LSL stream, or a recorded block CSV.

Both sources have the same interface:

    samples, timestamps = source.pull()   # everything that arrived since the last call

samples is (n, 9): 8 EEG channels then the marker channel.
"""

import time

import numpy as np
from pylsl import StreamInlet, resolve_byprop

from . import config as cfg


class LSLSource:
    def __init__(self, name=cfg.RAW_STREAM, timeout=10.0):
        streams = resolve_byprop("name", name, timeout=timeout)
        if not streams:
            raise RuntimeError(f"No LSL stream '{name}'. Is acquire.py running?")
        self.inlet = StreamInlet(streams[0], max_buflen=60)
        n = self.inlet.info().channel_count()
        if n != cfg.N_EEG + 1:
            raise RuntimeError(f"'{name}' has {n} channels, expected {cfg.N_EEG} EEG + 1 marker.")
        self.done = False  # a live stream never ends

    def pull(self, timeout=0.0):
        chunk, stamps = self.inlet.pull_chunk(timeout=timeout)
        if not chunk:
            return np.empty((0, cfg.N_EEG + 1)), np.empty(0)
        return np.asarray(chunk), np.asarray(stamps)


class ReplaySource:
    """Plays a recorded block back in real time (x speed). speed=0: all at once."""

    def __init__(self, path, speed=1.0):
        self.data, self.times = load_block(path)
        self.speed = speed
        self.pos = 0
        self.t0 = None

    @property
    def done(self):
        return self.pos >= len(self.data)

    def pull(self, timeout=0.0):
        if self.t0 is None:
            self.t0 = time.monotonic()
        if self.speed <= 0:
            end = len(self.data)
        else:
            elapsed = time.monotonic() - self.t0
            end = min(len(self.data), int(elapsed * self.speed * cfg.FS))
        start, self.pos = self.pos, max(self.pos, end)
        return self.data[start:self.pos], self.times[start:self.pos]


def load_block(path):
    """(data (n, 9), timestamps (n,)) from a block CSV written by dance.record."""
    table = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return table[:, 1:], table[:, 0]


def sample_labels(markers, skip_s=0.0):
    """Per-sample move code and trial number (both 0 outside trials).

    A move code in the marker channel opens a trial, TRIAL_END closes it.
    The first skip_s seconds of each trial are left unlabelled (reaction time).
    """
    m = np.rint(markers).astype(int)
    codes = np.zeros(len(m), int)
    trials = np.zeros(len(m), int)
    previous = np.r_[0, m[:-1]]
    onsets = np.flatnonzero((m != 0) & (m != previous))
    start, code, n = None, None, 0
    for i in onsets:
        if m[i] in cfg.MOVES:
            start, code = i, m[i]
        elif m[i] == cfg.TRIAL_END and start is not None:
            n += 1
            first = start + int(skip_s * cfg.FS)
            codes[first:i] = code
            trials[first:i] = n
            start = None
    return codes, trials
