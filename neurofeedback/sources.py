"""Where samples come from: live LSL, a recorded CSV, or a simulator.

All three have the same interface as dance/stream.py:

    samples, timestamps = source.pull(timeout)   # everything new since the last call
    source.labels, source.fs, source.done

samples is (n, len(labels)); missing values may be NaN.
"""

import time

import numpy as np
from scipy.signal import lfilter

from . import config as cfg


class LSLSource:
    """An LSL stream, e.g. acquire.py's core8_eeg."""

    def __init__(self, name=cfg.LSL_STREAM, timeout=10.0):
        from pylsl import StreamInlet, resolve_byprop

        streams = resolve_byprop("name", name, timeout=timeout)
        if not streams:
            raise RuntimeError(f"No LSL stream '{name}'. Is acquire.py running?")
        self.inlet = StreamInlet(streams[0], max_buflen=60)
        info = self.inlet.info()
        self.fs = info.nominal_srate() or cfg.FS
        self.labels = _lsl_labels(info)
        self.done = False

    def pull(self, timeout=0.1):
        chunk, stamps = self.inlet.pull_chunk(timeout=timeout)
        if not chunk:
            return np.empty((0, len(self.labels))), np.empty(0)
        return np.asarray(chunk, float), np.asarray(stamps, float)


def _lsl_labels(info):
    labels = []
    ch = info.desc().child("channels").child("channel")
    while not ch.empty():
        labels.append(ch.child_value("label"))
        ch = ch.next_sibling()
    n = info.channel_count()
    if len(labels) == n and all(labels):
        return labels
    # No channel description: assume acquire.py's layout (8 EEG, then the marker channel)
    if n in (len(cfg.MONTAGE), len(cfg.MONTAGE) + 1):
        return (cfg.MONTAGE + ["marker"])[:n]
    return [f"ch{i + 1}" for i in range(n)]


class CSVReplaySource:
    """Plays back a CSV with a header row, in real time (x speed; speed=0: as fast as possible).

    A time column (time / timestamp / lsl_timestamp) is used for timestamps and to
    estimate the sample rate; without one, samples are assumed to be at cfg.FS.
    Marker/trigger columns are ignored. Empty fields become NaN (missing samples).
    Reads both dance/ block CSVs and receive.py --csv output.
    """

    TIME_COLS = {"time", "timestamp", "lsl_timestamp", "t"}
    SKIP_COLS = {"marker", "markers", "trigger", "trig"}

    def __init__(self, path, speed=1.0, fs=None):
        with open(path) as f:
            header = [h.strip() for h in f.readline().split(",")]
        table = np.genfromtxt(path, delimiter=",", skip_header=1, ndmin=2,
                              missing_values="", filling_values=np.nan)
        if table.shape[1] != len(header):
            raise RuntimeError(f"{path}: header has {len(header)} columns, data has {table.shape[1]}")
        lower = [h.lower() for h in header]
        t_col = next((i for i, h in enumerate(lower) if h in self.TIME_COLS), None)
        eeg = [i for i, h in enumerate(lower) if i != t_col and h not in self.SKIP_COLS]
        self.labels = [header[i] for i in eeg]
        self.data = table[:, eeg]
        if t_col is not None:
            self.times = table[:, t_col]
            est = 1.0 / np.nanmedian(np.diff(self.times))
            self.fs = fs or float(round(est))
        else:
            self.fs = fs or cfg.FS
            self.times = np.arange(len(table)) / self.fs
        self.speed = speed
        self.pos = 0
        self.t0 = None

    @property
    def done(self):
        return self.pos >= len(self.data)

    def pull(self, timeout=0.1):
        if self.t0 is None:
            self.t0 = time.monotonic()
        if self.speed <= 0:
            end = min(len(self.data), self.pos + int(10 * self.fs))  # 10 s per pull
        else:
            end = min(len(self.data), int((time.monotonic() - self.t0) * self.speed * self.fs))
        if end <= self.pos:
            time.sleep(timeout)
            return self.data[:0], self.times[:0]
        start, self.pos = self.pos, end
        return self.data[start:end], self.times[start:end]


class SimulatedSource:
    """Synthetic 8-channel EEG on the Core-8 montage, for testing without the headset.

    Background: 1/f-ish noise plus a little 50 Hz and a DC offset per channel.
    Alpha: 10 Hz with a wandering phase, strongest at PO7/POz/PO8, whose amplitude
    slowly rises and falls (period `period_s`) so the reward has something to track.
    Every ~40 s an artifact (head movement or muscle burst) tests the rejection.
    `alpha_gain` scales the alpha (it glides there over a few seconds); the app raises it during
    sessions, standing in for eyes closing after the eyes-open baseline.
    `speed` > 1 runs faster than real time.
    """

    GAIN = {"Fz": 0.1, "C3": 0.3, "Cz": 0.3, "C4": 0.3, "Pz": 0.7, "PO7": 1.0, "POz": 1.0, "PO8": 1.0}

    def __init__(self, fs=cfg.FS, speed=1.0, period_s=90.0, alpha_uv=8.0, artifacts=True, seed=0):
        self.fs, self.speed, self.period, self.alpha_uv = fs, speed, period_s, alpha_uv
        self.labels = list(cfg.MONTAGE)
        self.done = False
        self.rng = np.random.default_rng(seed)
        n = len(self.labels)
        self.gain = np.array([self.GAIN.get(c, 0.3) for c in self.labels])
        self.offset = self.rng.uniform(-3e4, 3e4, n)
        self.ar_zi = np.zeros((1, n))
        self.phase = 0.0
        self.alpha_gain, self._gain = 1.0, 1.0
        self.produced = 0
        self.t_start = time.monotonic()
        self.artifacts = artifacts
        self.next_artifact = self.rng.uniform(30, 50) if artifacts else np.inf

    def pull(self, timeout=0.1):
        target = int((time.monotonic() - self.t_start) * self.speed * self.fs)
        n = target - self.produced
        if n <= 0:
            time.sleep(timeout)
            return np.empty((0, len(self.labels))), np.empty(0)
        t = (self.produced + np.arange(n)) / self.fs
        x = self._background(n) + self._alpha(t)[:, None] * self.gain
        x += 2.0 * np.sin(2 * np.pi * 50 * t)[:, None]
        x += self._artifact(t)
        self.produced = target
        return x + self.offset, self.t_start + t

    def _background(self, n):
        white = self.rng.normal(0, 1, (n, len(self.labels)))
        slow, self.ar_zi = lfilter([1.5], [1, -0.97], white, axis=0, zi=self.ar_zi)
        return slow + 2.0 * white

    def _alpha(self, t):
        # phase random walk keeps the peak a few tenths of a Hz wide
        dphi = 2 * np.pi * 10.0 / self.fs + self.rng.normal(0, 0.05, len(t))
        phi = self.phase + np.cumsum(dphi)
        self.phase = phi[-1] % (2 * np.pi)
        k = 1 - np.exp(-np.arange(1, len(t) + 1) / (3.0 * self.fs))  # ~3 s glide
        gain = self._gain + (self.alpha_gain - self._gain) * k
        self._gain = gain[-1]
        envelope = gain * self.alpha_uv * (1 + 0.6 * np.sin(2 * np.pi * t / self.period))
        return envelope * np.sin(phi)

    def _artifact(self, t):
        out = np.zeros((len(t), len(self.labels)))
        a = self.next_artifact
        hit = (t >= a) & (t < a + 1.5)
        if hit.any():
            if int(a) % 2:   # movement: big slow swing on every channel
                out[hit] = 200 * np.sin(2 * np.pi * 0.7 * (t[hit] - a))[:, None]
            else:            # muscle: broadband 20-60 uV noise
                out[hit] = self.rng.normal(0, 30, (hit.sum(), len(self.labels)))
        if t[-1] >= a + 1.5:
            self.next_artifact = a + self.rng.uniform(30, 50)
        return out


def make_source(args):
    if args.source == "lsl":
        return LSLSource(args.stream_name)
    if args.source == "csv":
        if not args.csv:
            raise SystemExit("--source csv needs --csv PATH")
        return CSVReplaySource(args.csv, speed=args.speed)
    return SimulatedSource(speed=args.speed, artifacts=not args.no_artifacts)
