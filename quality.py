"""Live per-channel EEG signal quality from the core8_eeg LSL stream.

The BCI Core-8 can't measure impedance (neither gtec_ble nor gpype exposes
it), so quality is judged from the signal itself, on acquire.py's 1-40 Hz
filtered stream:

  amplitude  median over 1 s chunks of each chunk's std. Robust to the
             odd blink. Good EEG at rest sits around 8-17 uV.
  drift      1-4 Hz rms. Poor contact shows up here (slow wander).
  muscle     20-40 Hz rms. Jaw, forehead or neck tension.

The thresholds come from session p01_s1. There the five good channels had
amplitude 9-12 uV (90th percentile 15-17); Fz, Cz and C4 had poor contact
at 28-57 uV, almost all of it in 1-4 Hz. They are set loosely: replaying
that session, the good channels are green in 91-100% of 5 s windows, while
Fz and C4 are still red most of the time.
"""

import threading
import time
from collections import deque

import numpy as np
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import welch

GOOD_UV, BAD_UV, FLAT_UV = 25.0, 40.0, 0.5      # amplitude
MUSCLE_WARN_UV, MUSCLE_BAD_UV = 6.0, 10.0       # 20-40 Hz rms
POSTERIOR = ("Pz", "PO7", "POz", "PO8")         # where eyes-closed alpha shows
COLORS = {"good": "\033[32m", "warn": "\033[33m", "bad": "\033[31m", "ignored": "\033[90m"}


def band_rms(x, fs, lo, hi):
    f, pxx = welch(x, fs=fs, nperseg=min(len(x), int(fs)), axis=0)
    return np.sqrt(pxx[(f >= lo) & (f < hi)].sum(axis=0) * (f[1] - f[0]))


def band_power(x, fs, lo, hi):
    f, pxx = welch(x, fs=fs, nperseg=min(len(x), int(fs)), axis=0)
    return pxx[(f >= lo) & (f < hi)].mean(axis=0)


def assess(x, fs, labels, ignore=()):
    """One dict per channel for x, a (samples, channels) array of 1-40 Hz EEG."""
    sec = int(fs)
    amp = np.median([x[i:i + sec].std(axis=0) for i in range(0, len(x) - sec + 1, sec)], axis=0)
    drift, muscle = band_rms(x, fs, 1, 4), band_rms(x, fs, 20, 40)
    rows = []
    for lab, a, d, m in zip(labels, amp, drift, muscle):
        if lab in ignore:
            status, hint = "ignored", ""
        elif a < FLAT_UV:
            status, hint = "bad", "flat: no contact, or headset not streaming"
        else:
            status = ("bad" if a > BAD_UV or m > MUSCLE_BAD_UV else
                      "warn" if a > GOOD_UV or m > MUSCLE_WARN_UV else "good")
            if status == "good":
                hint = ""
            elif m > MUSCLE_WARN_UV:
                hint = "muscle: relax jaw, forehead and neck"
            elif d > 0.7 * a:
                hint = "slow drift: poor contact. Part the hair, re-wet, press the electrode on"
            else:
                hint = "noisy: re-seat the electrode, keep the cable still"
        rows.append(dict(label=lab, amplitude=float(a), drift=float(d), muscle=float(m),
                         status=status, hint=hint))
    return rows


def ok(rows, strict=True):
    """strict: every channel good. Otherwise: no channel bad."""
    allowed = ("good", "ignored") if strict else ("good", "warn", "ignored")
    return all(r["status"] in allowed for r in rows)


def format_table(rows):
    lines = [f"{'ch':>4s}  {'ampl':>6s}  {'1-4Hz':>6s}  {'20-40':>6s}  status   (uV; good <= "
             f"{GOOD_UV:g}, bad > {BAD_UV:g})"]
    for r in rows:
        c = COLORS[r["status"]]
        lines.append(f"{r['label']:>4s}  {r['amplitude']:6.1f}  {r['drift']:6.1f}  {r['muscle']:6.1f}  "
                     f"{c}{r['status']:7s}\033[0m  {r['hint']}")
    return "\n".join(lines)


class QualityMonitor:
    """Keeps the last `buffer` seconds of the stream's EEG channels, in a background thread."""

    def __init__(self, stream_name="core8_eeg", buffer=20.0, timeout=10.0):
        streams = resolve_byprop("name", stream_name, timeout=timeout)
        if not streams:
            raise RuntimeError(f"No LSL stream '{stream_name}'. Is acquire.py running?")
        self.inlet = StreamInlet(streams[0])
        info = self.inlet.info()
        self.fs = info.nominal_srate()
        labels, types = [], []
        ch = info.desc().child("channels").child("channel")
        while not ch.empty():
            labels.append(ch.child_value("label"))
            types.append(ch.child_value("type"))
            ch = ch.next_sibling()
        if len(labels) != info.channel_count():  # no description: assume all EEG
            labels = [f"ch{i + 1}" for i in range(info.channel_count())]
            types = ["EEG"] * len(labels)
        self.idx = [i for i, t in enumerate(types) if t == "EEG"]
        self.labels = [labels[i] for i in self.idx]
        self._buf = deque(maxlen=int(buffer * self.fs))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.is_set():
            chunk, _ = self.inlet.pull_chunk(timeout=0.2)
            if chunk:
                with self._lock:
                    self._buf.extend(np.asarray(chunk)[:, self.idx])

    def latest(self, seconds):
        """The last `seconds` of EEG, waiting until that much has arrived."""
        n = int(seconds * self.fs)
        while True:
            with self._lock:
                if len(self._buf) >= n:
                    return np.array(self._buf)[-n:]
            time.sleep(0.1)

    def assess(self, seconds=5.0, ignore=()):
        return assess(self.latest(seconds), self.fs, self.labels, ignore)

    def close(self):
        self._stop.set()
