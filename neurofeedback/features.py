"""Preprocessing, alpha-band power and artifact checks."""

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos, welch

from quality import band_rms

from . import config as cfg


def select_channels(labels, wanted, fallback=cfg.FALLBACK):
    """Column indices and names of the wanted channels present in `labels`, plus warnings."""
    lookup = {lab.lower(): i for i, lab in enumerate(labels)}
    idx, names, warnings = [], [], []
    for ch in wanted:
        if ch.lower() in lookup:
            pick = ch
        elif fallback.get(ch, "").lower() in lookup:
            pick = fallback[ch]
            warnings.append(f"{ch} is not in the montage; using {pick} instead")
        else:
            warnings.append(f"{ch} is not in the montage; skipped")
            continue
        i = lookup[pick.lower()]
        if i not in idx:
            idx.append(i)
            names.append(labels[i])
    if not idx:
        raise RuntimeError(f"None of {wanted} are in the stream's channels {labels}")
    return idx, names, warnings


class Preprocessor:
    """Causal 1-40 Hz bandpass + mains notch, with filter state kept across chunks.

    Missing samples (NaN) are replaced by the last good value before filtering so they
    cannot poison the filter state; the returned mask marks them so the window is rejected.
    """

    def __init__(self, fs, n_channels, band=cfg.FILTER_BAND, mains=cfg.MAINS_HZ):
        sos = [butter(4, band, btype="bandpass", fs=fs, output="sos")]
        if mains and mains < fs / 2:
            sos.append(tf2sos(*iirnotch(mains, 30, fs=fs)))
        self.sos = np.vstack(sos)
        self.zi = None
        self.last = np.zeros(n_channels)

    def process(self, x):
        x = np.array(x, float)
        bad = ~np.isfinite(x)
        for i in np.flatnonzero(bad.any(axis=1)):  # forward-fill (rows before i are already filled)
            prev = x[i - 1] if i > 0 else self.last
            x[i, bad[i]] = prev[bad[i]]
        if len(x):
            self.last = x[-1].copy()
        if self.zi is None and len(x):
            # start in steady state for the first sample, so a DC offset doesn't ring
            self.zi = sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y, bad


def log_alpha_power(x, fs, band=cfg.ALPHA_BAND, seg_s=cfg.WELCH_SEG_S):
    """log10 alpha-band power (uV^2) per channel of x (samples, channels), via Welch."""
    f, pxx = welch(x, fs=fs, nperseg=min(len(x), int(seg_s * fs)), axis=0)
    m = (f >= band[0]) & (f <= band[1])
    power = pxx[m].sum(axis=0) * (f[1] - f[0])
    return np.log10(np.maximum(power, 1e-12))


def stream_problem(times):
    """A problem affecting every channel in the window ('' if none): dropped samples."""
    if len(times) > 1:
        gap = float(np.max(np.diff(times)))
        if gap > cfg.MAX_GAP_S:
            return f"dropped samples ({gap:.2f} s gap)"
    return ""


def channel_problems(x, fs, missing):
    """Per channel of the window x (samples, channels): why it is unusable ('' if clean)."""
    checks = [
        (missing.any(axis=0), "missing samples"),
        (x.std(axis=0) < cfg.FLAT_STD_UV, "flat/no contact"),
        (np.ptp(x, axis=0) > cfg.MAX_PTP_UV, "movement/large artifact"),
        (band_rms(x, fs, 20, 40) > cfg.MUSCLE_UV, "muscle"),
    ]
    out = []
    for c in range(x.shape[1]):
        out.append(", ".join(what for bad, what in checks if bad[c]))
    return out
