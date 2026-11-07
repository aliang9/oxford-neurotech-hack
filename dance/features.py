"""Window features. Used unchanged by training and live decoding.

No filtering beyond removing each channel's mean: slow motion artifacts and
high-frequency EMG are the signal here. Mains (50 Hz) is avoided by leaving
those frequencies out of the bands instead of notch-filtering.
"""

import numpy as np
from scipy.signal import welch

from . import config as cfg

BANDS = {
    "motion": (0.5, 3),
    "theta": (3, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
    "emg": (55, 95),
}
EPS = 1e-6
PAIRS = np.triu_indices(cfg.N_EEG, k=1)  # 28 channel pairs


def windows(data):
    """(start, window) for every WIN-sample window, every STEP samples, EEG columns only."""
    for start in range(0, len(data) - cfg.WIN + 1, cfg.STEP):
        yield start, data[start:start + cfg.WIN, :cfg.N_EEG]


def extract(window):
    """1-D feature vector from one (WIN, 8) window of raw EEG."""
    x = window - window.mean(axis=0)

    rms = np.log(np.sqrt((x ** 2).mean(axis=0)) + EPS)
    diff_rms = np.log(np.sqrt((np.diff(x, axis=0) ** 2).mean(axis=0)) + EPS)

    f, pxx = welch(x, fs=cfg.FS, nperseg=len(x) // 2, axis=0)
    bands = [np.log(pxx[(f >= lo) & (f < hi)].mean(axis=0) + EPS) for lo, hi in BANDS.values()]

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.nan_to_num(np.corrcoef(x.T)[PAIRS])  # flat channel -> 0

    return np.concatenate([rms, diff_rms, *bands, corr])


def keep_mask(exclude=()):
    """Boolean mask over the features: False for any feature that uses an excluded channel."""
    exclude = set(exclude)
    per_channel = [ch for _ in ["rms", "diff_rms", *BANDS] for ch in cfg.CHANNELS]
    pairs = [{cfg.CHANNELS[i], cfg.CHANNELS[j]} for i, j in zip(*PAIRS)]
    return np.array([ch not in exclude for ch in per_channel] + [not (p & exclude) for p in pairs])


def feature_names():
    per_channel = ["rms", "diff_rms", *BANDS]
    names = [f"{ch}:{kind}" for kind in per_channel for ch in cfg.CHANNELS]
    names += [f"corr:{cfg.CHANNELS[i]}-{cfg.CHANNELS[j]}" for i, j in zip(*PAIRS)]
    return names
