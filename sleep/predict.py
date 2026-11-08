"""The prediction interface used by the (future) smart alarm.

    from sleep.predict import load_models, predict_future
    models = load_models()                       # models/sleep/models_all.joblib
    p = predict_future([0.1, 0.1, 0.6, 0.1, 0.1], stage_history=[2] * 40,
                       time_since_sleep_onset=180, horizons_minutes=[5, 10, 30],
                       models=models)
    # p: (3, 5) array, rows = horizons, columns = W, N1, N2, N3, REM
"""

import os

import joblib
import numpy as np

from . import config as cfg
from .models import K, evidence_from_labels, filter_states, forecast, forecast_bins, \
    from_dict, time_bin_minutes

DEFAULT_PATH = os.path.join(cfg.MODELS_DIR, "models_all.joblib")
_cache = {}


def load_models(path=DEFAULT_PATH):
    """{name: model} from a file written by sleep.train (cached)."""
    if path not in _cache:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found: run `python -m sleep.train` first")
        blob = joblib.load(path)
        _cache[path] = {name: from_dict(d) for name, d in blob["models"].items()}
    return _cache[path]


def _as_evidence(seq):
    """Stage history -> (T, K) evidence. Accepts labels (ints, -1/None = unknown) or
    per-epoch probability vectors (T, K)."""
    if seq is None or len(seq) == 0:
        return np.zeros((0, K))
    arr = np.asarray([-1 if s is None else s for s in seq] if np.ndim(seq) == 1 else seq,
                     float)
    if arr.ndim == 1:
        return evidence_from_labels(arr.astype(int))
    if arr.shape[1] != K:
        raise ValueError(f"stage probabilities must have {K} columns ({', '.join(cfg.STAGES)})")
    return arr


def predict_future(stage_probabilities, stage_history=None, time_since_sleep_onset=None,
                   horizons_minutes=cfg.HORIZONS_MIN, model="hsmm_time", models=None):
    """Probability of each sleep stage at each future horizon.

    stage_probabilities     length-5 belief over (W, N1, N2, N3, REM) for the current
                            30 s epoch, e.g. an EEG classifier's output or a one-hot label.
    stage_history           past epochs, oldest first, not including the current one: stage
                            labels (0-4, -1 for unknown) or (T, 5) probability vectors.
                            The HSMM uses it to infer how long the current stage has lasted;
                            without it, elapsed time is marginalised over its prior.
    time_since_sleep_onset  minutes from the first sleep epoch to the current epoch;
                            None or negative = not asleep yet.
    horizons_minutes        multiples of 0.5 min.
    model                   'hsmm_time' (default), 'hsmm', 'hmm_time', 'hmm' or 'persistence'.

    Probabilities are treated as likelihoods in the forward filter (i.e. as if the
    classifier had a flat prior). Returns an array (len(horizons), 5).
    """
    models = models or load_models()
    m = models[model]
    ks = []
    for h in np.atleast_1d(horizons_minutes):
        k = int(round(h * cfg.EPOCHS_PER_MIN))
        if k < 1 or abs(h * cfg.EPOCHS_PER_MIN - k) > 1e-6:
            raise ValueError(f"horizon {h} min is not a positive multiple of 0.5 min")
        ks.append(k)

    current = np.asarray(stage_probabilities, float).reshape(1, K)
    if model == "persistence":
        return np.vstack([m.forecast([int(np.argmax(current))], k) for k in ks])

    ev = np.vstack([_as_evidence(stage_history), current])
    now = -1.0 if time_since_sleep_onset is None else float(time_since_sleep_onset)
    minutes = now - np.arange(len(ev))[::-1] / cfg.EPOCHS_PER_MIN    # each history epoch
    if now < 0:
        minutes[:] = -1.0
    tb = time_bin_minutes(minutes) if m.time_dependent else np.zeros(len(ev), int)
    state = filter_states(m, ev, tb)[-1]
    bins = forecast_bins([now], max(ks), m.time_dependent)
    out = forecast(m, state, bins, ks)
    return np.vstack([out[k][0] for k in ks])
