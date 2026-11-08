"""Smart-alarm settings. Costs are heuristic proxies, not validated sleep-inertia measures."""

from sleep import config as sleep_cfg

STAGES = sleep_cfg.STAGES                  # W, N1, N2, N3, REM
EPOCH_S = sleep_cfg.EPOCH_S
MODEL_FILE = "models/sleep/models.joblib"  # trained on the training subjects only
MODEL_NAME = "hsmm_time"

# Awakening cost per stage (arbitrary units). N3 is the worst; REM is not assumed ideal.
# A brief awakening costs the same as N1: both are easy moments to wake, and pricing W
# below N1 makes the policy pass up N1 while holding out for an awakening that may not come.
STAGE_COST = {"W": 1.0, "N1": 1.0, "N2": 3.0, "N3": 10.0, "REM": 4.0}
# Cost per minute of sleep given up by waking before the latest time
SLEEP_VALUE_PER_MIN = 0.05                 # halved from 0.1: favour lighter stages a bit more

# Reactive baseline: wake at the first epoch in the window where the estimated stage is light
REACTIVE_STAGES = ("W", "N1")
REACTIVE_THRESHOLD = 0.5

# Online sleep-onset detection: first of ONSET_RUN consecutive epochs with P(asleep) > 0.5.
# 6 epochs (3 min): with the simulated stager, median error 0.5 min on held-out nights
# (shorter runs fire on stager W/N1 confusions at bedtime).
ONSET_RUN = 6
# Tracking starts at bedtime, so the first epoch's prior is mostly wake
INITIAL_WAKE_PROB = 0.9

# Simulated EEG stager (replay). Log-likelihoods centred on SIM_CORRECT for the true stage
# plus typical confusions, with AR(1) noise so errors come in runs, as a real stager's do.
# Defaults give ~77% epoch accuracy at ~78% mean confidence.
SIM_CORRECT = 0.80
SIM_SIGMA = 1.5                            # noise on the log-likelihoods
SIM_RHO = 0.8                              # epoch-to-epoch noise correlation
SIM_DROPOUT = 0.02                         # chance an epoch is missing
# Typical stager confusions: where the 1 - SIM_CORRECT mass goes, per true stage
SIM_CONFUSION = {
    "W":   {"N1": 0.7, "N2": 0.1, "REM": 0.2},
    "N1":  {"W": 0.4, "N2": 0.4, "REM": 0.2},
    "N2":  {"N1": 0.3, "N3": 0.5, "REM": 0.2},
    "N3":  {"N2": 0.9, "N1": 0.1},
    "REM": {"N1": 0.5, "N2": 0.3, "W": 0.2},
}

# Replay clock: the trimmed recording starts at this time of night (recordings start
# 30 min before sleep onset). The default window ends 7.5 h after onset and lasts 30 min.
REPLAY_START = "22:30"
DEFAULT_WINDOW_MIN = 30
DEFAULT_LATEST_AFTER_ONSET_H = 7.5

SESSIONS_DIR = "data/alarm_sessions"
