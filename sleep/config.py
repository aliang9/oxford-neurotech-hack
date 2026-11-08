"""Settings for the sleep-stage forecasting models. Most can be overridden on the command line."""

# Stages, in model index order. Original R&K stages 3 and 4 are merged into N3.
STAGES = ["W", "N1", "N2", "N3", "REM"]
W, N1, N2, N3, REM = range(5)
K = len(STAGES)
UNSCORED = -1                  # movement time, "?" and gaps: missing, never a transition
LABEL_MAP = {
    "Sleep stage W": W,
    "Sleep stage 1": N1,
    "Sleep stage 2": N2,
    "Sleep stage 3": N3,
    "Sleep stage 4": N3,
    "Sleep stage R": REM,
    "Sleep stage ?": UNSCORED,
    "Movement time": UNSCORED,
}
EPOCH_S = 30.0
EPOCHS_PER_MIN = 2

# Sleep-EDF Expanded, sleep-cassette (healthy, unmedicated) subset
BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/"
SUBSET = "sleep-cassette"
RAW_DIR = "data/sleep-edf/hypnograms"   # downloaded *-Hypnogram.edf files
EPOCHS_CSV = "data/sleep-edf/epochs.csv"  # one row per trimmed 30 s epoch

# Trimming: keep from TRIM_WAKE_MIN before the first sleep epoch (any non-W stage)
# to TRIM_WAKE_MIN after the last one. The cassette recordings run ~20 h, mostly wake.
TRIM_WAKE_MIN = 30

# Time since sleep onset, in hours. Bin 0 is "not asleep yet" (before the first sleep
# epoch); bins 1.. are [0, 2), [2, 4), [4, 6), [6, inf) hours after onset.
TIME_EDGES_H = [0.0, 2.0, 4.0, 6.0]
TIME_LABELS = ["pre-onset", "0-2 h", "2-4 h", "4-6 h", "6 h+"]

# Elapsed time in the current stage (epochs). The HSMM tracks d = 1..MAX_DUR, where
# d = MAX_DUR means "MAX_DUR or longer" (the hazard is constant from there on).
MAX_DUR = 120                  # 60 min
# Hazard is estimated per duration bin (constant inside a bin); bins widen with d.
# Left edges, in epochs; the last bin runs to MAX_DUR and beyond.
DUR_EDGES = [1, 2, 3, 4, 5, 6, 8, 10, 13, 17, 21, 26, 31, 41, 51, 61, 81, 101]
# Destination of a stage exit is conditioned on a coarse elapsed-duration bin:
# [1, 2], [3, 10], [11, inf) epochs  (short arousal-like runs vs. settled bouts)
DEST_EDGES = [1, 3, 11]

# Regularisation (pseudo-counts)
LAPLACE = 1.0                  # Dirichlet prior on each transition row (stationary HMM)
KAPPA_TIME = 30.0              # time-bin estimate shrunk toward the stationary one
KAPPA_DUR = 20.0               # duration-bin hazard shrunk toward the stage's mean hazard
KAPPA_DEST = 20.0              # duration-bin destination shrunk toward the stage's mean
HAZARD_CLIP = (1e-4, 1 - 1e-4)

# Evaluation
HORIZONS_MIN = [5, 10, 15, 30]
TEST_FRACTION = 0.2            # of subjects
SEED = 0
MODELS_DIR = "models/sleep"
