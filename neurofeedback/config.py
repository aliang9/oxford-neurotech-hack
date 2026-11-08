"""Defaults for the alpha neurofeedback app. Most can be overridden on the command line."""

from quality import FLAT_UV, MUSCLE_BAD_UV  # same contact/muscle thresholds as calibrate.py

FS = 250.0
MONTAGE = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "POz", "PO8"]  # acquire.py / Core-8 layout
LSL_STREAM = "core8_eeg"       # acquire.py's filtered stream (8 EEG + marker)

# Electrodes whose alpha z-scores are averaged, each against its own baseline.
# A missing channel is replaced by its FALLBACK if that exists, otherwise dropped (with a warning).
CHANNELS = ["PO7", "Oz", "PO8"]
FALLBACK = {"Oz": "POz", "O1": "PO7", "O2": "PO8"}

# Feature
WIN_S = 4.0                    # rolling window for the PSD
STEP_S = 0.5                   # update interval
WELCH_SEG_S = 2.0              # Welch segment length (0.5 Hz resolution, 3 segments at 50% overlap)
ALPHA_BAND = (8.0, 12.0)
FILTER_BAND = (1.0, 40.0)      # causal bandpass applied to the incoming samples
MAINS_HZ = 50.0

# Baseline: eyes open, resting
BASELINE_S = 60.0
MIN_BASELINE_VALID = 0.5       # fraction of baseline updates that must be artifact-free
MIN_BASELINE_STD = 0.05        # log10 units (~12% power); floor against near-zero variance

# Reward
EMA_TAU_S = 3.0
THRESHOLD = 2.0                # dR/dt = max(0, z - THRESHOLD)
MAX_DT_S = 1.0                 # cap on the time credited per update (after stalls or gaps)

# Sessions
HISTORY_FILE = "data/neurofeedback_sessions.csv"  # one row per completed session
DEFAULT_MINUTES = 10           # preset for timed sessions

# Artifact rejection, per selected channel, on the 1-40 Hz filtered 4 s window (uV)
MAX_PTP_UV = 150.0             # movement / electrode pop (same as decode.py)
MUSCLE_UV = MUSCLE_BAD_UV      # 20-40 Hz rms
FLAT_STD_UV = FLAT_UV          # no contact
MAX_GAP_S = 0.1                # timestamp gap = dropped samples
MIN_CHANNELS = 1               # clean electrodes needed for an update to count
