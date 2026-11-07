"""Everything about the dance experiment you might want to change.

To add or remove a move, edit MOVES. Codes must be unique, not 99, and
should stay fixed once you have recorded data with them. Never reuse a
retired code: old recordings still contain it.
"""

# marker code: (name, cue text shown on screen)
# Retired codes: blocks recorded before still contain them, and those
# trials are skipped. 11 step_touch (confused with spin), 12 rock_step
# (confused with spin and step_touch), 14 shimmy (confused with the arm moves).
MOVES = {
    13: ("spin", "SPIN\npivot turn"),
    15: ("arm_pumps", "ARM PUMPS"),
    16: ("bounce", "BOUNCE\nsmall hops"),
    17: ("rest", "STAND STILL"),
    18: ("robot", "THE ROBOT\nstiff, jerky arms"),
    19: ("floss", "THE FLOSS\nswing arms hip to hip"),
    20: ("head_bob", "HEAD BOB\nnod to the beat"),
}
TRIAL_END = 99  # marker sent when a move stops

# Device and streams (see acquire.py)
FS = 250
CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "POz", "PO8"]
N_EEG = len(CHANNELS)
MARKER_CH = N_EEG          # column 8 of the raw stream holds the markers
RAW_STREAM = "core8_raw"   # unfiltered EEG + markers: what we record and decode
QUALITY_STREAM = "core8_eeg"  # 1-40 Hz filtered: used only for the signal check
MARKER_PORT = 5005

# Windows: identical for training and live decoding
WIN_S, STEP_S = 1.5, 0.25
WIN = int(WIN_S * FS)              # 375 samples
STEP = int(round(STEP_S * FS))     # 62 samples
ONSET_SKIP_S = 0.5                 # ignore the reaction time after each cue
MIN_PURITY = 1.0                  # training window: >= this fraction must be one move

# Online decoding
SMOOTH_ALPHA = 0.4   # weight of the newest probabilities in the moving average
THRESHOLD = 0.5      # smoothed probability needed to switch move
MIN_DUR_S = 0.75     # a new move must win for this long before we switch
