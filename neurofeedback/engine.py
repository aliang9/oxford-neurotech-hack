"""Samples in -> feature, baseline, reward and time series out. Runs on its own threads.

    acquisition thread:  source.pull() -> queue
    processing thread:   queue -> Engine.process() (filter, every 0.5 s: feature + reward)
    GUI thread:          Engine.snapshot() / commands, under a lock
"""

import csv
import queue
import threading
import time

import numpy as np

from . import config as cfg
from .baseline import BaselineRecorder
from .features import (Preprocessor, channel_problems, log_alpha_power, select_channels,
                       stream_problem)
from .reward import EMA, RewardAccumulator

WAITING, NEEDS_BASELINE, CALIBRATING, READY, RECORDING, PAUSED, FINISHED = (
    "waiting for data", "needs baseline", "calibrating", "ready", "recording", "paused", "finished")
TIMED, INDEFINITE = "timed", "indefinite"
POOR = "poor signal"

COLUMNS = ["time_s", "session_s", "stream_time", "state", "valid", "reason", "channels_used", "z_alpha",
           "z_smooth", "reward_rate", "cumulative_reward", "threshold"]


class Engine:
    def __init__(self, fs, labels, channels=cfg.CHANNELS, threshold=cfg.THRESHOLD,
                 baseline_s=cfg.BASELINE_S, tau_s=cfg.EMA_TAU_S, store=None):
        self.fs = fs
        self.idx, self.channels, self.warnings = select_channels(labels, channels)
        self.win = int(round(cfg.WIN_S * fs))
        self.step = int(round(cfg.STEP_S * fs))
        self.pre = Preprocessor(fs, len(self.idx))
        self.buf = np.empty((0, len(self.idx)))
        self.tbuf = np.empty(0)
        self.mbuf = np.empty((0, len(self.idx)), bool)
        self.count = 0                  # samples since the last update
        self.include = np.ones(len(self.idx), bool)  # channels not excluded by hand

        self.lock = threading.Lock()
        self.baseline_s = baseline_s
        self.state = WAITING
        self.baseline = None
        self.calibration = None
        self.ema = EMA(tau_s)
        self.reward = RewardAccumulator(threshold)
        self.t_prev = None              # stream time of the previous update
        self.t0 = None                  # stream time at session start (first update / reset)
        self.history = []
        self.last = {}                  # newest update, for the readouts
        self.message = "; ".join(self.warnings)
        self.source_done = False
        self.store = store              # history.SessionStore, or None to save nothing
        self.mode, self.minutes = INDEFINITE, cfg.DEFAULT_MINUTES  # chosen before starting
        self.session = None             # the running (or just finished) session
        self.result = None              # the last finished session's saved row

    # ---- processing thread -------------------------------------------------
    def process(self, samples, times):
        y, missing = self.pre.process(samples[:, self.idx])
        pos = 0
        while pos < len(y):
            take = min(len(y) - pos, self.step - self.count)
            sl = slice(pos, pos + take)
            self.buf = np.concatenate([self.buf, y[sl]])[-self.win:]
            self.tbuf = np.concatenate([self.tbuf, times[sl]])[-self.win:]
            self.mbuf = np.concatenate([self.mbuf, missing[sl]])[-self.win:]
            pos += take
            self.count += take
            if self.count == self.step:
                self.count = 0
                if len(self.buf) == self.win:
                    self._update()

    def _update(self):
        t = float(self.tbuf[-1])
        stream = stream_problem(self.tbuf)
        problems = channel_problems(self.buf, self.fs, self.mbuf)
        if stream:
            problems = [stream] * len(problems)
        clean = np.array([not p for p in problems])
        log_alpha = log_alpha_power(self.buf, self.fs)  # per channel
        with self.lock:
            dt = cfg.STEP_S if self.t_prev is None else min(max(t - self.t_prev, 0.0), cfg.MAX_DT_S)
            self.t_prev = t
            if self.t0 is None:
                self.t0 = t
            if self.state == WAITING:
                self.state = NEEDS_BASELINE

            if self.state == CALIBRATING:
                # every clean channel, excluded or not, so it can be re-included later
                self.calibration.add(np.where(clean, log_alpha, np.nan), dt)
                if self.calibration.done:
                    baseline, self.message = self.calibration.finish(self.channels)
                    if baseline is not None:
                        self.baseline = baseline
                        self.ema.reset()
                    self.calibration = None
                    self.state = READY if self.baseline else NEEDS_BASELINE

            usable = clean & self.include
            if self.baseline is not None:
                usable &= self.baseline.available
            valid = usable.sum() >= cfg.MIN_CHANNELS
            z_ch = np.full(len(self.channels), np.nan)
            z = np.nan
            if self.baseline is not None and self.state != CALIBRATING:
                z_ch = self.baseline.z(log_alpha)
                if valid:
                    z = float(np.mean(z_ch[usable]))
                    self.ema.update(z, dt)
            smooth = self.ema.value if self.ema.value is not None else np.nan
            rate = self.reward.step(self.ema.value, dt, active=valid and self.state == RECORDING)
            if self.state == RECORDING:
                # pauses stop the clock; poor-signal windows don't (they just earn nothing)
                self.session["elapsed"] += dt
                if self.session["mode"] == TIMED and \
                        self.session["elapsed"] >= self.session["minutes"] * 60 - 1e-6:
                    self._finish("timer")

            reason = "; ".join(f"{name}: {p}" for name, p, inc in
                               zip(self.channels, problems, self.include) if p and inc)
            if not self.include.any():
                reason = "every channel excluded"
            elif self.baseline is not None and not (self.include & self.baseline.available).any():
                reason = "no included channel has a baseline"
            row = dict(time_s=t - self.t0, stream_time=t, state=self.state if valid else POOR,
                       session_s=self.session["elapsed"] if self.session else np.nan,
                       valid=valid, reason=reason,
                       channels_used=" ".join(np.asarray(self.channels)[usable]),
                       problems=problems, z_alpha=z, z_smooth=smooth, reward_rate=rate,
                       cumulative_reward=self.reward.total, threshold=self.reward.threshold)
            for c, name in enumerate(self.channels):
                row[f"log_alpha_{name}"] = float(log_alpha[c])
                row[f"z_{name}"] = float(z_ch[c])
            self.history.append(row)
            self.last = row

    # ---- commands (GUI thread) ---------------------------------------------
    def start_calibration(self):
        with self.lock:
            if self.state in (NEEDS_BASELINE, READY, FINISHED):
                self.calibration = BaselineRecorder(len(self.channels), self.baseline_s)
                self.state = CALIBRATING
                self.message = f"Calibrating: eyes open, resting, {self.baseline_s:g} s"

    def set_session(self, mode, minutes=None):
        """Choose the next session: TIMED with `minutes`, or INDEFINITE."""
        with self.lock:
            self.mode = mode
            if minutes:
                self.minutes = float(minutes)

    def start_session(self):
        """A new session from zero, in the chosen mode. Needs a baseline."""
        with self.lock:
            if self.state not in (READY, FINISHED):
                return
            self.reward.reset()
            self.history = []
            self.t0 = None
            best = self.store.best(self.mode) if self.store else None
            self.session = dict(mode=self.mode, minutes=self.minutes if self.mode == TIMED else None,
                                elapsed=0.0, best_before=best)
            self.result = None
            self.state = RECORDING

    def toggle_feedback(self):
        """Start (when ready), pause or resume."""
        if self.state in (READY, FINISHED):
            return self.start_session()
        with self.lock:
            if self.state == RECORDING:
                self.state = PAUSED
            elif self.state == PAUSED:
                self.state = RECORDING

    def stop_session(self):
        with self.lock:
            if self.state in (RECORDING, PAUSED):
                self._finish("participant")

    def _finish(self, ended_by):
        """End the session and save it (caller holds the lock)."""
        s = self.session
        self.state = FINISHED
        self.result = dict(ended_at=time.strftime("%Y-%m-%d %H:%M:%S"), mode=s["mode"],
                           planned_min=s["minutes"] or "", score=round(self.reward.total, 3),
                           active_duration_s=round(s["elapsed"], 1),
                           valid_duration_s=round(self.reward.active_time, 1), ended_by=ended_by,
                           threshold=self.reward.threshold, channels=" ".join(self.channels))
        new_best = self.store.save(self.result) if self.store else False
        self.result["personal_best"] = new_best
        self.message = (f"Session saved: {self.result['score']:.1f} points in "
                        f"{s['elapsed'] / 60:.1f} min ({s['mode']})" + (", new personal best" if new_best else ""))

    def reset(self):
        """Zero the reward and clear the time series (baseline kept). Not during a session:
        the score never goes down while one is running."""
        with self.lock:
            if self.state in (RECORDING, PAUSED):
                return
            self.reward.reset()
            self.history = []
            self.t0 = None

    def set_included(self, channel, included):
        """Exclude a channel by hand (or put it back); its baseline is kept either way."""
        with self.lock:
            self.include[channel] = bool(included)

    def set_threshold(self, value):
        with self.lock:
            self.reward.threshold = float(value)

    def snapshot(self):
        with self.lock:
            hist = list(self.history)
            calib = self.calibration
            snap = dict(state=self.state, last=dict(self.last), total=self.reward.total,
                        feedback_time=self.reward.active_time, threshold=self.reward.threshold,
                        baseline=self.baseline, message=self.message, done=self.source_done,
                        session=dict(self.session) if self.session else None,
                        result=dict(self.result) if self.result else None,
                        mode=self.mode, minutes=self.minutes,
                        calib_progress=(calib.elapsed, calib.duration) if calib else None)
        for key in ("time_s", "session_s", "z_alpha", "z_smooth", "cumulative_reward", "valid"):
            snap[key] = np.array([r[key] for r in hist], float)
        return snap

    def export_csv(self, path):
        with self.lock:
            hist = list(self.history)
            b = self.baseline
        per_channel = [f"{k}_{name}" for name in self.channels for k in ("log_alpha", "z")]
        base = []
        for c, name in enumerate(self.channels):
            base += [b.mean[c] if b else "", b.std[c] if b else ""]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS + per_channel +
                       [f"baseline_{k}_{name}" for name in self.channels for k in ("mean", "std")])
            for r in hist:
                w.writerow([r[c] for c in COLUMNS + per_channel] + base)
        return len(hist)


class Worker:
    """Acquisition and processing on two daemon threads, so neither blocks the other or the GUI."""

    def __init__(self, source, engine):
        self.source, self.engine = source, engine
        self.q = queue.Queue()
        self.stop_event = threading.Event()
        self.threads = [threading.Thread(target=self._acquire, daemon=True),
                        threading.Thread(target=self._process, daemon=True)]

    def start(self):
        for t in self.threads:
            t.start()

    def stop(self):
        self.stop_event.set()

    def _acquire(self):
        while not self.stop_event.is_set():
            samples, times = self.source.pull(timeout=0.05)
            if len(samples):
                self.q.put((samples, times))
            elif self.source.done:
                break
        self.q.put(None)

    def _process(self):
        while not self.stop_event.is_set():
            try:
                item = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            self.engine.process(*item)
        self.engine.source_done = True
