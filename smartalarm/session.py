"""One night: source -> online HSMM -> policy -> alarm, with a model-independent deadline.

    s = Session(source, model, earliest, latest)      # epochs since the source's start
    s.advance(n)            # process up to n new epochs; returns True once the alarm fired

The hard deadline is checked on the epoch counter before anything model-related, and every
model call is wrapped, so a model error or a dead stream cannot stop the alarm at `latest`
(a stalled live stream still advances the clock: PushSource emits missing epochs).
Everything is logged as JSON lines: settings, every epoch's observation, posterior and
decision, and the outcome.
"""

import datetime as dt
import json
import os
import traceback

import numpy as np

from . import config as cfg
from .inference import OnlineHSMM
from .policy import BayesPolicy, FixedPolicy, ReactivePolicy, explain


def make_policy(name, model, stage_cost=None, sleep_value=cfg.SLEEP_VALUE_PER_MIN):
    if name == "bayes":
        return BayesPolicy(model, stage_cost, sleep_value)
    if name == "reactive":
        return ReactivePolicy()
    return FixedPolicy()


class Session:
    def __init__(self, source, model, earliest, latest, policy="bayes", stage_cost=None,
                 sleep_value=cfg.SLEEP_VALUE_PER_MIN, start_clock=None, log_dir=cfg.SESSIONS_DIR,
                 log=True):
        self.source, self.model = source, model
        self.earliest, self.latest = int(earliest), int(latest)
        self.policy_name = policy
        self.stage_cost, self.sleep_value = stage_cost, sleep_value
        self.start_clock = start_clock or dt.datetime.combine(
            dt.date.today(), dt.time.fromisoformat(cfg.REPLAY_START))
        self.log_dir, self.logging = log_dir, log
        self.log_file = None
        self.reset()

    # --- lifecycle --------------------------------------------------------------------
    def reset(self):
        self.inf = OnlineHSMM(self.model, getattr(self.source, "class_prior", None))
        self.policy = make_policy(self.policy_name, self.model, self.stage_cost,
                                  self.sleep_value)
        self.fired = None          # dict once the alarm went off
        self.decision = {}
        self.text = "Waiting for the first observation."
        self.error = None
        if hasattr(self.source, "seek"):
            self.source.seek(0)
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def seek(self, epoch):
        """Jump to `epoch` (replay): rebuild the filter causally from the start of the night.
        Decisions are re-made along the way, so an alarm can fire during the catch-up."""
        self.reset()
        self.advance(max(int(epoch), 0), log=False)

    def clock(self, epoch):
        return self.start_clock + dt.timedelta(seconds=epoch * cfg.EPOCH_S)

    @property
    def now(self):
        return max(self.inf.t, 0)

    # --- processing -------------------------------------------------------------------
    def advance(self, n=1, log=True):
        if self.fired:
            return True
        for obs in self.source.pull(n):
            self._epoch(obs, log)
            if self.fired:
                break
        return bool(self.fired)

    def _epoch(self, obs, log):
        t = obs.epoch
        # 1. hard deadline, independent of the model
        if t >= self.latest:
            self._fire(t, "deadline", log)
            return
        # 2. Bayesian update + policy, guarded
        try:
            self.inf.observe(t, obs.probs)
            self.decision = self.policy.decide(self.inf, self.earliest, self.latest)
            post = self.inf.posterior()
            self.text = explain(self.decision, post, t, self.earliest, self.latest)
            if obs.probs is None:
                self.text = "No signal for this epoch (missing); predicting from the model. " \
                            + self.text
            self.error = None
        except Exception as e:  # noqa: BLE001 - the deadline must survive any model failure
            err = f"{type(e).__name__}: {e}"
            if err != self.error:
                traceback.print_exc()
            self.error = err
            self.decision = {"wake": False}
            self.text = "Model error; the deadline alarm is still set."
        if log:
            self._log(dict(kind="epoch", epoch=t, clock=self.clock(t).strftime("%H:%M:%S"),
                           obs=None if obs.probs is None else np.round(obs.probs, 4).tolist(),
                           posterior=np.round(self.inf.posterior(), 4).tolist()
                           if self.inf.alpha is not None else None,
                           onset=self.inf.onset, error=self.error,
                           **{k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in self.decision.items()}))
        if self.decision.get("wake"):
            self._fire(t, "policy", log)

    def _fire(self, t, reason, log=True):
        self.fired = dict(epoch=t, clock=self.clock(t).strftime("%H:%M:%S"), reason=reason,
                          minutes_before_latest=(self.latest - t) / 2.0)
        if reason == "deadline":
            self.text = "Latest wake-up time reached: alarm."
        if log:
            self._log(dict(kind="alarm", **self.fired, outcome=self.outcome()))
            if self.log_file:
                self.log_file.close()
                self.log_file = None

    def outcome(self):
        """True stage at the alarm and its cost (replay only: uses the ground truth)."""
        truth = getattr(self.source, "truth", None)
        if truth is None or not self.fired:
            return None
        t = min(self.fired["epoch"], len(truth) - 1)
        s = int(truth[t])
        while s < 0 and t > 0:
            t -= 1
            s = int(truth[t])
        cost = (self.stage_cost or cfg.STAGE_COST)[cfg.STAGES[s]] + self.sleep_value * self.fired[
            "minutes_before_latest"]
        return dict(true_stage=cfg.STAGES[s], realised_cost=round(cost, 3))

    # --- logging ----------------------------------------------------------------------
    def _log(self, row):
        if not self.logging:
            return
        if self.log_file is None:
            os.makedirs(self.log_dir, exist_ok=True)
            name = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            rec = getattr(getattr(self.source, "rec", None), "rec_id", "live")
            self.log_file = open(os.path.join(self.log_dir, f"{name}_{rec}.jsonl"), "w")
            self.log_file.write(json.dumps(dict(
                kind="settings", recording=rec, policy=self.policy_name,
                earliest=self.clock(self.earliest).strftime("%H:%M"),
                latest=self.clock(self.latest).strftime("%H:%M"),
                stage_cost=self.stage_cost or cfg.STAGE_COST, sleep_value=self.sleep_value,
                model=cfg.MODEL_FILE + ":" + cfg.MODEL_NAME)) + "\n")
        self.log_file.write(json.dumps(row) + "\n")
        self.log_file.flush()
