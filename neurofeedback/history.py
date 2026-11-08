"""Completed sessions, one CSV row each; personal bests are read back from it."""

import csv
import os

FIELDS = ["participant", "ended_at", "mode", "planned_min", "score", "active_duration_s",
          "valid_duration_s", "ended_by", "personal_best", "threshold", "channels"]


class SessionStore:
    def __init__(self, path, participant):
        self.path, self.participant = path, participant

    def _rows(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path, newline="") as f:
            return list(csv.DictReader(f))

    def best(self, mode):
        """This participant's best score so far in this mode ('timed' / 'indefinite'), or None."""
        scores = [float(r["score"]) for r in self._rows()
                  if r["participant"] == self.participant and r["mode"] == mode]
        return max(scores) if scores else None

    def save(self, result):
        """Append a session (a dict with FIELDS minus participant/personal_best). Returns True on a new best."""
        previous = self.best(result["mode"])
        new_best = previous is None or result["score"] > previous
        row = dict(result, participant=self.participant, personal_best=int(new_best))
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        fresh = not os.path.exists(self.path)
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if fresh:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in FIELDS})
        return new_best
