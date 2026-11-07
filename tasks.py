"""Cue a participant through randomised task trials, sending markers to acquire.py.

Each trial:  fixation cross (rest, jittered) -> task cue for --duration s.
At cue onset the task's code is sent as UDP text to acquire.py's
UDPReceiver. gpype stamps it on its own timeline and records it as a
trigger channel next to the EEG. A schedule CSV is also written.

Keys: Space = start / continue after a break, Esc = abort.
"""

import argparse
import csv
import os
import random
import socket
import sys
import time

from PySide6 import QtCore, QtGui, QtWidgets

# code: (name, instruction). Codes are stable; decode.py reads them from here.
TASKS = {
    1: ("rest_open", "Relax with your eyes open.\nLook at the cross."),
    2: ("rest_closed", "Close your eyes and relax.\nOpen them at the beep."),
    3: ("left_hand", "Imagine squeezing your LEFT hand.\nDon't move."),
    4: ("right_hand", "Imagine squeezing your RIGHT hand.\nDon't move."),
    5: ("feet", "Imagine wiggling your toes.\nDon't move."),
    6: ("math", "Count down from {n} in steps of 7."),
    7: ("words", "Silently list words starting with '{letter}'."),
}
NAME_TO_CODE = {name: code for code, (name, _) in TASKS.items()}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", nargs="+", default=["rest_open", "rest_closed", "math"],
                    choices=list(NAME_TO_CODE), help="tasks to run (default: rest_open rest_closed math)")
    ap.add_argument("--trials", type=int, default=15, help="trials per task (default: 15)")
    ap.add_argument("--duration", type=float, default=5.0, help="task period, s (default: 5)")
    ap.add_argument("--rest", type=float, nargs=2, default=(2.0, 3.0), metavar=("MIN", "MAX"),
                    help="fixation period between trials, s (default: 2 3)")
    ap.add_argument("--break-every", type=int, default=20,
                    help="pause for a break every N trials (default: 20)")
    ap.add_argument("--port", type=int, default=5005, help="acquire.py's marker port")
    ap.add_argument("--seed", type=int, default=None, help="random seed for the trial order")
    ap.add_argument("--log", default=None, help="schedule CSV (default: tasks_<time>.csv)")
    ap.add_argument("--windowed", action="store_true", help="don't go full screen")
    ap.add_argument("--autostart", action="store_true", help="start without waiting for Space")
    return ap.parse_args(argv)


def make_schedule(tasks, trials, seed=None):
    """Balanced, shuffled list of task codes, with no task more than twice in a row."""
    rng = random.Random(seed)
    codes = [NAME_TO_CODE[t] for t in tasks] * trials
    while True:
        rng.shuffle(codes)
        if all(not (codes[i] == codes[i + 1] == codes[i + 2]) for i in range(len(codes) - 2)):
            return codes


class Cues(QtWidgets.QLabel):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.rng = random.Random(args.seed)
        self.schedule = make_schedule(args.tasks, args.trials, args.seed)
        self.i = 0
        self.waiting = True  # for Space
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        log = args.log or time.strftime("tasks_%Y%m%d_%H%M%S.csv")
        os.makedirs(os.path.dirname(log) or ".", exist_ok=True)
        self.log = open(log, "w", newline="")
        self.writer = csv.writer(self.log)
        self.writer.writerow(["trial", "unix_time", "code", "task", "cue"])
        self.done = False
        self.last_break = 0

        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setWordWrap(True)
        self.setFont(QtGui.QFont("Helvetica", 44))
        self.setStyleSheet("background: black; color: white;")
        n = len(self.schedule)
        mins = n * (args.duration + sum(args.rest) / 2) / 60
        self.show_text(f"{n} trials, about {mins:.0f} min.\n\nSit still, relax your jaw,\n"
                       "and keep blinks for the cross.\n\nPress Space to start.")
        if args.autostart:
            self.waiting = False
            self.after(1.0, self.fixation)

    def show_text(self, text):
        self.setText(text)
        self.repaint()  # draw now, so the marker goes out with the cue

    def send(self, code):
        self.sock.sendto(str(code).encode(), ("127.0.0.1", self.args.port))

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape:
            self.finish("Aborted.")
        elif e.key() == QtCore.Qt.Key_Space and self.waiting:
            self.waiting = False
            self.fixation()

    def fixation(self):
        if self.i >= len(self.schedule):
            return self.finish("Done, thank you!")
        if self.i and self.i % self.args.break_every == 0 and self.last_break != self.i:
            self.waiting, self.last_break = True, self.i
            return self.show_text(f"Break ({self.i}/{len(self.schedule)} done).\n\n"
                                  "Press Space to continue.")
        self.show_text("+")
        self.after(self.rng.uniform(*self.args.rest), self.cue)

    def after(self, seconds, fn):
        t = QtCore.QTimer(self, singleShot=True, timerType=QtCore.Qt.PreciseTimer)
        t.timeout.connect(lambda: None if self.done else fn())
        t.timeout.connect(t.deleteLater)
        t.start(int(1000 * seconds))

    def cue(self):
        code = self.schedule[self.i]
        name, text = TASKS[code]
        text = text.format(n=self.rng.randint(300, 999), letter=self.rng.choice("BCDFGHLMPRST"))
        self.show_text(text)
        self.send(code)
        QtWidgets.QApplication.beep()
        self.writer.writerow([self.i + 1, f"{time.time():.3f}", code, name, text.replace("\n", " ")])
        self.i += 1
        self.after(self.args.duration, self.end_trial)

    def end_trial(self):
        QtWidgets.QApplication.beep()
        self.fixation()

    def finish(self, text):
        if self.done:
            return
        self.done = True
        self.log.close()
        self.show_text(text)
        print(f"{text} {self.i} trials; schedule written to {self.log.name}")
        QtCore.QTimer.singleShot(1500, QtWidgets.QApplication.quit)


def main(argv=None):
    args = parse_args(argv)
    app = QtWidgets.QApplication(sys.argv[:1])
    w = Cues(args)
    w.resize(1000, 700)
    w.show() if args.windowed else w.showFullScreen()
    app.exec()


if __name__ == "__main__":
    main()
