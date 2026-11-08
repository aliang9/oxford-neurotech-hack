"""Download Sleep-EDF hypnograms and turn them into trimmed 30 s epoch sequences.

    python -m sleep.data            # download (skips files already there) + preprocess

Only the *-Hypnogram.edf annotation files are fetched (~5 kB each), never the PSG signals.

Preprocessing, per recording:
1. Annotations (MNE) -> one label per 30 s epoch. W, 1, 2, 3+4 -> N3, R; "?", movement
   time and any stretch with no annotation -> UNSCORED (-1). Unscored epochs are kept in
   place as gaps, so the models never see a "transition" across them.
2. Trim: keep from TRIM_WAKE_MIN (30 min) before the first sleep epoch (first non-W
   scored epoch) to 30 min after the last sleep epoch. Recordings with no sleep are dropped.
3. Write EPOCHS_CSV: recording, subject, night, epoch (index in the trimmed record),
   stage (-1..4), onset_epoch (index of the first sleep epoch in the trimmed record).
"""

import argparse
import concurrent.futures as cf
import csv
import os
import re
import sys
import urllib.request
from dataclasses import dataclass

import numpy as np

from . import config as cfg


@dataclass
class Recording:
    rec_id: str          # e.g. "SC4001"
    subject: int         # e.g. 0
    night: int           # 1 or 2
    stages: np.ndarray   # int8 per 30 s epoch, UNSCORED = -1
    onset: int           # index of the first sleep epoch

    def __len__(self):
        return len(self.stages)


# --- download ---------------------------------------------------------------------------

def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "sleep-forecast/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_hypnograms():
    """File names of every hypnogram in the subset, from the PhysioNet directory listing."""
    html = _get(cfg.BASE_URL + cfg.SUBSET + "/").decode("utf-8", "replace")
    names = sorted(set(re.findall(r"(SC4\d{3}[A-Z0-9]{2}-Hypnogram\.edf)", html)))
    if not names:
        raise RuntimeError("no hypnogram files found in the PhysioNet listing")
    return names


def download(raw_dir=cfg.RAW_DIR, workers=8, retries=3):
    os.makedirs(raw_dir, exist_ok=True)
    names = list_hypnograms()
    todo = [n for n in names if not os.path.exists(os.path.join(raw_dir, n))]
    print(f"{len(names)} hypnograms listed, {len(names) - len(todo)} already downloaded")

    def fetch(name):
        for attempt in range(retries):
            try:
                data = _get(cfg.BASE_URL + cfg.SUBSET + "/" + name)
                tmp = os.path.join(raw_dir, name + ".part")
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, os.path.join(raw_dir, name))
                return None
            except Exception as e:  # noqa: BLE001 - report and retry any network error
                err = e
        return f"{name}: {err}"

    failed = []
    with cf.ThreadPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(fetch, todo), 1):
            if res:
                failed.append(res)
            if i % 25 == 0 or i == len(todo):
                print(f"  downloaded {i}/{len(todo)}")
    for f in failed:
        print("  FAILED", f, file=sys.stderr)
    return len(names) - len(failed)


# --- parsing ----------------------------------------------------------------------------

def parse_hypnogram(path):
    """Per-epoch stage labels (int8, UNSCORED for anything missing) from one hypnogram."""
    import mne  # imported here so the models can be used without MNE installed

    ann = mne.read_annotations(path)
    n = int(round(max(o + d for o, d in zip(ann.onset, ann.duration)) / cfg.EPOCH_S))
    stages = np.full(n, cfg.UNSCORED, np.int8)
    unknown = set()
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        label = cfg.LABEL_MAP.get(desc)
        if label is None:
            unknown.add(desc)
            continue
        a = int(round(onset / cfg.EPOCH_S))
        b = int(round((onset + dur) / cfg.EPOCH_S))
        stages[a:b] = label
    if unknown:
        print(f"  {os.path.basename(path)}: ignored annotations {sorted(unknown)}")
    return stages


def trim(stages, pad=cfg.TRIM_WAKE_MIN * cfg.EPOCHS_PER_MIN):
    """Trimmed stages and the onset index in them, or None if there is no sleep."""
    sleep = np.flatnonzero(stages > cfg.W)
    if len(sleep) == 0:
        return None
    a = max(0, sleep[0] - pad)
    b = min(len(stages), sleep[-1] + 1 + pad)
    return stages[a:b].copy(), int(sleep[0] - a)


def rec_info(name):
    """('SC4001', subject 0, night 1) from 'SC4001EC-Hypnogram.edf'."""
    m = re.match(r"SC4(\d\d)(\d)", name)
    return name[:6], int(m.group(1)), int(m.group(2))


def preprocess(raw_dir=cfg.RAW_DIR, out=cfg.EPOCHS_CSV):
    names = sorted(n for n in os.listdir(raw_dir) if n.endswith("-Hypnogram.edf"))
    recs = []
    for name in names:
        rec_id, subj, night = rec_info(name)
        res = trim(parse_hypnogram(os.path.join(raw_dir, name)))
        if res is None:
            print(f"  {rec_id}: no sleep, dropped")
            continue
        recs.append(Recording(rec_id, subj, night, *res))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recording", "subject", "night", "epoch", "stage", "onset_epoch"])
        for r in recs:
            for i, s in enumerate(r.stages):
                w.writerow([r.rec_id, r.subject, r.night, i, int(s), r.onset])
    summarize(recs)
    print(f"wrote {out}")
    return recs


def load(path=cfg.EPOCHS_CSV):
    """List of Recording from EPOCHS_CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: run `python -m sleep.data` first")
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.setdefault(row["recording"], []).append(row)
    recs = []
    for rec_id, rs in rows.items():
        rs.sort(key=lambda r: int(r["epoch"]))
        recs.append(Recording(rec_id, int(rs[0]["subject"]), int(rs[0]["night"]),
                              np.array([int(r["stage"]) for r in rs], np.int8),
                              int(rs[0]["onset_epoch"])))
    return recs


def summarize(recs):
    allst = np.concatenate([r.stages for r in recs])
    scored = allst[allst >= 0]
    print(f"{len(recs)} recordings, {len({r.subject for r in recs})} subjects, "
          f"{len(allst)} epochs ({len(allst) / 120:.0f} h), "
          f"{np.mean(allst < 0):.2%} unscored")
    frac = np.bincount(scored, minlength=cfg.K) / len(scored)
    print("  stage mix:", "  ".join(f"{s} {p:.1%}" for s, p in zip(cfg.STAGES, frac)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", default=cfg.RAW_DIR)
    ap.add_argument("--out", default=cfg.EPOCHS_CSV)
    ap.add_argument("--no-download", action="store_true", help="only preprocess local files")
    args = ap.parse_args(argv)
    if not args.no_download:
        try:
            download(args.raw_dir)
        except Exception as e:  # noqa: BLE001
            print(f"download failed ({e}); preprocessing whatever is already in {args.raw_dir}",
                  file=sys.stderr)
    preprocess(args.raw_dir, args.out)


if __name__ == "__main__":
    main()
