"""Read an EEG stream from LSL and print per-channel mean/std every second.

Standalone: needs only pylsl and numpy, not gpype.
"""

import argparse
import csv
import sys
import time

import numpy as np
from pylsl import StreamInlet, resolve_byprop


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stream-name", default="core8_eeg",
                    help="LSL stream name to look for (default: core8_eeg)")
    ap.add_argument("--timeout", type=float, default=10.0,
                    help="seconds to search for the stream (default: 10)")
    ap.add_argument("--save", metavar="FILE.csv", default=None,
                    help="also write every sample to CSV with its LSL timestamp")
    ap.add_argument("--duration", type=float, default=None,
                    help="stop after this many seconds (default: run until Ctrl-C)")
    return ap.parse_args(argv)


def channel_labels(info):
    """Labels from the stream's desc/channels, or ch1..chN if absent."""
    labels = []
    ch = info.desc().child("channels").child("channel")
    while not ch.empty():
        labels.append(ch.child_value("label"))
        ch = ch.next_sibling()
    n = info.channel_count()
    if len(labels) != n or not all(labels):
        labels = [f"ch{i + 1}" for i in range(n)]
    return labels


def main(argv=None):
    args = parse_args(argv)

    print(f"Looking for LSL stream '{args.stream_name}' ...")
    streams = resolve_byprop("name", args.stream_name, timeout=args.timeout)
    if not streams:
        sys.exit(f"No LSL stream named '{args.stream_name}' found "
                 f"within {args.timeout:g} s.")

    inlet = StreamInlet(streams[0])
    info = inlet.info()
    labels = channel_labels(info)
    print(f"Connected: {info.name()} ({info.type()}), "
          f"{info.channel_count()} channels @ {info.nominal_srate():g} Hz")
    print("Channels:", ", ".join(labels))

    writer = out = None
    if args.save:
        out = open(args.save, "w", newline="")
        writer = csv.writer(out)
        writer.writerow(["lsl_timestamp", *labels])
        print(f"Saving to {args.save}")

    buffer = []
    t_start = t_last = time.monotonic()
    try:
        while args.duration is None or time.monotonic() - t_start < args.duration:
            samples, stamps = inlet.pull_chunk(timeout=0.2)
            if samples:
                buffer.extend(samples)
                if writer:
                    writer.writerows([t, *s] for t, s in zip(stamps, samples))

            if time.monotonic() - t_last >= 1.0:
                t_last = time.monotonic()
                if buffer:
                    x = np.asarray(buffer)
                    stats = "  ".join(f"{lab}={m:+8.2f}±{s:6.2f}"
                                      for lab, m, s in zip(labels, x.mean(0), x.std(0)))
                    print(f"[{len(x):4d} samples] {stats}")
                else:
                    print("[   0 samples] no data in the last second")
                buffer.clear()
    except KeyboardInterrupt:
        pass
    finally:
        if out:
            out.close()
        inlet.close_stream()


if __name__ == "__main__":
    main()
