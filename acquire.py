"""Stream a g.tec BCI Core-8 into a live scope, LSL and (optionally) a file.

    BCICore ------+-> Router -+-> HDF5Writer (raw EEG + markers)    [--record]
    UDPReceiver --+  (markers)+-> LSLSender "core8_raw" (unfiltered, for dance/)
                              +-> 50 Hz notch -> 1-40 Hz bandpass -+-> TimeSeriesScope
                                                                   +-> LSLSender "core8_eeg"
    BCICore's accelerometer/gyroscope -> LSL "core8_motion"         (see motion.py)

Task markers are numbers sent as UDP text to --marker-port (tasks.py does
this). gpype stamps them on its own timeline and adds them as a trigger
channel next to the EEG. They are never filtered, and they line up with the
EEG samples both in the file and in the LSL stream.

Wiring follows gpype's examples s3e1 (BCICore -> filters -> scope),
s4e2 (LSLSender), s4e4 (UDPReceiver + Router) and s4e7 (HDF5Writer).
"""

import argparse
import os

import gpype as gp

from motion import MotionBCICore

EEG_CHANNELS = 8  # the marker channel is index 8, after the EEG


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--serial", default=None,
                    help="headset serial, e.g. U8-xxxx.xx.xx (default: first one found)")
    ap.add_argument("--stream-name", default="core8_eeg",
                    help="LSL stream name for filtered EEG + markers (default: core8_eeg)")
    ap.add_argument("--raw-stream", default="core8_raw",
                    help="LSL stream name for unfiltered EEG + markers (default: core8_raw)")
    ap.add_argument("--motion-stream", default="core8_motion",
                    help="LSL stream name for accelerometer/gyroscope (default: core8_motion)")
    ap.add_argument("--mains", type=float, default=50.0,
                    help="mains frequency to notch out, Hz (default: 50 for UK)")
    ap.add_argument("--band", type=float, nargs=2, default=(1.0, 40.0),
                    metavar=("LO", "HI"), help="bandpass edges, Hz (default: 1 40)")
    ap.add_argument("--oscar", action="store_true",
                    help="enable g.tec's OSCAR artifact removal (adds ~0.33 s delay)")
    ap.add_argument("--record", metavar="NAME.h5", default=None,
                    help="record raw EEG + markers to HDF5 (gpype adds a date/time to the name)")
    ap.add_argument("--marker-port", type=int, default=5005,
                    help="UDP port for task markers (default: 5005)")
    ap.add_argument("--simulate", action="store_true",
                    help="use gpype's Generator instead of the headset (for testing)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    app = gp.MainApp(caption=f"BCI Core-8 -> LSL '{args.stream_name}'")
    p = gp.Pipeline()

    if args.simulate:
        # A Generator named like the Core-8 montage, standing in for the headset
        generator = gp.Generator(sampling_rate=250, channel_count=EEG_CHANNELS,
                                 signal_frequency=10, signal_amplitude=10,
                                 noise_amplitude=5)
        source = gp.ChannelLabeler(labels=["Fz", "C3", "Cz", "C4", "Pz", "PO7", "POz", "PO8"])
        p.connect(generator, source)
    else:
        # serial=None takes the first g.tec BCI Core the BLE scan finds.
        # 8 channels at 250 Hz, labelled Fz C3 Cz C4 Pz PO7 POz PO8.
        source = MotionBCICore(serial=args.serial, enable_oscar=args.oscar,
                               motion_stream=args.motion_stream)

    markers = gp.UDPReceiver(port=args.marker_port)
    with_markers = gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL])
    notch = gp.Bandstop(f_lo=args.mains - 2, f_hi=args.mains + 2)
    bandpass = gp.Bandpass(f_lo=args.band[0], f_hi=args.band[1])
    scope = gp.TimeSeriesScope(
        amplitude_limit=50, time_window=10, hidden_channels=[EEG_CHANNELS],
        markers=[gp.TimeSeriesScope.Markers(color=c, label=str(v),
                                            channel=EEG_CHANNELS, value=v)
                 for v, c in zip([*range(1, 10), *range(11, 18), 99], "rgbcmyrgbrgbcmygw")])
    lsl = gp.LSLSender(stream_name=args.stream_name)
    raw_lsl = gp.LSLSender(stream_name=args.raw_stream)  # dance/ needs motion + EMG unfiltered

    p.connect(source, with_markers["in1"])
    p.connect(markers, with_markers["in2"])
    if args.record:
        os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
        p.connect(with_markers, gp.HDF5Writer(file_name=args.record))
    p.connect(with_markers, raw_lsl)
    p.connect(with_markers, notch)
    p.connect(notch, bandpass)
    p.connect(bandpass, scope)
    p.connect(bandpass, lsl)
    app.add_widget(scope)

    print(f"Streaming filtered EEG + markers on LSL as '{args.stream_name}', "
          f"unfiltered as '{args.raw_stream}'; "
          f"markers on UDP port {args.marker_port}"
          + (f"; recording to {args.record}" if args.record else "")
          + ". Close the scope window to stop.", flush=True)
    p.start()
    try:
        app.run()  # blocks until the window is closed
    finally:
        p.stop()
        p.close()


if __name__ == "__main__":
    main()
