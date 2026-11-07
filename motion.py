"""BCICore that also publishes the headset's accelerometer/gyroscope on LSL.

gpype's BCICore keeps only the EEG columns of each sample and drops the rest
(accelerometer, gyroscope, battery, counter...). HybridBlack has
include_accel/include_gyro; BCICore (gpype 4.1.0) has no equivalent. This
subclass hooks the two places where the full device row is visible:

  _resolve_channels(device)  -- once, at connect: find the ACC/GYR columns
  _assemble(block, arrival)  -- per BLE notification: push those columns

These are private gpype methods, so this is pinned to gpype 4.1.x
(see requirements.txt). The EEG path is untouched.

Timestamps: `arrival` is gpype's own host clock (time.monotonic), the same
clock gpype's LSLSender stamps the EEG stream with, so the two streams line
up (to within BLE notification jitter, a few tens of ms).
"""

import numpy as np
from pylsl import StreamInfo, StreamOutlet

import gpype as gp

# gtec_ble.Amplifier.ChannelType values
ACC, GYR = 2, 3
AXES = "XYZ"


def _type_value(t):
    return getattr(t, "value", t)


class MotionBCICore(gp.BCICore):

    def __init__(self, *args, motion_stream="core8_motion", **kwargs):
        super().__init__(*args, **kwargs)
        self._motion_stream = motion_stream
        self._motion_idx = None
        self._motion_outlet = None

    def _resolve_channels(self, device):
        super()._resolve_channels(device)
        types = [_type_value(t) for t in device.channel_types]
        print(f"[motion] device channel map ({len(types)} columns): {types} "
              "(1=EEG 2=ACC 3=GYR 4=BAT 5=CNT 6=LINK 9=VALID)")
        idx, labels = [], []
        for kind, name in ((ACC, "Acc"), (GYR, "Gyr")):
            cols = [i for i, t in enumerate(types) if t == kind]
            idx += cols
            labels += [f"{name}{AXES[k] if k < 3 else k}" for k in range(len(cols))]
        if not idx:
            print("[motion] this headset reports no accelerometer/gyroscope "
                  "channels; no motion stream.")
            return
        self._motion_idx = np.asarray(idx)
        if self._motion_outlet is None:  # keep one outlet across reconnects
            info = StreamInfo(self._motion_stream, "Motion", len(idx),
                              float(device.sampling_rate or 250), "float32",
                              f"{self._motion_stream}-{device.serial_number}")
            chans = info.desc().append_child("channels")
            # Units are whatever gtec_ble delivers (for the Unicorn gpype
            # documents g and deg/s; unverified for the Core-8).
            for lab in labels:
                chans.append_child("channel").append_child_value("label", lab)
            self._motion_outlet = StreamOutlet(info)
        print(f"[motion] publishing {labels} on LSL as '{self._motion_stream}'")

    def _assemble(self, block, arrival):
        super()._assemble(block, arrival)
        if self._motion_outlet is None or self._sample_width is None:
            return
        block = np.asarray(block)
        if block.size % self._sample_width:
            return
        rows = block.reshape(-1, self._sample_width)[:, self._motion_idx]
        # LSL takes the timestamp of the chunk's last sample
        self._motion_outlet.push_chunk(rows.tolist(), arrival)
