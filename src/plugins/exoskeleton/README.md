<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Exoskeleton Plugin

Self-contained IsaacTeleop plugin for a dual-arm Dynamixel-based exoskeleton (e.g. the [Dexmate Vega exoskeleton](https://docs.dexmate.ai/BnmmWshTJeOxvO8H2Ucl/tutorial/teleoperation/teleoperation-exoskeleton/hardware-setup)).

The plugin opens the Dynamixel serial chain itself via the [`dynamixel-sdk`](https://pypi.org/project/dynamixel-sdk/), bulk-reads the joint positions (and optionally velocities) at a configurable rate, and pushes each snapshot as an `ExoArmsOutput` FlatBuffer into an OpenXR tensor collection.

Pair it with [`ExoArmsTracker`](../../core/deviceio_trackers/cpp/inc/deviceio_trackers/exo_arms_tracker.hpp) on the consumer side (e.g. in `groot/control/teleop/device/isaac_teleop_server.py`).

## Data flow

```
Dynamixel chain on /dev/ttyUSB0 (or similar)
    └─ ROBOTIS Protocol 2.0, single bulk-read per cycle
        ↓
exoskeleton_plugin.py (Python, this plugin)
    ├─ dynamixel_sdk: PortHandler + PacketHandler + GroupBulkRead
    └─ SchemaPusher.push_buffer(ExoArmsOutput-serialized FlatBuffer, ts_local, ts_device)
        ↓ OpenXR tensor extensions (XR_NVX1_push_tensor / XR_NVX1_tensor_data)
ExoArmsTracker (C++ in consumer process; e.g. gr00t IsaacTeleopServer)
```

The plugin runs as its own process alongside the IsaacTeleop consumer; the two only share the OpenXR runtime, not any in-process imports.

## Usage

Install the optional extra (brings in `dynamixel-sdk`):

```bash
pip install "isaacteleop[exoskeleton]"
```

Connect the U2D2 (or equivalent Dynamixel-to-USB adapter), make sure the user running the plugin has access to the serial device (e.g. is in the `dialout` group on Linux), then run:

```bash
python3 exoskeleton_plugin.py \
    --port /dev/ttyUSB0 \
    --baud-rate 3000000 \
    --left-arm-motor-ids 1,2,3,4,5,6,7 \
    --right-arm-motor-ids 11,12,13,14,15,16,17 \
    --read-velocity \
    --read-rate-hz 40
```

CLI options:

- `--port` (default `/dev/ttyUSB0`): serial port the Dynamixel adapter is on.
- `--baud-rate` (default `3000000`): Dynamixel bus baud rate.
- `--left-arm-motor-ids`, `--right-arm-motor-ids` (defaults `1..7` / `11..17`): comma-separated motor IDs per arm. Either side can be empty (e.g. for single-arm rigs).
- `--left-arm-polarities`, `--right-arm-polarities` (default all `+1`): comma-separated `+1`/`-1` per motor to flip the sign of the decoded joint (useful when an arm is mirrored physically).
- `--read-velocity` (default off): also bulk-read present velocity; otherwise the `*_vel` FlatBuffer fields are emitted as empty lists.
- `--read-rate-hz` (default `40`): target bulk-read + push frequency. Hard upper bound depends on the bus / motor count.
- `--collection-id` (default `exo_arms`): OpenXR tensor collection ID. Must match `ExoArmsTracker`'s constructor argument.
- `--tensor-identifier` (default `exo_arms`): OpenXR tensor identifier within the collection.
- `--max-flatbuffer-size` (default `1024`): upper bound on the serialized FlatBuffer payload size. 4 vectors x ~16 floats x 4 bytes + overhead fits easily.

The plugin pings each configured motor at startup; unreachable motors are reported on stderr and replaced with `0.0` in the published snapshot (so the shape of `left_arm_pos`/`right_arm_pos` stays stable across power cycles).

## Schema

The wire schema lives in [`src/core/schema/fbs/exo_arms.fbs`](../../core/schema/fbs/exo_arms.fbs):

```fbs
table ExoArmsOutput {
  left_arm_pos:  [float] (id: 0);  // radians, ordered as --left-arm-motor-ids
  left_arm_vel:  [float] (id: 1);  // rad/s, empty if --read-velocity is off
  right_arm_pos: [float] (id: 2);  // radians, ordered as --right-arm-motor-ids
  right_arm_vel: [float] (id: 3);
}
```

Conversions match the Dynamixel X-series defaults (position scaled from raw `[0, 4095]` ticks to `[-pi, pi]` radians; velocity from raw 0.229 rpm units to rad/s). Adjust the constants in `exoskeleton_plugin.py` if you wire a different motor model.
