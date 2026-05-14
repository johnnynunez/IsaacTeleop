#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IsaacTeleop exoskeleton plugin.

Self-contained plugin that owns a dual-arm exoskeleton (e.g. the
`Dexmate Vega exoskeleton <https://docs.dexmate.ai/BnmmWshTJeOxvO8H2Ucl/tutorial/teleoperation/teleoperation-exoskeleton/hardware-setup>`_):

1. Opens the Dynamixel chain directly over the serial bus using ROBOTIS's
   official ``dynamixel_sdk`` Python package (``PortHandler`` +
   ``PacketHandler`` + ``GroupBulkRead``).
2. Bulk-reads each motor's present position (and, optionally, present
   velocity) at the configured rate.
3. Packs the dual-arm snapshot into an ``ExoArmsOutput`` FlatBuffer and pushes
   it into an OpenXR tensor collection via IsaacTeleop's ``SchemaPusher``.

Consumer-side trackers like ``ExoArmsTracker`` (and gr00t's
``ExoskeletonStreamer``) read the collection and decode the joint snapshots
back into typed Python objects.

Architecturally this mirrors the C++ ``generic_3axis_pedal`` plugin: the
plugin owns its data source (here: the Dynamixel chain instead of
``/dev/input/js*``), its own ``OpenXRSession``, and its own ``SchemaPusher``,
running in a separate process from the IsaacTeleop consumer (e.g. gr00t's
``IsaacTeleopServer``).

Run with::

    pip install "isaacteleop[exoskeleton]"  # installs dynamixel-sdk
    python3 exoskeleton_plugin.py \\
        --port /dev/ttyUSB0 \\
        --baud-rate 3000000 \\
        --left-arm-motor-ids 1,2,3,4,5,6,7 \\
        --right-arm-motor-ids 11,12,13,14,15,16,17

Motor wire conversions match the Dynamixel X-series defaults used by the
Dexmate exoskeleton (position scaled to ``[-pi, pi]``; velocity in rad/s,
assuming the 0.229 rpm raw unit). Adjust the constants below if you wire a
different motor model.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import List, Optional, Sequence, Tuple

from isaacteleop import oxr
from isaacteleop.pusherio import SchemaPusher, SchemaPusherConfig
from isaacteleop.schema import ExoArmsOutput


# Dynamixel X-series register addresses and conversions (ROBOTIS Protocol 2.0
# Control Table). See https://emanual.robotis.com/docs/en/dxl/x/ for the spec.
ADDR_PRESENT_VELOCITY = 128
LEN_PRESENT_VELOCITY = 4
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4
PROTOCOL_VERSION = 2.0
POSITION_SCALE = 4095.0
VELOCITY_SCALE = 0.229
RPM_TO_RAD_S = 2.0 * math.pi / 60.0
COMM_SUCCESS = 0


def _parse_int_list(raw: str) -> List[int]:
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_polarity_list(raw: str) -> List[int]:
    parsed = _parse_int_list(raw)
    for value in parsed:
        if value not in (-1, 1):
            raise argparse.ArgumentTypeError(
                f"polarity entries must be either 1 or -1 (got {value})"
            )
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IsaacTeleop exoskeleton plugin (direct Dynamixel reader)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Hardware / Dynamixel chain.
    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="Serial port the Dynamixel U2D2/USB2Dynamixel adapter is on.",
    )
    parser.add_argument(
        "--baud-rate",
        type=int,
        default=3_000_000,
        help="Dynamixel bus baud rate.",
    )
    parser.add_argument(
        "--left-arm-motor-ids",
        type=_parse_int_list,
        default=_parse_int_list("1,2,3,4,5,6,7"),
        help="Comma-separated Dynamixel motor IDs for the left arm.",
    )
    parser.add_argument(
        "--left-arm-polarities",
        type=_parse_polarity_list,
        default=None,
        help=(
            "Comma-separated +1/-1 polarity correction per left-arm motor "
            "(defaults to all +1 if omitted)."
        ),
    )
    parser.add_argument(
        "--right-arm-motor-ids",
        type=_parse_int_list,
        default=_parse_int_list("11,12,13,14,15,16,17"),
        help="Comma-separated Dynamixel motor IDs for the right arm.",
    )
    parser.add_argument(
        "--right-arm-polarities",
        type=_parse_polarity_list,
        default=None,
        help=(
            "Comma-separated +1/-1 polarity correction per right-arm motor "
            "(defaults to all +1 if omitted)."
        ),
    )
    parser.add_argument(
        "--read-velocity",
        action="store_true",
        help=(
            "If set, also bulk-read each motor's present velocity. Otherwise "
            "the left_arm_vel/right_arm_vel FlatBuffer fields stay empty."
        ),
    )
    parser.add_argument(
        "--read-rate-hz",
        type=float,
        default=40.0,
        help=(
            "Target rate at which motors are bulk-read and snapshots are "
            "pushed to OpenXR. Should not exceed what the bus can sustain."
        ),
    )

    # OpenXR / SchemaPusher.
    parser.add_argument(
        "--collection-id",
        default="exo_arms",
        help="OpenXR tensor collection ID. Must match the consumer's ExoArmsTracker.",
    )
    parser.add_argument(
        "--tensor-identifier",
        default="exo_arms",
        help="OpenXR tensor identifier within the collection.",
    )
    parser.add_argument(
        "--max-flatbuffer-size",
        type=int,
        default=1024,
        help="Upper bound on the serialized FlatBuffer payload size, in bytes.",
    )
    return parser.parse_args()


def _validate_arm_config(
    name: str, motor_ids: Sequence[int], polarities: Optional[Sequence[int]]
) -> List[int]:
    """Return a polarity list matching the motor count (defaulting to +1s)."""
    if polarities is None:
        return [1] * len(motor_ids)
    if len(polarities) != len(motor_ids):
        raise SystemExit(
            f"exoskeleton_plugin: {name} polarity count ({len(polarities)}) does "
            f"not match motor count ({len(motor_ids)})"
        )
    return list(polarities)


class DynamixelArmReader:
    """Reads positions (and optionally velocities) for one or more Dynamixel arms.

    Owns a single serial port / packet handler / bulk reader for the entire
    chain so we can pull both arms in one round-trip per cycle.
    """

    def __init__(
        self,
        port: str,
        baud_rate: int,
        read_velocity: bool,
    ) -> None:
        from dynamixel_sdk import GroupBulkRead, PacketHandler, PortHandler

        self._read_velocity = read_velocity
        self._port_handler = PortHandler(port)
        self._packet_handler = PacketHandler(PROTOCOL_VERSION)

        if not self._port_handler.openPort():
            raise SystemExit(f"exoskeleton_plugin: failed to open port {port}")
        if not self._port_handler.setBaudRate(baud_rate):
            self._port_handler.closePort()
            raise SystemExit(
                f"exoskeleton_plugin: failed to set baud rate {baud_rate} on {port}"
            )

        self._bulk_reader = GroupBulkRead(self._port_handler, self._packet_handler)
        self._connected: set[int] = set()

        print(f"[exoskeleton_plugin] Opened {port} @ {baud_rate} baud")

    def register_motors(self, motor_ids: Sequence[int], label: str) -> None:
        """Ping each motor, register it for bulk reads if reachable."""
        for motor_id in motor_ids:
            _, comm_result, dxl_error = self._packet_handler.ping(
                self._port_handler, motor_id
            )
            if comm_result != COMM_SUCCESS:
                print(
                    f"[exoskeleton_plugin] WARN {label} motor {motor_id} ping failed "
                    f"(comm_result={comm_result})",
                    file=sys.stderr,
                )
                continue
            if dxl_error:
                print(
                    f"[exoskeleton_plugin] WARN {label} motor {motor_id} reported "
                    f"hardware error 0x{dxl_error:02x}",
                    file=sys.stderr,
                )

            if not self._bulk_reader.addParam(
                motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
            ):
                print(
                    f"[exoskeleton_plugin] WARN {label} motor {motor_id} could not be "
                    f"added to bulk reader for position",
                    file=sys.stderr,
                )
                continue

            if self._read_velocity:
                if not self._bulk_reader.addParam(
                    motor_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY
                ):
                    print(
                        f"[exoskeleton_plugin] WARN {label} motor {motor_id} could not "
                        f"be added to bulk reader for velocity",
                        file=sys.stderr,
                    )

            self._connected.add(motor_id)
            print(f"[exoskeleton_plugin] {label} motor {motor_id} connected")

    @property
    def num_connected(self) -> int:
        return len(self._connected)

    def refresh(self) -> bool:
        """Trigger one bulk read. Returns True if the wire transaction succeeded."""
        comm_result = self._bulk_reader.txRxPacket()
        if comm_result != COMM_SUCCESS:
            return False
        return True

    def read_arm(
        self, motor_ids: Sequence[int], polarities: Sequence[int]
    ) -> Tuple[List[float], List[float]]:
        """Decode one arm's positions (and velocities) from the latest bulk read."""
        positions: List[float] = []
        velocities: List[float] = []
        for motor_id, polarity in zip(motor_ids, polarities):
            if motor_id not in self._connected:
                positions.append(0.0)
                if self._read_velocity:
                    velocities.append(0.0)
                continue
            positions.append(self._read_position(motor_id) * polarity)
            if self._read_velocity:
                velocities.append(self._read_velocity_one(motor_id) * polarity)
        return positions, velocities

    def _read_position(self, motor_id: int) -> float:
        if not self._bulk_reader.isAvailable(
            motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
        ):
            return 0.0
        raw = self._bulk_reader.getData(
            motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
        )
        return (raw / POSITION_SCALE) * (2.0 * math.pi) - math.pi

    def _read_velocity_one(self, motor_id: int) -> float:
        if not self._bulk_reader.isAvailable(
            motor_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY
        ):
            return 0.0
        raw = self._bulk_reader.getData(
            motor_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY
        )
        # Dynamixel returns velocity as a signed 32-bit value; getData returns
        # it raw (unsigned), so two's-complement the high bit ourselves.
        if raw >= (1 << 31):
            raw -= 1 << 32
        return raw * VELOCITY_SCALE * RPM_TO_RAD_S

    def close(self) -> None:
        try:
            self._port_handler.closePort()
        except Exception:
            pass


def main() -> int:
    args = _parse_args()

    if args.read_rate_hz <= 0:
        raise SystemExit("exoskeleton_plugin: --read-rate-hz must be > 0")

    if not args.left_arm_motor_ids and not args.right_arm_motor_ids:
        raise SystemExit(
            "exoskeleton_plugin: at least one of --left-arm-motor-ids / "
            "--right-arm-motor-ids must be non-empty"
        )

    left_polarities = _validate_arm_config(
        "left-arm", args.left_arm_motor_ids, args.left_arm_polarities
    )
    right_polarities = _validate_arm_config(
        "right-arm", args.right_arm_motor_ids, args.right_arm_polarities
    )

    try:
        import dynamixel_sdk  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "exoskeleton_plugin: dynamixel_sdk is required. Install via "
            "`pip install isaacteleop[exoskeleton]` or `pip install dynamixel-sdk`."
        ) from exc

    reader = DynamixelArmReader(
        port=args.port,
        baud_rate=args.baud_rate,
        read_velocity=args.read_velocity,
    )
    if args.left_arm_motor_ids:
        reader.register_motors(args.left_arm_motor_ids, label="left-arm")
    if args.right_arm_motor_ids:
        reader.register_motors(args.right_arm_motor_ids, label="right-arm")

    if reader.num_connected == 0:
        reader.close()
        raise SystemExit(
            "exoskeleton_plugin: no Dynamixel motors responded to ping. Check the "
            "serial port, baud rate, and motor IDs."
        )

    session = oxr.OpenXRSession(
        "ExoskeletonPlugin", SchemaPusher.get_required_extensions()
    )
    pusher = SchemaPusher(
        session.get_handles(),
        SchemaPusherConfig(
            collection_id=args.collection_id,
            max_flatbuffer_size=args.max_flatbuffer_size,
            tensor_identifier=args.tensor_identifier,
            localized_name="Exoskeleton Arms",
            app_name="ExoskeletonPlugin",
        ),
    )
    print(
        f"[exoskeleton_plugin] SchemaPusher ready on collection_id='{args.collection_id}', "
        f"tensor_identifier='{args.tensor_identifier}'"
    )

    frame_duration_s = 1.0 / args.read_rate_hz

    try:
        next_deadline = time.monotonic()
        while True:
            success = reader.refresh()
            if success:
                left_pos, left_vel = reader.read_arm(
                    args.left_arm_motor_ids, left_polarities
                )
                right_pos, right_vel = reader.read_arm(
                    args.right_arm_motor_ids, right_polarities
                )

                output = ExoArmsOutput(
                    left_arm_pos=left_pos,
                    left_arm_vel=left_vel if args.read_velocity else [],
                    right_arm_pos=right_pos,
                    right_arm_vel=right_vel if args.read_velocity else [],
                )
                buffer = output.serialize(args.max_flatbuffer_size)

                # Reader-side DeviceDataTimestamp expects:
                #   sample_time_local_common_clock = first arg (monotonic ns)
                #   sample_time_raw_device_clock   = second arg (raw device ns)
                # The plugin owns the hardware read, so both clocks are simply
                # our own measurements taken right after the bulk read returned.
                local_common_ns = time.monotonic_ns()
                raw_device_ns = time.time_ns()
                pusher.push_buffer(buffer, local_common_ns, raw_device_ns)

            # Pace the loop to the configured read rate, even if a bulk read
            # transaction returns faster (or fails).
            next_deadline += frame_duration_s
            sleep_for = next_deadline - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Drifted behind schedule (slow bus / wire errors). Reset the
                # deadline so we don't tight-loop catching up.
                next_deadline = time.monotonic()
    except KeyboardInterrupt:
        print("[exoskeleton_plugin] Interrupted, shutting down")
        return 0
    finally:
        reader.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
