#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bridge from Dexmate's `dexcomm` exoskeleton topic to the IsaacTeleop
external_skeleton plugin.

Subscribes to the `exo_joints` topic published by `omniteleop`'s
`leader/arm_reader.py` and forwards each `ExoJointData` sample as a small
fixed-format binary UDP packet to the local `external_skeleton_plugin`
process. The C++ side (`DexmateExoSource`) decodes these packets and pushes
them onto the OpenXR runtime via `XR_NVX1_push_tensor`.

Wire format (little-endian; matches `DexmateExoSource` exactly):

    offset  size  field
    ------  ----  -----------------------------------------------------------
       0     4    magic       = b'DEXO'
       4     1    version     = 1
       5     1    has_velocity (0 or 1)
       6     1    n_left      (joint count for left arm,  0..32)
       7     1    n_right     (joint count for right arm, 0..32)
       8     8    timestamp_ns (int64, ExoJointData.timestamp_ns)
      16   n_left * 4   left_pos  (float32, radians)
      ...   n_left * 4   left_vel  (float32, rad/s, only if has_velocity)
      ...   n_right * 4  right_pos (float32, radians)
      ...   n_right * 4  right_vel (float32, rad/s, only if has_velocity)

Run alongside the `omni-arm` reader (see Dexmate omniteleop docs):

    omni-arm                                        # produces the dexcomm topic
    ./external_skeleton_plugin dexmate \\           # consumes the UDP feed
        external_skeleton 127.0.0.1 53700
    python3 dexmate_bridge.py --port 53700          # this script

Install the dexmate Python deps once via `install_dexmate.sh`.
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
from typing import Any, Mapping, Optional, Sequence

# Imports from the dexmate omniteleop package (installed by install_dexmate.sh).
# Imported lazily inside main() so that --help works even without dexmate
# installed (useful in CI / sandboxes).


_MAGIC = b"DEXO"
_VERSION = 1
_HEADER_FMT = "<4sBBBBq"  # magic, version, has_vel, n_left, n_right, ts_ns
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAX_JOINTS_PER_ARM = 32


def _coerce_float_list(seq: Optional[Sequence[Any]]) -> list[float]:
    if not seq:
        return []
    return [float(x) for x in seq]


def _encode_packet(payload: Mapping[str, Any]) -> Optional[bytes]:
    """Encode an ExoJointData-shaped dict into the wire packet.

    Returns None if the payload is malformed or exceeds bounds; the bridge
    drops bad samples rather than crashing the loop.
    """
    left_pos = _coerce_float_list(payload.get("left_arm_pos"))
    right_pos = _coerce_float_list(payload.get("right_arm_pos"))
    left_vel = _coerce_float_list(payload.get("left_arm_vel"))
    right_vel = _coerce_float_list(payload.get("right_arm_vel"))
    ts_ns = int(payload.get("timestamp_ns", 0))

    has_vel_bool = bool(left_vel) or bool(right_vel)
    if has_vel_bool:
        # Velocities, when present, must match positions in length per arm.
        if (len(left_vel) not in (0, len(left_pos))) or (len(right_vel) not in (0, len(right_pos))):
            print(
                f"dexmate_bridge: dropping sample with mismatched vel/pos lengths "
                f"(left_pos={len(left_pos)}, left_vel={len(left_vel)}, "
                f"right_pos={len(right_pos)}, right_vel={len(right_vel)})",
                file=sys.stderr,
            )
            return None
        # Pad missing arm vel with zeros so the packet has a consistent shape.
        if not left_vel:
            left_vel = [0.0] * len(left_pos)
        if not right_vel:
            right_vel = [0.0] * len(right_pos)

    if len(left_pos) > _MAX_JOINTS_PER_ARM or len(right_pos) > _MAX_JOINTS_PER_ARM:
        print(
            f"dexmate_bridge: dropping sample, joint count exceeds cap "
            f"(left={len(left_pos)}, right={len(right_pos)}, cap={_MAX_JOINTS_PER_ARM})",
            file=sys.stderr,
        )
        return None

    has_vel_byte = 1 if has_vel_bool else 0
    header = struct.pack(_HEADER_FMT, _MAGIC, _VERSION, has_vel_byte, len(left_pos), len(right_pos), ts_ns)

    body = struct.pack(f"<{len(left_pos)}f", *left_pos)
    if has_vel_bool:
        body += struct.pack(f"<{len(left_vel)}f", *left_vel)
    body += struct.pack(f"<{len(right_pos)}f", *right_pos)
    if has_vel_bool:
        body += struct.pack(f"<{len(right_vel)}f", *right_vel)

    return header + body


class _ForwardingSubscriber:
    """Glue: dexcomm subscriber callback → UDP send."""

    def __init__(self, host: str, port: int) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (host, port)
        self._send_count = 0
        self._drop_count = 0

    def __call__(self, payload: Mapping[str, Any]) -> None:
        packet = _encode_packet(payload)
        if packet is None:
            self._drop_count += 1
            return
        try:
            self._sock.sendto(packet, self._addr)
            self._send_count += 1
        except OSError as exc:
            # Log but don't propagate — the C++ plugin may not be up yet, and
            # the bridge should keep running and reconnect implicitly when it
            # comes back (UDP is connectionless).
            print(f"dexmate_bridge: sendto({self._addr}) failed: {exc}", file=sys.stderr)
            self._drop_count += 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", default="", help="dexcomm namespace (default: empty / global)")
    parser.add_argument(
        "--topic",
        default="exo_joints",
        help="dexcomm topic name as published by omniteleop's arm_reader (default: 'exo_joints')",
    )
    parser.add_argument("--host", default="127.0.0.1", help="UDP host the C++ plugin listens on")
    parser.add_argument("--port", type=int, default=53700, help="UDP port the C++ plugin listens on")
    args = parser.parse_args(argv)

    try:
        from dexcomm import Node
        from dexcomm.codecs import DictDataCodec
    except ImportError as exc:
        print(
            "dexmate_bridge: dexcomm is not importable. Run "
            "src/plugins/external_skeleton/install_dexmate.sh to clone and install "
            "Dexmate's omniteleop and dynamixelAPI packages, then re-run this bridge.",
            file=sys.stderr,
        )
        print(f"  underlying error: {exc}", file=sys.stderr)
        return 1

    forwarder = _ForwardingSubscriber(args.host, args.port)
    node = Node(name="external_skeleton_bridge", namespace=args.namespace)
    node.create_subscriber(args.topic, callback=forwarder, decoder=DictDataCodec.decode)

    resolved = node.resolve_topic(args.topic) if hasattr(node, "resolve_topic") else args.topic
    print(
        f"dexmate_bridge: subscribed to dexcomm topic '{resolved}' (namespace='{args.namespace}'), "
        f"forwarding to {args.host}:{args.port}",
        flush=True,
    )

    try:
        # dexcomm's Node spawns its own threads for subscribers; this main
        # thread just stays alive until the user kills it.
        import time

        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("dexmate_bridge: shutting down")
    finally:
        if hasattr(node, "shutdown"):
            node.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
