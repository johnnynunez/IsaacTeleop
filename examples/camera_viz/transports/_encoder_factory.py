# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Encoder backend selection for camera_streamer.

Three backends, all opt-in via ``encoder:`` (top-level default) or
``rtp.encoder:`` (per-camera override):

  * :class:`NvH264Encoder` (``native``) — desktop, native NVENC at
    ``examples/camera_viz/codec/``. Zero-copy from CuPy into NVENC.
  * :class:`GstNvH264Encoder` (``gstreamer``) — Jetson / portable,
    GStreamer's ``nvv4l2h264enc`` (or ``nvh264enc`` / ``x264enc``
    fallbacks). ~1-3 ms D2H download at 720p.
  * VPU passthrough (``vpu``) — OAK-D only. The source itself runs
    ``dai.node.VideoEncoder`` on the device's Myriad-X and emits H.264
    NALs that go straight to RTP, no host-side encode. ``make_encoder``
    returns ``None`` for this backend (sentinel "no host encoder") —
    the RTP sender detects ``Frame.encoded_packet`` and short-circuits.

``"auto"`` tries the native codec first; falls back to GStreamer. It
NEVER picks ``vpu`` — the VPU path has a different bitstream contract
(no host-side bitrate adaptation, no per-frame error recovery) and is
camera-specific, so it must be opted into explicitly.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _has_native_codec() -> bool:
    # Both ImportError (no package) and RuntimeError (raised by
    # codec/__init__.py on missing .so) fall through to GStreamer.
    try:
        import codec  # noqa: F401

        return True
    except (ImportError, RuntimeError):
        return False


def make_encoder(
    backend: str,
    *,
    width: int,
    height: int,
    bitrate: int,
    fps: int,
    gop: int,
    gpu_id: int,
):
    """Build a sender-side H.264 encoder.

    ``backend`` is ``"auto"``, ``"native"``, ``"gstreamer"``, or
    ``"vpu"``. Returns an object with
    ``encode(rgba_cupy_array) -> List[bytes]`` +
    ``end_of_stream() -> List[bytes]`` + ``reset() -> None`` — except
    for ``"vpu"`` which returns ``None`` (the source emits pre-encoded
    NALs; the sender short-circuits when ``Frame.encoded_packet`` is set).
    """
    chosen = backend.lower() if isinstance(backend, str) else "auto"
    if chosen == "auto":
        chosen = "native" if _has_native_codec() else "gstreamer"
        logger.info("encoder backend (auto): %s", chosen)

    if chosen == "native":
        from ._nv_encode import NvH264Encoder

        return NvH264Encoder(
            width=width, height=height, bitrate=bitrate, fps=fps, gop=gop, gpu_id=gpu_id
        )
    if chosen == "gstreamer":
        from ._nv_encode_gst import GstNvH264Encoder

        return GstNvH264Encoder(
            width=width, height=height, bitrate=bitrate, fps=fps, gop=gop, gpu_id=gpu_id
        )
    if chosen == "vpu":
        # No host-side encoder; the OAK-D source runs dai.node.VideoEncoder
        # on the Myriad-X VPU. RtpH264Sender keys off Frame.encoded_packet
        # and pushes the NALs through h264parse / rtph264pay directly.
        return None
    raise ValueError(
        f"unknown encoder backend {backend!r} (known: auto | native | gstreamer | vpu)"
    )
