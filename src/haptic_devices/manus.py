# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manus Metagloves Pro Haptic adapter.

Forwards per-finger powers to the Manus plugin singleton via the
``_manus_haptic`` pybind module, which routes to
``CoreSdk_VibrateFingersForGlove``. Assumes the Manus hand-tracking plugin
has already started the singleton; vendor SDK linkage stays inside
``src/plugins/manus/`` per the AGENTS.md boundary.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from isaacteleop.retargeting_engine.interface.tensor_group_type import TensorGroupType
from isaacteleop.retargeting_engine.tensor_types import (
    FingerPowerVector,
    NUM_HAPTIC_FINGERS,
)

from .interface import IHapticDevice


logger = logging.getLogger(__name__)


class ManusHapticDevice(IHapticDevice):
    """:class:`IHapticDevice` adapter for the Manus Metagloves Pro Haptic glove.

    Consumes ``FingerPowerVector(5)`` in Manus order
    ``[Thumb, Index, Middle, Ring, Pinky]`` with values in ``[0, 1]``. The
    pybind module is imported lazily so importing this module does not
    require the Manus SDK to be installed; if the SDK or plugin is missing,
    :meth:`apply` logs once per side and silently no-ops so the pipeline
    keeps running.
    """

    def __init__(self, num_fingers: int = NUM_HAPTIC_FINGERS) -> None:
        self._num_fingers = num_fingers
        self._pybind = None
        self._error_logged: dict[str, bool] = {"left": False, "right": False}

    def accepted_type(self) -> TensorGroupType:
        return FingerPowerVector(self._num_fingers)

    def apply(self, side: Literal["left", "right"], values: np.ndarray) -> None:
        pybind = self._get_pybind()
        if pybind is None:
            return

        arr = np.asarray(values, dtype=np.float32).ravel()
        if arr.size != self._num_fingers:
            raise ValueError(
                f"ManusHapticDevice.apply expects a {self._num_fingers}-element "
                f"FingerPowerVector (order [Thumb, Index, Middle, Ring, Pinky]), "
                f"got shape {np.asarray(values).shape}"
            )
        try:
            pybind.apply_haptic_command(side, arr)
        except Exception as exc:
            if not self._error_logged[side]:
                logger.warning(
                    "ManusHapticDevice.apply(%s) failed (further errors for this "
                    "side will be silenced): %s",
                    side,
                    exc,
                )
                self._error_logged[side] = True

    def _get_pybind(self):
        if self._pybind is not None:
            return self._pybind
        try:
            from . import _manus_haptic  # type: ignore[import-not-found]
        except ImportError as exc:
            if not self._error_logged["left"]:
                logger.warning(
                    "ManusHapticDevice unavailable: %s. Build the Manus plugin "
                    "(src/plugins/manus/) with the Manus SDK installed to "
                    "enable haptic output.",
                    exc,
                )
                self._error_logged["left"] = True
                self._error_logged["right"] = True
            return None
        self._pybind = _manus_haptic
        return self._pybind
