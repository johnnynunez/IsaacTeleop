# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rate-limiting safety harness retargeters (EE-pose and joint-space).

Low-cost follower arms such as the SO-101 run bare position servos with no
controller-side handling of anomalous targets: whatever position command reaches the
bus is tracked at full servo torque. A single bad frame -- an IK jump near a
singularity, a controller tracking glitch, a leader-stream hiccup, an operator
flick -- becomes a full-speed slew that can strip gears or stall motors on real
hardware.

This module provides two composable harness nodes that bound the *step per frame*
of an existing command stream without changing its contract, so they can be
inserted between a producing retargeter and the downstream action wiring:

* :class:`EePoseRateLimiter` -- wraps a 7-D absolute ``ee_pose`` stream (e.g. the
  output of ``SO101ClutchRetargeter`` or ``Se3AbsRetargeter``), clamping linear
  velocity [m/s] and angular velocity [rad/s].
* :class:`JointRateLimiter` -- wraps a name-keyed joint-target group (e.g. the
  ``joint`` mode output of ``JointStateRetargeter``), clamping per-joint velocity
  [rad/s or m/s], with optional per-joint overrides.

Both limiters are pure trajectory governors:

- The **first valid frame** after construction or a pipeline reset passes through
  unclamped and latches the baseline. Engage-time placement is owned by the
  upstream clutch / align logic (which already guarantees no snap on engage); the
  limiter bounds *motion between frames*, not absolute placement.
- Each subsequent frame may move at most ``max_velocity * dt`` from the last
  *emitted* command (not the last input), so a persistent far-away target is
  approached at bounded speed instead of in one jump.
- ``dt`` is derived from ``context.graph_time.real_time_ns`` and clamped to
  ``[min_dt, max_dt]``: a stalled or resumed pipeline (huge wall-clock gap) must
  not authorize a proportionally huge step, and a duplicate timestamp must not
  freeze the limiter. When no usable timestamp is available (both deltas zero or
  negative), ``nominal_dt`` is used.
- A dropped input frame holds the last emitted command (matching the upstream
  retargeters' hold-last convention) and does not advance the time baseline.
- A pipeline **reset** clears the baseline: the next valid frame passes through
  and re-latches (the owning task resets the arm pose; re-clamping against a
  stale pre-reset baseline would slew the arm across the workspace).

The clamp is intentionally a hard per-frame velocity bound (a first-order rate
limiter), not a smoothing filter: it adds no lag below the limit -- teleop feel is
untouched until the harness actually has to intervene.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from isaacteleop.retargeting_engine.interface import (
    BaseRetargeter,
    RetargeterIOType,
)
from isaacteleop.retargeting_engine.interface.retargeter_core_types import RetargeterIO
from isaacteleop.retargeting_engine.interface.tensor_group_type import (
    OptionalType,
    TensorGroupType,
)
from isaacteleop.retargeting_engine.tensor_types import (
    DLDataType,
    FloatType,
    NDArrayType,
)

# Output group key of the EE-pose limiter -- matches the upstream SO-101 clutch /
# Se3 retargeters so the limiter is a drop-in insertion in the pipeline wiring.
EE_POSE_KEY = "ee_pose"
# Output group key of the joint limiter -- matches JointStateRetargeter's
# ``joint`` mode output so the limiter is a drop-in insertion.
JOINT_TARGETS_KEY = "joint_targets"

# Numerical floor below which a quaternion norm is treated as degenerate and the
# previous orientation is held instead (never divide by ~zero; mirrors the
# defense-in-depth net in the SO-101 clutch retargeter).
_MIN_QUAT_NORM = 1e-6
# Angular steps below this [rad] pass through without axis extraction: the
# rotation axis is numerically meaningless near identity and the step is
# guaranteed under any sane limit.
_MIN_ANGLE_RAD = 1e-9


@dataclass
class RateLimiterConfig:
    """Configuration shared by the rate-limiting harness nodes.

    Args:
        max_linear_velocity: EE limiter only -- maximum linear speed [m/s] of the
            emitted position. The default is a deliberately conservative
            tabletop-arm bring-up value; raise it once the setup is trusted.
        max_angular_velocity: EE limiter only -- maximum angular speed [rad/s] of
            the emitted orientation (geodesic angle per second).
        max_joint_velocity: Joint limiter only -- default maximum per-joint speed
            [rad/s (revolute) or m/s (prismatic)] applied to every joint not
            overridden in ``joint_velocity_overrides``.
        joint_velocity_overrides: Joint limiter only -- per-joint ``{name: limit}``
            overrides of ``max_joint_velocity`` (e.g. a slower shoulder, a faster
            gripper).
        nominal_dt: Fallback frame period [s] used when the graph timestamps do
            not advance (first frame after a latch, duplicate timestamps).
        max_dt: Upper clamp [s] on the wall-clock frame delta. Bounds the step
            authorized after a pipeline stall/resume; a gap larger than this is
            treated as ``max_dt``.
        min_dt: Lower clamp [s] on the wall-clock frame delta, guarding against
            zero/near-zero deltas producing a frozen limiter.
        startup_duration_s: Length [s] of the startup homing window. While active
            (the first ``startup_duration_s`` seconds after construction and after
            every reset), the limiter **ignores its input entirely** and drives
            toward the node's configured home command at the normal velocity
            limits, so session start / episode reset always returns the arm to a
            known pose regardless of the teleop device's state. ``0`` (default)
            disables the window. Nodes require a home command when this is > 0.
    """

    max_linear_velocity: float = 0.25
    max_angular_velocity: float = 1.5
    max_joint_velocity: float = 1.5
    joint_velocity_overrides: dict[str, float] = field(default_factory=dict)
    nominal_dt: float = 1.0 / 60.0
    max_dt: float = 0.1
    min_dt: float = 1e-4
    startup_duration_s: float = 0.0

    def __post_init__(self) -> None:
        if self.max_linear_velocity <= 0.0:
            raise ValueError("max_linear_velocity must be > 0")
        if self.max_angular_velocity <= 0.0:
            raise ValueError("max_angular_velocity must be > 0")
        if self.max_joint_velocity <= 0.0:
            raise ValueError("max_joint_velocity must be > 0")
        for name, limit in self.joint_velocity_overrides.items():
            if limit <= 0.0:
                raise ValueError(f"joint_velocity_overrides[{name!r}] must be > 0")
        if not 0.0 < self.min_dt <= self.nominal_dt <= self.max_dt:
            raise ValueError("require 0 < min_dt <= nominal_dt <= max_dt")
        if self.startup_duration_s < 0.0:
            raise ValueError("startup_duration_s must be >= 0")


def _clamped_dt(
    prev_ns: int | None, now_ns: int, nominal: float, lo: float, hi: float
) -> float:
    """Frame delta [s] from two ``real_time_ns`` stamps, clamped to ``[lo, hi]``.

    Falls back to ``nominal`` when there is no previous stamp or the clock did not
    advance (duplicate or non-monotonic timestamps).
    """
    if prev_ns is None:
        return nominal
    delta = (now_ns - prev_ns) * 1e-9
    if delta <= 0.0:
        return nominal
    return min(max(delta, lo), hi)


def _clamp_position_step(
    target: np.ndarray, previous: np.ndarray, max_step: float
) -> np.ndarray:
    """Clamp the Euclidean step from ``previous`` toward ``target`` to ``max_step`` [m].

    Returns ``target`` unchanged when it is within reach, else the point at
    ``max_step`` along the straight line from ``previous`` to ``target``.
    """
    delta = target - previous
    dist = float(np.linalg.norm(delta))
    if dist <= max_step or dist == 0.0:
        return target
    return previous + delta * (max_step / dist)


def _quat_normalize(q: np.ndarray) -> np.ndarray | None:
    """Return ``q`` normalized, or ``None`` when degenerate (zero / non-finite)."""
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < _MIN_QUAT_NORM:
        return None
    return q / norm


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a (x) b`` of two ``[x, y, z, w]`` quaternions (scalar-last)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate (== inverse for unit quaternions) of an ``[x, y, z, w]`` quaternion."""
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _clamp_orientation_step(
    target: np.ndarray, previous: np.ndarray, max_step: float
) -> np.ndarray:
    """Clamp the geodesic rotation from ``previous`` toward ``target`` to ``max_step`` [rad].

    Both quaternions are ``[x, y, z, w]`` unit quaternions. Returns ``target``
    (sign-aligned with the shortest arc) when the relative rotation is within
    ``max_step``, else the orientation reached by rotating ``max_step`` along the
    shortest arc from ``previous`` toward ``target``.
    """
    # Relative rotation in the previous frame's body frame: q_rel = prev^-1 (x) target.
    q_rel = _quat_mul(_quat_conjugate(previous), target)
    # Shortest arc: flip the double-cover sign so w >= 0.
    if q_rel[3] < 0.0:
        q_rel = -q_rel
    sin_half = float(np.linalg.norm(q_rel[:3]))
    angle = 2.0 * float(np.arctan2(sin_half, q_rel[3]))
    if angle <= max_step or angle < _MIN_ANGLE_RAD:
        # Within the limit: emit the target, but from the shortest-arc branch so
        # consecutive emitted quaternions never flip hemisphere spuriously.
        return _quat_mul(previous, q_rel)
    axis = q_rel[:3] / sin_half
    half = 0.5 * max_step
    q_step = np.array(
        [
            axis[0] * np.sin(half),
            axis[1] * np.sin(half),
            axis[2] * np.sin(half),
            np.cos(half),
        ],
        dtype=np.float64,
    )
    return _quat_mul(previous, q_step)


class EePoseRateLimiter(BaseRetargeter):
    """Bounds the per-frame linear/angular velocity of an absolute 7-D ``ee_pose`` stream.

    Input and output are both a single 7-D ``ee_pose`` (position [m] + orientation
    quaternion ``[x, y, z, w]``), the exact contract of ``SO101ClutchRetargeter`` /
    ``Se3AbsRetargeter``, so this node inserts between the pose producer and the
    downstream reorderer without any wiring changes.

    The first valid frame after construction / reset passes through and latches the
    baseline; each later frame is clamped to ``max_linear_velocity * dt`` [m] and
    ``max_angular_velocity * dt`` [rad] from the last **emitted** pose (see the
    module docstring for the dt and reset semantics).

    With ``config.startup_duration_s > 0`` (requires ``home_pose``), the node adds
    a **startup homing window**: for that long after construction and after every
    reset the input is ignored entirely and the emitted pose slews from the last
    emitted command toward ``home_pose`` at the configured velocity limits, so a
    session start / episode reset always returns the arm to a known pose no matter
    what the teleop device is doing. On reset the baseline is deliberately kept
    (not cleared): the last emitted command is the best available proxy for where
    the follower physically is, and homing from it is what makes the return
    bounded instead of a snap.
    """

    def __init__(
        self,
        name: str,
        config: RateLimiterConfig | None = None,
        home_pose: np.ndarray | None = None,
    ) -> None:
        """Initialize the EE-pose rate limiter.

        Args:
            name: Name identifier for this retargeter node.
            config: Velocity limits, dt clamping, and startup homing window;
                ``None`` uses the conservative :class:`RateLimiterConfig` defaults.
            home_pose: 7-D home command ``[x, y, z, qx, qy, qz, qw]`` (base frame,
                same convention as the governed stream) targeted during the
                startup homing window. Required when
                ``config.startup_duration_s > 0``; ignored otherwise.
        """
        self._cfg = config if config is not None else RateLimiterConfig()
        if self._cfg.startup_duration_s > 0.0:
            if home_pose is None:
                raise ValueError(
                    "startup_duration_s > 0 requires a home_pose for EePoseRateLimiter"
                )
            home = np.asarray(home_pose, dtype=np.float64)
            if home.shape != (7,):
                raise ValueError(f"home_pose must have shape (7,), got {home.shape}")
            home_ori = _quat_normalize(home[3:7])
            if home_ori is None:
                raise ValueError("home_pose orientation quaternion is degenerate")
            self._home_pose: np.ndarray | None = np.concatenate([home[:3], home_ori])
        else:
            self._home_pose = None
        super().__init__(name=name)
        self._last_pose: np.ndarray | None = None
        self._last_time_ns: int | None = None
        # Startup homing window bookkeeping: armed at construction and on every
        # reset; the deadline is materialized from the first context seen after
        # arming (the node has no clock of its own).
        self._homing_armed = self._home_pose is not None
        self._homing_until_ns: int | None = None

    def input_spec(self) -> RetargeterIOType:
        """Requires an (optional) absolute 7-D ``ee_pose`` to govern."""
        return {
            EE_POSE_KEY: OptionalType(
                TensorGroupType(
                    EE_POSE_KEY,
                    [
                        NDArrayType(
                            "pose", shape=(7,), dtype=DLDataType.FLOAT, dtype_bits=32
                        )
                    ],
                )
            )
        }

    def output_spec(self) -> RetargeterIOType:
        """Outputs the rate-limited absolute 7-D ``ee_pose``."""
        return {
            EE_POSE_KEY: TensorGroupType(
                EE_POSE_KEY,
                [
                    NDArrayType(
                        "pose", shape=(7,), dtype=DLDataType.FLOAT, dtype_bits=32
                    )
                ],
            )
        }

    def _homing_active(self, now_ns: int) -> bool:
        """Whether the startup homing window is active at ``now_ns`` (and lazily arms it)."""
        if self._home_pose is None:
            return False
        if self._homing_armed:
            self._homing_armed = False
            self._homing_until_ns = now_ns + int(self._cfg.startup_duration_s * 1e9)
        return self._homing_until_ns is not None and now_ns < self._homing_until_ns

    def _step_toward(self, target: np.ndarray, now_ns: int) -> np.ndarray:
        """One rate-limited step from the last emitted pose toward ``target`` (7-D)."""
        assert self._last_pose is not None
        dt = _clamped_dt(
            self._last_time_ns,
            now_ns,
            self._cfg.nominal_dt,
            self._cfg.min_dt,
            self._cfg.max_dt,
        )
        pos = _clamp_position_step(
            target[:3], self._last_pose[:3], self._cfg.max_linear_velocity * dt
        )
        ori = _clamp_orientation_step(
            target[3:7], self._last_pose[3:7], self._cfg.max_angular_velocity * dt
        )
        return np.concatenate([pos, ori])

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        """Emits the input pose clamped to the configured per-frame step."""
        now_ns = int(context.graph_time.real_time_ns)

        if context.execution_events.reset:
            if self._home_pose is not None:
                # Fresh episode with startup homing: KEEP the baseline (the last
                # emitted command is the best proxy for where the follower
                # physically is) and re-arm the homing window, so the return to
                # home is a bounded slew instead of a snap.
                self._homing_armed = True
                self._homing_until_ns = None
            else:
                # Fresh episode without homing: forget the baseline. The next
                # valid frame passes through and re-latches -- clamping against
                # the stale pre-reset pose would command a long slew.
                self._last_pose = None
                self._last_time_ns = None

        out = outputs[EE_POSE_KEY]

        if self._homing_active(now_ns):
            # Startup homing window: ignore the input entirely and slew toward
            # the configured home at the normal velocity limits.
            assert self._home_pose is not None
            if self._last_pose is None:
                # Nothing emitted yet (cold start): command home directly. The
                # deployment contract is the same as the upstream clutch's
                # reset-origin -- the owning task parks the arm at/near home, so
                # this does not command a cross-workspace jump.
                homed = self._home_pose.copy()
            else:
                homed = self._step_toward(self._home_pose, now_ns)
            self._last_pose = homed
            self._last_time_ns = now_ns
            out[0] = homed.astype(np.float32)
            return

        inp = inputs[EE_POSE_KEY]
        if inp.is_none:
            # Dropped frame: hold the last emitted pose (upstream convention).
            # Keep the time baseline -- max_dt bounds the catch-up step anyway.
            if self._last_pose is not None:
                out[0] = self._last_pose.astype(np.float32)
            return

        pose = np.asarray(np.from_dlpack(inp[0]), dtype=np.float64)
        ori = _quat_normalize(pose[3:7])

        if self._last_pose is None:
            if ori is None:
                # Degenerate first orientation: nothing sane to latch; hold off.
                return
            # First valid frame: pass through, latch the baseline.
            latched = np.concatenate([pose[:3], ori])
            self._last_pose = latched
            self._last_time_ns = now_ns
            out[0] = latched.astype(np.float32)
            return

        if ori is None:
            # Degenerate target orientation: keep the last emitted one.
            target = np.concatenate([pose[:3], self._last_pose[3:7]])
        else:
            target = np.concatenate([pose[:3], ori])

        limited = self._step_toward(target, now_ns)
        self._last_pose = limited
        self._last_time_ns = now_ns
        out[0] = limited.astype(np.float32)


class JointRateLimiter(BaseRetargeter):
    """Bounds the per-frame velocity of a name-keyed joint-target group.

    Input and output are both a ``joint_targets`` group with one ``FloatType`` per
    configured joint name, the exact contract of ``JointStateRetargeter``'s
    ``joint`` mode, so this node inserts between the leader mirror and the action
    wiring without changes. Every joint is clamped to ``limit * dt`` per frame,
    where ``limit`` is ``config.joint_velocity_overrides[name]`` when present,
    else ``config.max_joint_velocity``.

    The first valid frame after construction / reset passes through and latches the
    baseline (startup alignment is owned upstream); later frames are clamped
    against the last **emitted** targets (see the module docstring).

    With ``config.startup_duration_s > 0`` (requires ``home_targets``), the node
    adds a **startup homing window**: for that long after construction and after
    every reset the input is ignored entirely and the emitted targets slew from
    the last emitted command toward ``home_targets`` at the per-joint limits (see
    :class:`EePoseRateLimiter` for the rationale; the reset baseline is kept, not
    cleared, so the return to home is a bounded slew).
    """

    def __init__(
        self,
        name: str,
        joint_names: list[str],
        config: RateLimiterConfig | None = None,
        home_targets: list[float] | np.ndarray | None = None,
    ) -> None:
        """Initialize the joint-space rate limiter.

        Args:
            name: Name identifier for this retargeter node.
            joint_names: Ordered joint names; must match the upstream producer's
                output element names/order (checked by the graph at connect time).
            config: Velocity limits, dt clamping, and startup homing window;
                ``None`` uses the conservative :class:`RateLimiterConfig` defaults.
            home_targets: Per-joint home command (same order and units as
                ``joint_names``) targeted during the startup homing window.
                Required when ``config.startup_duration_s > 0``; ignored otherwise.
        """
        if not joint_names:
            raise ValueError("joint_names must be non-empty")
        self._cfg = config if config is not None else RateLimiterConfig()
        self._joint_names = list(joint_names)
        unknown = set(self._cfg.joint_velocity_overrides) - set(self._joint_names)
        if unknown:
            raise ValueError(
                f"joint_velocity_overrides for unknown joints: {sorted(unknown)}"
            )
        if self._cfg.startup_duration_s > 0.0:
            if home_targets is None:
                raise ValueError(
                    "startup_duration_s > 0 requires home_targets for JointRateLimiter"
                )
            home = np.asarray(home_targets, dtype=np.float64)
            if home.shape != (len(self._joint_names),):
                raise ValueError(
                    f"home_targets must have shape ({len(self._joint_names)},), "
                    f"got {home.shape}"
                )
            self._home_targets: np.ndarray | None = home.copy()
        else:
            self._home_targets = None
        super().__init__(name=name)
        self._limits = np.array(
            [
                self._cfg.joint_velocity_overrides.get(n, self._cfg.max_joint_velocity)
                for n in self._joint_names
            ],
            dtype=np.float64,
        )
        self._last_targets: np.ndarray | None = None
        self._last_time_ns: int | None = None
        # Startup homing window bookkeeping (see EePoseRateLimiter).
        self._homing_armed = self._home_targets is not None
        self._homing_until_ns: int | None = None

    def input_spec(self) -> RetargeterIOType:
        """Requires an (optional) name-keyed joint-target group to govern."""
        return {
            JOINT_TARGETS_KEY: OptionalType(
                TensorGroupType(
                    JOINT_TARGETS_KEY, [FloatType(n) for n in self._joint_names]
                )
            )
        }

    def output_spec(self) -> RetargeterIOType:
        """Outputs the rate-limited joint targets, one ``FloatType`` per joint."""
        return {
            JOINT_TARGETS_KEY: TensorGroupType(
                JOINT_TARGETS_KEY, [FloatType(n) for n in self._joint_names]
            )
        }

    def _homing_active(self, now_ns: int) -> bool:
        """Whether the startup homing window is active at ``now_ns`` (and lazily arms it)."""
        if self._home_targets is None:
            return False
        if self._homing_armed:
            self._homing_armed = False
            self._homing_until_ns = now_ns + int(self._cfg.startup_duration_s * 1e9)
        return self._homing_until_ns is not None and now_ns < self._homing_until_ns

    def _step_toward(self, targets: np.ndarray, now_ns: int) -> np.ndarray:
        """One rate-limited step from the last emitted targets toward ``targets``."""
        assert self._last_targets is not None
        dt = _clamped_dt(
            self._last_time_ns,
            now_ns,
            self._cfg.nominal_dt,
            self._cfg.min_dt,
            self._cfg.max_dt,
        )
        max_step = self._limits * dt
        delta = np.clip(targets - self._last_targets, -max_step, max_step)
        return self._last_targets + delta

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        """Emits the input targets clamped to the configured per-frame step."""
        now_ns = int(context.graph_time.real_time_ns)

        if context.execution_events.reset:
            if self._home_targets is not None:
                # Fresh episode with startup homing: keep the baseline and re-arm
                # the homing window (see EePoseRateLimiter).
                self._homing_armed = True
                self._homing_until_ns = None
            else:
                # Fresh episode without homing: forget the baseline.
                self._last_targets = None
                self._last_time_ns = None

        out = outputs[JOINT_TARGETS_KEY]
        n = len(self._joint_names)

        if self._homing_active(now_ns):
            # Startup homing window: ignore the input entirely and slew toward
            # the configured home targets at the per-joint limits.
            assert self._home_targets is not None
            if self._last_targets is None:
                # Cold start: command home directly (the owning task parks the
                # arm at/near home; same contract as the upstream align slew).
                homed = self._home_targets.copy()
            else:
                homed = self._step_toward(self._home_targets, now_ns)
            self._last_targets = homed
            self._last_time_ns = now_ns
            for i in range(n):
                out[i] = float(homed[i])
            return

        jin = inputs[JOINT_TARGETS_KEY]
        if jin.is_none:
            # Dropped frame: hold the last emitted targets.
            if self._last_targets is not None:
                for i in range(n):
                    out[i] = float(self._last_targets[i])
            return

        targets = np.array([float(jin[i]) for i in range(n)], dtype=np.float64)

        if self._last_targets is None:
            # First valid frame: pass through, latch the baseline.
            self._last_targets = targets
            self._last_time_ns = now_ns
            for i in range(n):
                out[i] = float(targets[i])
            return

        limited = self._step_toward(targets, now_ns)
        self._last_targets = limited
        self._last_time_ns = now_ns
        for i in range(n):
            out[i] = float(limited[i])
