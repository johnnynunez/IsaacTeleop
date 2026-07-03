# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sim-free unit tests for the rate-limiting safety harness retargeters.

Covers the safety-harness nodes that bound per-frame command velocity for bare
position-servo followers (e.g. the SO-101):

* :class:`~isaacteleop.retargeters.EePoseRateLimiter` -- linear/angular velocity
  bounds on an absolute 7-D ``ee_pose`` stream.
* :class:`~isaacteleop.retargeters.JointRateLimiter` -- per-joint velocity bounds
  on a name-keyed ``joint_targets`` group.

Each limiter is exercised at the pure-math level (the module-private clamp
helpers) and at the ``BaseRetargeter.compute`` level (build inputs/outputs, drive
frames with controlled ``GraphTime`` stamps, read the emitted tensors), with no
``gym.make``, USD, GPU, or XR device.
"""

import math

import numpy as np
import pytest

from isaacteleop.retargeting_engine.interface import (
    ComputeContext,
    ExecutionEvents,
    ExecutionState,
    OptionalTensorGroup,
    TensorGroup,
)
from isaacteleop.retargeting_engine.interface.retargeter_core_types import GraphTime
from isaacteleop.retargeting_engine.interface.tensor_group_type import (
    OptionalTensorGroupType,
)
from isaacteleop.retargeters import (
    EePoseRateLimiter,
    JointRateLimiter,
    RateLimiterConfig,
)
from isaacteleop.retargeters.rate_limiter import (
    EE_POSE_KEY,
    JOINT_TARGETS_KEY,
    _clamp_orientation_step,
    _clamp_position_step,
    _clamped_dt,
)

# ---------------------------------------------------------------------------
# Helpers (mirror the patterns in test_so101_retargeters.py)
# ---------------------------------------------------------------------------

_ID_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
_NS = 1_000_000_000


def _make_context(
    *,
    reset: bool = False,
    state: ExecutionState = ExecutionState.RUNNING,
    t_ns: int = 0,
) -> ComputeContext:
    """Build a ComputeContext with the given reset flag, state, and timestamp."""
    return ComputeContext(
        graph_time=GraphTime(sim_time_ns=t_ns, real_time_ns=t_ns),
        execution_events=ExecutionEvents(reset=reset, execution_state=state),
    )


def _build_io(retargeter):
    """Construct empty input/output containers for a retargeter (optionals start absent)."""
    inputs = {}
    for k, v in retargeter.input_spec().items():
        inputs[k] = (
            OptionalTensorGroup(v)
            if isinstance(v, OptionalTensorGroupType)
            else TensorGroup(v)
        )
    outputs = {}
    for k, v in retargeter.output_spec().items():
        outputs[k] = (
            OptionalTensorGroup(v)
            if isinstance(v, OptionalTensorGroupType)
            else TensorGroup(v)
        )
    return inputs, outputs


def _pose_group(spec_type, pos, ori=_ID_QUAT) -> TensorGroup:
    """Build a present 7-D ee_pose TensorGroup from a position and quaternion."""
    tg = TensorGroup(spec_type.inner_type)
    tg[0] = np.concatenate(
        [np.asarray(pos, dtype=np.float32), np.asarray(ori, dtype=np.float32)]
    )
    return tg


def _set_pose_input(limiter, inputs, pos, ori=_ID_QUAT) -> None:
    """Replace the limiter's ee_pose input with a present pose group."""
    spec = limiter.input_spec()[EE_POSE_KEY]
    inputs[EE_POSE_KEY] = _pose_group(spec, pos, ori)


def _set_joint_input(limiter, inputs, values) -> None:
    """Replace the limiter's joint_targets input with a present group."""
    spec = limiter.input_spec()[JOINT_TARGETS_KEY]
    tg = TensorGroup(spec.inner_type)
    for i, v in enumerate(values):
        tg[i] = float(v)
    inputs[JOINT_TARGETS_KEY] = tg


def _read_pose(outputs) -> np.ndarray:
    """Read the 7-D ee_pose output as a numpy array."""
    return np.asarray(np.from_dlpack(outputs[EE_POSE_KEY][0]), dtype=np.float64)


def _read_joints(outputs, n: int) -> np.ndarray:
    """Read the joint_targets output as a numpy array."""
    return np.array(
        [float(outputs[JOINT_TARGETS_KEY][i]) for i in range(n)], dtype=np.float64
    )


def _quat_xyzw(axis, angle_rad: float) -> np.ndarray:
    """Build an [x, y, z, w] quaternion for a rotation of ``angle_rad`` about ``axis``."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = 0.5 * angle_rad
    xyz = axis * math.sin(half)
    return np.array([xyz[0], xyz[1], xyz[2], math.cos(half)], dtype=np.float64)


def _quat_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic angle [rad] between two unit quaternions (double-cover aware)."""
    dot = abs(float(np.dot(a, b)))
    return 2.0 * math.acos(min(1.0, dot))


# ===========================================================================
# Pure math helpers
# ===========================================================================


class TestClampedDt:
    """The ``_clamped_dt`` frame-delta helper."""

    def test_no_previous_uses_nominal(self):
        """With no previous stamp the nominal dt is returned."""
        assert _clamped_dt(None, 123, 0.02, 1e-4, 0.1) == pytest.approx(0.02)

    def test_duplicate_stamp_uses_nominal(self):
        """A non-advancing clock falls back to the nominal dt (no frozen limiter)."""
        assert _clamped_dt(500, 500, 0.02, 1e-4, 0.1) == pytest.approx(0.02)
        assert _clamped_dt(500, 400, 0.02, 1e-4, 0.1) == pytest.approx(0.02)

    def test_normal_delta_passes(self):
        """A delta within [min_dt, max_dt] is returned as-is."""
        assert _clamped_dt(0, 20_000_000, 0.02, 1e-4, 0.1) == pytest.approx(0.02)

    def test_stall_clamps_to_max(self):
        """A multi-second stall is clamped to max_dt (no huge authorized step)."""
        assert _clamped_dt(0, 5 * _NS, 0.02, 1e-4, 0.1) == pytest.approx(0.1)

    def test_tiny_delta_clamps_to_min(self):
        """A near-zero delta is clamped up to min_dt."""
        assert _clamped_dt(0, 10, 0.02, 1e-4, 0.1) == pytest.approx(1e-4)


class TestClampPositionStep:
    """The ``_clamp_position_step`` Euclidean step clamp."""

    def test_within_limit_passes_through(self):
        """A step under the limit is returned untouched."""
        prev = np.zeros(3)
        tgt = np.array([0.001, 0.0, 0.0])
        out = _clamp_position_step(tgt, prev, 0.01)
        np.testing.assert_allclose(out, tgt)

    def test_over_limit_is_clamped_along_line(self):
        """An over-limit step lands exactly max_step along the straight line."""
        prev = np.zeros(3)
        tgt = np.array([1.0, 0.0, 0.0])
        out = _clamp_position_step(tgt, prev, 0.01)
        np.testing.assert_allclose(out, [0.01, 0.0, 0.0], atol=1e-12)

    def test_zero_step_is_stable(self):
        """target == previous returns target (no divide-by-zero)."""
        p = np.array([0.2, 0.1, 0.3])
        out = _clamp_position_step(p.copy(), p.copy(), 0.01)
        np.testing.assert_allclose(out, p)


class TestClampOrientationStep:
    """The ``_clamp_orientation_step`` geodesic rotation clamp."""

    def test_within_limit_passes_through(self):
        """A rotation under the limit reaches the target orientation."""
        tgt = _quat_xyzw([0, 0, 1], 0.01)
        out = _clamp_orientation_step(tgt, _ID_QUAT.copy(), 0.1)
        assert _quat_angle(out, tgt) == pytest.approx(0.0, abs=1e-9)

    def test_over_limit_advances_exactly_max_step(self):
        """An over-limit rotation advances exactly max_step along the arc."""
        tgt = _quat_xyzw([0, 0, 1], 1.0)
        out = _clamp_orientation_step(tgt, _ID_QUAT.copy(), 0.1)
        assert _quat_angle(out, _ID_QUAT) == pytest.approx(0.1, abs=1e-9)

    def test_double_cover_takes_shortest_arc(self):
        """A sign-flipped target quaternion still rotates along the shortest arc."""
        tgt = -_quat_xyzw([0, 0, 1], 1.0)  # same rotation, other hemisphere
        out = _clamp_orientation_step(tgt, _ID_QUAT.copy(), 0.1)
        assert _quat_angle(out, _ID_QUAT) == pytest.approx(0.1, abs=1e-9)

    def test_identity_step_is_stable(self):
        """target == previous is returned without axis extraction blowing up."""
        q = _quat_xyzw([0, 1, 0], 0.3)
        out = _clamp_orientation_step(q.copy(), q.copy(), 0.1)
        # acos-based angle recovery amplifies float rounding near 0; 1e-6 rad is
        # far below any meaningful command resolution.
        assert _quat_angle(out, q) == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# RateLimiterConfig validation
# ===========================================================================


class TestRateLimiterConfig:
    """Constructor validation of the shared config."""

    def test_defaults_are_valid(self):
        """The default config constructs without error."""
        RateLimiterConfig()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_linear_velocity": 0.0},
            {"max_angular_velocity": -1.0},
            {"max_joint_velocity": 0.0},
            {"joint_velocity_overrides": {"elbow": 0.0}},
            {"min_dt": 0.0},
            {"min_dt": 0.05, "nominal_dt": 0.02},
            {"nominal_dt": 0.2, "max_dt": 0.1},
        ],
    )
    def test_invalid_values_raise(self, kwargs):
        """Non-positive limits and inconsistent dt bounds are rejected."""
        with pytest.raises(ValueError):
            RateLimiterConfig(**kwargs)


# ===========================================================================
# EePoseRateLimiter
# ===========================================================================


class TestEePoseRateLimiter:
    """End-to-end ``compute`` behavior of the EE-pose limiter."""

    def _limiter(self, **cfg_kwargs) -> EePoseRateLimiter:
        cfg = RateLimiterConfig(**cfg_kwargs) if cfg_kwargs else RateLimiterConfig()
        return EePoseRateLimiter(name="ee_limiter", config=cfg)

    def test_first_frame_passes_through(self):
        """The first valid frame latches and is emitted unclamped."""
        r = self._limiter()
        inputs, outputs = _build_io(r)
        _set_pose_input(r, inputs, [0.3, 0.1, 0.2])
        r.compute(inputs, outputs, _make_context(t_ns=0))
        np.testing.assert_allclose(_read_pose(outputs)[:3], [0.3, 0.1, 0.2], atol=1e-6)

    def test_slow_motion_is_untouched(self):
        """Motion under the velocity limit is emitted exactly (no lag)."""
        r = self._limiter(max_linear_velocity=0.25)
        inputs, outputs = _build_io(r)
        _set_pose_input(r, inputs, [0.2, 0.0, 0.1])
        r.compute(inputs, outputs, _make_context(t_ns=0))

        # 1 mm in 20 ms = 0.05 m/s, well under the 0.25 m/s limit.
        _set_pose_input(r, inputs, [0.201, 0.0, 0.1])
        r.compute(inputs, outputs, _make_context(t_ns=20_000_000))
        np.testing.assert_allclose(
            _read_pose(outputs)[:3], [0.201, 0.0, 0.1], atol=1e-6
        )

    def test_position_jump_is_rate_limited(self):
        """A teleport-sized position jump advances only max_linear_velocity * dt."""
        r = self._limiter(max_linear_velocity=0.25)
        inputs, outputs = _build_io(r)
        _set_pose_input(r, inputs, [0.0, 0.0, 0.0])
        r.compute(inputs, outputs, _make_context(t_ns=0))

        # 0.5 m jump in 20 ms -> clamp to 0.25 * 0.02 = 5 mm.
        _set_pose_input(r, inputs, [0.5, 0.0, 0.0])
        r.compute(inputs, outputs, _make_context(t_ns=20_000_000))
        np.testing.assert_allclose(
            _read_pose(outputs)[:3], [0.005, 0.0, 0.0], atol=1e-6
        )

    def test_persistent_target_is_approached_at_bounded_speed(self):
        """A held far target is approached stepwise, never faster than the limit."""
        r = self._limiter(max_linear_velocity=0.25)
        inputs, outputs = _build_io(r)
        _set_pose_input(r, inputs, [0.0, 0.0, 0.0])
        r.compute(inputs, outputs, _make_context(t_ns=0))

        last = np.zeros(3)
        for k in range(1, 6):
            _set_pose_input(r, inputs, [0.5, 0.0, 0.0])
            r.compute(inputs, outputs, _make_context(t_ns=k * 20_000_000))
            pos = _read_pose(outputs)[:3]
            step = float(np.linalg.norm(pos - last))
            assert step <= 0.25 * 0.02 + 1e-9
            last = pos
        # Monotonic progress toward the target.
        assert last[0] == pytest.approx(5 * 0.005, abs=1e-6)

    def test_orientation_jump_is_rate_limited(self):
        """A large orientation flip advances only max_angular_velocity * dt."""
        r = self._limiter(max_angular_velocity=1.5)
        inputs, outputs = _build_io(r)
        _set_pose_input(r, inputs, [0.2, 0.0, 0.1], _ID_QUAT)
        r.compute(inputs, outputs, _make_context(t_ns=0))

        # pi/2 flip in 20 ms -> clamp to 1.5 * 0.02 = 0.03 rad.
        tgt = _quat_xyzw([0, 0, 1], math.pi / 2)
        _set_pose_input(r, inputs, [0.2, 0.0, 0.1], tgt)
        r.compute(inputs, outputs, _make_context(t_ns=20_000_000))
        emitted = _read_pose(outputs)[3:7]
        # The emitted pose is float32; angle recovery through acos loses a few
        # ulps, so compare at 1e-4 rad (~0.006 deg), far below command resolution.
        assert _quat_angle(emitted, _ID_QUAT) == pytest.approx(0.03, abs=1e-4)

    def test_stall_does_not_authorize_teleport(self):
        """A 5 s pipeline stall authorizes at most max_linear_velocity * max_dt."""
        r = self._limiter(max_linear_velocity=0.25, max_dt=0.1)
        inputs, outputs = _build_io(r)
        _set_pose_input(r, inputs, [0.0, 0.0, 0.0])
        r.compute(inputs, outputs, _make_context(t_ns=0))

        _set_pose_input(r, inputs, [1.0, 0.0, 0.0])
        r.compute(inputs, outputs, _make_context(t_ns=5 * _NS))
        # 0.25 m/s * 0.1 s = 25 mm, not 1 m.
        np.testing.assert_allclose(
            _read_pose(outputs)[:3], [0.025, 0.0, 0.0], atol=1e-6
        )

    def test_dropped_frame_holds_last(self):
        """An absent input frame re-emits the last limited pose."""
        r = self._limiter()
        inputs, outputs = _build_io(r)
        _set_pose_input(r, inputs, [0.3, 0.1, 0.2])
        r.compute(inputs, outputs, _make_context(t_ns=0))

        inputs2, outputs2 = _build_io(r)  # optionals start absent
        r.compute(inputs2, outputs2, _make_context(t_ns=20_000_000))
        np.testing.assert_allclose(_read_pose(outputs2)[:3], [0.3, 0.1, 0.2], atol=1e-6)

    def test_reset_relatches_without_clamping(self):
        """After a reset the next frame passes through instead of slewing from the old pose."""
        r = self._limiter(max_linear_velocity=0.25)
        inputs, outputs = _build_io(r)
        _set_pose_input(r, inputs, [0.0, 0.0, 0.0])
        r.compute(inputs, outputs, _make_context(t_ns=0))

        # Reset, then a far-away pose (the task has physically reset the arm).
        _set_pose_input(r, inputs, [0.4, 0.2, 0.1])
        r.compute(inputs, outputs, _make_context(reset=True, t_ns=20_000_000))
        np.testing.assert_allclose(_read_pose(outputs)[:3], [0.4, 0.2, 0.1], atol=1e-6)

    def test_degenerate_orientation_holds_previous(self):
        """A zero quaternion in the target holds the last emitted orientation."""
        r = self._limiter()
        inputs, outputs = _build_io(r)
        start_ori = _quat_xyzw([0, 1, 0], 0.2)
        _set_pose_input(r, inputs, [0.2, 0.0, 0.1], start_ori)
        r.compute(inputs, outputs, _make_context(t_ns=0))

        _set_pose_input(r, inputs, [0.2, 0.0, 0.1], np.zeros(4))
        r.compute(inputs, outputs, _make_context(t_ns=20_000_000))
        emitted = _read_pose(outputs)[3:7]
        assert _quat_angle(emitted, start_ori) == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# JointRateLimiter
# ===========================================================================

_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


class TestJointRateLimiter:
    """End-to-end ``compute`` behavior of the joint-space limiter."""

    def _limiter(self, **cfg_kwargs) -> JointRateLimiter:
        cfg = RateLimiterConfig(**cfg_kwargs) if cfg_kwargs else RateLimiterConfig()
        return JointRateLimiter(name="joint_limiter", joint_names=_JOINTS, config=cfg)

    def test_empty_joint_names_raises(self):
        """An empty joint list is rejected at construction."""
        with pytest.raises(ValueError):
            JointRateLimiter(name="bad", joint_names=[])

    def test_unknown_override_raises(self):
        """A velocity override for a joint not in joint_names is rejected."""
        with pytest.raises(ValueError):
            JointRateLimiter(
                name="bad",
                joint_names=["a"],
                config=RateLimiterConfig(joint_velocity_overrides={"zz": 1.0}),
            )

    def test_first_frame_passes_through(self):
        """The first valid frame latches and is emitted unclamped."""
        r = self._limiter()
        inputs, outputs = _build_io(r)
        _set_joint_input(r, inputs, [0.1, -0.5, 1.2, 0.0, 0.3])
        r.compute(inputs, outputs, _make_context(t_ns=0))
        np.testing.assert_allclose(
            _read_joints(outputs, 5), [0.1, -0.5, 1.2, 0.0, 0.3], atol=1e-6
        )

    def test_slow_motion_is_untouched(self):
        """Per-joint motion under the limit is emitted exactly."""
        r = self._limiter(max_joint_velocity=1.5)
        inputs, outputs = _build_io(r)
        _set_joint_input(r, inputs, [0.0] * 5)
        r.compute(inputs, outputs, _make_context(t_ns=0))

        # 0.01 rad in 20 ms = 0.5 rad/s, under the 1.5 rad/s limit.
        _set_joint_input(r, inputs, [0.01] * 5)
        r.compute(inputs, outputs, _make_context(t_ns=20_000_000))
        np.testing.assert_allclose(_read_joints(outputs, 5), [0.01] * 5, atol=1e-6)

    def test_jump_is_rate_limited_per_joint(self):
        """An over-limit jump advances each joint by at most limit * dt, signed."""
        r = self._limiter(max_joint_velocity=1.5)
        inputs, outputs = _build_io(r)
        _set_joint_input(r, inputs, [0.0] * 5)
        r.compute(inputs, outputs, _make_context(t_ns=0))

        # +-2 rad jumps in 20 ms -> clamp to +-1.5 * 0.02 = +-0.03 rad.
        _set_joint_input(r, inputs, [2.0, -2.0, 2.0, -2.0, 2.0])
        r.compute(inputs, outputs, _make_context(t_ns=20_000_000))
        np.testing.assert_allclose(
            _read_joints(outputs, 5), [0.03, -0.03, 0.03, -0.03, 0.03], atol=1e-6
        )

    def test_per_joint_override_applies(self):
        """A per-joint override clamps that joint tighter than the default."""
        r = self._limiter(
            max_joint_velocity=1.5,
            joint_velocity_overrides={"shoulder_lift": 0.5},
        )
        inputs, outputs = _build_io(r)
        _set_joint_input(r, inputs, [0.0] * 5)
        r.compute(inputs, outputs, _make_context(t_ns=0))

        _set_joint_input(r, inputs, [2.0] * 5)
        r.compute(inputs, outputs, _make_context(t_ns=20_000_000))
        joints = _read_joints(outputs, 5)
        assert joints[1] == pytest.approx(0.5 * 0.02, abs=1e-6)  # overridden
        assert joints[0] == pytest.approx(1.5 * 0.02, abs=1e-6)  # default

    def test_dropped_frame_holds_last(self):
        """An absent input frame re-emits the last limited targets."""
        r = self._limiter()
        inputs, outputs = _build_io(r)
        _set_joint_input(r, inputs, [0.1, 0.2, 0.3, 0.4, 0.5])
        r.compute(inputs, outputs, _make_context(t_ns=0))

        inputs2, outputs2 = _build_io(r)  # optionals start absent
        r.compute(inputs2, outputs2, _make_context(t_ns=20_000_000))
        np.testing.assert_allclose(
            _read_joints(outputs2, 5), [0.1, 0.2, 0.3, 0.4, 0.5], atol=1e-6
        )

    def test_reset_relatches_without_clamping(self):
        """After a reset the next frame passes through (fresh episode, fresh baseline)."""
        r = self._limiter(max_joint_velocity=1.5)
        inputs, outputs = _build_io(r)
        _set_joint_input(r, inputs, [0.0] * 5)
        r.compute(inputs, outputs, _make_context(t_ns=0))

        _set_joint_input(r, inputs, [1.0, -1.0, 0.5, -0.5, 0.2])
        r.compute(inputs, outputs, _make_context(reset=True, t_ns=20_000_000))
        np.testing.assert_allclose(
            _read_joints(outputs, 5), [1.0, -1.0, 0.5, -0.5, 0.2], atol=1e-6
        )

    def test_stall_does_not_authorize_jump(self):
        """A long stall authorizes at most limit * max_dt per joint."""
        r = self._limiter(max_joint_velocity=1.5, max_dt=0.1)
        inputs, outputs = _build_io(r)
        _set_joint_input(r, inputs, [0.0] * 5)
        r.compute(inputs, outputs, _make_context(t_ns=0))

        _set_joint_input(r, inputs, [3.0] * 5)
        r.compute(inputs, outputs, _make_context(t_ns=10 * _NS))
        np.testing.assert_allclose(_read_joints(outputs, 5), [0.15] * 5, atol=1e-6)


# ===========================================================================
# Startup homing window
# ===========================================================================

_HOME_POSE = np.array([0.22, 0.0, 0.12, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
_HOME_JOINTS = [0.0, -0.3, 0.6, 0.1, 0.0]


class TestEePoseStartupHoming:
    """The startup homing window of the EE-pose limiter."""

    def _limiter(self, duration_s: float = 5.0, **cfg_kwargs) -> EePoseRateLimiter:
        cfg = RateLimiterConfig(startup_duration_s=duration_s, **cfg_kwargs)
        return EePoseRateLimiter(name="ee_homing", config=cfg, home_pose=_HOME_POSE)

    def test_homing_requires_home_pose(self):
        """startup_duration_s > 0 without a home_pose is rejected."""
        with pytest.raises(ValueError):
            EePoseRateLimiter(
                name="bad", config=RateLimiterConfig(startup_duration_s=5.0)
            )

    def test_degenerate_home_orientation_rejected(self):
        """A zero home quaternion is rejected at construction."""
        with pytest.raises(ValueError):
            EePoseRateLimiter(
                name="bad",
                config=RateLimiterConfig(startup_duration_s=5.0),
                home_pose=np.zeros(7),
            )

    def test_negative_duration_rejected(self):
        """A negative startup duration is rejected by the config."""
        with pytest.raises(ValueError):
            RateLimiterConfig(startup_duration_s=-1.0)

    def test_input_ignored_during_window(self):
        """During the window the emitted pose tracks home, not the (wild) input."""
        r = self._limiter(duration_s=5.0, max_linear_velocity=0.25)
        inputs, outputs = _build_io(r)
        # Controller starts far away and keeps moving: must be ignored.
        _set_pose_input(r, inputs, [0.9, -0.5, 0.8])
        r.compute(inputs, outputs, _make_context(t_ns=0))
        np.testing.assert_allclose(_read_pose(outputs)[:3], _HOME_POSE[:3], atol=1e-6)

        _set_pose_input(r, inputs, [-0.9, 0.5, -0.8])
        r.compute(inputs, outputs, _make_context(t_ns=20_000_000))
        np.testing.assert_allclose(_read_pose(outputs)[:3], _HOME_POSE[:3], atol=1e-6)

    def test_after_window_input_is_tracked(self):
        """Past the window the limiter resumes normal (rate-limited) tracking."""
        r = self._limiter(duration_s=1.0, max_linear_velocity=0.25)
        inputs, outputs = _build_io(r)
        _set_pose_input(r, inputs, [0.9, 0.0, 0.0])
        r.compute(inputs, outputs, _make_context(t_ns=0))  # homing

        # t = 2 s > 1 s window: input close to home is tracked exactly.
        near = _HOME_POSE[:3] + np.array([0.001, 0.0, 0.0])
        _set_pose_input(r, inputs, near)
        r.compute(inputs, outputs, _make_context(t_ns=2 * _NS))
        np.testing.assert_allclose(_read_pose(outputs)[:3], near, atol=1e-6)

    def test_reset_rehomes_boundedly_from_last_command(self):
        """A reset re-arms the window and slews home from the last emitted pose."""
        r = self._limiter(duration_s=5.0, max_linear_velocity=0.25, max_dt=0.1)
        inputs, outputs = _build_io(r)
        r.compute(inputs, outputs, _make_context(t_ns=0))  # homing latch at home

        # Drive away after the window would matter -- simulate mid-task pose by
        # walking the limiter to a far pose past the window.
        far = _HOME_POSE[:3] + np.array([0.3, 0.0, 0.0])
        t = 6 * _NS
        for k in range(200):
            _set_pose_input(r, inputs, far)
            r.compute(inputs, outputs, _make_context(t_ns=t + k * 20_000_000))
        drifted = _read_pose(outputs)[:3]
        assert drifted[0] > _HOME_POSE[0] + 0.2  # actually moved away

        # Reset: homing re-arms; first frame steps TOWARD home from the last
        # command, bounded by max_linear_velocity * dt (no snap).
        _set_pose_input(r, inputs, far)  # controller still parked far away
        t2 = t + 300 * 20_000_000
        r.compute(inputs, outputs, _make_context(reset=True, t_ns=t2))
        after_reset = _read_pose(outputs)[:3]
        step = float(np.linalg.norm(after_reset - drifted))
        # float32 pose storage rounds the step by a few ulps; 1e-6 m slack.
        assert step <= 0.25 * 0.1 + 1e-6  # bounded first homing step
        # And it moved toward home, not toward the controller.
        assert abs(after_reset[0] - _HOME_POSE[0]) < abs(drifted[0] - _HOME_POSE[0])

    def test_homing_converges_to_home(self):
        """Within the window the emitted pose converges to home and stays there."""
        r = self._limiter(duration_s=5.0, max_linear_velocity=0.25)
        inputs, outputs = _build_io(r)
        r.compute(inputs, outputs, _make_context(t_ns=0))
        # Walk 3 s of frames with the controller somewhere wild.
        for k in range(1, 150):
            _set_pose_input(r, inputs, [0.9, -0.5, 0.8])
            r.compute(inputs, outputs, _make_context(t_ns=k * 20_000_000))
        np.testing.assert_allclose(_read_pose(outputs)[:3], _HOME_POSE[:3], atol=1e-6)


class TestJointStartupHoming:
    """The startup homing window of the joint-space limiter."""

    def _limiter(self, duration_s: float = 5.0, **cfg_kwargs) -> JointRateLimiter:
        cfg = RateLimiterConfig(startup_duration_s=duration_s, **cfg_kwargs)
        return JointRateLimiter(
            name="joint_homing",
            joint_names=_JOINTS,
            config=cfg,
            home_targets=_HOME_JOINTS,
        )

    def test_homing_requires_home_targets(self):
        """startup_duration_s > 0 without home_targets is rejected."""
        with pytest.raises(ValueError):
            JointRateLimiter(
                name="bad",
                joint_names=_JOINTS,
                config=RateLimiterConfig(startup_duration_s=5.0),
            )

    def test_home_targets_shape_checked(self):
        """home_targets with the wrong length is rejected."""
        with pytest.raises(ValueError):
            JointRateLimiter(
                name="bad",
                joint_names=_JOINTS,
                config=RateLimiterConfig(startup_duration_s=5.0),
                home_targets=[0.0, 1.0],
            )

    def test_input_ignored_during_window(self):
        """During the window the emitted targets track home, not the leader."""
        r = self._limiter(duration_s=5.0)
        inputs, outputs = _build_io(r)
        _set_joint_input(r, inputs, [2.0] * 5)  # leader parked somewhere wild
        r.compute(inputs, outputs, _make_context(t_ns=0))
        np.testing.assert_allclose(_read_joints(outputs, 5), _HOME_JOINTS, atol=1e-6)

    def test_after_window_input_is_tracked(self):
        """Past the window the limiter resumes normal (rate-limited) tracking."""
        r = self._limiter(duration_s=1.0, max_joint_velocity=1.5)
        inputs, outputs = _build_io(r)
        r.compute(inputs, outputs, _make_context(t_ns=0))  # homing

        near = [h + 0.01 for h in _HOME_JOINTS]
        _set_joint_input(r, inputs, near)
        r.compute(inputs, outputs, _make_context(t_ns=2 * _NS))
        np.testing.assert_allclose(_read_joints(outputs, 5), near, atol=1e-6)

    def test_reset_rehomes_boundedly(self):
        """A reset re-arms homing; the first step toward home is velocity-bounded."""
        r = self._limiter(duration_s=5.0, max_joint_velocity=1.5, max_dt=0.1)
        inputs, outputs = _build_io(r)
        r.compute(inputs, outputs, _make_context(t_ns=0))

        # Walk past the window to a far leader pose.
        t = 6 * _NS
        for k in range(100):
            _set_joint_input(r, inputs, [2.0] * 5)
            r.compute(inputs, outputs, _make_context(t_ns=t + k * 20_000_000))
        drifted = _read_joints(outputs, 5)
        assert drifted[0] > 1.0  # actually moved away from home

        # Reset with the leader still far away: step must go TOWARD home,
        # bounded by max_joint_velocity * max_dt.
        _set_joint_input(r, inputs, [2.0] * 5)
        r.compute(inputs, outputs, _make_context(reset=True, t_ns=t + 200 * 20_000_000))
        after = _read_joints(outputs, 5)
        np.testing.assert_array_less(np.abs(after - drifted), 1.5 * 0.1 + 1e-9)
        assert abs(after[0] - _HOME_JOINTS[0]) < abs(drifted[0] - _HOME_JOINTS[0])

    def test_homing_converges_to_home(self):
        """Within the window the emitted targets converge to home and stay."""
        r = self._limiter(duration_s=5.0, max_joint_velocity=1.5)
        inputs, outputs = _build_io(r)
        r.compute(inputs, outputs, _make_context(t_ns=0))
        for k in range(1, 150):
            _set_joint_input(r, inputs, [2.0] * 5)
            r.compute(inputs, outputs, _make_context(t_ns=k * 20_000_000))
        np.testing.assert_allclose(_read_joints(outputs, 5), _HOME_JOINTS, atol=1e-6)


class TestPlayEdgeRehoming:
    """Every transition into RUNNING (the client's Play) re-arms the homing window."""

    def _ee_limiter(self, duration_s: float = 2.0) -> EePoseRateLimiter:
        cfg = RateLimiterConfig(
            startup_duration_s=duration_s, max_linear_velocity=0.25, max_dt=0.1
        )
        return EePoseRateLimiter(name="ee_play", config=cfg, home_pose=_HOME_POSE)

    def _run_past_window(self, r, inputs, outputs, t0_ns: int) -> int:
        """Walk the limiter past its homing window to a far pose; returns end time."""
        t = t0_ns
        far = _HOME_POSE[:3] + np.array([0.3, 0.0, 0.0])
        for k in range(300):
            t = t0_ns + k * 20_000_000
            _set_pose_input(r, inputs, far)
            r.compute(inputs, outputs, _make_context(t_ns=t))
        return t

    def test_pause_then_play_rehomes(self):
        """PAUSED -> RUNNING re-arms homing: the arm returns toward home on Play."""
        r = self._ee_limiter(duration_s=2.0)
        inputs, outputs = _build_io(r)
        r.compute(inputs, outputs, _make_context(t_ns=0))  # first frame: homing
        t = self._run_past_window(r, inputs, outputs, 3 * _NS)
        drifted = _read_pose(outputs)[:3]
        assert drifted[0] > _HOME_POSE[0] + 0.2

        # Pause (state != RUNNING): the operator parks; the held command stays at
        # the drifted pose. Then Play again with the controller far away.
        _set_pose_input(r, inputs, drifted)
        r.compute(
            inputs, outputs, _make_context(state=ExecutionState.PAUSED, t_ns=t + _NS)
        )
        _set_pose_input(r, inputs, drifted + np.array([0.2, 0.0, 0.0]))
        r.compute(inputs, outputs, _make_context(t_ns=t + 2 * _NS))  # Play edge
        after_play = _read_pose(outputs)[:3]
        # Moved toward home (input ignored), by a bounded step.
        assert abs(after_play[0] - _HOME_POSE[0]) < abs(drifted[0] - _HOME_POSE[0])
        assert np.linalg.norm(after_play - drifted) <= 0.25 * 0.1 + 1e-6

    def test_stop_then_play_rehomes_joint(self):
        """STOPPED -> RUNNING re-arms homing on the joint limiter too."""
        cfg = RateLimiterConfig(
            startup_duration_s=2.0, max_joint_velocity=1.5, max_dt=0.1
        )
        r = JointRateLimiter(
            name="joint_play",
            joint_names=_JOINTS,
            config=cfg,
            home_targets=_HOME_JOINTS,
        )
        inputs, outputs = _build_io(r)
        r.compute(inputs, outputs, _make_context(t_ns=0))
        # Past the window, drift to a far pose.
        t = 3 * _NS
        for k in range(300):
            t = 3 * _NS + k * 20_000_000
            _set_joint_input(r, inputs, [2.0] * 5)
            r.compute(inputs, outputs, _make_context(t_ns=t))
        drifted = _read_joints(outputs, 5)
        assert drifted[0] > 1.0

        _set_joint_input(r, inputs, [2.0] * 5)
        r.compute(
            inputs, outputs, _make_context(state=ExecutionState.STOPPED, t_ns=t + _NS)
        )
        _set_joint_input(r, inputs, [2.0] * 5)
        r.compute(inputs, outputs, _make_context(t_ns=t + 2 * _NS))  # Play edge
        after = _read_joints(outputs, 5)
        assert abs(after[0] - _HOME_JOINTS[0]) < abs(drifted[0] - _HOME_JOINTS[0])
        np.testing.assert_array_less(np.abs(after - drifted), 1.5 * 0.1 + 1e-6)

    def test_no_rehome_while_continuously_running(self):
        """Staying in RUNNING (no edge) does NOT re-arm homing mid-task."""
        r = self._ee_limiter(duration_s=1.0)
        inputs, outputs = _build_io(r)
        r.compute(inputs, outputs, _make_context(t_ns=0))
        t = self._run_past_window(r, inputs, outputs, 2 * _NS)
        drifted = _read_pose(outputs)[:3]
        assert drifted[0] > _HOME_POSE[0] + 0.2

        # Next RUNNING frame with a nearby target: normal tracking, no homing.
        near = drifted + np.array([0.001, 0.0, 0.0])
        _set_pose_input(r, inputs, near)
        r.compute(inputs, outputs, _make_context(t_ns=t + 20_000_000))
        np.testing.assert_allclose(_read_pose(outputs)[:3], near, atol=1e-6)

    def test_play_edge_without_homing_keeps_plain_behavior(self):
        """A limiter without a homing window ignores Play edges (no state reset)."""
        cfg = RateLimiterConfig(max_linear_velocity=0.25)
        r = EePoseRateLimiter(name="plain", config=cfg)
        inputs, outputs = _build_io(r)
        _set_pose_input(r, inputs, [0.2, 0.0, 0.1])
        r.compute(inputs, outputs, _make_context(t_ns=0))
        # Pause then Play: baseline must be kept (still rate-limits from it).
        _set_pose_input(r, inputs, [0.2, 0.0, 0.1])
        r.compute(inputs, outputs, _make_context(state=ExecutionState.PAUSED, t_ns=_NS))
        _set_pose_input(r, inputs, [0.9, 0.0, 0.1])  # far target on the Play frame
        r.compute(inputs, outputs, _make_context(t_ns=2 * _NS))
        pos = _read_pose(outputs)[:3]
        # Clamped from the old baseline, not a pass-through re-latch.
        assert pos[0] < 0.3
