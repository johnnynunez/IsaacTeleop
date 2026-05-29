#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Teleop ROS2 Reference Node.

Publishes teleoperation data over ROS2 topics using isaacteleop TeleopSession.
The `mode` parameter selects the teleoperation scenario and which topics are
published:

  - controller_teleop (default): ee_poses (from controller aim pose), root_twist,
                       root_pose, finger_joints (retargeted TriHand angles),
                       controller_data, and TF transforms for left/right wrists
  - hand_teleop: ee_poses (from hand tracking wrist), hand (finger joint poses),
                 finger_joints (retargeted Sharpa joint angles),
                 root_twist/root_pose (from foot pedal locomotion), and TF
                 transforms for left/right wrists
  - controller_raw: controller_data only
  - full_body: full_body and controller_data

Topic names (remappable via ROS 2 remapping):
  - xr_teleop/hand (PoseArray): [finger_joint_poses...]
  - xr_teleop/ee_poses (PoseArray): [left_ee, right_ee]
  - xr_teleop/root_twist (TwistStamped): root velocity command
  - xr_teleop/root_pose (PoseStamped): root pose command (height only)
  - xr_teleop/controller_data (ByteMultiArray): msgpack-encoded controller data
  - xr_teleop/full_body (ByteMultiArray): msgpack-encoded full body tracking data
  - xr_teleop/finger_joints (JointState): retargeted finger joint angles

TF frames published in hand_teleop and controller_teleop modes (configurable via parameters):
  - world_frame -> right_wrist_frame
  - world_frame -> left_wrist_frame
"""

import os
import signal
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path
from typing import List

import msgpack
import msgpack_numpy as mnp
import numpy as np
import rclpy
from geometry_msgs.msg import (
    Pose,
    PoseArray,
    PoseStamped,
    TransformStamped,
    TwistStamped,
)
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import ByteMultiArray
from tf2_ros import TransformBroadcaster

from isaacteleop.cloudxr import CloudXRLauncher
from isaacteleop.teleop_session_manager import TeleopSession
from geometry import make_transform
from messages import (
    build_controller_payload,
    build_ee_msg_from_controllers,
    build_ee_msg_from_hands,
    build_finger_joints_msg,
    build_full_body_payload,
    build_hand_msg_from_hands,
    controller_aim_is_valid,
    hand_wrist_is_valid,
)
from node_parameters import (
    NodeParameters,
    create_node_parameters,
)
from session_config import build_session_config


# Streaming SDK TCP port that CloudXR binds. The bundled launcher's
# _cleanup_stale_runtime only handles the IPC unix socket, so a zombie that
# released the socket but still holds this port slips through and the next
# start dies with ERROR_STREAMSDK_PORT_UNAVAILABLE.
_CLOUDXR_STREAMSDK_PORT = 49100


def _kill_zombie_cloudxr_runtime(logger, install_dir: Path) -> None:
    """Kill any leftover CloudXR runtime holding the streaming SDK port.

    Only kills processes whose ``/proc/<pid>/cmdline`` mentions cloudxr or
    isaacteleop, so an unrelated service that happens to bind 49100 is left
    alone. SIGTERM first, then SIGKILL after a 2 s grace period. Finally
    removes the IPC sentinel files under ``install_dir/run`` so the next
    launch starts from a clean slate.
    """
    try:
        out = subprocess.run(
            ["lsof", "-ti", f"tcp:{_CLOUDXR_STREAMSDK_PORT}"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        pids = sorted({int(p) for p in out.stdout.split() if p.strip().isdigit()})
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pids = []

    for pid in pids:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = f.read().replace(b"\x00", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError):
            cmdline = ""
        low = cmdline.lower()
        if "cloudxr" not in low and "isaacteleop" not in low:
            logger.warn(
                f"CloudXR streaming port {_CLOUDXR_STREAMSDK_PORT} held by "
                f"non-cloudxr pid={pid} ({cmdline[:80]!r}); leaving it alone"
            )
            continue
        logger.info(f"Killing stale CloudXR runtime pid={pid}")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        for _ in range(20):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    run_dir = install_dir / "run"
    for name in ("ipc_cloudxr", "runtime_started", "monado.pid", "cloudxr.pid"):
        try:
            (run_dir / name).unlink()
        except FileNotFoundError:
            pass


class TeleopRos2Node(Node):
    """ROS 2 node that publishes teleop data."""

    def __init__(self) -> None:
        super().__init__("teleop_ros2_node")
        self._params: NodeParameters = create_node_parameters(self)
        self._tf_broadcaster = TransformBroadcaster(self)
        self._create_publishers()
        self._config = build_session_config(self._params)

    def _build_wrist_tfs(
        self,
        ee_msg: PoseArray,
        *,
        right_available: bool,
        left_available: bool,
        now,
    ) -> List[TransformStamped]:
        """Build wrist TF transforms from a pre-built ee_poses PoseArray (left pose at index 0, right at index 1)."""
        tfs = []

        def _get_orientation(pose: Pose) -> List[float]:
            return [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]

        if left_available:
            pose = ee_msg.poses[0]
            tfs.append(
                make_transform(
                    now,
                    self._params.world_frame,
                    self._params.left_wrist_frame,
                    [pose.position.x, pose.position.y, pose.position.z],
                    _get_orientation(pose),
                )
            )
        if right_available:
            pose = ee_msg.poses[1]
            tfs.append(
                make_transform(
                    now,
                    self._params.world_frame,
                    self._params.right_wrist_frame,
                    [pose.position.x, pose.position.y, pose.position.z],
                    _get_orientation(pose),
                )
            )
        return tfs

    def _create_publishers(self) -> None:
        self._pub_hand = self.create_publisher(PoseArray, "xr_teleop/hand", 10)
        self._pub_ee_pose = self.create_publisher(PoseArray, "xr_teleop/ee_poses", 10)
        self._pub_root_twist = self.create_publisher(
            TwistStamped, "xr_teleop/root_twist", 10
        )
        self._pub_root_pose = self.create_publisher(
            PoseStamped, "xr_teleop/root_pose", 10
        )
        self._pub_controller = self.create_publisher(
            ByteMultiArray, "xr_teleop/controller_data", 10
        )
        self._pub_full_body = self.create_publisher(
            ByteMultiArray, "xr_teleop/full_body", 10
        )
        self._pub_finger_joints = self.create_publisher(
            JointState, "xr_teleop/finger_joints", 10
        )

    def _publish_controller_outputs(self, result: dict, now) -> None:
        left_ctrl = result["controller_left"]
        right_ctrl = result["controller_right"]
        ee_msg = build_ee_msg_from_controllers(
            left_ctrl,
            right_ctrl,
            now,
            self._params.world_frame,
            self._params.transform_rotation,
            self._params.transform_translation,
            self._params.controller_uses_hands_source,
        )
        if ee_msg.poses:
            self._pub_ee_pose.publish(ee_msg)
        wrist_tfs = self._build_wrist_tfs(
            ee_msg,
            right_available=controller_aim_is_valid(right_ctrl),
            left_available=controller_aim_is_valid(left_ctrl),
            now=now,
        )
        if wrist_tfs:
            self._tf_broadcaster.sendTransform(wrist_tfs)
        if self._params.controller_uses_hands_source:
            hand_msg = build_hand_msg_from_hands(
                result["hand_left"],
                result["hand_right"],
                now,
                self._params.world_frame,
                self._params.transform_rotation,
                self._params.transform_translation,
            )
            if hand_msg.poses:
                self._pub_hand.publish(hand_msg)

    def _publish_controller_payload(self, result: dict) -> None:
        if self._params.mode not in (
            "controller_raw",
            "controller_teleop",
            "full_body",
        ):
            return

        left_ctrl = result["controller_left"]
        right_ctrl = result["controller_right"]
        if left_ctrl.is_none and right_ctrl.is_none:
            return

        controller_payload = build_controller_payload(left_ctrl, right_ctrl)
        payload = msgpack.packb(controller_payload, default=mnp.encode)
        controller_msg = ByteMultiArray()
        controller_msg.data = tuple(bytes([a]) for a in payload)
        self._pub_controller.publish(controller_msg)

    def _publish_finger_joints(self, result: dict, now) -> None:
        if self._params.mode not in ("controller_teleop", "hand_teleop"):
            return

        finger_joints_msg = build_finger_joints_msg(
            result["finger_joints_left"],
            result["finger_joints_right"],
            now,
            self._params.world_frame,
        )
        if finger_joints_msg is not None:
            self._pub_finger_joints.publish(finger_joints_msg)

    def _publish_full_body_payload(self, result: dict) -> None:
        if self._params.mode != "full_body":
            return

        full_body_data = result["full_body"]
        if full_body_data.is_none:
            return

        body_payload = build_full_body_payload(full_body_data)
        payload = msgpack.packb(body_payload, default=mnp.encode)
        body_msg = ByteMultiArray()
        body_msg.data = tuple(bytes([a]) for a in payload)
        self._pub_full_body.publish(body_msg)

    def _publish_hand_tracking_outputs(self, result: dict, now) -> None:
        left_hand = result["hand_left"]
        right_hand = result["hand_right"]
        hand_msg = build_hand_msg_from_hands(
            left_hand,
            right_hand,
            now,
            self._params.world_frame,
            self._params.transform_rotation,
            self._params.transform_translation,
        )
        if hand_msg.poses:
            self._pub_hand.publish(hand_msg)

        ee_msg = build_ee_msg_from_hands(
            left_hand,
            right_hand,
            now,
            self._params.world_frame,
            self._params.transform_rotation,
            self._params.transform_translation,
        )
        if ee_msg.poses:
            self._pub_ee_pose.publish(ee_msg)
        wrist_tfs = self._build_wrist_tfs(
            ee_msg,
            right_available=hand_wrist_is_valid(right_hand),
            left_available=hand_wrist_is_valid(left_hand),
            now=now,
        )
        if wrist_tfs:
            self._tf_broadcaster.sendTransform(wrist_tfs)

    def _publish_root_command(self, result: dict, now) -> None:
        if self._params.mode not in ("hand_teleop", "controller_teleop"):
            return

        root_command = result["root_command"]
        if root_command.is_none:
            return

        cmd = np.asarray(root_command[0])
        twist_msg = TwistStamped()
        twist_msg.header.stamp = now
        twist_msg.header.frame_id = self._params.world_frame
        twist_msg.twist.linear.x = float(cmd[0])
        twist_msg.twist.linear.y = float(cmd[1])
        twist_msg.twist.linear.z = 0.0
        twist_msg.twist.angular.z = float(cmd[2])
        self._pub_root_twist.publish(twist_msg)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = self._params.world_frame
        pose_msg.pose.position.z = float(cmd[3])
        pose_msg.pose.orientation.w = 1.0
        self._pub_root_pose.publish(pose_msg)

    def _run_teleop_loop(self) -> int:
        while rclpy.ok():
            try:
                with TeleopSession(self._config) as session:
                    self.get_logger().info("TeleopSession started successfully")
                    while rclpy.ok():
                        result = session.step()

                        # Keep ROS time and other callbacks updated in this
                        # manual loop so stamped messages progress with /clock.
                        rclpy.spin_once(self, timeout_sec=0.0)

                        now = self.get_clock().now().to_msg()

                        if self._params.mode == "hand_teleop":
                            self._publish_hand_tracking_outputs(result, now)
                        elif self._params.mode == "controller_teleop":
                            self._publish_controller_outputs(result, now)

                        self._publish_root_command(result, now)
                        self._publish_finger_joints(result, now)
                        self._publish_controller_payload(result)
                        self._publish_full_body_payload(result)

                        time.sleep(self._params.sleep_period_s)
            except RuntimeError as e:
                if "Failed to get OpenXR system" not in str(e):
                    raise
                self.get_logger().warn(
                    f"No XR client connected ({e}), retrying in 2s..."
                )
                time.sleep(2.0)

        return 0

    def _start_cloudxr(self) -> CloudXRLauncher | None:
        """Spawn the CloudXR runtime + WSS proxy via :class:`CloudXRLauncher`.

        Returns ``None`` when ``cloudxr_enable`` is false (e.g. MCAP replay or
        an externally-managed runtime such as ``run_cloudxr_via_docker.sh``);
        otherwise blocks until the runtime signals readiness (~10 s).
        """
        if not self._params.cloudxr_enable:
            return None

        install_dir = Path(
            os.path.expandvars(os.path.expanduser(self._params.cloudxr_install_dir))
        )
        _kill_zombie_cloudxr_runtime(self.get_logger(), install_dir)

        self.get_logger().info(
            f"Starting CloudXR runtime via CloudXRLauncher (install_dir={install_dir})"
        )
        return CloudXRLauncher(
            install_dir=str(install_dir),
            env_config=self._params.cloudxr_env_config,
            accept_eula=self._params.cloudxr_accept_eula,
            setup_oob=self._params.cloudxr_use_adb,
            usb_local=self._params.cloudxr_use_adb,
        )

    def _stop_cloudxr(self, launcher: CloudXRLauncher) -> None:
        """Stop the launcher and log instead of raising on shutdown failures."""
        try:
            launcher.stop()
        except Exception as e:
            self.get_logger().error(f"CloudXRLauncher.stop() failed: {e}")

    def run(self) -> int:
        with ExitStack() as stack:
            cloudxr_launcher = self._start_cloudxr()
            if cloudxr_launcher is not None:
                # ExitStack.callback runs on normal exit *and* on exception,
                # mirroring the try/finally in isaac_teleop_server.activate().
                stack.callback(self._stop_cloudxr, cloudxr_launcher)
            return self._run_teleop_loop()


def main() -> int:
    rclpy.init()
    node = None
    try:
        node = TeleopRos2Node()
        return node.run()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
