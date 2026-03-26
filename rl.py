#!/usr/bin/env python3

"""
Simplified demonstration replay using MoveIt Cartesian planning + execute.

This version keeps simulation/real mode detection so you can test in simulation
first, then run on hardware with the same script.
"""

import argparse
import math
import os
import sys

import numpy as np
import rospy
from geometry_msgs.msg import Pose
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown
from tf.transformations import quaternion_from_matrix


def detect_real_robot(robot_name, timeout=3.0):
    """Return True if xs_sdk service is reachable (real robot mode)."""
    service_name = f"/{robot_name}/get_robot_info"
    try:
        rospy.wait_for_service(service_name, timeout=timeout)
        return True
    except rospy.exceptions.ROSException:
        return False


class DemoReplayer:
    """Replay recorded EEF demonstrations through MoveIt."""

    def __init__(
        self,
        robot_model="wx250s",
        robot_name="wx250s",
        eef_step=0.01,
        jump_threshold=0.0,
        min_fraction=0.90,
        base_vel_scale=0.5,
        base_accel_scale=0.5,
        avoid_collisions=False,
    ):
        self.robot_model = robot_model
        self.robot_name = robot_name
        self.eef_step = eef_step
        self.jump_threshold = jump_threshold
        self.min_fraction = min_fraction
        self.base_vel_scale = base_vel_scale
        self.base_accel_scale = base_accel_scale
        self.avoid_collisions = avoid_collisions

        rospy.init_node("demo_replayer_moveit", anonymous=True)

        self.sim_mode = not detect_real_robot(robot_name)
        print("\nDetecting robot mode...")
        print("  Mode: SIMULATION" if self.sim_mode else "  Mode: REAL ROBOT")

        self._init_moveit()

    def _init_moveit(self):
        """Initialize MoveGroupCommander in the robot namespace."""
        roscpp_initialize(sys.argv)

        self.robot_description = f"/{self.robot_name}/robot_description"
        semantic_ns = f"/{self.robot_name}/robot_description_semantic"

        if not rospy.has_param(semantic_ns):
            print(f"Error: missing MoveIt semantic parameter: {semantic_ns}")
            print("Launch MoveIt first.")
            print("  Simulation:")
            print(
                "  roslaunch interbotix_xsarm_moveit xsarm_moveit.launch "
                f"robot_model:={self.robot_model} robot_name:={self.robot_name} "
                "use_fake:=true dof:=6"
            )
            print("  Real robot:")
            print(
                "  roslaunch interbotix_xsarm_moveit xsarm_moveit.launch "
                f"robot_model:={self.robot_model} robot_name:={self.robot_name} "
                "use_actual:=true dof:=6"
            )
            sys.exit(1)

        try:
            self.move_group = MoveGroupCommander(
                "interbotix_arm",
                robot_description=self.robot_description,
                ns=f"/{self.robot_name}",
            )
        except TypeError:
            # Compatibility with older MoveIt Python APIs.
            self.move_group = MoveGroupCommander(
                "interbotix_arm",
                robot_description=self.robot_description,
            )

        self.move_group.set_max_velocity_scaling_factor(self.base_vel_scale)
        self.move_group.set_max_acceleration_scaling_factor(self.base_accel_scale)
        self.move_group.set_planning_time(15.0)
        self.move_group.set_num_planning_attempts(15)
        self.move_group.set_goal_position_tolerance(0.01)
        self.move_group.set_goal_orientation_tolerance(0.15)
        try:
            self.move_group.allow_replanning(True)
        except Exception:
            pass

        print("  MoveIt initialized")
        print(f"  Planning frame: {self.move_group.get_planning_frame()}")
        print(f"  End effector:   {self.move_group.get_end_effector_link()}")

    def load_demo(self, demo_dir):
        """Load demonstration arrays from directory."""
        demo = {}

        eef_poses_path = os.path.join(demo_dir, "eef_poses.npy")
        twists_path = os.path.join(demo_dir, "demo_eef_twists.npy")
        timestamps_path = os.path.join(demo_dir, "timestamps.npy")

        if not os.path.exists(eef_poses_path):
            print(f"Error: {eef_poses_path} not found.")
            sys.exit(1)

        demo["eef_poses"] = np.load(eef_poses_path)
        demo["eef_twists"] = np.load(twists_path) if os.path.exists(twists_path) else None

        if os.path.exists(timestamps_path):
            demo["timestamps"] = np.load(timestamps_path)
        else:
            T = len(demo["eef_poses"])
            demo["timestamps"] = np.arange(T) / 30.0

        print(f"\nLoaded demo from: {demo_dir}")
        print(f"  Timesteps: {len(demo['eef_poses'])}")
        print(f"  Duration:  {demo['timestamps'][-1]:.2f}s")
        return demo

    @staticmethod
    def T_to_pose(T):
        """Convert a 4x4 SE(3) matrix to geometry_msgs/Pose."""
        pose = Pose()
        pose.position.x = float(T[0, 3])
        pose.position.y = float(T[1, 3])
        pose.position.z = float(T[2, 3])
        quat = quaternion_from_matrix(T)
        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])
        return pose

    @staticmethod
    def _build_indices(length, downsample):
        indices = list(range(0, length, downsample))
        if indices[-1] != length - 1:
            indices.append(length - 1)
        return indices

    @staticmethod
    def _filter_waypoints(waypoints, position_only=False):
        """Drop nearly-duplicate waypoints that can break Cartesian interpolation."""
        if len(waypoints) <= 1:
            return waypoints

        filtered = [waypoints[0]]
        pos_eps = 1e-4
        ang_eps = 1e-3  # radians

        for wp in waypoints[1:]:
            prev = filtered[-1]
            dp = math.sqrt(
                (wp.position.x - prev.position.x) ** 2 +
                (wp.position.y - prev.position.y) ** 2 +
                (wp.position.z - prev.position.z) ** 2
            )

            if position_only:
                changed = dp > pos_eps
            else:
                dot = (
                    prev.orientation.x * wp.orientation.x +
                    prev.orientation.y * wp.orientation.y +
                    prev.orientation.z * wp.orientation.z +
                    prev.orientation.w * wp.orientation.w
                )
                dot = max(min(abs(dot), 1.0), -1.0)
                dang = 2.0 * math.acos(dot)
                changed = (dp > pos_eps) or (dang > ang_eps)

            if changed:
                filtered.append(wp)

        return filtered

    @staticmethod
    def _adaptive_eef_step(waypoints, requested_step):
        """Choose an eef_step that matches waypoint spacing."""
        if len(waypoints) < 2:
            return requested_step

        dists = []
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            d = math.sqrt(
                (a.position.x - b.position.x) ** 2 +
                (a.position.y - b.position.y) ** 2 +
                (a.position.z - b.position.z) ** 2
            )
            if d > 1e-6:
                dists.append(d)

        if not dists:
            return requested_step

        dists = np.array(dists)
        # Keep interpolation finer than typical waypoint spacing.
        auto_step = max(5e-4, float(np.percentile(dists, 50)) * 0.8)
        return min(requested_step, auto_step)

    def _plan_cartesian(self, eef_poses, downsample=3, position_only=False):
        """Plan Cartesian path from EEF pose matrices."""
        if len(eef_poses) == 0:
            return None, 0.0, []

        indices = self._build_indices(len(eef_poses), downsample)
        waypoints = [self.T_to_pose(eef_poses[idx]) for idx in indices]

        if position_only and waypoints:
            # Keep a fixed orientation while following positional waypoints.
            qx = waypoints[0].orientation.x
            qy = waypoints[0].orientation.y
            qz = waypoints[0].orientation.z
            qw = waypoints[0].orientation.w
            for wp in waypoints:
                wp.orientation.x = qx
                wp.orientation.y = qy
                wp.orientation.z = qz
                wp.orientation.w = qw

        raw_count = len(waypoints)
        waypoints = self._filter_waypoints(waypoints, position_only=position_only)
        eef_step = self._adaptive_eef_step(waypoints, self.eef_step)

        if len(waypoints) < 2:
            print("  Warning: not enough unique waypoints after filtering.")
            return None, 0.0, []

        print(
            f"\nPlanning Cartesian path with {len(waypoints)} waypoints "
            f"(downsampled {downsample}x, "
            f"{'position-only' if position_only else 'full-pose'})..."
        )
        if raw_count != len(waypoints):
            print(f"  Filtered duplicate waypoints: {raw_count} -> {len(waypoints)}")
        print(f"  Effective eef_step: {eef_step:.6f} m")

        self.move_group.set_start_state_to_current_state()

        # MoveIt Python API differs across versions:
        # - some expect (waypoints, eef_step, jump_threshold)
        # - others expect (waypoints, eef_step, avoid_collisions)
        try:
            plan, fraction = self.move_group.compute_cartesian_path(
                waypoints,
                eef_step,
                self.jump_threshold,
            )
        except Exception:
            plan, fraction = self.move_group.compute_cartesian_path(
                waypoints,
                eef_step,
                self.avoid_collisions,
            )
            if abs(float(self.jump_threshold)) > 1e-12:
                print(
                    "  Note: this MoveIt version does not expose jump_threshold "
                    "in compute_cartesian_path; ignoring --jump_threshold."
                )

        n_points = len(plan.joint_trajectory.points)
        print(f"  Cartesian fraction: {fraction*100:.1f}%")
        print(f"  Planned points:     {n_points}")
        return plan, float(fraction), indices

    def _move_to_start_pose(self, start_pose):
        """Move to the first demo pose before Cartesian waypoint tracking."""
        print("\nMoving to trajectory start pose...")
        self.move_group.set_start_state_to_current_state()

        # Try full pose first.
        self.move_group.set_pose_target(start_pose)
        success = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        if success:
            print("  Reached start pose.")
            return True

        # Fallback: position-only goal (ignore orientation).
        print("  Full pose goal failed; trying position-only start alignment...")
        self.move_group.set_start_state_to_current_state()
        self.move_group.set_position_target([
            start_pose.position.x,
            start_pose.position.y,
            start_pose.position.z,
        ])
        success = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

        if success:
            print("  Reached start position.")
            return True

        print("  Failed to reach start pose/position.")
        return False

    def _set_speed(self, speed_factor):
        """Scale velocity/acceleration from base settings."""
        vel = min(max(self.base_vel_scale * speed_factor, 0.01), 1.0)
        acc = min(max(self.base_accel_scale * speed_factor, 0.01), 1.0)
        self.move_group.set_max_velocity_scaling_factor(vel)
        self.move_group.set_max_acceleration_scaling_factor(acc)

    def _execute_plan(self, plan, fraction, dry_run=False):
        """Execute planned trajectory if fraction is acceptable."""
        if fraction < self.min_fraction:
            print(
                f"Error: fraction too low ({fraction:.3f} < {self.min_fraction:.3f}). "
                "Not executing."
            )
            return False

        if dry_run:
            print("\n[DRY RUN] Plan computed. Not executing.")
            return True

        print("\nExecuting MoveIt plan...")
        success = self.move_group.execute(plan, wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

        if success:
            print("Done.")
        else:
            print("Execution failed.")
        return bool(success)

    def _execute_chunked_cartesian(self, eef_poses, chunk_size=35, dry_run=False):
        """Fallback: execute the trajectory as several Cartesian chunks."""
        if len(eef_poses) < 2:
            print("Error: not enough poses for chunked fallback.")
            return False

        print(
            f"\nChunked Cartesian fallback: {len(eef_poses)} poses, "
            f"chunk_size={chunk_size}"
        )

        if dry_run:
            print("[DRY RUN] Skipping chunked execution.")
            return True

        seg_idx = 0
        start = 0
        executed_segments = 0
        min_seg_fraction = max(0.20, self.min_fraction * 0.5)

        while start < len(eef_poses) - 1:
            end = min(start + chunk_size, len(eef_poses))
            seg = eef_poses[start:end]
            seg_idx += 1

            print(f"\n  Segment {seg_idx}: poses [{start}:{end}]")
            plan, fraction, _ = self._plan_cartesian(seg, downsample=1, position_only=False)

            if plan is None or fraction < min_seg_fraction:
                print("    Full-pose segment short, retrying position-only...")
                plan, fraction, _ = self._plan_cartesian(seg, downsample=1, position_only=True)

            if plan is None or len(plan.joint_trajectory.points) < 2 or fraction < 0.15:
                print("    Segment planning failed; skipping this segment.")
                start = end - 1
                continue

            ok = self.move_group.execute(plan, wait=True)
            self.move_group.stop()
            self.move_group.clear_pose_targets()

            if ok:
                executed_segments += 1
                print(f"    Executed (fraction={fraction:.3f})")
            else:
                print("    Execution failed for this segment.")

            # Overlap by one point for continuity.
            start = end - 1

        print(f"\nChunked execution summary: {executed_segments} segments executed.")
        return executed_segments > 0

    def run(self, demo_dir, speed_factor=1.0, downsample=3, dry_run=False, mode="trajectory"):
        """Replay from saved demo directory."""
        _ = mode  # kept for CLI compatibility
        demo = self.load_demo(demo_dir)
        self._set_speed(speed_factor)

        if len(demo["eef_poses"]) < 2:
            print("Error: need at least 2 EEF poses.")
            return

        start_pose = self.T_to_pose(demo["eef_poses"][0])
        eef_for_cartesian = demo["eef_poses"][1:]
        if not dry_run:
            if not self._move_to_start_pose(start_pose):
                print("  Continuing without start alignment.")
                eef_for_cartesian = demo["eef_poses"]

        # Start Cartesian tracking from the next pose to avoid immediate failure
        # due to mismatch between current state and first waypoint.
        plan, fraction, _ = self._plan_cartesian(
            eef_for_cartesian,
            downsample=max(1, downsample),
            position_only=False,
        )
        if plan is None:
            print("Error: no poses to plan.")
            return

        if fraction < self.min_fraction:
            print("  Full-pose Cartesian path too short, trying position-only fallback...")
            plan, fraction, _ = self._plan_cartesian(
                eef_for_cartesian,
                downsample=max(1, downsample * 2),
                position_only=True,
            )
            if plan is None:
                print("Error: no poses to plan in fallback.")
                return

        if fraction < self.min_fraction:
            print(
                "  Cartesian planning still below threshold. "
                "Trying chunked Cartesian fallback..."
            )
            self._execute_chunked_cartesian(
                eef_for_cartesian,
                chunk_size=max(20, 50 // max(1, downsample)),
                dry_run=dry_run,
            )
            return

        self._execute_plan(plan, fraction, dry_run=dry_run)

    def set_live_data(self, live_bottleneck_pose, end_effector_twists):
        """Set live bottleneck pose and twists directly."""
        self.live_bottleneck_pose = live_bottleneck_pose
        self.end_effector_twists = end_effector_twists
        print(f"  Set live bottleneck pose: {live_bottleneck_pose.shape}")
        print(f"  Set end-effector twists: {end_effector_twists.shape}")

    def se3_exp(self, vec):
        """Build a transformation matrix using exponential map."""
        assert len(vec) == 6

        phi = vec[3:]
        rho = vec[:3]

        angle = np.linalg.norm(phi)
        if angle < 1e-12:
            R = np.eye(3)
            J = np.eye(3)
        else:
            axis = phi / angle
            cp = np.cos(angle)
            sp = np.sin(angle)
            axis_skew = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ])
            R = cp * np.eye(3) + (1 - cp) * np.outer(axis, axis) + sp * axis_skew
            J = (
                (sp / angle) * np.eye(3)
                + (1 - sp / angle) * np.outer(axis, axis)
                + ((1 - cp) / angle) * axis_skew
            )

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = J @ rho
        return T

    def compute_trajectory_from_twists(self):
        """Compute EEF poses by integrating twists from bottleneck pose."""
        poses = [self.live_bottleneck_pose.copy()]
        current_pose = self.live_bottleneck_pose.copy()

        for twist in self.end_effector_twists:
            dt = 1.0 / 30.0
            twist_6d = twist[:6] * dt
            T_inc = self.se3_exp(twist_6d)
            current_pose = current_pose @ T_inc
            poses.append(current_pose.copy())

        return np.array(poses[:-1])

    def run_with_live_data(self, speed_factor=1.0, dry_run=False, mode="trajectory"):
        """Replay from live bottleneck pose + twists."""
        _ = mode  # kept for CLI compatibility
        print("\nRunning replay with live data:")
        print(f"  Bottleneck pose: {self.live_bottleneck_pose.shape}")
        print(f"  Twists: {self.end_effector_twists.shape}")

        eef_poses = self.compute_trajectory_from_twists()
        self._set_speed(speed_factor)

        if len(eef_poses) < 2:
            print("Error: need at least 2 EEF poses from twists.")
            return

        start_pose = self.T_to_pose(eef_poses[0])
        eef_for_cartesian = eef_poses[1:]
        if not dry_run:
            if not self._move_to_start_pose(start_pose):
                print("  Continuing without start alignment.")
                eef_for_cartesian = eef_poses

        plan, fraction, _ = self._plan_cartesian(
            eef_for_cartesian,
            downsample=1,
            position_only=False,
        )
        if plan is None:
            print("Error: no poses to plan.")
            return

        if fraction < self.min_fraction:
            print("  Full-pose Cartesian path too short, trying position-only fallback...")
            plan, fraction, _ = self._plan_cartesian(
                eef_for_cartesian,
                downsample=2,
                position_only=True,
            )
            if plan is None:
                print("Error: no poses to plan in fallback.")
                return

        if fraction < self.min_fraction:
            print(
                "  Cartesian planning still below threshold. "
                "Trying chunked Cartesian fallback..."
            )
            self._execute_chunked_cartesian(eef_for_cartesian, chunk_size=30, dry_run=dry_run)
            return

        self._execute_plan(plan, fraction, dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(
        description="Replay demonstrations via MoveIt Cartesian planning"
    )
    parser.add_argument(
        "--demo_dir",
        "-d",
        type=str,
        required=True,
        help="Path to demo directory (containing eef_poses.npy, etc.)",
    )
    parser.add_argument("--robot_model", type=str, default="wx250s")
    parser.add_argument("--robot_name", type=str, default="wx250s")
    parser.add_argument(
        "--speed_factor",
        "-s",
        type=float,
        default=2.0,
        help="Speed multiplier (default: 2.0)",
    )
    parser.add_argument(
        "--downsample",
        "-n",
        type=int,
        default=3,
        help="Use every Nth waypoint (default: 3)",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="trajectory",
        choices=["trajectory", "point_by_point"],
        help="Kept for compatibility; ignored in MoveIt-direct mode.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only plan, do not execute")
    parser.add_argument(
        "--eef_step",
        type=float,
        default=0.01,
        help="Cartesian interpolation step in meters",
    )
    parser.add_argument(
        "--jump_threshold",
        type=float,
        default=0.0,
        help="MoveIt jump threshold",
    )
    parser.add_argument(
        "--min_fraction",
        type=float,
        default=0.90,
        help="Minimum Cartesian fraction required for execution",
    )
    parser.add_argument(
        "--avoid_collisions",
        action="store_true",
        help="Enable collision checking during Cartesian path planning",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("DEMONSTRATION REPLAY (MOVEIT DIRECT)")
    print("=" * 60)
    print(f"Demo:         {args.demo_dir}")
    print(f"Speed:        {args.speed_factor}x")
    print(f"Downsample:   {args.downsample}x")
    print(f"EEF step:     {args.eef_step}")
    print(f"Min fraction: {args.min_fraction}")
    print(f"Avoid coll.:  {args.avoid_collisions}")
    print(f"Dry run:      {args.dry_run}")
    print("=" * 60)

    try:
        replayer = DemoReplayer(
            robot_model=args.robot_model,
            robot_name=args.robot_name,
            eef_step=args.eef_step,
            jump_threshold=args.jump_threshold,
            min_fraction=args.min_fraction,
            avoid_collisions=args.avoid_collisions,
        )
        replayer.run(
            demo_dir=args.demo_dir,
            speed_factor=args.speed_factor,
            downsample=args.downsample,
            dry_run=args.dry_run,
            mode=args.mode,
        )
    finally:
        roscpp_shutdown()


if __name__ == "__main__":
    main()
