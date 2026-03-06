#!/usr/bin/env python

"""
Demonstration Replay Script for Interbotix WX250S

Replays demonstrations saved in learning_thousand_tasks format by
demo_collect_v2.py. Works in two modes, auto-detected at startup:

  1. Real robot (xsarm_control.launch running):
     Uses InterbotixManipulatorXS SDK to command the arm via trajectory messages.

  2. Simulation / RViz (xsarm_description.launch running):
     Publishes JointState messages directly so robot_state_publisher
     updates the TF tree and RViz shows the motion.

Usage:
  # With real robot:
  roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=wx250s
  python replay_demo_v2_last.py -d collected_demos/session_XXXXXXXX/demo_0000

  # With RViz simulation:
  roslaunch interbotix_xsarm_descriptions xsarm_description.launch robot_model:=wx250s use_joint_pub_gui:=true
  python replay_demo_v2_last.py -d collected_demos/session_XXXXXXXX/demo_0000

Options:
  --speed_factor   Speed multiplier (default: 1.0, slower: 0.5, faster: 2.0)
  --downsample     Use every Nth waypoint (default: 3, i.e. ~10Hz from 30Hz)
  --dry_run        Only compute IK, don't move the robot
"""

import rospy
import numpy as np
import modern_robotics as mr
import os
import sys
import argparse
import math
from sensor_msgs.msg import JointState
import interbotix_xs_modules.mr_descriptions as mrd


# ---------------------------------------------------------------------------
# WX250S constants (used in simulation mode when xs_sdk is not available)
# ---------------------------------------------------------------------------
WX250S_ARM_JOINT_NAMES = [
    'waist', 'shoulder', 'elbow',
    'forearm_roll', 'wrist_angle', 'wrist_rotate'
]

# Dynamixel position register to radians: (val - 2048) * 2pi / 4096
WX250S_JOINT_LOWER_LIMITS = [
    -3.14159,   # waist         (reg 0)
    -1.88496,   # shoulder      (reg 819)
    -2.14675,   # elbow         (reg 648)
    -3.14159,   # forearm_roll  (reg 0)
    -1.74533,   # wrist_angle   (reg 910)
    -3.14159,   # wrist_rotate  (reg 0)
]

WX250S_JOINT_UPPER_LIMITS = [
     3.14159,   # waist         (reg 4095)
     1.98968,   # shoulder      (reg 3345)
     1.60429,   # elbow         (reg 3094)
     3.14159,   # forearm_roll  (reg 4095)
     2.14675,   # wrist_angle   (reg 3447)
     3.14159,   # wrist_rotate  (reg 4095)
]

WX250S_SLEEP_POSITIONS = [0, -1.80, 1.55, 0, 0.8, 0]

# Gripper finger joint limits (prismatic, meters)
GRIPPER_OPEN_POS = 0.037       # left_finger when open
GRIPPER_CLOSED_POS = 0.015     # left_finger when closed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def detect_real_robot(robot_name, timeout=3.0):
    """Return True if the xs_sdk services are reachable (real robot mode)."""
    service_name = "/" + robot_name + "/get_robot_info"
    try:
        rospy.wait_for_service(service_name, timeout=timeout)
        return True
    except rospy.exceptions.ROSException:
        return False


def normalize_angle(angle):
    """Normalize an angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# DemoReplayer
# ---------------------------------------------------------------------------
class DemoReplayer:
    """Replays recorded EEF demonstrations on the Interbotix arm."""

    def __init__(self, robot_model="wx250s", robot_name="wx250s"):
        self.robot_model = robot_model
        self.robot_name = robot_name

        # Kinematics (always available, no ROS dependency)
        self.robot_des = getattr(mrd, robot_model)
        self.Slist = self.robot_des.Slist
        self.M = self.robot_des.M
        self.rev = 2 * math.pi

        # Initialize ROS node
        rospy.init_node('demo_replayer', anonymous=True)

        # Auto-detect mode
        print("\nDetecting robot mode...")
        self.sim_mode = not detect_real_robot(robot_name)

        if self.sim_mode:
            self._init_sim_mode()
        else:
            self._init_real_mode()

    # -- Initialisation per mode ------------------------------------------

    def _init_sim_mode(self):
        """Initialise for RViz-only (no xs_sdk)."""
        print("  xs_sdk not found -> SIMULATION mode (RViz)")

        self.joint_names = list(WX250S_ARM_JOINT_NAMES)
        self.num_joints = len(self.joint_names)
        self.joint_lower_limits = list(WX250S_JOINT_LOWER_LIMITS)
        self.joint_upper_limits = list(WX250S_JOINT_UPPER_LIMITS)
        self.sleep_positions = list(WX250S_SLEEP_POSITIONS)

        # All joint names published in the JointState message (arm + fingers)
        self.all_joint_names = self.joint_names + ['left_finger', 'right_finger']

        # Current state for JointState publishing
        self.current_arm_positions = list(self.sleep_positions)
        self.current_gripper_closed = False

        # Publisher for visualisation
        self.js_pub = rospy.Publisher(
            '/' + self.robot_name + '/joint_states',
            JointState,
            queue_size=10
        )
        # Give publisher time to register with subscribers
        rospy.sleep(0.5)

        print(f"  Publishing JointState on: /{self.robot_name}/joint_states")
        print(f"  Joints: {self.joint_names}")

    def _init_real_mode(self):
        """Initialise for real robot (xs_sdk running)."""
        print("  xs_sdk found -> REAL ROBOT mode")

        from interbotix_xs_modules.arm import InterbotixManipulatorXS
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from interbotix_xs_msgs.msg import JointTrajectoryCommand

        # Store these for later use in real-robot methods
        self._JointTrajectory = JointTrajectory
        self._JointTrajectoryPoint = JointTrajectoryPoint
        self._JointTrajectoryCommand = JointTrajectoryCommand

        self.bot = InterbotixManipulatorXS(
            robot_model=self.robot_model,
            robot_name=self.robot_name,
            moving_time=2.0,
            accel_time=0.5,
            init_node=False
        )

        self.joint_names = list(self.bot.arm.group_info.joint_names)
        self.num_joints = self.bot.arm.group_info.num_joints
        self.joint_lower_limits = list(self.bot.arm.group_info.joint_lower_limits)
        self.joint_upper_limits = list(self.bot.arm.group_info.joint_upper_limits)

        print(f"  Robot initialised with {self.num_joints} joints: {self.joint_names}")

    # -- Demo loading (shared) --------------------------------------------

    def load_demo(self, demo_dir):
        """Load a demonstration directory into a dict."""
        demo = {}

        eef_poses_path = os.path.join(demo_dir, "eef_poses.npy")
        twists_path = os.path.join(demo_dir, "demo_eef_twists.npy")
        timestamps_path = os.path.join(demo_dir, "timestamps.npy")
        bottleneck_path = os.path.join(demo_dir, "bottleneck_pose.npy")

        if not os.path.exists(eef_poses_path):
            print(f"Error: {eef_poses_path} not found.")
            sys.exit(1)

        demo['eef_poses'] = np.load(eef_poses_path)
        demo['eef_twists'] = np.load(twists_path)
        demo['bottleneck_pose'] = np.load(bottleneck_path)

        if os.path.exists(timestamps_path):
            demo['timestamps'] = np.load(timestamps_path)
        else:
            T = len(demo['eef_poses'])
            demo['timestamps'] = np.arange(T) / 30.0

        print(f"\nLoaded demo from: {demo_dir}")
        print(f"  Timesteps: {len(demo['eef_poses'])}")
        print(f"  Duration:  {demo['timestamps'][-1]:.2f}s")

        return demo

    # -- IK (shared, silent) ----------------------------------------------

    def _check_joint_limits(self, theta_list):
        """Check that all joints are within limits (no warnings)."""
        for i in range(self.num_joints):
            val = round(theta_list[i], 3)
            if val < round(self.joint_lower_limits[i], 3):
                return False
            if val > round(self.joint_upper_limits[i], 3):
                return False
        return True

    def _wrap_ik_solution(self, theta_list):
        """
        Normalize IK solution angles into joint limits.

        Strategy:
          1. Normalize every angle to [-pi, pi].
          2. If still out of a joint's specific limits, try +2pi and -2pi.
        """
        for x in range(len(theta_list)):
            # Step 1: normalize to [-pi, pi]
            theta_list[x] = normalize_angle(theta_list[x])

            ll = self.joint_lower_limits[x]
            ul = self.joint_upper_limits[x]

            # Step 2: if out of limits, try the +-2pi candidates and pick
            # whichever one is inside the limits
            if round(theta_list[x], 3) < round(ll, 3):
                candidate = theta_list[x] + self.rev
                if round(candidate, 3) <= round(ul, 3):
                    theta_list[x] = candidate
            elif round(theta_list[x], 3) > round(ul, 3):
                candidate = theta_list[x] - self.rev
                if round(candidate, 3) >= round(ll, 3):
                    theta_list[x] = candidate

        return theta_list

    def _solve_ik(self, T_sd, initial_guess):
        """
        Solve IK with multiple initial guesses. Returns (theta_list, success).

        Uses a silent joint-limit check to avoid flooding the console with
        warnings on every failed guess.
        """
        guesses = [initial_guess]
        guesses.append([0.0] * self.num_joints)

        neg_guess = [0.0] * self.num_joints
        neg_guess[0] = np.deg2rad(-120)
        guesses.append(neg_guess)

        pos_guess = [0.0] * self.num_joints
        pos_guess[0] = np.deg2rad(120)
        guesses.append(pos_guess)

        # Extra guesses biased toward common demonstration poses
        mid_guess = list(initial_guess)
        mid_guess[1] = 0.0   # shoulder centred
        mid_guess[2] = 0.0   # elbow centred
        guesses.append(mid_guess)

        for guess in guesses:
            theta_list, success = mr.IKinSpace(
                self.Slist, self.M, T_sd, guess, 0.001, 0.001
            )
            if not success:
                continue

            theta_list = self._wrap_ik_solution(theta_list)

            if self._check_joint_limits(theta_list):
                return theta_list, True

        return theta_list, False

    def compute_joint_trajectory(self, demo, downsample=3):
        """Pre-compute the full joint trajectory from EEF poses via IK."""
        eef_poses = demo['eef_poses']
        timestamps = demo['timestamps']
        gripper_col = demo['eef_twists'][:, 6]

        indices = list(range(0, len(eef_poses), downsample))
        if indices[-1] != len(eef_poses) - 1:
            indices.append(len(eef_poses) - 1)

        print(f"\nComputing IK for {len(indices)} waypoints "
              f"(downsampled {downsample}x from {len(eef_poses)} frames)...")

        joint_positions = []
        waypoint_times = []
        gripper_states = []
        ik_failures = 0

        if self.sim_mode:
            current_guess = list(self.sleep_positions)
        else:
            current_guess = list(self.bot.arm.joint_commands)

        for count, idx in enumerate(indices):
            T_sd = eef_poses[idx]
            theta_list, success = self._solve_ik(T_sd, current_guess)

            if success:
                joint_positions.append(list(theta_list))
                waypoint_times.append(timestamps[idx])
                gripper_states.append(gripper_col[idx] > 0.5)
                current_guess = list(theta_list)
            else:
                ik_failures += 1
                if joint_positions:
                    joint_positions.append(list(joint_positions[-1]))
                    waypoint_times.append(timestamps[idx])
                    gripper_states.append(gripper_col[idx] > 0.5)

            if (count + 1) % 50 == 0 or count == len(indices) - 1:
                print(f"  IK progress: {count + 1}/{len(indices)} "
                      f"({ik_failures} failures)")

        success_rate = 1.0 - (ik_failures / len(indices)) if indices else 0
        print(f"  IK success rate: {success_rate*100:.1f}%")

        if ik_failures > 0:
            print(f"  ({ik_failures} waypoints used previous valid position)")

        return joint_positions, waypoint_times, gripper_states, success_rate

    # update here

    def se3_exp(self, vec):  
        """Build a transformation matrix using the exponential map."""  
        assert len(vec) == 6  
        
        phi = vec[3:]  # Angular part  
        rho = vec[:3]  # Linear part  
        
        # Compute rotation matrix from angular part  
        angle = np.linalg.norm(phi)  
        if angle < 1e-12:  
            R = np.eye(3)  
        else:  
            axis = phi / angle  
            cp = np.cos(angle)  
            sp = np.sin(angle)  
            R = cp * np.eye(3) + (1 - cp) * np.outer(axis, axis) + sp * np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])  
        
        # Compute the J matrix for translation  
        if angle < 1e-12:  
            J = np.eye(3)  
        else:  
            axis = phi / angle  
            cp = np.cos(angle)  
            sp = np.sin(angle)  
            J = sp/angle * np.eye(3) + (1 - sp/angle) * np.outer(axis, axis) + (1 - cp)/angle * np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])  
        
        # Build transformation matrix  
        T = np.eye(4)  
        T[:3, :3] = R  
        T[:3, 3] = J @ rho  
        
        return T
  
    def set_live_data(self, live_bottleneck_pose, end_effector_twists):  
      """Set live bottleneck pose and twists directly."""  
      self.live_bottleneck_pose = live_bottleneck_pose  
      self.end_effector_twists = end_effector_twists  
      print(f"  Set live bottleneck pose: {live_bottleneck_pose.shape}")  
      print(f"  Set end-effector twists: {end_effector_twists.shape}")  
    
    def compute_trajectory_from_twists(self):  
        """Compute EEF poses by integrating twists from bottleneck pose."""  
        poses = [self.live_bottleneck_pose.copy()]  
        current_pose = self.live_bottleneck_pose.copy()  
          
        for twist in self.end_effector_twists:  
            # Convert twist to transformation matrix  
            dt = 1.0/30.0  # Assuming 30Hz recording  
            twist_6d = twist[:6] * dt  
              
            # Use modern_robotics to compute exponential of twist  
            #T_inc = mr.TwistExp(twist_6d)  
            #T_inc = mr.MatrixExp6(twist_6d)
            #se3_mat = mr.Vectortose3(twist_6d)  
            #T_inc = mr.MatrixExp6(se3_mat)
            T_inc = self.se3_exp(twist_6d)
            #from thousand_tasks.core.utils.se3_tools import se3_exp  
            #T_inc = se3_exp(twist_6d)
            current_pose = current_pose @ T_inc  
            poses.append(current_pose.copy())  
          
        return np.array(poses[:-1])  # Return T poses for T twists
  
    # =====================================================================
    #  SIMULATION MODE - publish JointState to RViz
    # =====================================================================

    def _publish_joint_state(self, arm_positions, gripper_closed):
        """Publish a single JointState message (arm + gripper fingers)."""
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = self.all_joint_names

        left_finger = GRIPPER_CLOSED_POS if gripper_closed else GRIPPER_OPEN_POS
        right_finger = -left_finger

        msg.position = list(arm_positions) + [left_finger, right_finger]
        msg.velocity = [0.0] * len(msg.name)
        msg.effort = [0.0] * len(msg.name)

        self.js_pub.publish(msg)

    def _sim_move_to(self, target_positions, gripper_closed,
                     duration=2.0, rate_hz=30):
        """Smoothly interpolate from current position to target in RViz."""
        start = np.array(self.current_arm_positions)
        end = np.array(target_positions)
        rate = rospy.Rate(rate_hz)
        steps = max(int(duration * rate_hz), 1)

        for i in range(steps + 1):
            if rospy.is_shutdown():
                return
            alpha = float(i) / steps
            interp = start + alpha * (end - start)
            self._publish_joint_state(interp, gripper_closed)
            rate.sleep()

        self.current_arm_positions = list(target_positions)
        self.current_gripper_closed = gripper_closed

    def replay_sim(self, joint_positions, waypoint_times,
                   gripper_states, speed_factor=1.0):
        """Replay in simulation by publishing JointState at the correct rate."""
        if len(joint_positions) < 2:
            print("Not enough waypoints to replay.")
            return

        total_time = (waypoint_times[-1] - waypoint_times[0]) / speed_factor
        print(f"\nReplaying in SIMULATION mode:")
        print(f"  Waypoints: {len(joint_positions)}")
        print(f"  Duration:  {total_time:.2f}s (speed_factor={speed_factor}x)")

        t0 = waypoint_times[0]
        replay_start = rospy.Time.now()

        idx = 0
        rate = rospy.Rate(60)  # 60 Hz for smooth visualisation
        last_pct_logged = -1

        while not rospy.is_shutdown() and idx < len(joint_positions) - 1:
            elapsed = (rospy.Time.now() - replay_start).to_sec()
            target_demo_time = t0 + elapsed * speed_factor

            # Advance index to match target time
            while (idx < len(waypoint_times) - 1 and
                   waypoint_times[idx + 1] <= target_demo_time):
                idx += 1

            # Interpolate between current and next waypoint
            if idx < len(waypoint_times) - 1:
                t_start = waypoint_times[idx]
                t_end = waypoint_times[idx + 1]
                seg_len = t_end - t_start
                if seg_len > 0:
                    alpha = min((target_demo_time - t_start) / seg_len, 1.0)
                else:
                    alpha = 1.0
                pos = (np.array(joint_positions[idx]) * (1 - alpha) +
                       np.array(joint_positions[idx + 1]) * alpha)
            else:
                pos = np.array(joint_positions[idx])

            gripper_closed = gripper_states[idx]

            if gripper_closed != self.current_gripper_closed:
                print(f"  Gripper {'CLOSED' if gripper_closed else 'OPENED'}")
                self.current_gripper_closed = gripper_closed

            self._publish_joint_state(pos, gripper_closed)
            self.current_arm_positions = list(pos)
            rate.sleep()

            # Progress logging (every 25%)
            if total_time > 0:
                pct = int(elapsed / total_time * 4) * 25
                if pct != last_pct_logged and pct <= 100:
                    print(f"  Progress: {pct}%")
                    last_pct_logged = pct

        # Hold final pose briefly
        self._publish_joint_state(joint_positions[-1], gripper_states[-1])
        self.current_arm_positions = list(joint_positions[-1])
        print("  Replay complete.")

    # =====================================================================
    #  REAL ROBOT MODE - use InterbotixManipulatorXS SDK
    # =====================================================================

    def replay_real_trajectory(self, joint_positions, waypoint_times,
                               gripper_states, speed_factor=1.0):
        """Replay on real robot using JointTrajectory messages."""
        JointTrajectory = self._JointTrajectory
        JointTrajectoryPoint = self._JointTrajectoryPoint
        JointTrajectoryCommand = self._JointTrajectoryCommand

        if len(joint_positions) < 2:
            print("Not enough waypoints to replay.")
            return

        # Split at gripper transitions
        segments = []
        seg_start = 0
        for i in range(1, len(gripper_states)):
            if gripper_states[i] != gripper_states[i - 1]:
                segments.append((seg_start, i, gripper_states[i - 1]))
                seg_start = i
        segments.append((seg_start, len(gripper_states), gripper_states[-1]))

        total_time = (waypoint_times[-1] - waypoint_times[0]) / speed_factor
        print(f"\nReplaying on REAL ROBOT:")
        print(f"  Waypoints: {len(joint_positions)}")
        print(f"  Duration:  {total_time:.2f}s (speed_factor={speed_factor}x)")
        print(f"  Segments:  {len(segments)} (split at gripper transitions)")

        for seg_idx, (start, end, gripper_closed) in enumerate(segments):
            seg_positions = joint_positions[start:end]
            seg_times = waypoint_times[start:end]

            if len(seg_positions) < 2:
                if gripper_closed:
                    self.bot.gripper.close(delay=0.5)
                    print("  Gripper CLOSED")
                else:
                    self.bot.gripper.open(delay=0.5)
                    print("  Gripper OPENED")
                continue

            # Build JointTrajectory
            joint_traj = JointTrajectory()
            joint_traj.joint_names = self.joint_names

            t0 = seg_times[0]
            for pos, t in zip(seg_positions, seg_times):
                point = JointTrajectoryPoint()
                point.positions = pos
                point.time_from_start = rospy.Duration.from_sec(
                    (t - t0) / speed_factor
                )
                joint_traj.points.append(point)

            # Snap first point to actual position to avoid a jump
            current_positions = []
            with self.bot.dxl.js_mutex:
                for name in self.joint_names:
                    current_positions.append(
                        self.bot.dxl.joint_states.position[
                            self.bot.dxl.js_index_map[name]
                        ]
                    )
            joint_traj.points[0].positions = current_positions

            seg_duration = (seg_times[-1] - seg_times[0]) / speed_factor
            avg_wp_time = seg_duration / max(len(seg_positions) - 1, 1)
            wp_moving_time = max(avg_wp_time, 0.1)
            wp_accel_time = min(wp_moving_time / 2.0, 0.1)
            self.bot.arm.set_trajectory_time(wp_moving_time, wp_accel_time)

            joint_traj.header.stamp = rospy.Time.now()
            traj_cmd = JointTrajectoryCommand(
                "group", self.bot.arm.group_name, joint_traj
            )
            self.bot.dxl.pub_traj.publish(traj_cmd)

            print(f"  Segment {seg_idx + 1}/{len(segments)}: "
                  f"{len(seg_positions)} wp, {seg_duration:.2f}s, "
                  f"gripper={'CLOSED' if gripper_closed else 'OPEN'}")

            rospy.sleep(seg_duration + wp_moving_time)

            self.bot.arm.joint_commands = list(seg_positions[-1])
            self.bot.arm.T_sb = mr.FKinSpace(
                self.M, self.Slist, seg_positions[-1]
            )

            # Gripper transition
            if seg_idx < len(segments) - 1:
                next_gripper = segments[seg_idx + 1][2]
                if next_gripper and not gripper_closed:
                    self.bot.gripper.close(delay=0.5)
                    print("  Gripper CLOSED")
                elif not next_gripper and gripper_closed:
                    self.bot.gripper.open(delay=0.5)
                    print("  Gripper OPENED")

    def replay_real_point_by_point(self, joint_positions, waypoint_times,
                                    gripper_states, speed_factor=1.0):
        """Simple point-by-point replay on real robot."""
        if not joint_positions:
            print("No waypoints to replay.")
            return

        print(f"\nReplaying point-by-point on REAL ROBOT:")
        print(f"  Waypoints: {len(joint_positions)}")

        gripper_is_closed = False

        for i in range(len(joint_positions)):
            if gripper_states[i] and not gripper_is_closed:
                self.bot.gripper.close(delay=0.3)
                gripper_is_closed = True
                print(f"  [{i}] Gripper CLOSED")
            elif not gripper_states[i] and gripper_is_closed:
                self.bot.gripper.open(delay=0.3)
                gripper_is_closed = False
                print(f"  [{i}] Gripper OPENED")

            if i < len(joint_positions) - 1:
                dt = (waypoint_times[i + 1] - waypoint_times[i]) / speed_factor
                dt = max(dt, 0.05)
            else:
                dt = 0.2

            moving_time = max(dt, 0.1)
            accel_time = min(moving_time / 2.0, 0.1)

            self.bot.arm.publish_positions(
                joint_positions[i],
                moving_time=moving_time,
                accel_time=accel_time,
                blocking=False
            )
            rospy.sleep(dt)

            if (i + 1) % 25 == 0:
                print(f"  Progress: {i + 1}/{len(joint_positions)}")

        rospy.sleep(0.3)
        print("  Replay complete.")

    # =====================================================================
    #  Main entry point
    # =====================================================================

    def run(self, demo_dir, speed_factor=1.0, downsample=3,
            dry_run=False, mode="trajectory"):
        # Load
        demo = self.load_demo(demo_dir)

        # IK
        joint_positions, waypoint_times, gripper_states, success_rate = \
            self.compute_joint_trajectory(demo, downsample)

        if not joint_positions:
            print("Error: IK failed for all waypoints. Cannot replay.")
            return

        if success_rate < 0.5:
            print(f"Warning: IK success rate is low ({success_rate*100:.1f}%).")

        if dry_run:
            print("\n[DRY RUN] IK complete. Not moving the robot.")
            return

        # -- Move to start pose --
        print("\nMoving to start pose...")
        if self.sim_mode:
            self._sim_move_to(joint_positions[0], gripper_states[0], duration=2.0)
        else:
            self.bot.arm.set_trajectory_time(2.0, 0.5)
            self.bot.arm.publish_positions(joint_positions[0], moving_time=2.0,
                                            accel_time=0.5, blocking=True)
            if gripper_states[0]:
                self.bot.gripper.close(delay=0.5)
            else:
                self.bot.gripper.open(delay=0.5)

        # -- Replay --
        if self.sim_mode:
            self.replay_sim(
                joint_positions, waypoint_times, gripper_states, speed_factor
            )
        elif mode == "trajectory":
            self.replay_real_trajectory(
                joint_positions, waypoint_times, gripper_states, speed_factor
            )
        else:
            self.replay_real_point_by_point(
                joint_positions, waypoint_times, gripper_states, speed_factor
            )

        # -- Return to sleep --
        print("\nReturning to sleep pose...")
        if self.sim_mode:
            self._sim_move_to(self.sleep_positions, False, duration=2.0)
        else:
            self.bot.arm.set_trajectory_time(2.0, 0.5)
            self.bot.arm.go_to_sleep_pose()

        print("Done.")


    def run_with_live_data(self, speed_factor=1.0, dry_run=False, mode="trajectory"):  
      """Run replay using live bottleneck pose and twists."""  
      print(f"\nRunning replay with live data:")  
      print(f"  Bottleneck pose: {self.live_bottleneck_pose.shape}")  
      print(f"  Twists: {self.end_effector_twists.shape}")  
        
      # Compute poses from twists  
      eef_poses = self.compute_trajectory_from_twists()  
      timestamps = np.arange(len(self.end_effector_twists)) / 30.0  
      gripper_states = self.end_effector_twists[:, 6] > 0.5  
        
      # Compute IK for all poses  
      joint_positions = []  
      waypoint_times = []  
      ik_failures = 0  
        
      current_guess = list(self.sleep_positions) if self.sim_mode else list(self.bot.arm.joint_commands)  
        
      for i, pose in enumerate(eef_poses):  
          theta_list, success = self._solve_ik(pose, current_guess)  
            
          if success:  
              joint_positions.append(list(theta_list))  
              waypoint_times.append(timestamps[i])  
              current_guess = list(theta_list)  
          else:  
              ik_failures += 1  
              if joint_positions:  
                  joint_positions.append(list(joint_positions[-1]))  
                  waypoint_times.append(timestamps[i])  
        
      success_rate = 1.0 - (ik_failures / len(eef_poses))  
      print(f"  IK success rate: {success_rate*100:.1f}%")  
        
      if dry_run:  
          print("\n[DRY RUN] IK complete. Not moving the robot.")  
          return  
        
      # Move to bottleneck pose first  
      print("\nMoving to bottleneck pose...")  
      if self.sim_mode:  
          self._sim_move_to(joint_positions[0], gripper_states[0], duration=2.0)  
      else:  
          self.bot.arm.publish_positions(joint_positions[0], moving_time=2.0,   
                                          accel_time=0.5, blocking=True)  
        
      # Replay the trajectory  
      if self.sim_mode:  
          self.replay_sim(joint_positions, waypoint_times, gripper_states, speed_factor)  
      elif mode == "trajectory":  
          self.replay_real_trajectory(joint_positions, waypoint_times, gripper_states, speed_factor)  
      else:  
          self.replay_real_point_by_point(joint_positions, waypoint_times, gripper_states, speed_factor)

      # -- Return to sleep --
      print("\nReturning to sleep pose...")
      if self.sim_mode:
          self._sim_move_to(self.sleep_positions, False, duration=2.0)
      else:
          self.bot.arm.set_trajectory_time(2.0, 0.5)
          self.bot.arm.go_to_sleep_pose()

      print("Done.")
      

def main():
    parser = argparse.ArgumentParser(
        description='Replay demonstrations (real robot or RViz simulation)'
    )
    parser.add_argument(
        '--demo_dir', '-d', type=str, required=True,
        help='Path to demo directory (containing demo_eef_twists.npy, etc.)'
    )
    parser.add_argument(
        '--robot_model', type=str, default='wx250s',
        help='Robot model (default: wx250s)'
    )
    parser.add_argument(
        '--robot_name', type=str, default='wx250s',
        help='Robot name/namespace (default: wx250s)'
    )
    parser.add_argument(
        '--speed_factor', '-s', type=float, default=1.0,
        help='Speed multiplier (default: 1.0, slower: 0.5, faster: 2.0)'
    )
    parser.add_argument(
        '--downsample', '-n', type=int, default=3,
        help='Use every Nth waypoint (default: 3, i.e. ~10Hz from 30Hz)'
    )
    parser.add_argument(
        '--mode', '-m', type=str, default='trajectory',
        choices=['trajectory', 'point_by_point'],
        help='Replay mode for real robot (default: trajectory)'
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='Only compute IK, do not move'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("DEMONSTRATION REPLAY v2")
    print("="*60)
    print(f"Demo:       {args.demo_dir}")
    print(f"Speed:      {args.speed_factor}x")
    print(f"Downsample: {args.downsample}x")
    print(f"Mode:       {args.mode}")
    print(f"Dry run:    {args.dry_run}")
    print("="*60)

    replayer = DemoReplayer(
        robot_model=args.robot_model,
        robot_name=args.robot_name
    )

    replayer.run(
        demo_dir=args.demo_dir,
        speed_factor=args.speed_factor,
        downsample=args.downsample,
        dry_run=args.dry_run,
        mode=args.mode
    )


if __name__ == '__main__':
    main()
