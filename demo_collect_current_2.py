#!/usr/bin/env python

"""
Demonstration Collection Script v2 for Interbotix WX250S

Records demonstrations directly in the learning_thousand_tasks format:
  - demo_eef_twists.npy: (T, 7) velocity commands [vx, vy, vz, wx, wy, wz, gripper]
  - bottleneck_pose.npy: (4, 4) final end-effector pose SE(3)

At each timestep, joint positions and velocities are read from /joint_states,
then converted on the fly to EEF spatial twists via the space Jacobian.

Usage:
1. Launch the arm controller first:
   roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=wx250s

2. Run this script:
   python demo_collect_v2.py

3. Follow the on-screen instructions to collect demonstrations.

Controls during recording:
  - 'o': Open gripper
  - 'c': Close gripper
  - 's': Start recording demonstration
  - 'e': End recording demonstration
  - 'r': Go to ready position
  - 't': Toggle teaching mode
  - 'p': Print current EEF pose
  - 'q': Quit and save demonstrations
"""

import sys
import rospy
import numpy as np
import modern_robotics as mr
import os
import time
import threading
import pickle
from datetime import datetime
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image, CameraInfo
from interbotix_xs_modules.arm import InterbotixManipulatorXS
import interbotix_xs_modules.mr_descriptions as mrd
from moveit_commander import MoveGroupCommander, roscpp_initialize
from tf.transformations import quaternion_matrix
from cv_bridge import CvBridge
import cv2
from test_sam import mobile_sam_segmap_function

class DemonstrationCollectorV2:  
    """Collects kinesthetic demonstrations directly in EEF twist format."""  
  
    def __init__(  
        self,  
        robot_model="wx250s",  
        robot_name="wx250s",  
        num_demos=5,  
        record_rate=50,  
        task_name="pick_up_cube",  
    ):  
        self.robot_model = robot_model  
        self.robot_name = robot_name  
        self.num_demos = num_demos  
        self.record_rate = record_rate  
        self.task_name = task_name  
  
        # Load kinematics for the robot model  
        self.robot_des = getattr(mrd, robot_model)  
        self.Slist = self.robot_des.Slist  # (6, num_joints)  
        self.M = self.robot_des.M          # (4, 4)  
  
        # Data storage: each demo is a dict with eef_twists and bottleneck_pose  
        self.demonstrations = []  
        self._reset_current_demo()  
  
        # State variables  
        self.recording = False  
        self.gripper_is_closed = False  
        self.latest_joint_state = None  
        self.joint_state_lock = threading.Lock()  
        self.cv_bridge = CvBridge()  
        self.pending_camera_data = None  
        self.mobile_sam_point = None
          
        # Segmentation parameters for orange cube extraction - more conservative values  
        self.orange_hsv_lower = np.array([8, 80, 80], dtype=np.uint8)    # Conservative orange range  
        self.orange_hsv_upper = np.array([20, 255, 255], dtype=np.uint8)  # Conservative orange range  
        self.seg_min_area_px = 50      # Increased from 10 to avoid tiny noise  
        self.depth_margin_mm = 15      # Reduced from 20 to be less restrictive  
        self.object_depth_band_mm = 40 # Increased from 30 to allow more depth variation  
        self.seg_fill_kernel = 5       # Reduced from 7 for more precise morphological operations  
        self.save_segmentation_debug = True  

        # Initialize ROS node
        rospy.init_node('demo_collector_v2', anonymous=True)
        roscpp_initialize(sys.argv)

        # Initialize MoveIt interface so recorded EEF poses come from MoveIt FK.
        semantic_ns = f"/{robot_name}/robot_description_semantic"
        if not rospy.has_param(semantic_ns):
            print(f"Error: MoveIt not running (missing {semantic_ns}).")
            print("Launch MoveIt first, e.g.:")
            print(
                f"  roslaunch interbotix_xsarm_moveit xsarm_moveit.launch "
                f"robot_model:={robot_model} robot_name:={robot_name} use_actual:=true dof:=6"
            )
            sys.exit(1)
        self.move_group = MoveGroupCommander(
            "interbotix_arm",
            robot_description=f"/{robot_name}/robot_description",
            ns=f"/{robot_name}",
        )
        print("MoveIt interface ready for EEF pose sampling.")

        # Initialize robot
        print("\nInitializing robot...")
        self.bot = InterbotixManipulatorXS(
            robot_model=robot_model,
            robot_name=robot_name,
            moving_time=2.0,
            accel_time=0.5,
            init_node=False
        )

        # Get joint info
        self.joint_names = list(self.bot.arm.group_info.joint_names)
        self.num_joints = self.bot.arm.group_info.num_joints
        print(f"Robot initialized with {self.num_joints} joints: {self.joint_names}")

        # Subscribe to joint states
        self.joint_state_sub = rospy.Subscriber(
            f"/{robot_name}/joint_states",
            JointState,
            self._joint_state_callback
        )

        # Wait for joint states
        print("Waiting for joint state messages...")
        timeout = 5.0
        start_time = time.time()
        while self.latest_joint_state is None and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        if self.latest_joint_state is None:
            print("Warning: No joint states received. Trying default topic...")
            self.joint_state_sub.unregister()
            self.joint_state_sub = rospy.Subscriber(
                "/joint_states",
                JointState,
                self._joint_state_callback
            )
            time.sleep(1.0)

        # Output directory
        self.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'collected_demos'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Demonstrations will be saved to: {self.output_dir}")

    def _reset_current_demo(self):
        """Reset the current demonstration buffers."""
        self.current_demo = {
            'eef_twists': [],    # Will become (T, 7) [vx, vy, vz, wx, wy, wz, gripper]
            'eef_poses': [],     # Will become (T, 4, 4) for reference
            'timestamps': [],
            'camera_data': None,
        }

    def _capture_workspace_camera_data(self, timeout=20.0):
        """
        Capture a single RGB frame, depth frame, and camera intrinsics.

        Returns:
            dict with keys: rgb_image, depth_image, segmap, intrinsic_matrix
            or None if capture fails.
        """
        try:
            rgb_msg = rospy.wait_for_message(
                "/camera/color/image_raw",
                Image,
                timeout=timeout
            )
            depth_msg = rospy.wait_for_message(
                "/camera/aligned_depth_to_color/image_raw",
                Image,
                timeout=timeout
            )
            camera_info_msg = rospy.wait_for_message(
                "/camera/color/camera_info",
                CameraInfo,
                timeout=timeout
            )
        except rospy.ROSException as e:
            rospy.logwarn(f"Failed to capture camera data before recording: {e}")
            return None

        try:
            rgb_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8') # (720, 1280, 3)
            depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough') # (720, 1280)

            # MT3 expects a 16-bit depth image.
            depth_image = self._to_uint16_depth_mm(depth_image)

            intrinsic_matrix = np.array(camera_info_msg.K, dtype=np.float64).reshape(3, 3)
            print(f"RGB SHAPE: {rgb_image.shape}")
            print(f"DEPTH SHAPE: {depth_image.shape}") 

            selected_point = self._select_mobile_sam_point(
                rgb_image,
                current_point=self.mobile_sam_point
            )
            if selected_point is not None:
                self.mobile_sam_point = selected_point
                print(
                    f"Using MobileSAM point: "
                    f"({self.mobile_sam_point[0]}, {self.mobile_sam_point[1]})"
                )
            elif self.mobile_sam_point is not None:
                print(
                    "No new point selected. Reusing previous MobileSAM point: "
                    f"({self.mobile_sam_point[0]}, {self.mobile_sam_point[1]})"
                )
            else:
                rospy.logwarn(
                    "No MobileSAM point selected. Falling back to default point (300, 200)."
                )
            
            #segmap = self.simple_orange_segmentation_with_depth(rgb_image, depth_image, intrinsic_matrix)
            segmap = self.mobile_sam_segmap(rgb_image, point=self.mobile_sam_point)
            #segmap = self.get_cube_segmentation_mask(rgb_image, depth_image)
            #segmap = self.langsam_orange_segmentation(rgb_image, depth_image, intrinsic_matrix)
            depth_image = self._refine_object_depth(depth_image, segmap)
            print(f"SEGMAP SHAPE: {segmap.shape}, TRUE PIXELS: {np.count_nonzero(segmap)}")
            if not segmap.any():
                rospy.logwarn(
                    "Segmentation mask is empty. Tune HSV/depth thresholds or check object visibility."
                )
            else:
                seg_depths = depth_image[segmap & (depth_image > 0)]
                if seg_depths.size > 0:
                    print(
                        f"OBJECT DEPTH [mm] min={int(seg_depths.min())} "
                        f"median={int(np.median(seg_depths))} max={int(seg_depths.max())}"
                    )
                if self.save_segmentation_debug:
                    debug_vis = rgb_image.copy()
                    debug_vis[segmap] = (0.35 * debug_vis[segmap] + 0.65 * np.array([0, 255, 0])).astype(np.uint8)
                    debug_path = os.path.join(self.output_dir, "latest_segmentation_debug.png")
                    cv2.imwrite(debug_path, debug_vis)
                    print(f"Saved segmentation debug overlay: {debug_path}")

            return {
                'rgb_image': rgb_image,
                'depth_image': depth_image,
                'segmap': segmap,
                'intrinsic_matrix': intrinsic_matrix,
            }
        except Exception as e:
            rospy.logwarn(f"Error converting camera messages: {e}")
            return None

    def _select_mobile_sam_point(self, rgb_image, current_point=None, timeout=60.0):
        """
        Open an image preview and let the user pick the prompt point for MobileSAM.

        Returns:
            (x, y) tuple if selected/confirmed, otherwise None.
        """
        state = {
            'point': current_point,
            'clicked': False,
        }

        window_name = "Select Cube Point For MobileSAM"

        def _mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                state['point'] = (int(x), int(y))
                state['clicked'] = True

        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, _mouse_callback)
        except cv2.error as e:
            rospy.logwarn(f"Could not open point-selection window: {e}")
            return current_point

        print(
            "Point selector: left-click cube location. "
            "Press Enter/Space to confirm current point, 'r' to clear, ESC to cancel."
        )

        start_time = time.time()
        confirmed = False

        try:
            while not rospy.is_shutdown():
                preview = rgb_image.copy()
                if state['point'] is not None:
                    x, y = state['point']
                    cv2.drawMarker(
                        preview, (x, y), (0, 0, 255),
                        markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2
                    )
                    cv2.circle(preview, (x, y), 6, (0, 255, 0), -1)
                    cv2.putText(
                        preview,
                        f"Current point: ({x}, {y})",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )
                else:
                    cv2.putText(
                        preview,
                        "Left-click to select cube point",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                cv2.imshow(window_name, preview)

                if state['clicked']:
                    confirmed = True
                    break

                key = cv2.waitKey(30) & 0xFF
                if key in (13, 10, 32) and state['point'] is not None:
                    confirmed = True
                    break
                if key == ord('r'):
                    state['point'] = None
                    state['clicked'] = False
                if key == 27:  # ESC
                    break

                if (time.time() - start_time) > timeout:
                    print("Point selection timed out.")
                    break
        finally:
            try:
                cv2.destroyWindow(window_name)
                cv2.waitKey(1)
            except cv2.error:
                pass

        if confirmed and state['point'] is not None:
            return state['point']
        return None

    def _to_uint16_depth_mm(self, depth_image):  
        """Convert depth image to uint16 millimeters, handling float meter inputs safely."""  
        if depth_image.dtype == np.uint16:  
            return depth_image  
  
        depth_float = depth_image.astype(np.float32)  
        depth_float[~np.isfinite(depth_float)] = 0.0  
        max_depth = float(np.max(depth_float)) if depth_float.size else 0.0  
  
        # Heuristic: values <= 20 are likely meters; larger values likely already in mm.  
        if max_depth <= 20.0:  
            depth_mm = np.rint(depth_float * 1000.0)  
        else:  
            depth_mm = np.rint(depth_float)  
  
        depth_mm = np.clip(depth_mm, 0, np.iinfo(np.uint16).max).astype(np.uint16)  
        return depth_mm  

    def _compute_object_segmap(self, rgb_image, depth_image):  
        """  
        Build a boolean segmentation map for the orange target object.  
  
        Strategy:  
          1. Debug HSV values of orange pixels  
          2. Segment orange pixels in HSV with conservative ranges  
          3. Prefer pixels that are closer than dominant table/background depth  
          4. Clean the mask and keep the best connected component  
          5. Fallback mechanisms if primary detection fails  
        """  
        segmap = np.zeros(depth_image.shape, dtype=bool)  
        valid_mask = depth_image > 0  
        if not valid_mask.any():  
            return segmap  
  
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)  
          
        # Debug: Check HSV values of a sample orange region if we have any previous detection  
        if self.save_segmentation_debug:  
            # Save original RGB for debugging  
            cv2.imwrite('debug_original_rgb.png', rgb_image)  
            cv2.imwrite('debug_hsv.png', hsv_image)  
          
        # Primary orange detection with conservative range  
        orange_mask = cv2.inRange(hsv_image, self.orange_hsv_lower, self.orange_hsv_upper) > 0  
        orange_mask = orange_mask & valid_mask  
          
        # Debug: Save intermediate masks  
        if self.save_segmentation_debug:  
            cv2.imwrite('debug_orange_mask.png', orange_mask.astype(np.uint8) * 255)  
            cv2.imwrite('debug_valid_mask.png', valid_mask.astype(np.uint8) * 255)  
  
        # If primary detection fails, try alternative orange ranges  
        if not orange_mask.any():  
            print("Primary orange detection failed, trying alternative ranges...")  
              
            # Try reddish-orange range  
            alt_lower1 = np.array([0, 70, 70])  
            alt_upper1 = np.array([10, 255, 255])  
            alt_mask1 = cv2.inRange(hsv_image, alt_lower1, alt_upper1) > 0  
            alt_mask1 = alt_mask1 & valid_mask  
              
            # Try yellowish-orange range  
            alt_lower2 = np.array([18, 70, 70])  
            alt_upper2 = np.array([30, 255, 255])  
            alt_mask2 = cv2.inRange(hsv_image, alt_lower2, alt_upper2) > 0  
            alt_mask2 = alt_mask2 & valid_mask  
              
            # Use the best alternative  
            if alt_mask1.sum() > alt_mask2.sum():  
                orange_mask = alt_mask1  
                print("Using reddish-orange range")  
            else:  
                orange_mask = alt_mask2  
                print("Using yellowish-orange range")  
              
            if self.save_segmentation_debug:  
                cv2.imwrite('debug_alt_orange_mask.png', orange_mask.astype(np.uint8) * 255)  
  
        # If still no detection, try simple RGB fallback  
        if not orange_mask.any():  
            print("HSV detection failed, trying RGB fallback...")  
            lower_orange = np.array([150, 100, 50])   # R, G, B  
            upper_orange = np.array([255, 200, 150])  
            orange_mask = cv2.inRange(rgb_image, lower_orange, upper_orange) > 0  
            orange_mask = orange_mask & valid_mask  
            print(f"RGB fallback found {orange_mask.sum()} pixels")  
              
            if self.save_segmentation_debug:  
                cv2.imwrite('debug_rgb_fallback.png', orange_mask.astype(np.uint8) * 255)  
  
        if not orange_mask.any():  
            print("No orange pixels detected in any method")  
            return segmap  
  
        # Calculate dominant depth for foreground filtering  
        valid_depths = depth_image[valid_mask]  
        dominant_depth = int(np.bincount(valid_depths).argmax())  
  
        # Prefer raised orange object pixels over tabletop-like regions  
        foreground_mask = valid_mask & (depth_image < (dominant_depth - self.depth_margin_mm))  
        candidate = orange_mask & foreground_mask  
          
        # Fallback: if depth gating removes everything, keep color-only mask  
        if not candidate.any():  
            print("Depth gating removed all pixels, using color-only mask")  
            candidate = orange_mask  
          
        if not candidate.any():  
            return segmap  
  
        # Morphological operations to clean up the mask  
        candidate_u8 = candidate.astype(np.uint8)  
        kernel_open = np.ones((3, 3), np.uint8)  
        kernel_close = np.ones((self.seg_fill_kernel, self.seg_fill_kernel), np.uint8)  
        candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_OPEN, kernel_open)  
        candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_CLOSE, kernel_close)  
  
        if self.save_segmentation_debug:  
            cv2.imwrite('debug_after_morphology.png', candidate_u8)  
  
        # Connected component analysis  
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_u8, connectivity=8)  
        if num_labels <= 1:  
            print(f"Only background found, returning mask with {candidate_u8.sum()} pixels")  
            return candidate_u8.astype(bool)  
  
        # Find valid components (above minimum area)  
        valid_component_labels = []  
        for label in range(1, num_labels):  
            area = int(stats[label, cv2.CC_STAT_AREA])  
            if area >= self.seg_min_area_px:  
                valid_component_labels.append(label)  
  
        if not valid_component_labels:  
            print("No components meet minimum area requirement")  
            return segmap  
  
        print(f"Found {len(valid_component_labels)} valid components")  
  
        # Choose the best component: prefer closest, but fall back to largest if needed  
        best_label = None  
        best_depth = None  
          
        # First try: closest component  
        for label in valid_component_labels:  
            component_depths = depth_image[labels == label]  
            component_depths = component_depths[component_depths > 0]  
            if component_depths.size == 0:  
                continue  
            median_depth = float(np.median(component_depths))  
            if best_depth is None or median_depth < best_depth:  
                best_depth = median_depth  
                best_label = label  
  
        # If closest component is too far, try largest component  
        if best_label is not None and best_depth > (dominant_depth - self.depth_margin_mm * 2):  
            print("Closest component too far, trying largest component")  
            best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  
  
        if best_label is None:  
            print("No valid component selected")  
            return segmap  
  
        segmap = labels == best_label  
        print(f"Selected component {best_label} with {segmap.sum()} pixels")  
  
        # Remove depth outliers that break point cloud continuity  
        seg_depths = depth_image[segmap & (depth_image > 0)]  
        if seg_depths.size > 0:  
            median_depth = int(np.median(seg_depths))  
            depth_consistent = (  
                segmap  
                & (depth_image > 0)  
                & (np.abs(depth_image.astype(np.int32) - median_depth) <= self.object_depth_band_mm)  
            )  
            if np.count_nonzero(depth_consistent) > (self.seg_min_area_px // 2):  
                segmap = depth_consistent  
                print(f"After depth consistency filtering: {segmap.sum()} pixels")  
  
        # Final morphological cleanup  
        segmap_u8 = segmap.astype(np.uint8)  
        segmap_u8 = cv2.morphologyEx(segmap_u8, cv2.MORPH_CLOSE, kernel_close)  
        segmap_u8 = cv2.morphologyEx(segmap_u8, cv2.MORPH_OPEN, kernel_open)  
          
        # Final connected component to ensure single region  
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(segmap_u8, connectivity=8)  
        if num_labels > 1:  
            largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  
            segmap_u8 = (labels == largest_label).astype(np.uint8)  
          
        segmap = segmap_u8.astype(bool)  
        print(f"Final segmentation: {segmap.sum()} pixels")  
          
        if self.save_segmentation_debug:  
            cv2.imwrite('debug_final_segmap.png', segmap.astype(np.uint8) * 255)  
            # Create overlay visualization  
            overlay = rgb_image.copy()  
            overlay[segmap] = [0, 255, 0]  # Green overlay on segmented pixels  
            cv2.imwrite('debug_overlay.png', overlay)  
          
        return segmap.astype(bool)  

    def simple_orange_segmentation1(self, rgb_image, depth_image):  
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)  
        
        # Tighter orange range to avoid wood  
        lower_orange = np.array([8, 80, 80])  
        upper_orange = np.array([20, 255, 255])  
        
        mask = cv2.inRange(hsv, lower_orange, upper_orange)  
        
        # Remove distant objects (wooden background)  
        valid_depth = depth_image > 0  
        median_depth = np.median(depth_image[valid_depth])  
        depth_mask = depth_image < (median_depth - 50)  # Keep closer objects  
        mask = mask & depth_mask  
        
        # Clean up  
        kernel = np.ones((5, 5), np.uint8)  
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  
        
        return mask.astype(bool)


    def simple_depth_filter(self, depth_image, intrinsic_matrix, max_depth=1000):  
        """  
        Simple depth filter that doesn't require open3d  
        """  
        # Remove invalid depth values  
        valid_depth = depth_image > 0  
        
        # Remove objects beyond max_depth (in millimeters)  
        depth_mask = depth_image < max_depth  
        
        # Combine filters  
        return valid_depth & depth_mask  
      
  
    def langsam_orange_segmentation(self, rgb_image, text_prompt="orange cube"):  
        """  
        Lightweight text-prompt segmentation using LangSAM  
        """  
        model = LangSAM()  
        masks, boxes, phrases, logits = model.predict(rgb_image, text_prompt)  
        
        # Return the best mask (highest confidence)  
        if len(masks) > 0:  
            return masks[0].astype(bool)  
        else:  
            return np.zeros(rgb_image.shape[:2], dtype=bool)
  
    def simple_orange_segmentation_with_depth(self, rgb_image, depth_image, intrinsic_matrix):  
        # Your existing HSV segmentation  
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)  
        #lower_orange = np.array([5, 50, 50])  
        #upper_orange = np.array([25, 255, 255])
        
        #lower_orange = np.array([0, 40, 40])    # Lower hue, saturation, value  
        #upper_orange = np.array([30, 255, 255]) # Higher hue range  
        lower_orange = np.array([5, 50, 50])      # [Hue, Saturation, Value]  
        upper_orange = np.array([25, 255, 255])  
        mask = cv2.inRange(hsv, lower_orange, upper_orange)  
        
        # Apply simple depth filtering  
        valid_depth_mask = self.simple_depth_filter(depth_image, intrinsic_matrix)  
        
        # Combine HSV and depth filters  
        mask = mask & valid_depth_mask  
        
        # Clean up with morphological operations  
        kernel = np.ones((5, 5), np.uint8)  
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  
        
        return mask.astype(bool)

    def mobile_sam_segmap(self, rgb_image, point=None):
        if point is None:
            return mobile_sam_segmap_function(rgb_image)
        point_x, point_y = point
        return mobile_sam_segmap_function(rgb_image, point_x=point_x, point_y=point_y)

    def get_cube_segmentation_mask(self, rgb_image, depth_image):
        """
        Creates a boolean mask for a small cube.
        - rgb_image: (H, W, 3) uint8
        - depth_image: (H, W) float/uint16 (depth in mm)
        """
        # 1. Convert to HSV for better color thresholding
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        
        # Example: Range for a GREEN cube (Adjust these for your cube color!)
        lower_color = np.array([35, 50, 50])
        upper_color = np.array([85, 255, 255])
        color_mask = cv2.inRange(hsv, lower_color, upper_color)

        # 2. Depth Thresholding (Assumes table is at ~600mm, cube is ~20mm tall)
        # Filter points between 550mm and 595mm (just above table surface)
        depth_min, depth_max = 550, 595 
        depth_mask = cv2.inRange(depth_image, depth_min, depth_max)

        # 3. Combine Masks
        combined_mask = cv2.bitwise_and(color_mask, depth_mask)

        # 4. Morphological Cleanup (Remove noise)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask.astype(bool)

    def simple_orange_segmentation_with_depth11(self, rgb_image, depth_image, intrinsic_matrix):  
        # Your existing HSV segmentation  
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)  
        lower_orange = np.array([5, 50, 50])  
        upper_orange = np.array([25, 255, 255])  
        mask = cv2.inRange(hsv, lower_orange, upper_orange)  
        
        # Apply workspace-based depth filtering (from codebase)  
        from thousand_tasks.core.utils.segmentation_utils import get_valid_pixels_based_on_workspace  
        valid_depth_mask = get_valid_pixels_based_on_workspace(depth_image, intrinsic_matrix, T_WC)  
        
        # Combine HSV and depth filters  
        mask = mask & valid_depth_mask  
        
        # Clean up with morphological operations  
        kernel = np.ones((5, 5), np.uint8)  
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  
        
        return mask.astype(bool)

    
    def simple_orange_segmentation(self, rgb_image):  
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)  
        
        # Simple orange range (wider than your original)  
        lower_orange = np.array([8, 80, 80])  
        upper_orange = np.array([20, 255, 255])
        #lower_orange = np.array([8, 80, 80])  
        #upper_orange = np.array([20, 255, 255])    
        
        mask = cv2.inRange(hsv, lower_orange, upper_orange)  
        
        # Clean up with morphological operations  
        kernel = np.ones((5, 5), np.uint8)  
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  
        return mask.astype(bool)

    def lab_orange_segmentation(self, rgb_image):  
        lab = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2LAB)  
        
        # Orange in LAB space (more stable under different lighting)  
        lower_orange = np.array([120, 120, 150])  
        upper_orange = np.array([200, 200, 255])  
        
        mask = cv2.inRange(lab, lower_orange, upper_orange)  
        
        # Fill holes  
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  
        for cnt in contours:  
            cv2.drawContours(mask, [cnt], 0, 255, -1)  
        
        return mask.astype(bool)


    def _refine_object_depth(self, depth_image, segmap):  
        """Reduce depth holes/outliers inside object mask to avoid fragmented object point clouds."""  
        if not segmap.any():  
            return depth_image  
  
        refined = depth_image.copy()  
        obj_valid = segmap & (refined > 0)  
        if not obj_valid.any():  
            return refined  
  
        obj_depths = refined[obj_valid]  
        median_depth = int(np.median(obj_depths))  
        low = max(1, median_depth - self.object_depth_band_mm)  
        high = median_depth + self.object_depth_band_mm  
  
        outlier_or_missing = segmap & ((refined == 0) | (refined < low) | (refined > high))  
        refined[outlier_or_missing] = np.uint16(median_depth)  
  
        ys, xs = np.where(segmap)  
        if ys.size == 0:  
            return refined  
  
        y0, y1 = int(ys.min()), int(ys.max())  
        x0, x1 = int(xs.min()), int(xs.max())  
        roi = refined[y0:y1 + 1, x0:x1 + 1]  
        roi_mask = segmap[y0:y1 + 1, x0:x1 + 1]  
  
        if roi.shape[0] >= 5 and roi.shape[1] >= 5:  
            roi_blur = cv2.medianBlur(roi, 5)  
            roi[roi_mask] = roi_blur[roi_mask]  
            refined[y0:y1 + 1, x0:x1 + 1] = roi  
  
        return refined

    def _joint_state_callback(self, msg):
        """Callback for joint state messages."""
        with self.joint_state_lock:
            self.latest_joint_state = msg

    def _get_current_joint_state(self):
        """Get the current joint state."""
        with self.joint_state_lock:
            return self.latest_joint_state

    def _extract_arm_state(self, joint_state):
        """
        Extract arm joint positions and velocities from a JointState message.

        Returns:
            (positions, velocities) - numpy arrays of shape (num_joints,)
            or (None, None) if extraction fails
        """

        #print(f"joint_state.name: {joint_state.name}")
        positions = np.zeros(self.num_joints)
        velocities = np.zeros(self.num_joints)

        for i, name in enumerate(self.joint_names):
            try:
                idx = list(joint_state.name).index(name)
                positions[i] = joint_state.position[idx]
                if joint_state.velocity:
                    velocities[i] = joint_state.velocity[idx]
            except (ValueError, IndexError):
                return None, None

        #print(f"self.num_joints: {self.num_joints}")
        #print(f"positions: {positions}")
        #print(f"velocities: {velocities}")
        return positions, velocities

    def _get_moveit_eef_pose_matrix(self):
        """Get current EEF pose from MoveIt and convert to a 4x4 matrix."""
        pose = self.move_group.get_current_pose().pose
        q = np.array([
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ], dtype=np.float64)
        T_sb = quaternion_matrix(q)
        T_sb[0, 3] = pose.position.x
        T_sb[1, 3] = pose.position.y
        T_sb[2, 3] = pose.position.z
        return T_sb

    def _compute_eef_twist(self, joint_positions, joint_velocities):
        """
        Compute EEF body-frame twist from joint state.

        Uses:
          - space-frame twist:  V_s = J_s(q) * q_dot
          - body-frame twist:   V_b = Adjoint(T_sb^{-1}) * V_s

        Modern Robotics twist ordering is [wx, wy, wz, vx, vy, vz].
        We reorder to [vx, vy, vz, wx, wy, wz] for learning_thousand_tasks.

        Returns:
            twist: (6,) array in end-effector frame [vx, vy, vz, wx, wy, wz]
            T_sb: (4, 4) EEF pose
        """
        # Forward kinematics pose from MoveIt (planning model / kinematics).
        T_sb = self._get_moveit_eef_pose_matrix()

        # Space Jacobian and spatial twist [wx, wy, wz, vx, vy, vz]
        Js = mr.JacobianSpace(self.Slist, joint_positions)
        twist_spatial = Js @ joint_velocities

        # Convert to body frame (end-effector frame).
        twist_body = mr.Adjoint(mr.TransInv(T_sb)) @ twist_spatial

        # Reorder to [vx, vy, vz, wx, wy, wz].
        twist = np.array([
            twist_body[3], twist_body[4], twist_body[5],  # vx, vy, vz
            twist_body[0], twist_body[1], twist_body[2],  # wx, wy, wz
        ])

        return twist, T_sb

    def enable_teaching_mode(self):
        """Enable teaching mode by disabling motor torques."""
        print("\nEnabling teaching mode (torques OFF)...")
        self.bot.dxl.robot_torque_enable("group", "arm", False)
        print("Teaching mode enabled - you can now move the arm freely.")

    def disable_teaching_mode(self):
        """Disable teaching mode by enabling motor torques."""
        print("\nDisabling teaching mode (torques ON)...")
        self.bot.arm.capture_joint_positions()
        self.bot.dxl.robot_torque_enable("group", "arm", True)
        print("Teaching mode disabled - arm is now holding position.")

    def open_gripper(self):
        """Open the gripper."""
        self.bot.gripper.open(delay=0.5)
        self.gripper_is_closed = False
        print("Gripper OPENED")

    def close_gripper(self):
        """Close the gripper."""
        self.bot.gripper.close(delay=0.5)
        self.gripper_is_closed = True
        print("Gripper CLOSED")

    def start_recording(self):
        """Start recording a demonstration."""
        self._reset_current_demo()
        if self.pending_camera_data is None:
            print("No ready-position camera snapshot available. Capturing now...")
            self.pending_camera_data = self._capture_workspace_camera_data()

        if self.pending_camera_data is None:
            print("Could not capture camera data. Demonstration was not started.")
            return False

        self.current_demo['camera_data'] = self.pending_camera_data
        self.recording = True
        self.record_start_time = time.time()
        print("\n>>> RECORDING STARTED <<<")
        print("    Recording EEF twists [vx, vy, vz, wx, wy, wz, gripper]")
        print("    Using ready-position snapshot for MT3 camera files")
        return True

    def stop_recording(self):
        """Stop recording and finalize the demonstration."""
        self.recording = False
        print("\n>>> RECORDING STOPPED <<<")

        if len(self.current_demo['timestamps']) == 0:
            print("No data recorded in this demonstration.")
            return False

        # Convert to numpy arrays
        eef_twists = np.array(self.current_demo['eef_twists'])    # (T, 7)
        eef_poses = np.array(self.current_demo['eef_poses'])      # (T, 4, 4)
        timestamps = np.array(self.current_demo['timestamps'])    # (T,)

        # bottleneck_pose = first EEF pose
        bottleneck_pose = eef_poses[0]  # (4, 4)

        demo = {
            'eef_twists': eef_twists,
            'eef_poses': eef_poses,
            'bottleneck_pose': bottleneck_pose,
            'timestamps': timestamps,
            'camera_data': self.current_demo['camera_data'],
            'metadata': {
                'robot_model': self.robot_model,
                'joint_names': self.joint_names,
                'num_joints': self.num_joints,
                'record_rate': self.record_rate,
                'duration': timestamps[-1],
                'num_samples': len(timestamps),
                'timestamp': datetime.now().isoformat(),
                'format': '[vx, vy, vz, wx, wy, wz, gripper]',
                'task_name': self.task_name,
            }
        }

        self.demonstrations.append(demo)

        demo_num = len(self.demonstrations)
        duration = demo['metadata']['duration']
        samples = demo['metadata']['num_samples']
        print(f"Demo {demo_num} saved: {samples} samples over {duration:.2f} seconds")
        print(f"  EEF twists shape: {eef_twists.shape}")
        print(f"  Bottleneck pose (first EEF position): "
              f"[{bottleneck_pose[0,3]:.4f}, {bottleneck_pose[1,3]:.4f}, {bottleneck_pose[2,3]:.4f}]")
        return True

    def record_step(self):
        """Record a single timestep: compute EEF twist from current joint state."""
        if not self.recording:
            return

        joint_state = self._get_current_joint_state()
        if joint_state is None:
            return

        try:
            positions, velocities = self._extract_arm_state(joint_state)
            if positions is None:
                return

            # Compute EEF twist and pose
            twist, T_sb = self._compute_eef_twist(positions, velocities)

            # Append gripper state to form the 7-element vector
            gripper = 1.0 if self.gripper_is_closed else 0.0
            eef_twist_with_gripper = np.append(twist, gripper)  # (7,)

            timestamp = time.time() - self.record_start_time

            self.current_demo['eef_twists'].append(eef_twist_with_gripper)
            self.current_demo['eef_poses'].append(T_sb.copy())
            self.current_demo['timestamps'].append(timestamp)

        except Exception as e:
            rospy.logwarn(f"Error recording step: {e}")

    def save_demonstrations(self):
        """Save all collected demonstrations in learning_thousand_tasks format."""
        if not self.demonstrations:
            print("No demonstrations to save.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.output_dir, f"session_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)

        for i, demo in enumerate(self.demonstrations):
            demo_dir = os.path.join(session_dir, f"demo_{i:04d}")
            os.makedirs(demo_dir, exist_ok=True)

            # Save demo_eef_twists.npy - (T, 7) [vx, vy, vz, wx, wy, wz, gripper]
            np.save(os.path.join(demo_dir, "demo_eef_twists.npy"), demo['eef_twists'])

            # Save bottleneck_pose.npy - (4, 4) SE(3)
            np.save(os.path.join(demo_dir, "bottleneck_pose.npy"), demo['bottleneck_pose'])

            # Save task name
            with open(os.path.join(demo_dir, "task_name.txt"), 'w') as f:
                f.write(self.task_name)

            # Save MT3 camera files
            camera_data = demo.get('camera_data')
            if camera_data is not None:
                cv2.imwrite(
                    os.path.join(demo_dir, "head_camera_ws_rgb.png"),
                    camera_data['rgb_image']
                )
                cv2.imwrite(
                    os.path.join(demo_dir, "head_camera_ws_depth_to_rgb.png"),
                    camera_data['depth_image']
                )
                np.save(
                    os.path.join(demo_dir, "head_camera_ws_segmap.npy"),
                    camera_data['segmap']
                )
                np.save(
                    os.path.join(demo_dir, "head_camera_rgb_intrinsic_matrix.npy"),
                    camera_data['intrinsic_matrix']
                )
            else:
                rospy.logwarn(
                    f"No camera data for demo_{i:04d}; skipping MT3 camera files."
                )

            # Save additional data for reference
            np.save(os.path.join(demo_dir, "eef_poses.npy"), demo['eef_poses'])
            np.save(os.path.join(demo_dir, "timestamps.npy"), demo['timestamps'])
            with open(os.path.join(demo_dir, "metadata.pkl"), 'wb') as f:
                pickle.dump(demo['metadata'], f)

            print(f"  Saved demo_{i:04d}/")
            print(f"    demo_eef_twists.npy  {demo['eef_twists'].shape}")
            print(f"    bottleneck_pose.npy  {demo['bottleneck_pose'].shape}")

        print(f"\nSaved {len(self.demonstrations)} demonstrations to: {session_dir}")

    def go_to_ready_position(self):
        """Move the arm to a ready position for demonstration."""
        print("\nMoving to ready position...")
        self.disable_teaching_mode()
        self.bot.arm.go_to_sleep_pose()
        self.open_gripper()
        print("Capturing ready-position camera snapshot...")
        camera_data = self._capture_workspace_camera_data()
        if camera_data is not None:
            self.pending_camera_data = camera_data
            print("Ready-position snapshot captured.")
        else:
            print("Warning: Camera snapshot failed. Last valid snapshot will be reused.")
        print("Ready position reached.")

    def go_to_sleep(self):
        """Move the arm to sleep position."""
        print("\nMoving to sleep position...")
        self.disable_teaching_mode()
        self.bot.arm.go_to_sleep_pose()
        print("Sleep position reached.")

    def print_instructions(self):
        """Print usage instructions."""
        print("\n" + "="*60)
        print("DEMONSTRATION COLLECTION v2 - CONTROLS")
        print("  Output: [vx, vy, vz, wx, wy, wz, gripper]")
        print("="*60)
        print("  [o] - Open gripper")
        print("  [c] - Close gripper")
        print("  [s] - Start recording demonstration")
        print("  [e] - End recording demonstration")
        print("  [r] - Go to ready position")
        print("  [t] - Toggle teaching mode")
        print("  [p] - Print current EEF pose")
        print("  [q] - Quit and save demonstrations")
        print("="*60)

    def run_collection(self):
        """Main collection loop."""
        print("\n" + "="*60)
        print(f"{self.task_name.upper()} DEMONSTRATION COLLECTION v2")
        print(f"Target: {self.num_demos} demonstrations")
        print(f"Output format: learning_thousand_tasks")
        print("="*60)

        # Move to ready position
        self.go_to_ready_position()

        # Enable teaching mode
        self.enable_teaching_mode()

        self.print_instructions()

        teaching_mode_enabled = True
        rate = rospy.Rate(self.record_rate)

        print(f"\nCollected: 0/{self.num_demos} demonstrations")
        print("Press 's' to start recording when ready...")

        try:
            while not rospy.is_shutdown():
                # Get keyboard input
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1).lower()

                    if key == 'o':
                        self.open_gripper()

                    elif key == 'c':
                        self.close_gripper()

                    elif key == 's':
                        if not self.recording:
                            self.start_recording()

                    elif key == 'e':
                        if self.recording:
                            if self.stop_recording():
                                print(f"\nCollected: {len(self.demonstrations)}/{self.num_demos} demonstrations")
                                if len(self.demonstrations) >= self.num_demos:
                                    print("\nTarget number of demonstrations reached!")
                                    print("Press 'q' to save and quit, or continue collecting more.")
                                else:
                                    print("Press 's' to start the next demonstration...")

                    elif key == 'r':
                        if self.recording:
                            print("Cannot go to ready position while recording. Press 'e' first.")
                        else:
                            self.go_to_ready_position()
                            self.enable_teaching_mode()
                            teaching_mode_enabled = True

                    elif key == 't':
                        if self.recording:
                            print("Cannot toggle teaching mode while recording.")
                        elif teaching_mode_enabled:
                            self.disable_teaching_mode()
                            teaching_mode_enabled = False
                        else:
                            self.enable_teaching_mode()
                            teaching_mode_enabled = True

                    elif key == 'p':
                        joint_state = self._get_current_joint_state()
                        if joint_state:
                            positions, _ = self._extract_arm_state(joint_state)
                            if positions is not None:
                                T_sb = self._get_moveit_eef_pose_matrix()
                                print("\nCurrent EEF pose (SE(3)):")
                                print(f"  Position: x={T_sb[0,3]:.4f}, y={T_sb[1,3]:.4f}, z={T_sb[2,3]:.4f}")
                                print(f"  Rotation matrix:")
                                for row in T_sb[:3, :3]:
                                    print(f"    [{row[0]:7.4f} {row[1]:7.4f} {row[2]:7.4f}]")

                    elif key == 'q':
                        print("\nQuitting...")
                        break

                # Record data if recording
                if self.recording:
                    self.record_step()

                rate.sleep()

        except KeyboardInterrupt:
            print("\nInterrupted by user.")

        finally:
            # Cleanup
            if self.recording:
                self.stop_recording()

            # Save demonstrations
            if self.demonstrations:
                self.save_demonstrations()

            # Move to sleep
            try:
                self.go_to_sleep()
            except:
                pass

            print("\nDemonstration collection complete!")


def main():
    import select
    global select

    import argparse
    parser = argparse.ArgumentParser(
        description='Collect kinesthetic demonstrations directly in EEF twist format'
    )
    parser.add_argument('--robot_model', type=str, default='wx250s',
                        help='Robot model (default: wx250s)')
    parser.add_argument('--robot_name', type=str, default='wx250s',
                        help='Robot name/namespace (default: wx250s)')
    parser.add_argument('--num_demos', type=int, default=5,
                        help='Number of demonstrations to collect (default: 5)')
    parser.add_argument('--record_rate', type=int, default=30,
                        help='Recording rate in Hz (default: 30)')
    parser.add_argument('--task_name', type=str, default='pick_up_cube',
                        help='Task description saved in task_name.txt (default: pick_up_cube)')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("INTERBOTIX WX250S DEMONSTRATION COLLECTOR v2")
    print("  Output: learning_thousand_tasks format")
    print("="*60)
    print(f"Robot Model: {args.robot_model}")
    print(f"Robot Name: {args.robot_name}")
    print(f"Target Demos: {args.num_demos}")
    print(f"Record Rate: {args.record_rate} Hz")
    print(f"Task Name: {args.task_name}")
    print(f"Format: [vx, vy, vz, wx, wy, wz, gripper]")
    print("="*60)

    collector = DemonstrationCollectorV2(
        robot_model=args.robot_model,
        robot_name=args.robot_name,
        num_demos=args.num_demos,
        record_rate=args.record_rate,
        task_name=args.task_name,
    )

    collector.run_collection()


if __name__ == '__main__':
    main()
