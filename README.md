# Pipeline Worflow

## Camera and Interbotix Arm Setup

run in a terminal:

`roslaunch realsense2_camera rs_camera.launch align_depth:=true filters:=pointcloud` 

in a second terminal:

`roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=wx250s` 

you can check the topics of both the arm and intelisense camera doing:

`rostopic list` 

<br>
In another terminal, for simplicity, launch another rviz window: <br><br>

`rviz` 

- In the "Global Options" panel on the left, change the Fixed Frame from `map` to `wx250s/base_link`

- Click the Add button at the bottom left, and select `RobotModel`. You should now see the 3D model of your WidowX arm.

- In the "Displays" panel on the left, find the field labeled "Description Topic" and change it to `/wx250s/robot_description`

- Click "Add" again, go to the "By topic" tab, and find /camera/depth/color/points. Select `PointCloud2`
  
<br>
Connect the Camera to the Wrist running in another terminal:<br><br>

`rosrun tf static_transform_publisher 0.05 0 0.04 0 0 0 wx250s/ee_arm_link camera_link 100` <br><br>

## Scripts:

In a terminal, go to:
`interbotix_ws/src/interbotix_ros_manipulators/interbotix_ros_xsarms/examples/python_demos`

here you will see different files but the relevant ones are three:
- demo_collect_current.py
- replay_live.py
- call_replay_live_with_data.py

There is another script we will be using but this one is located in the docker image that has been built, the script can be found here: <br>
`1000_tasks/learning_thousand_tasks/deployments/deploy_mt3.py` <br><br>

## Demonstration Collection Pipeline:

### Code Architecture & Key Functions

The `DemonstrationCollectorV2` class manages the lifecycle of a robot recording session. Below are the primary functional blocks:

#### 1. Kinematics & State Estimation
* **`_compute_eef_twist`**: Uses the **Modern Robotics** library to compute the end-effector (EEF) body-frame twist. It calculates the Space Jacobian ($J_s$) from joint positions and converts the resulting spatial twist into the body frame ($V_b$).
* **`_extract_arm_state`**: Parses incoming ROS `JointState` messages to isolate the specific joint positions and velocities for the `wx250s` arm.

#### 2. Computer Vision & Segmentation
* **`_capture_workspace_camera_data`**: A blocking call that synchronizes and captures a single frame of RGB, Depth, and Camera Intrinsics via ROS topics.
* **`simple_orange_segmentation_with_depth`**: The primary perception pipeline. It combines **HSV color thresholding** with **depth-gating** to isolate the target object (orange cube) from the table surface.
* **`_refine_object_depth`**: A cleanup utility that uses a median filter on the segmented depth map to remove "holes" or outliers, ensuring a clean point cloud for later processing.

#### 3. Recording & Hardware Control
* **`start_recording` / `stop_recording`**: Manages the data buffers. Recording only begins once a "Ready Position" camera snapshot is successfully verified.
* **`enable_teaching_mode`**: Disables motor torques on the Interbotix arm, allowing for **kinesthetic teaching** (moving the arm by hand).
* **`record_step`**: The high-frequency loop function (running at the defined `record_rate`) that samples the current twist and appends it to the trajectory buffer.

#### 4. Data Serialization
* **`save_demonstrations`**: Handles the directory creation and converts Python lists into `.npy` (NumPy) and `.png` files. It enforces the `learning_thousand_tasks` naming convention required for training.

<br> 

Inside the `python_demos` directory run the demonstration collector file to record a demonstration:

`python3 demo_collect_current.py` 

Example of how to do it: [https://drive.google.com/drive/u/0/home](https://drive.google.com/file/d/1_yAW-ArX_D4vTTU7kbD1qVi1sqcG0pif/view?usp=sharing)

Instructions:

Controls during recording:

- `'o': Open gripper`
- `'c': Close gripper`
- `'s': Start recording demonstration`
- `'e': End recording demonstration`
- `'r': Go to ready position`
- `'t': Toggle teaching mode`
- `'p': Print current EEF pose`
- `'q': Quit and save demonstrations`

- press `Ctrl + C` to stop the script

The following data will save within the collected_demos/session_[TIMESTAMP]/demo0000/ directory.

  ### Demonstration Data Structure (`demo0000`)

| File Name | Data Type | Dimensions / Size | Description |
| :--- | :--- | :--- | :--- |
| **demo_eef_twists.npy** | NumPy Array (`float64`) | (T, 7) | Time-series of EEF twists: [vx, vy, vz, wx, wy, wz, gripper] |
| **bottleneck_pose.npy** | NumPy Array (`float64`) | (4, 4) | The initial SE(3) transformation matrix of the end-effector |
| **task_name.txt** | Plain Text | N/A | The string name of the task (e.g., pick_up_cube) |
| **head_camera_ws_rgb.png** | Image (uint8) | (720, 1280, 3) | RGB workspace snapshot taken at the "Ready Position" |
| **head_camera_ws_depth_to_rgb.png** | Image (uint16) | (720, 1280) | Aligned depth map in millimeters (mm) |
| **head_camera_ws_segmap.npy** | NumPy Array (`bool`) | (720, 1280) | Boolean mask where True identifies the target object |
| **head_camera_rgb_intrinsic_matrix.npy** | NumPy Array (`float64`) | (3, 3) | The camera intrinsic matrix K |
| **eef_poses.npy** | NumPy Array (`float64`) | (T, 4, 4) | Full sequence of SE(3) matrices for the entire trajectory |
| **timestamps.npy** | NumPy Array (`float64`) | (T,) | Relative time in seconds for each recorded timestep |
| **metadata.pkl** | Python Pickle | Dictionary | Serialized metadata (robot model, joint names, rate, etc.) |

**Note:** `T` represents the total number of timesteps recorded.

Stop the `roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=wx250s` process before moving on to pass our collected demonstration to the mt3 pipeline. We are going to check the the demonstration has been correctly recorded by replaying it in the rviz simulator (real life robot will not replay it): <br><br>
`roslaunch interbotix_xsarm_descriptions xsarm_description.launch robot_model:=wx200 use_joint_pub_gui:=true`
<br><br>
## Demonstration Replay Pipeline

To replay a previously recorded demonstration, run the `replay_live.py` script in the `python_demos` directory. The system will automatically detect if the physical robot is active or if it should run in RViz simulation mode.

### Example: Replaying a specific demonstration
`python3 replay_live.py --demo_dir collected_demos/session_20260306_152850/demo_0000`

### Code Architecture & Key Functions

The `replay_live.py` script utilizes a **Modern Robotics** based IK solver to translate end-effector targets into joint angles.

1. **Initialization:** The script checks for the `xs_sdk` node. If missing, it enters **Simulation Mode**.
2. **Trajectory Synthesis:** It loads the `.npy` files and uses `se3_exp` to integrate twists if necessary.
3. **IK Verification:** It runs an IK pass for every waypoint, ensuring the path is physically reachable.
4. **Execution:**
   - In **Real Mode**, it sends trajectory segments to the Dynamixel controllers.
   - In **Sim Mode**, it publishes JointStates to update the RViz model.
<br><br>
## Updating Inference and Demonstration Files in Docker Image
If everything looks as expected, then we shouold run the bash script `update_demo.sh` inside `collected_demos`: <br><br>
`./update_demo.sh -s session....` <br><br>
What this file does is basically update the demonstration `pick_up_shoe` in `1000_tasks/learning_thousand_tasks/assets/demonstrations/pick_up_shoe` and also update the files inside `1000_tasks/learning_thousand_tasks/assets/inference_example`.
<br><br>
## Deploying MT3 Pipeline
Now we can execute `make deploy_mt3` (inside `1000_tasks/learning_thousand_tasks/`). This runs the docker image, it is the main entry point of the mt3 pipeline and it also executes the file `1000_tasks/learning_thousand_tasks/deployments/deploy_mt3.py` which we will also be working with. The file has been updated towards the end to inlcude the following:

    save_dir = Path('/workspace/saved_data')  # This is mounted to your host  
    save_dir.mkdir(parents=True, exist_ok=True)  
    
    np.save(save_dir / 'live_bottleneck_pose.npy', live_bottleneck_pose)  
    np.save(save_dir / 'end_effector_twists.npy', end_effector_twists)

This way, once the updated `live_bottleneck_pose` and `end_effector_twists` have been produced, we can save them to a directory in: `1000_tasks/learning_thousand_tasks/saved_data` we can access from outside the docker image.

Now we can run the last script: `python3 call_replay_live_with_saved_data.py` which will basically load saved MT3 data and replay it using the DemoReplayer. This script loads the live_bottleneck_pose and end_effector_twists saved by deploy_mt3.py and passes them to the ROS replay system. 
Make sure to execute this file with rviz simulator first (not real life) to see if the output of the mt3 pipeline looks correct and safe.


# TODO

1. Update segmentation mask pipeline.
   What learning 1000 tasks do:
For the first frame, they use LangSAM (Language Segment Anything Model) to generate workspace segmentation from language prompts like "grey shoe on table" README.md:179 . This creates the binary mask head_camera_ws_segmap.npy that identifies the target object. We should use the same approach as them and not hardcode a function for orange cubes as we are currently doing in `simple_orange_segmentation_with_depth`
2. We should subsititue our inverse kinematics functions and use Moveit instead.
