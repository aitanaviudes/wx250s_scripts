# pipeline worflow

## Camera and Interbotix arm setup

run in a terminal:

`roslaunch realsense2_camera rs_camera.launch align_depth:=true filters:=pointcloud` 

in a second terminal:

`roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=wx250s` 

you can check the topics of both the arm and intelisense camera doing:

`rostopic list` 

in another terminal, for simplicity, launch another rviz window:

`rviz` 

- Add a ‘`RobotModel`’

wx250s/base_link

/wx250s/robot_description

- Add ’`PointCloud2`’

Connect the Camera to the Wrist running in another terminal:

`rosrun tf static_transform_publisher 0.05 0 0.04 0 0 0 wx250s/ee_arm_link camera_link 100`

## Demonstration Collection pipeline:

run the demonstration collector file

`python3 demo_collect_current.py` 

instructions:

Controls during recording:

- `'o': Open gripper`
- `'c': Close gripper`
- `'s': Start recording demonstration`
- `'e': End recording demonstration`
- `'r': Go to ready position`
- `'t': Toggle teaching mode`
- `'p': Print current EEF pose`
- `'q': Quit and save demonstrations`

replay to check (no mt3 pipeline)

python3 replay_demo_v2_last_2.py --demo_dir collected_demos/session_20260306_152850/demo_0000

make deploy_mt3 

python3 call_replay_live_with_saved_data.py
