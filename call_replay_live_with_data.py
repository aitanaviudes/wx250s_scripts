#!/usr/bin/env python3    
"""      
Script to load saved MT3 data and replay it using the DemoReplayer.      
This script loads the live_bottleneck_pose and end_effector_twists      
saved by deploy_mt3.py and passes them to the ROS replay system.      
"""      
      
import numpy as np      
import rospy      
from pathlib import Path      
from replay_live import DemoReplayer      
      
      
def main():      
    """Load saved data and run replay."""      
    # REMOVE THIS LINE: rospy.init_node('mt3_replay_node')      
          
    # Path to saved data directory - UPDATED PATH    
    save_dir = Path('/home/aitana_viudes/1000_tasks/learning_thousand_tasks/saved_data')      
          
    try:      
        # Load the saved bottleneck pose and twists      
        print(f"Loading data from: {save_dir}")      
        live_bottleneck_pose = np.load(save_dir / 'live_bottleneck_pose.npy')      
        end_effector_twists = np.load(save_dir / 'end_effector_twists.npy')      
              
        print(f"Loaded live_bottleneck_pose: {live_bottleneck_pose.shape}")      
        print(f"Loaded end_effector_twists: {end_effector_twists.shape}")      
              
        # Initialize and run the replayer      
        replayer = DemoReplayer()      
        replayer.set_live_data(live_bottleneck_pose, end_effector_twists)      
        replayer.run_with_live_data(speed_factor=1.0, dry_run=False)      
        print("Replay completed successfully!")      
              
    except FileNotFoundError as e:      
        print(f"Error: Could not find saved data files. {e}")      
        print("Please run deploy_mt3.py first to generate the saved data.")      
    except Exception as e:      
        print(f"Error during replay: {e}")      
        import traceback      
        traceback.print_exc()      
      
      
if __name__ == '__main__':      
    main()
