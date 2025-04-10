# main.py
import numpy as np
import time
import argparse

# Import modules
from . import config
from . import kuka_api as api
from . import utils
from .camera_handler import CameraHandler
from . import sequences
from . import gui

# run me with python -m KMR_communication.main --mode sequence --item "mustard bottle" --clean

def main():
    # Set numpy print options
    np.set_printoptions(suppress=True, precision=4) # Added precision formatting

    # Argument Parser (optional, to choose between GUI and sequences)
    parser = argparse.ArgumentParser(description="KUKA KMR IIWA Control Interface")
    parser.add_argument('--mode', type=str, default='gui', choices=['gui', 'sequence', 'calibrate', 'pick', "object"],
                        help="Operation mode: 'gui', 'sequence', 'calibrate', 'pick'")
    parser.add_argument('--item', type=str, default='plug-in outlet expander', #'foam brick', #'mustard bottle',
                        help="Item name for detection/pose estimation in sequence mode")
    parser.add_argument('--clean', action='store_true',
                        help="Clean output folder before running sequence/calibration")
    args = parser.parse_args()

    # Initialize Camera
    print("Initializing camera...")
    # Select camera type based on config or availability
    try:
        cam_handler = CameraHandler(camera_type='basler') # Or 'realsense'
    except Exception as e:
        print(f"Camera initialization failed: {e}")
        cam_handler = None

    if not cam_handler.is_ready():
        print("CRITICAL ERROR: Camera initialization failed. Exiting.")
        # Optionally, allow running GUI without camera?
        if args.mode == 'gui' or args.mode == 'sequence' or args.mode == 'pick':
            print("Running  without camera functionality.")
        else:
            return

    # --- Run selected mode ---
    try:
        if args.mode == 'gui':
            print("Starting GUI mode...")
            gui.create_gui(cam_handler)
        elif args.mode == 'sequence':
            print(f"Starting Execution Sequence mode for item: '{args.item}'")
            # Define sequence parameters
            # sequences.execute_sequence(
            #     cam_handler,
            #     Only_current=True,
            #     do_camera_around=True,
            #     take_images=True,
            #     do_detection=True,
            #     do_6d_estimation=True,
            #     detection_item=args.item,
            #     clean_folder=args.clean,
            #     output_folder=config.DEFAULT_GO_AROUND_OUTPUT_FOLDER # Or customize
            # )
            sequences.Go_to_the_position()
        elif args.mode == 'calibrate':
             print("Starting Calibration Capture mode...")
             sequences.move_to_hand_poses_and_capture(
                 cam_handler,
                 num_sets=5 # Or get from args
             )
        elif args.mode == 'pick':
            print("Starting JustPickIt Sequence mode...")
            sequences.just_pick_it_full_sequence()
        
        elif args.mode == 'object':
            print("Starting Object Pose Estimation mode...")
            # Define parameters for object pose estimation
            sequences.Object_to_world()

        else:
            print(f"Unknown mode: {args.mode}")

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging

    finally:
        # Clean up resources
        print("Closing camera...")
        cam_handler.close()
        print("Program finished.")


if __name__ == "__main__":
    main()