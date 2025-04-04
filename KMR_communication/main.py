# main.py
import numpy as np
import time
import argparse

# Import modules
from . import config
from . import kuka_api as api
from . import utils
# Assuming camera_handler.py is directly inside KMR_communication
from .camera_handler import CameraHandler
from . import sequences
from . import gui

def main():
    # Set numpy print options
    np.set_printoptions(suppress=True, precision=4) # Added precision formatting

    # Argument Parser (optional, to choose between GUI and sequences)
    parser = argparse.ArgumentParser(description="KUKA KMR IIWA Control Interface")
    parser.add_argument('--mode', type=str, default='gui', choices=['gui', 'sequence', 'calibrate', 'pick'],
                        help="Operation mode: 'gui', 'sequence', 'calibrate', 'pick'")
    parser.add_argument('--item', type=str, default='plug-in outlet expander', #'foam brick', #'mustard bottle',
                        help="Item name for detection/pose estimation in sequence mode")
    parser.add_argument('--clean', action='store_true',
                        help="Clean output folder before running sequence/calibration")
    args = parser.parse_args()

    # Initialize Camera
    print("Initializing camera...")
    # Select camera type based on config or availability
    cam_handler = CameraHandler(camera_type='basler') # Or 'realsense'

    if not cam_handler.is_ready():
        print("CRITICAL ERROR: Camera initialization failed. Exiting.")
        # Optionally, allow running GUI without camera?
        # if args.mode == 'gui':
        #     print("Running GUI without camera functionality.")
        #     gui.create_gui(camera_handler=None) # Pass None or dummy object
        # else:
        #     return # Cannot run sequences without camera
        return # Exit if camera failed

    # --- Run selected mode ---
    try:
        if args.mode == 'gui':
            print("Starting GUI mode...")
            gui.create_gui(cam_handler)
        elif args.mode == 'sequence':
            print(f"Starting Execution Sequence mode for item: '{args.item}'")
            # Define sequence parameters
            sequences.execute_sequence(
                cam_handler,
                do_detection=True,
                do_6d_estimation=True,
                detection_item=args.item,
                clean_folder=args.clean,
                output_folder=config.DEFAULT_GO_AROUND_OUTPUT_FOLDER # Or customize
            )
        elif args.mode == 'calibrate':
             print("Starting Calibration Capture mode...")
             sequences.move_to_hand_poses_and_capture(
                 cam_handler,
                 num_sets=5 # Or get from args
             )
        elif args.mode == 'pick':
             print("Starting JustPickIt Sequence mode...")
             # Ensure the 'estimated_pose.json' file exists and is correct before running
             input("Ensure 'images/JustPickIt/estimated_pose.json' is ready. Press Enter to continue...")
             sequences.just_pick_it_full_sequence(cam_handler)

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