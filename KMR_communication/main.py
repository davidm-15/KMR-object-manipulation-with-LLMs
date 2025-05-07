# main.py
import numpy as np
import time
import argparse

# Import modules
from utils import config
from . import kuka_api as api
from utils import utils
from .camera_handler import CameraHandler
from . import sequences
from . import gui

# run me with python -m KMR_communication.main --mode sequence --prompt "Find me a brown foam brick" --clean



# python -m KMR_communication.main --mode grabTest --item "box of jello" --clean
# python -m KMR_communication.main --mode gui
# plug-in outlet expander
def main():
    # Set numpy print options
    np.set_printoptions(suppress=True, precision=4) # Added precision formatting

    # Argument Parser (optional, to choose between GUI and sequences)
    parser = argparse.ArgumentParser(description="KUKA KMR IIWA Control Interface")
    parser.add_argument('--mode', type=str, default='gui', choices=['gui', 'sequence', 'calibrate', 'pick', "object", "scan", "estimate", "calc", "viz", "grabTest"],
                        help="Operation mode: 'gui', 'sequence', 'calibrate', 'pick'")
    parser.add_argument('--item', type=str, default='plug-in outlet expander', #'foam brick', #'mustard bottle',
                        help="Item name for detection/pose estimation in sequence mode")
    parser.add_argument('--prompt', type=str, default='find me a plug-in outlet expander')
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
        if args.mode == 'gui' or args.mode == 'sequence' or args.mode == 'pick' or args.mode == "scan" or args.mode == "estimate" or args.mode == "calc" or args.mode == "viz":
            print("Running  without camera functionality.")
        else:
            return

    # --- Run selected mode ---
    try:
        if args.mode == 'gui':
            print("Starting GUI mode...")
            gui.create_gui(cam_handler)
            # api.IsPositionInZone(13.666, 14.643, 179.0876, 4)
        elif args.mode == 'grabTest':
            sequences.test_grabbing(detection_item=args.item, camera_handler=cam_handler, clean_folder=args.clean)
        elif args.mode == 'estimate':
            sequences.estimate_the_transformation()
        elif args.mode == 'viz':
            sequences.visualize_transformations()
        elif args.mode == 'scan':
            sequences.find_object_6D_pose(camera_handler=cam_handler, detection_item="tuna fish can")
        elif args.mode == 'calc':
            sequences.Object_to_world()
        elif args.mode == 'sequence':
            print(f"Starting Execution Sequence mode for prompt: '{args.prompt}'")
            # Define sequence parameters
            sequences.execute_sequence(
                cam_handler,
                Only_current=False,
                do_camera_around=True,
                take_images=True,
                do_detection=True,
                do_6d_estimation=True,
                go_to_object=True,
                prompt=args.prompt,
                clean_folder=args.clean,
                output_folder=config.DEFAULT_GO_AROUND_OUTPUT_FOLDER # Or customize
            )
            # sequences.just_grab_the_object(np.array([
            #     [0.4548, 0.8138, 0.3618, 12658.4371],
            #     [-0.6125, 0.5807, -0.5363, 15149.0346],
            #     [-0.6466, 0.0223, 0.7625, 1203.2101],
            #     [0., 0., 0., 1.]
            # ]))
            # sequences.Object_to_world()
        elif args.mode == 'calibrate':
             print("Starting Calibration Capture mode...")
             sequences.move_to_hand_poses_and_capture(
                cam_handler,
                num_sets=1 # Or get from args
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