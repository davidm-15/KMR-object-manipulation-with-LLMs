# sequences.py
import time
import json
import numpy as np
import cv2
import os
import winsound
import open3d as o3d

from . import config
from . import kuka_api as api
from . import utils
from .camera_handler import CameraHandler # If used directly
from communication.client import send_image_to_server
from communication.client import send_for_pose_estimation
from scipy.spatial.transform import Rotation as R
from robot import iiwa
import utils.image_utils as image_utils

def get_current_state_data(camera_handler: CameraHandler, image_filename_base: str) -> dict | None:
    """Captures image, gets robot state, and prepares data dict for saving."""
    timestamp = int(time.time())
    image_filename = f"{image_filename_base}_{timestamp}.png"
    image_full_path = os.path.join(config.DEFAULT_GO_AROUND_OUTPUT_FOLDER, image_filename) # Example path

    # Capture Image
    image = camera_handler.capture_image()
    if image is None:
        print("Failed to capture image.")
        return None # Indicate failure

    # Get Robot State
    kmr_pose = api.get_pose()
    iiwa_pos = api.get_iiwa_position()
    iiwa_joints = api.get_iiwa_joint_position()

    if not all([kmr_pose, iiwa_pos, iiwa_joints]):
        print("Failed to get complete robot state.")
        return None # Indicate failure

    # Calculate Camera in World
    T_world_cam = utils.calculate_camera_in_world(kmr_pose, iiwa_pos)
    if T_world_cam is None:
        print("Failed to calculate camera pose in world.")
        # Continue without it or return None? Decide based on requirements.
        T_world_cam_list = None
    else:
        T_world_cam_list = T_world_cam.tolist()


    # Save Image (consider moving save outside if detection adds overlays)
    # camera_handler.save_image(image, image_full_path) # Moved after potential detection overlay

    data = {
        "image_timestamp": timestamp,
        "image_filename": image_filename, # Relative filename
        "kmr_pose": kmr_pose,
        "iiwa_position": iiwa_pos,
        "iiwa_joints": iiwa_joints,
        "T_world_camera": T_world_cam_list, # Store as list for JSON
        # Store raw image here? Only if necessary, usually just filename
    }
    # Return image separately if needed for processing
    return data, image


def process_image_and_estimate_pose(image: np.ndarray, item_name: str, T_world_cam: np.ndarray, output_path_detected: str) -> np.ndarray | None:
    """Performs detection and 6D pose estimation, returns object pose in world."""
    image_bytes = utils.ndarray_to_bytes(image)
    if not image_bytes:
        return None

    print(f"Sending image for detection of '{item_name}'...")
    bounding_boxes = send_image_to_server(image_bytes, item_name)

    if not bounding_boxes:
        print("Object not detected.")
        # Save original image if no detection
        cv2.imwrite(output_path_detected.replace('_detected', ''), image)
        print(f"Saved original image to {output_path_detected.replace('_detected', '')}")
        return None # No object found

    print("\n" * 2 + "OBJECT DETECTED!" + "\n" * 2)
    print("Bounding boxes:", bounding_boxes)

    # Draw boxes on image
    img_detected = image.copy()
    for (x1, y1, x2, y2) in bounding_boxes:
        cv2.rectangle(img_detected, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.imwrite(output_path_detected, img_detected) # Save image with detection boxes
    print(f"Saved detected image to {output_path_detected}")


    # --- Perform 6D Pose Estimation (using the first detection) ---
    print("Sending for 6D pose estimation...")
    
    pose_result_dict = send_for_pose_estimation(
        image_bytes,
        bounding_boxes,
        item_name
    )


    # Check for errors returned from the server/worker
    if pose_result_dict is None or "error" in pose_result_dict:
        error_msg = pose_result_dict.get("error", "Unknown error") if pose_result_dict else "Returned None"
        print(f"6D Pose estimation failed: {error_msg}")
        return None # Indicate failure

    # Check if poses were actually found
    if not pose_result_dict.get("pose"):
        print(f"6D Pose estimation ran successfully but found no poses for '{item_name}'.")
        return None # Indicate no pose found

    # Extract 4x4 pose matrix (assuming server returns it correctly)
    # Pose from server is T_cam_obj (Camera -> Object)
    T_cam_obj_list = pose_result_dict["pose"]
    T_cam_obj = np.array(T_cam_obj_list)
    T_cam_obj[:3, 3] *= 1000

    print(f"Estimated T_cam_obj:\n{T_cam_obj}")

    # Calculate Object in World
    if T_world_cam is None:
        print("Cannot calculate object in world: T_world_cam is None.")
        return None


    T_world_obj = utils.calculate_object_in_world(T_world_cam, T_cam_obj)
    print(f"Calculated T_world_obj:\n{T_world_obj}")

    return T_world_obj # Return the object pose in world coordinates


def estimate_the_transformation():
    object_1_in_cam = np.array([
        [-0.1489, -0.7005, -0.698, 60.9806],
        [-0.9499, -0.0947, 0.2978, 124.7719],
        [-0.2747, 0.7073, -0.6513, 584.4992],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    object_2_in_cam = np.array([
        [-0.1336, 0.0815, -0.9877, 1.862],
        [-0.7777, -0.6264, 0.0536, 29.4034],
        [-0.6143, 0.7753, 0.1471, 375.9291],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    object_3_in_cam = np.array([
        [-0.0433, 0.4193, -0.9068, -13.5429],
        [-0.9092, -0.3928, -0.1383, 38.7004],
        [-0.4142, 0.8185, 0.3982, 512.3289],
        [0.0, 0.0, 0.0, 1.0]
    ])

    def estimate_object_world_position(objects_in_cam, cam_transforms):
        """Estimates object positions in world coordinates using camera transforms.
        
        Args:
            objects_in_cam: List of 4x4 homogeneous transformation matrices (object in camera frame)
            cam_transforms: List of camera poses in world coordinates (from robot state data)
            
        Returns:
            List of estimated object poses in world coordinates
        """
        if len(objects_in_cam) != len(cam_transforms):
            print(f"Warning: Number of objects ({len(objects_in_cam)}) doesn't match number of camera transforms ({len(cam_transforms)})")
            return None
        
        object_world_positions = []
        
        # Process each object-camera pair
        for i, (T_cam_obj, T_world_cam) in enumerate(zip(objects_in_cam, cam_transforms)):
            # Calculate Object in World using T_world_cam @ T_cam_obj
            T_world_obj = T_world_cam @ T_cam_obj
            object_world_positions.append(T_world_obj)
            print(f"Estimated position for object {i+1} in world frame: {T_world_obj[:3, 3]}")
        
        return object_world_positions

    def load_camera_transforms():
        """Loads camera transforms from the JSON data file."""
        try:
            with open("images/GoAround/data.json", "r") as f:
                data = json.load(f)
            
            # Extract camera transforms from the data
            camera_transforms = []
            for entry in data:
                if "T_world_camera" in entry and entry["T_world_camera"] is not None:
                    camera_transforms.append(np.array(entry["T_world_camera"]))
            
            if len(camera_transforms) < 3:
                print(f"Warning: Found only {len(camera_transforms)} camera transforms, expected at least 3")
            
            return camera_transforms[:3]  # Return the first three transforms
        except Exception as e:
            print(f"Error loading camera transforms: {e}")
            return None

    # Get the camera transforms from the data file
    camera_transforms = load_camera_transforms()

    if camera_transforms:        
        objects_in_cam = [object_1_in_cam, object_2_in_cam, object_3_in_cam]
        
        # Estimate object positions in world coordinates
        world_positions = estimate_object_world_position(objects_in_cam, camera_transforms)
        
        if world_positions:
            print("\nFinal estimated object positions in world frame:")
            for i, T_world_obj in enumerate(world_positions):
                print(f"Object {i+1}:\n{np.round(T_world_obj, 4)}")
    else:
        print("Failed to load camera transforms. Cannot estimate object world positions.")















def find_object_6D_pose(camera_handler: CameraHandler, **kwargs):
    output_folder = kwargs.get("output_folder", config.DEFAULT_GO_AROUND_OUTPUT_FOLDER)
    do_detection = kwargs.get("do_detection", True)
    do_6d_estimation = kwargs.get("do_6d_estimation", True)
    detection_item = kwargs.get("detection_item", "mustard bottle")
    json_filename = kwargs.get("json_filename", "data.json")
    input_positions_file = kwargs.get("input_positions_file", "image_processing\\grabbing_poses\\find_3D.json")
    clean_folder = kwargs.get("clean_folder", True) # Renamed from clean_fodler

    if clean_folder:
        utils.clean_directory(output_folder)

    
    with open(input_positions_file, "r") as f:
        input_positions = json.load(f)

    T_list = []

    for joints in input_positions:
        joints = joints["joints"]
        response = api.goto_joint(
            joints["A1"], joints["A2"], joints["A3"], joints["A4"],
            joints["A5"], joints["A6"], joints["A7"]
        )


        T_world_obj_list = get_and_save_image_data(
            camera_handler,
            output_folder=output_folder,
            do_detection=do_detection,
            do_6d_estimation=do_6d_estimation,
            detection_item=detection_item,
            json_filename=json_filename
        )
        if T_world_obj_list is not None:
            T_list.append(T_world_obj_list)

        print(f"Object pose in world coordinates:\n{T_world_obj_list}")

    # Calculate average transformation if we have multiple poses
    if len(T_list) > 0:
        # Save all detected object positions to the output folder
        output_file = os.path.join(output_folder, f"{detection_item}_positions.json")
        
        # Create data structure for the positions
        position_data = {
            "timestamp": int(time.time()),
            "detection_item": detection_item,
            "positions": T_list
        }
        
        # Save to JSON file
        print(f"Saving {len(T_list)} object position(s) to {output_file}")
        utils.save_json_data(output_file, position_data)
        
        # Also return the last detected position
        return T_list[-1]
    else:
        print("No object positions were detected.")
        return None
    


def get_and_save_image_data(camera_handler: CameraHandler, **kwargs):
    """
    Captures image, gets state, optionally performs detection/pose estimation,
    and saves all data.
    """
    output_folder = kwargs.get("output_folder", config.DEFAULT_GO_AROUND_OUTPUT_FOLDER)
    do_detection = kwargs.get("do_detection", False)
    do_6d_estimation = kwargs.get("do_6d_estimation", False)
    detection_item = kwargs.get("detection_item", "mustard bottle")
    json_filename = kwargs.get("json_filename", "data.json")

    os.makedirs(output_folder, exist_ok=True)
    json_filepath = os.path.join(output_folder, json_filename)

    print("-" * 10)
    print(f"Capturing data point: Detection={do_detection}, PoseEst={do_6d_estimation}, Item='{detection_item}'")

    time.sleep(kwargs.get("delay_before_capture", 2.0)) # Allow settling time

    # --- Get State Data and Image ---
    state_data, image = get_current_state_data(camera_handler, "image")
    if state_data is None or image is None:
        print("Failed to get state or image. Skipping this data point.")
        return # Or raise an error

    image_filename = state_data["image_filename"] # Relative filename
    image_save_path = os.path.join(output_folder, image_filename)
    image_detected_save_path = image_save_path.replace(".png", "_detected.png")


    # --- Perform Detection and Pose Estimation (if requested) ---
    T_world_obj_list = None # Initialize
    if do_detection and state_data.get("T_world_camera"):
        T_world_cam = np.array(state_data["T_world_camera"]) # Convert list back to array for calculations
        T_world_obj = process_image_and_estimate_pose(image, detection_item, T_world_cam, image_detected_save_path)
        if T_world_obj is not None:
            T_world_obj_list = T_world_obj.tolist() # Convert back to list for JSON storage
        else:
            # Save the original image if detection happened but pose failed, or no detection
            if not os.path.exists(image_detected_save_path): # Check if detection image was already saved
                camera_handler.save_image(image, image_save_path)

    elif not do_detection:
         # Save the original image if no detection is performed
         camera_handler.save_image(image, image_save_path)

    # Add results to the data dictionary
    state_data["detection_item"] = detection_item if do_detection else None
    state_data["T_world_object"] = T_world_obj_list # Will be None if pose est wasn't done or failed

    # --- Save Data to JSON ---
    utils.append_to_json_list(json_filepath, state_data)

    time.sleep(kwargs.get("delay_after_save", 1.0))

    if T_world_obj_list is not None:
        return T_world_obj_list # Return the object pose in world coordinates
    else:
        return None 


def go_to_handpose_position(idx: int):
    """Moves the robot to a specific position based on the HandPoses.json file with given index."""
    # Load hand poses from file
    hand_poses_file = config.HAND_POSES_FILE
    hand_poses_data = utils.load_json_data(hand_poses_file)
    
    if not isinstance(hand_poses_data, list) or not hand_poses_data:
        print(f"Error: No valid data found in {hand_poses_file}")
        return False
        
    # Check if index is within valid range
    if idx < 0 or idx >= len(hand_poses_data):
        print(f"Error: Index {idx} is out of range. File contains {len(hand_poses_data)} poses.")
        return False
        
    # Get the specified pose
    pose_data = hand_poses_data[idx]
    
    # Verify that position data exists
    if "position" not in pose_data:
        print(f"Error: Position data not found in pose at index {idx}")
        return False
        
    position = pose_data["position"]
    
    # Check if all required position keys exist
    required_keys = ["x", "y", "z", "A", "B", "C"]
    if not all(key in position for key in required_keys):
        print(f"Error: Missing position keys in pose at index {idx}")
        return False
        
    print(f"Moving to position from index {idx}: {position}")
    
    # Command robot to move to position
    response = api.goto_position(
        position["x"], position["y"], position["z"],
        position["A"], position["B"], position["C"]
    )
    
    if response and response.ok and response.text.strip() == "OK":
        print(f"Successfully moved to position from index {idx}")
        return True
    else:
        print(f"Failed to move to position from index {idx}: {response.text if response else 'No response'}")
        return False


def go_to_handpose_joints(idx: int):
    """Moves the robot to specific joint positions based on the HandPoses.json file with given index."""
    # Load hand poses from file
    hand_poses_file = config.HAND_POSES_FILE
    hand_poses_data = utils.load_json_data(hand_poses_file)
    
    if not isinstance(hand_poses_data, list) or not hand_poses_data:
        print(f"Error: No valid data found in {hand_poses_file}")
        return False
        
    # Check if index is within valid range
    if idx < 0 or idx >= len(hand_poses_data):
        print(f"Error: Index {idx} is out of range. File contains {len(hand_poses_data)} poses.")
        return False
        
    # Get the specified pose
    pose_data = hand_poses_data[idx]
    
    # Verify that joints data exists
    if "joints" not in pose_data:
        print(f"Error: Joints data not found in pose at index {idx}")
        return False
        
    joints = pose_data["joints"]
    
    # Check if all required joint keys exist
    required_keys = [f"A{i}" for i in range(1, 8)]
    if not all(key in joints for key in required_keys):
        print(f"Error: Missing joint keys in pose at index {idx}")
        return False
        
    print(f"Moving to joint configuration from index {idx}: {joints}")
    
    # Command robot to move to joint positions
    response = api.goto_joint(
        joints["A1"], joints["A2"], joints["A3"], joints["A4"],
        joints["A5"], joints["A6"], joints["A7"]
    )
    
    if response and response.ok and response.text.strip() == "OK":
        print(f"Successfully moved to joint configuration from index {idx}")
        return True
    else:
        print(f"Failed to move to joint configuration from index {idx}: {response.text if response else 'No response'}")
        return False

def go_around_positions(camera_handler: CameraHandler, **kwargs):
    """Moves the robot through predefined poses from a file and captures data."""
    output_folder = kwargs.get("output_folder", config.DEFAULT_GO_AROUND_OUTPUT_FOLDER)
    Only_current = kwargs.get("Only_current", False) # Default to False
    take_images = kwargs.get("take_images", False) # Default to False
    do_detection = kwargs.get("do_detection", False)
    do_6d_estimation = kwargs.get("do_6d_estimation", False)
    detection_item = kwargs.get("detection_item", "foam brick")
    pose_file = kwargs.get("pose_file", config.GO_AROUND_HAND_POSES_FILE)
    json_filename = kwargs.get("json_filename", "data.json")
    

    poses_data = utils.load_json_data(pose_file)
    if not poses_data or not isinstance(poses_data, list):
        print(f"Error: Could not load or parse poses from {pose_file}")
        return

    # Extract joint configurations, ensure keys match API ('A1'...'A7')
    Poses = []
    for pose_entry in poses_data:
        if "joints" in pose_entry and all(f"A{i}" in pose_entry["joints"] for i in range(1, 8)):
            joint_dict = pose_entry["joints"]
            # Make sure all joint values are floats
            try:
                for i in range(1, 8):
                    joint_dict[f"A{i}"] = float(joint_dict[f"A{i}"])
            except ValueError as e:
                print(f"Warning: Skipping pose due to non-float joint value: {e} in {pose_entry}")
                continue # Skip this pose entry

            joint_dict["speed"] = float(pose_entry.get("speed", config.DEFAULT_ARM_SPEED)) # Use default speed if not specified
            Poses.append(joint_dict)
        else:
            print(f"Warning: Skipping invalid pose entry in {pose_file}: {pose_entry}")

    if not Poses:
        print(f"No valid joint poses found in {pose_file}")
        return

    os.makedirs(output_folder, exist_ok=True)

    # --- START: Restore Direction Logic ---
    print("Checking current position to determine starting direction...")
    current_position = api.get_iiwa_joint_position() # Get current joints {'A1': val,...}
    # Optional short delay if needed, e.g., time.sleep(0.5)

    if current_position:
        try:
            # Define the joint keys to use for distance calculation
            joint_names = [f"A{i}" for i in range(1, 8)]

            # Calculate distance to the first pose in the list
            start_pose_joints = Poses[0]
            start_distance = sum(abs(start_pose_joints[joint] - current_position[joint]) for joint in joint_names)

            # Calculate distance to the last pose in the list
            end_pose_joints = Poses[-1]
            end_distance = sum(abs(end_pose_joints[joint] - current_position[joint]) for joint in joint_names)

            print(f"Distance to start pose (sum abs diff): {start_distance:.4f}")
            print(f"Distance to end pose (sum abs diff): {end_distance:.4f}")

            if end_distance < start_distance:
                print("Robot is closer to the end pose. Reversing pose order.")
                Poses = Poses[::-1] # Reverse the list in-place
            else:
                print("Robot is closer to the start pose. Using original pose order.")

        except KeyError as e:
            print(f"Warning: Could not calculate distances - missing joint key {e} in poses or current position. Using default order.")
        except Exception as e:
            print(f"Warning: An unexpected error occurred during distance calculation: {e}. Using default order.")
    else:
        print("Warning: Could not get current joint position. Using default pose order.")
    # --- END: Restore Direction Logic ---


    print(f"Starting GoAround sequence using {len(Poses)} poses from {pose_file}")
    # Loop through the (potentially reversed) Poses list

    T_world_obj_list = None 
    for i, pose in enumerate(Poses):
        if i < 8:
            continue
        # Make sure keys match API requirements ('A1'...'A7')
        print(f"Moving to pose {i+1}/{len(Poses)}: { {k: round(v, 3) for k, v in pose.items() if k != 'speed'} }")
        try:
            if Only_current:
                i = len(Poses)
                response = type('Response', (), {'ok': True, 'text': 'OK'})()
            else:
                response = api.goto_joint(
                    pose["A1"], pose["A2"], pose["A3"], pose["A4"],
                    pose["A5"], pose["A6"], pose["A7"], speed=pose["speed"]
                )
        except KeyError as e:
            print(f"  ERROR: Missing joint key {e} in pose data. Skipping move.")
            continue # Skip to the next pose

        time.sleep(0.5) # Give time for command acknowledgement

        if response and response.ok and response.text.strip() == "OK":
            print("  Move successful. Capturing data...")
            # Call the data capture function
            if take_images:
                T_world_obj_list = get_and_save_image_data(
                    camera_handler,
                    output_folder=output_folder,
                    do_detection=do_detection,
                    do_6d_estimation=do_6d_estimation,
                    detection_item=detection_item,
                    json_filename=json_filename,
                    delay_before_capture=1.5 # Allow extra time for settling after move
                )
            if T_world_obj_list != None:
                return T_world_obj_list
        else:
            print(f"  Error moving to pose {i+1} or response not OK: {response.text if response else 'No response'}")
            # Decide whether to stop or continue
            # break # Uncomment to stop sequence on move failure
            print("  Continuing to next pose...")

    print("GoAround sequence finished.")
    return T_world_obj_list

def execute_sequence(camera_handler: CameraHandler, **kwargs):
    """Executes a sequence involving moving the KMR and performing GoAround."""
    output_folder = kwargs.get("output_folder", config.DEFAULT_GO_AROUND_OUTPUT_FOLDER)
    Only_current = kwargs.get("Only_current", False) # Default to False
    do_camera_around = kwargs.get("do_camera_around", True) # Default to False
    take_images = kwargs.get("take_images", False) # Default to False
    do_detection = kwargs.get("do_detection", True) # Default to True
    do_6d_estimation = kwargs.get("do_6d_estimation", True) # Default to True
    go_to_object = kwargs.get("go_to_object", True) # Default to True
    detection_item = kwargs.get("detection_item", "mustard bottle")
    clean_folder = kwargs.get("clean_folder", False) # Renamed from clean_fodler
    json_filename = kwargs.get("json_filename", "sequence_data.json") # Use a different default filename


    if clean_folder:
        utils.clean_directory(output_folder)

    os.makedirs(output_folder, exist_ok=True) # Ensure directory exists

    locations = kwargs.get("locations", [7, 8]) # Use predefined locations 1 and 2

    T_world_obj_list = None
    for i, location in enumerate(locations):
        print(f"\n--- Moving to Location {location} ---")
        if Only_current:
            response = type('Response', (), {'ok': True, 'text': 'OK'})()
        else:
            response = api.move_to_location(location)
        # Add check for move success
        if response and response.ok and response.text.strip() == "OK":
            print(f"Successfully arrived at location {location}. Waiting for system to settle...")
            time.sleep(5) # Increased wait time after KMR move

            print(f"\n--- Starting GoAround at Location {location} ---")
            # Pass parameters down to go_around_positions
            if do_camera_around:
                T_world_obj_list = go_around_positions(
                    camera_handler,
                    output_folder=output_folder,
                    Only_current=Only_current,
                    take_images=take_images,
                    do_detection=do_detection,
                    do_6d_estimation=do_6d_estimation,
                    detection_item=detection_item,
                    # Use a location-specific json filename or append to the main one
                    json_filename=json_filename
                )
                if T_world_obj_list is not None:
                    break
        else:
            print(f"Failed to move to location {location}. Skipping GoAround.")
            # Optional: Add error handling or stop the sequence

    print("\n=== Full Sequence Execution Finished ===")

    T_world_obj = np.array(T_world_obj_list)

    if go_to_object:    
        drive_to_object(T_world_obj)

    just_grab_the_object(T_world_obj, prompt=detection_item)


    return T_world_obj_list

def just_grab_the_object(T_world_obj, **kwargs):
    use_before_grasp = kwargs.get("use_before_grasp", False)
    prompt = kwargs.get("prompt", "mustard bottle")


    if use_before_grasp:
        # Function to move to the pre-grasp position from the calibration file
        file_path = "image_processing\\calibration_data\\before_gripping_pose.json"
            
        with open(file_path, "r") as f:
            joint_data = json.load(f)
            
        joints = joint_data[0]["joints"]  # Access the first element in the list

        response = api.goto_joint(
            joints["A1"], joints["A2"], joints["A3"], joints["A4"],
            joints["A5"], joints["A6"], joints["A7"]
        )
        
        if response and response.ok and response.text.strip() == "OK":
            print("Successfully moved to pre-grasp position")
            time.sleep(3)  # Allow time for the robot to settle
        else:
            print(f"Failed to move to pre-grasp position: {response.text if response else 'No response'}")


    file_path = "object_models\\" + prompt + "\\grab_poses\\grab_poses.json"


    data = utils.load_json_data(file_path)
    iiwa_robot = iiwa.IIWA()

    # Load all grasp poses from the JSON file
    grasp_data = utils.load_json_data(file_path)
    
    if not isinstance(grasp_data, list) or not grasp_data:
        print(f"Error: No valid grasp data found in {file_path}")
        return
    
    print(f"Loaded {len(grasp_data)} grasp poses")
    
    # Get current end effector position
    current_ee_pose = api.get_iiwa_position()
    if not current_ee_pose:
        print("Error: Failed to get current end effector position")
        return
        
    # Current end effector position as array for distance calculation
    current_ee_pos = np.array([current_ee_pose['x'], current_ee_pose['y'], current_ee_pose['z']])
    
    # Calculate all possible grasp positions in the world frame and their distances
    grasp_options = []
    
    for i, grasp in enumerate(grasp_data):
        # Calculate pre-grasp position in world frame
        T_object_ee_before_grasp = np.array(grasp["T_object_ee_before_grasp"])
        
        # Calculate where the pre-grasp position would be in world coordinates
        T_world_ee_before_grasp = T_world_obj @ T_object_ee_before_grasp
        
        # Extract position part (translation vector)
        # world_ee_pos = T_world_ee_before_grasp[:3, 3]
        
        # Calculate distance to current end effector position
        # Convert world position to iiwa base frame for comparison
        kmr_pose_m_rad = api.get_pose()
        T_world_iiwabase = utils.get_T_world_iiwabase(kmr_pose_m_rad)
        T_inv_world_iiwabase = utils.inverse_homogeneous_transform(T_world_iiwabase)
        T_iiwabase_ee = T_inv_world_iiwabase @ T_world_ee_before_grasp
        iiwa_ee_pos = T_iiwabase_ee[:3, 3]
        
        # Calculate distance in iiwa base frame
        distance = np.linalg.norm(iiwa_ee_pos - current_ee_pos)
        
        grasp_option = {
            'index': i,
            'grasp_name': grasp.get('grabbing_pose_name', f"Grasp {i}"),
            'distance': distance,
            'T_object_ee_before_grasp': T_object_ee_before_grasp,
            'T_object_ee_grasp': np.array(grasp["T_object_ee_grasp"]),
            'T_world_ee_before_grasp': T_world_ee_before_grasp
        }
        grasp_options.append(grasp_option)
    
    # Sort grasp options by distance (closest first)
    sorted_options = sorted(grasp_options, key=lambda x: x['distance'])
    
    print("\nGrasp options sorted by distance to current end effector position:")
    for i, option in enumerate(sorted_options):
        print(f"{i+1}. {option['grasp_name']} - Distance: {option['distance']:.2f}mm")
    
    # Select the closest grasp
    selected_grasp = sorted_options[0]
    print(f"\nSelected grasp: {selected_grasp['grasp_name']} (distance: {selected_grasp['distance']:.2f}mm)")
    
    # Use the selected grasp's transformation matrices
    # T_world_obj = selected_grasp['T_world_obj']
    T_object_ee_before_grasp = selected_grasp['T_object_ee_before_grasp']
    T_object_ee_grasp = selected_grasp['T_object_ee_grasp']
    
    print(f"Using object pose in world:\n{np.round(T_world_obj, 3)}")
    print(f"Using object-to-ee pre-grasp transform:\n{np.round(T_object_ee_before_grasp, 3)}")
    print(f"Using object-to-ee grasp transform:\n{np.round(T_object_ee_grasp, 3)}")



    T_world_ee_before_grasp = T_world_obj @ T_object_ee_before_grasp
    T_world_ee_grasp = T_world_obj @ T_object_ee_grasp

    print(f"Target EE Pre-Grasp (World):\n{T_world_ee_before_grasp}")
    print(f"Target EE Grasp (World):\n{T_world_ee_grasp}")

    kmr_pose_m_rad = api.get_pose()
    T_world_iiwabase = utils.get_T_world_iiwabase(kmr_pose_m_rad)



    T_inv_world_iiwabase = utils.inverse_homogeneous_transform(T_world_iiwabase)

    print(f"IIWA Base Pose (World):\n{T_world_iiwabase}")
    print(f"Inv IIWA Base Pose (IIWA_in_World -> World_in_IIWA):\n{T_inv_world_iiwabase}")


    T_iiwabase_ee_before_grasp = T_inv_world_iiwabase @ T_world_ee_before_grasp
    T_iiwabase_ee_grasp = T_inv_world_iiwabase @ T_world_ee_grasp

    # T_object_ee_grasp = T_inv_world_object @ end_effector_world_pose_grab
    # T_object_ee_before_grasp = T_inv_world_object @ end_effector_world_pose_before_grab


    print(f"Target EE Pre-Grasp (IIWA Base):\n{T_iiwabase_ee_before_grasp}")
    print(f"Target EE Grasp (IIWA Base):\n{T_iiwabase_ee_grasp}")



    # T_iiwabase_ee_before_grasp[0:3, 0:3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    # T_iiwabase_ee_before_grasp[0:3, 0:3] = np.array([[0, 1, 0], [0, -1, 0], [-1, 0, 0]])
    # print(f"Target EE Pre-Grasp (IIWA Base) with Z-axis aligned:\n{T_iiwabase_ee_before_grasp}")




    # --- Extract Position and Orientation for API ---
    # Position (x, y, z)
    position_before_grasp = T_iiwabase_ee_before_grasp[0:3, 3]
    position_grasp = T_iiwabase_ee_grasp[0:3, 3]

    # Orientation (Rotation Matrix -> Euler Angles)
    # IMPORTANT: Ensure 'zyx' matches the convention expected by api.goto_position
    # Common alternatives: 'xyz', 'zxy', 'zyz', etc. Check your robot API docs!
    R_before_grasp = T_iiwabase_ee_before_grasp[0:3, 0:3]
    R_grasp = T_iiwabase_ee_grasp[0:3, 0:3]

    try:
        # orientation_before_grasp = utils.rotation_matrix_to_euler_zyx(R_before_grasp) # [roll, pitch, yaw]
        # orientation_grasp = utils.rotation_matrix_to_euler_zyx(R_grasp)               # [roll, pitch, yaw]
        orientation_before_grasp = image_utils.rotation_matrix_to_xyz_extrinsic(R_before_grasp) # [roll, pitch, yaw]
        orientation_grasp = image_utils.rotation_matrix_to_xyz_extrinsic(R_grasp)               # [roll, pitch, yaw]

    except Exception as e:
        print(f"Error converting rotation matrix to Euler angles: {e}")
        # Handle potential issues, e.g., gimbal lock if using Euler angles
        return

    print(f"Calculated Pre-Grasp Pose (IIWA): pos={position_before_grasp}, orient_rad={orientation_before_grasp}")
    print(f"Calculated Grasp Pose (IIWA): pos={position_grasp}, orient_rad={orientation_grasp}")


    # --- Command the Robot ---
    print("\n--- Commanding Robot ---")
    response = api.goto_position(position_before_grasp[0], position_before_grasp[1], position_before_grasp[2],
                      orientation_before_grasp[0], orientation_before_grasp[1], orientation_before_grasp[2])

    print("-"*50)
    print(f"Response from IIWA API: {response.text}")
    print("-"*50)

    # if response.text == "failed":
    #     iiwa_joints = api.get_iiwa_joint_position()
    #     joints = np.array([iiwa_joints["A1"], iiwa_joints["A2"], iiwa_joints["A3"], iiwa_joints["A4"], iiwa_joints["A5"], iiwa_joints["A6"], iiwa_joints["A7"]])
    #     print("IIWA solver failed to find a solution. Trying myself...")

    #     print(f"Current IIWA Joints: {joints}")
    #     print(f"T_iiwabase_ee_before_grasp:\n{T_iiwabase_ee_before_grasp}")
    #     theta_final, success, final_error, clamped_warning = iiwa_robot.inverse_kinematics_nr(T_iiwabase_ee_before_grasp, joints)


    #     print(f"Inverse Kinematics successful: {theta_final}")
    #     response = api.goto_joint(
    #         theta_final[0], theta_final[1], theta_final[2],
    #         theta_final[3], theta_final[4], theta_final[5], theta_final[6]
    #     )



    #     return

    print("Waiting after moving to pre-grasp...")
    time.sleep(10) # Use a non-blocking sleep if possible in a real application

    api.goto_position(position_grasp[0], position_grasp[1], position_grasp[2],
                      orientation_grasp[0], orientation_grasp[1], orientation_grasp[2])
    

    time.sleep(10) # Wait for the robot to reach the grasp position

    print("Grasping...")
    api.close_gripper(force=5)

    time.sleep(5) # Wait for the grasp to complete

    api.go_home()



def drive_to_object(T_world_obj):
    Object_position = np.array(T_world_obj[:2, 3])
    KMR_position = api.get_pose()
    KMR_orientation = KMR_position['theta']
    KMR_position = np.array([KMR_position['x']*1000, KMR_position['y']*1000])
    Rotate_KMR_to_object(KMR_position, KMR_orientation, Object_position)

    time.sleep(3)  # Allow time for KMR to rotate

    KMR_position = api.get_pose()
    KMR_orientation = KMR_position['theta']
    KMR_position = np.array([KMR_position['x']*1000, KMR_position['y']*1000])

    # Calculate the direction vector from the object to the IIWA base
    direction_vector = KMR_position - Object_position

    # Normalize the direction vector
    norm = np.linalg.norm(direction_vector)
    if norm == 0:
        print("Object and KMR are at the same position.")
        return  # Handle the case where the object and KMR are at the same position
    direction_vector = direction_vector / norm

    # Calculate the direction vector from the object to the IIWA base
    direction_vector = KMR_position - Object_position

    # Normalize the direction vector
    norm = np.linalg.norm(direction_vector)
    if norm == 0:
        print("Object and KMR are at the same position.")
        return  # Handle the case where the object and KMR are at the same position
    direction_vector = direction_vector / norm

    # Calculate the new KMR position that is 420mm away from the object

    minimal_iiwa_range = 420  # mm
    offset_from_object = 350  # mm

    new_KMR_position = Object_position + direction_vector * (minimal_iiwa_range + offset_from_object + config.LONGEST_LENGTH_KMR/2)

    # Calculate the offset in the world frame
    x_offset_world = new_KMR_position[0] - KMR_position[0]
    y_offset_world = new_KMR_position[1] - KMR_position[1]

    # Convert the offset to the KMR frame
    x_offset_kmr = x_offset_world * np.cos(KMR_orientation) + y_offset_world * np.sin(KMR_orientation)
    y_offset_kmr = -x_offset_world * np.sin(KMR_orientation) + y_offset_world * np.cos(KMR_orientation)

    print(f"Current KMR position: {KMR_position}")
    print(f"New KMR position: {new_KMR_position}")
    print(f"Offset to move KMR (World Frame): ({x_offset_world}, {y_offset_world})")
    print(f"Offset to move KMR (KMR Frame): ({x_offset_kmr}, {y_offset_kmr})")

    # Move the KMR to the new position
    print(f"Moving KMR to new position: {new_KMR_position}")
    x_offset_kmr_m = x_offset_kmr / 1000  # Convert mm to meters
    y_offset_kmr_m = y_offset_kmr / 1000  # Convert mm to meters
    
    # Max distance per move (m)
    max_distance = 1.3
    
    # Calculate total movement distance
    total_distance = np.sqrt(x_offset_kmr_m**2 + y_offset_kmr_m**2)
    
    if total_distance <= max_distance:
        # If we're already within the limit, just move once
        api.move(x_offset_kmr_m, y_offset_kmr_m, 0)
    else:
        # Calculate number of segments needed
        num_segments = int(np.ceil(total_distance / max_distance))
        print(f"Movement too large ({total_distance:.2f}m). Splitting into {num_segments} segments.")
        
        # Calculate movement per segment
        x_segment = x_offset_kmr_m / num_segments
        y_segment = y_offset_kmr_m / num_segments
        
        # Move in segments
        for i in range(num_segments):
            print(f"Movement segment {i+1}/{num_segments}: ({x_segment:.2f}m, {y_segment:.2f}m)")
            api.move(x_segment, y_segment, 0)
            time.sleep(3)  # Wait between movements


    print(f"Object position: {Object_position}")
    print(f"KMR position: {KMR_position}")


def Rotate_KMR_to_object(KMR_position, KMR_orientation, Object_position):
    # Calculate the angle to rotate KMR to face the object
    delta_x = Object_position[0] - KMR_position[0]
    delta_y = Object_position[1] - KMR_position[1]

    # Calculate the angle in radians
    angle_to_object = np.arctan2(delta_y, delta_x)


    # Calculate the rotation needed to align KMR with the object
    rotation_needed = angle_to_object - KMR_orientation

    # Normalize the rotation to be within -180 to 180 degrees
    if rotation_needed > np.pi:
        rotation_needed -= 2*np.pi
    elif rotation_needed < -np.pi:
        rotation_needed += 2*np.pi

    print(f"Rotation needed: {rotation_needed} radians")

    api.move(0, 0, rotation_needed)  # Move KMR to face the object

def save_current_joints_to_file(camera_handler: CameraHandler):
    """Gets current joints, position, pose, calculates T_world_cam and saves to HandPoses.json."""
    output_file = config.HAND_POSES_FILE # Use path from config
    output_folder = os.path.dirname(output_file) # Get directory from file path

    print("Getting current robot state...")
    joints_data = api.get_iiwa_joint_position()
    position_data = api.get_iiwa_position()
    pose_data = api.get_pose()

    if not all([joints_data, position_data, pose_data]):
        print("Error: Failed to retrieve complete robot state.")
        return

    print("Calculating camera pose in world...")
    T_world_cam = utils.calculate_camera_in_world(pose_data, position_data)
    T_world_cam_list = T_world_cam.tolist() if T_world_cam is not None else None

    data = {
        "timestamp": time.time(),
        "joints": joints_data,  # Already sorted by get_iiwa_joint_position
        "position": position_data, # EE pos in base frame
        "kmr_pose": pose_data, # KMR pos in world frame
        "T_world_camera": T_world_cam_list
    }

    print(f"Saving data to {output_file}...")
    utils.append_to_json_list(output_file, data)


def save_calibration_image(camera_handler: CameraHandler, out_path: str):
    """Captures image, gets robot state, and saves data for calibration."""
    # Ensure path ends with a separator
    if not out_path.endswith(os.path.sep):
        out_path += os.path.sep

    os.makedirs(out_path, exist_ok=True)
    json_file_path = os.path.join(out_path, "calibration_data.json")
    timestamp = int(time.time())

    # Capture Image
    print("Capturing calibration image...")
    image = camera_handler.capture_image()
    if image is None:
        print("Failed to capture calibration image.")
        return

    image_filename = f"image_{timestamp}.png"
    image_full_path = os.path.join(out_path, image_filename)
    camera_handler.save_image(image, image_full_path)

    # Get Robot State
    print("Getting robot state for calibration point...")
    joint_positions = api.get_iiwa_joint_position()
    end_pose = api.get_iiwa_position() # EE pose in base frame
    kmr_pose = api.get_pose()

    if not all([joint_positions, end_pose, kmr_pose]):
        print("Error: Failed to retrieve complete robot state for calibration.")
        # Optionally delete the saved image if state is missing
        # os.remove(image_full_path)
        return
    T_world_camera = utils.calculate_camera_in_world(kmr_pose, end_pose)

    # Prepare data for JSON
    data = {
        "image_filename": image_filename,
        "timestamp": timestamp,
        "joints": joint_positions,
        "ee_pose_in_base": end_pose,
        "kmr_pose_in_world": kmr_pose,
        "T_world_camera": T_world_camera.tolist() if T_world_camera is not None else None,
    }

    # Append data to the JSON file
    print(f"Appending calibration data to {json_file_path}")
    utils.append_to_json_list(json_file_path, data)
    print(f"Saved calibration image and data point.")


def move_to_hand_poses_and_capture(camera_handler: CameraHandler, num_sets: int = 5):
    """Moves through predefined hand poses multiple times, capturing calibration data."""
    input_file = config.GO_AROUND_HAND_POSES_FILE # Use the same poses for this example

    input_file = "communication\HandPoses.json"

    try:
        hand_poses_data = utils.load_json_data(input_file)
        if not hand_poses_data or not isinstance(hand_poses_data, list):
             print(f"Error: Could not load poses from {input_file}")
             return
        # Extract valid joint poses
        hand_poses = []
        for pose_entry in hand_poses_data:
             if "joints" in pose_entry and all(f"A{i}" in pose_entry["joints"] for i in range(1, 8)):
                 joint_dict = pose_entry["joints"]
                 # Make sure joints are in radians (assuming file stores radians)
                 hand_poses.append(joint_dict)
             else:
                 print(f"Warning: Skipping invalid pose entry: {pose_entry}")
        if not hand_poses:
             print("No valid poses found.")
             return

    except Exception as e:
        print(f"Error processing pose file {input_file}: {e}")
        return


    # Loop through calibration sets
    for i in range(1, num_sets + 1):
        print(f"\n--- Starting Calibration Set {i}/{num_sets} ---")
        output_path = os.path.join(config.DEFAULT_CALIBRATION_OUTPUT_FOLDER, f"ScanAround_{i}")
        utils.clean_directory(output_path) # Clean before starting set
        os.makedirs(output_path, exist_ok=True)

        print(f"Capturing {len(hand_poses)} points for set {i} in {output_path}...")

        for pose_idx, joints in enumerate(hand_poses):
            print(f"Moving to pose {pose_idx + 1}/{len(hand_poses)}...")
            response = api.goto_joint(
                joints["A1"], joints["A2"], joints["A3"], joints["A4"],
                joints["A5"], joints["A6"], joints["A7"],
                speed=0.4 # Use a moderate speed for calibration moves
            )
            time.sleep(0.5) # Command delay

            if response and response.ok and response.text.strip() == "OK":
                print("Move successful. Waiting to settle...")
                time.sleep(1.5) # Settling time
                save_calibration_image(camera_handler, output_path)
                time.sleep(0.5) # Delay before next move
            else:
                print(f"Error moving to pose {pose_idx + 1}. Skipping capture.")
                # Decide if sequence should stop

        print(f"--- Finished Calibration Set {i} ---")

        if i < num_sets:
             print("Waiting before next set...")
             # Countdown with beeps (consider making this optional)
             try:
                 print("Starting in 30 seconds...")
                 time.sleep(15)
                 winsound.Beep(440, 500) # Frequency: 440 Hz (A4 note)
                 print("Starting in 15 seconds...")
                 time.sleep(10)
                 winsound.Beep(440, 500)
                 print("Starting in 5 seconds...")
                 time.sleep(5)
                 winsound.Beep(660, 750) # Higher pitch beep
                 print("Starting next set now!")
             except Exception as e:
                 print(f"Could not play sound: {e}. Continuing after delay.")
                 time.sleep(1) # Short pause if winsound failed

    print("=== All Calibration Sets Finished ===")


# --- JustPickIt Functions (Need careful review of logic) ---

def just_pick_it_step1_calculate_world_pose(camera_handler: CameraHandler):
    """Reads estimated pose from camera, gets robot state, calculates object pose in world."""
    pose_file_path = os.path.join(config.DEFAULT_PICK_OUTPUT_FOLDER, "estimated_pose.json")
    output_file_path = os.path.join(config.DEFAULT_PICK_OUTPUT_FOLDER, "calculated_world_pose.json")
    os.makedirs(config.DEFAULT_PICK_OUTPUT_FOLDER, exist_ok=True)

    # 1. Load the estimated pose (T_cam_obj) - Assume this file is generated by another process
    print(f"Loading estimated pose from {pose_file_path}")
    estimated_data = utils.load_json_data(pose_file_path)
    if not isinstance(estimated_data, dict) or "pose_matrix" not in estimated_data:
        print(f"Error: Invalid or missing data in {pose_file_path}. Expected dict with 'pose_matrix'.")
        return None

    T_cam_obj = np.array(estimated_data["pose_matrix"])
    print(f"Loaded T_cam_obj:\n{T_cam_obj}")

     # Optional: Add unit conversion check/logic here as in process_image_and_estimate_pose
    if np.max(np.abs(T_cam_obj[:3, 3])) < 10.0: # Heuristic: if max translation < 10, assume meters
        print("Converting estimated pose translation from meters to mm.")
        T_cam_obj[:3, 3] *= 1000
    else:
        print("Assuming estimated pose translation is already in mm.")

    # 2. Get current robot state
    print("Getting current robot state...")
    kmr_pose = api.get_pose()
    iiwa_pos = api.get_iiwa_position()
    if not kmr_pose or not iiwa_pos:
        print("Error: Failed to get current robot state.")
        return None

    # 3. Calculate T_world_cam
    print("Calculating T_world_cam...")
    T_world_cam = utils.calculate_camera_in_world(kmr_pose, iiwa_pos)
    if T_world_cam is None:
        print("Error: Failed to calculate T_world_cam.")
        return None
    print(f"Calculated T_world_cam:\n{T_world_cam}")

    # 4. Calculate T_world_obj
    print("Calculating T_world_obj...")
    T_world_obj = utils.calculate_object_in_world(T_world_cam, T_cam_obj)
    print(f"Calculated T_world_obj:\n{T_world_obj}")

    # 5. Save calculated poses
    save_data = {
        "T_cam_obj_used": T_cam_obj.tolist(),
        "T_world_cam_at_capture": T_world_cam.tolist(),
        "T_world_obj_calculated": T_world_obj.tolist(),
        "kmr_pose_at_capture": kmr_pose,
        "iiwa_pos_at_capture": iiwa_pos,
        "timestamp": time.time()
    }
    utils.save_json_data(output_file_path, save_data)
    print(f"Saved calculated world pose data to {output_file_path}")

    return T_world_obj # Return the calculated world pose


def align_A1_with_object(T_world_obj: np.ndarray):
    """Aligns the robot's A1 joint towards the object's world position."""
    if T_world_obj is None:
        print("Error: T_world_obj is None, cannot align A1.")
        return

    print("--- Starting Align A1 ---")
    # 1. Get current robot state (KMR pose and IIWA joints)
    kmr_pose = api.get_pose()
    joints = api.get_iiwa_joint_position()
    if not kmr_pose or not joints:
        print("Error: Failed to get robot state for A1 alignment.")
        return

    iiwa_base_in_world_mm = utils.get_iiwa_base_in_world([kmr_pose['x'], kmr_pose['y'], kmr_pose['theta']])

    obj_pos_world_mm = T_world_obj[:3, 3]
    vector_base_to_obj = obj_pos_world_mm[:2] - iiwa_base_in_world_mm[:2] # Only consider X, Y

    world_angle_to_object_rad = np.arctan2(vector_base_to_obj[1], vector_base_to_obj[0])

    kmr_theta_rad = kmr_pose['theta']
    a1_desired_rad = world_angle_to_object_rad - kmr_theta_rad - np.pi/2

    a1_desired_rad = (a1_desired_rad + np.pi) % (2 * np.pi) - np.pi # Normalize to [-pi, pi]

    a1_min_rad = -2.96
    a1_max_rad = 2.96
    final_a1_angle = np.clip(a1_desired_rad, a1_min_rad, a1_max_rad)

    print(f"IIWA Base (mm): {np.round(iiwa_base_in_world_mm[:2],1)}")
    print(f"Object Pos (mm): {np.round(obj_pos_world_mm[:2],1)}")
    print(f"Vector Base->Obj: {np.round(vector_base_to_obj,1)}")
    print(f"World Angle to Obj (rad): {world_angle_to_object_rad:.3f} (deg: {np.degrees(world_angle_to_object_rad):.1f})")
    print(f"KMR Theta (rad): {kmr_theta_rad:.3f}")
    print(f"Desired A1 (rad): {a1_desired_rad:.3f}")
    print(f"Current A1 (rad): {joints['A1']:.3f}")
    print(f"Final Clamped A1 (rad): {final_a1_angle:.3f} (deg: {np.degrees(final_a1_angle):.1f})")


    # 6. Command the robot to move only A1
    print(f"Moving A1 to {final_a1_angle:.3f} radians...")
    response = api.goto_joint(
        final_a1_angle, joints["A2"], joints["A3"], joints["A4"],
        joints["A5"], joints["A6"], joints["A7"],
        speed=0.3
    )

    if response and response.ok and response.text.strip() == "OK":
        print("A1 alignment move successful.")
    else:
        print("Error during A1 alignment move.")


def just_pick_it_full_sequence():
    # Using the provided object-to-world transformation matrix
    object_in_world = np.array([
        [0.427, 0.903, -0.046, 12701.271],
        [-0.904, 0.427, -0.002, 15262.575],
        [0.018, 0.043, 0.999, 911.753],
        [0.0, 0.0, 0.0, 1.0]
    ])

    object_in_world = np.array([
        [0.427, 0.903, -0.046, 12601.271],
        [-0.904, 0.427, -0.002, 15262.575],
        [0.018, 0.043, 0.999, 911.753],
        [0.0, 0.0, 0.0, 1.0]
    ])

    just_grab_the_object(object_in_world)



def Object_to_world():

    grabbing_pose_name = "up"

    out_name = "lego brick.json"

    # Load the object pose from the JSON file
    object_pose_file = "image_processing\\grabbing_poses\\current_object_pose.json"
    
    with open(object_pose_file, "r") as f:
        T_world_obj = np.array(json.load(f))

    print(f"Object position in world: \n {np.round(T_world_obj, 3)}")



    # Load the pose data from the JSON file
    poses_file_path = "image_processing\\grabbing_poses\\pose_and_prepose.json"
    with open(poses_file_path, "r") as f:
        poses_data = json.load(f)
    
    # Extract before grab pose (first entry)
    kmr_pose_before_grab = poses_data[0]["kmr_pose"]
    end_pose_before_grab = poses_data[0]["position"]
    
    # Extract grab pose (second entry)
    kmr_pose_grab = poses_data[1]["kmr_pose"]
    end_pose_grab = poses_data[1]["position"]
    
    print(f"Loaded pre-grasp KMR pose: {kmr_pose_before_grab}")
    print(f"Loaded grasp KMR pose: {kmr_pose_grab}")



    end_effector_world_pose_before_grab = utils.calculate_end_effector_in_world(kmr_pose_before_grab, end_pose_before_grab)
    print(f"End effector position in world before grabing: \n {np.round(end_effector_world_pose_before_grab, 3)}")

    end_effector_world_pose_grab = utils.calculate_end_effector_in_world(kmr_pose_grab, end_pose_grab)
    print(f"End effector position in world while grabing: \n {np.round(end_effector_world_pose_grab, 3)}")



    R_world_object = T_world_obj[0:3, 0:3]
    p_world_object = T_world_obj[0:3, 3:4] # Keep as column vector

    R_inv = R_world_object.T
    p_inv = -R_inv @ p_world_object

    T_inv_world_object = np.identity(4)
    T_inv_world_object[0:3, 0:3] = R_inv
    T_inv_world_object[0:3, 3:4] = p_inv

    # Alternatively, use numpy's built-in inverse function (numerically stable)
    # T_inv_world_object = np.linalg.inv(T_world_object)

    # Calculate the relative grasp pose
    T_object_ee_grasp = T_inv_world_object @ end_effector_world_pose_grab
    T_object_ee_before_grasp = T_inv_world_object @ end_effector_world_pose_before_grab


    print(f"T_object_ee_grasp: \n{T_object_ee_grasp}")
    print(f"T_object_ee_before_grasp: \n{T_object_ee_before_grasp}")

    # Save the transformation matrices to JSON file
    output_data = {
        "grabbing_pose_name": grabbing_pose_name,
        "T_world_obj": T_world_obj.tolist(),
        "kmr_pose_before_grab": kmr_pose_before_grab,
        "end_pose_before_grab": end_pose_before_grab,
        "T_object_ee_grasp": T_object_ee_grasp.tolist(),
        "T_object_ee_before_grasp": T_object_ee_before_grasp.tolist()
    }
    output_file_path = os.path.join("image_processing", "grabbing_poses", out_name)

    utils.append_to_json_list(output_file_path, output_data)




def Go_to_the_position():
    T_world_obj = np.array([
        [-0.7611, 0.5502, 0.3436, 12621.1097],
        [-0.5215, -0.2041, -0.8285, 15043.4196],
        [-0.3857, -0.8097, 0.4423, 1341.2886],
        [0.0, 0.0, 0.0, 1.0]
    ])


    kmr_pose = api.get_pose()
    iiwa_pos = api.get_iiwa_position()
    T_world_cam = utils.calculate_camera_in_world(kmr_pose, iiwa_pos)
    iiwa_base_pos_mm = utils.get_iiwa_base_in_world([kmr_pose['x'], kmr_pose['y'], kmr_pose['theta']])

    t_iiwa_obj = T_world_obj[:3, 3] - iiwa_base_pos_mm[:3]
    print(f"{t_iiwa_obj=}")
    print(f"{iiwa_pos=}")

    api.goto_position(t_iiwa_obj[0], t_iiwa_obj[1], t_iiwa_obj[2], iiwa_pos["A"], iiwa_pos["B"], iiwa_pos["C"])


def visualize_transformations():
    """
    Loads a mesh and visualizes ALL grasp and pre-grasp poses
    defined in the specified JSON file using Open3D coordinate frames.
    Assumes specific relative paths for the JSON and OBJ files unless
    overridden by absolute paths.
    """
    objects = {1:"box of jello",
               2: "cracker box",
               3: "foam brick",
               4: "gray box",
               5: "lego brick",
               6: "mustard bottle",
               7: "plug-in outlet expander",
               8: "tuna fish can"}
    
    object_id = 8

    base_path = "object_models"
    json_file_path = os.path.join(base_path, f"{objects[object_id]}/grab_poses/grab_poses.json")
    obj_file_path = os.path.join(base_path, f"{objects[object_id]}/mesh/{objects[object_id]}.ply")


    # obj_file_path = "C:\\Users\\siram\\OneDrive\\Plocha\\Skola - CVUT\\4.semestr Mag\\Diplomka\\KMR-object-manipulation-with-LLMs\\YCB_Objects\\plug-in outlet expander\\meshes\\plug-in outlet expander\\obj_000021.ply"

    # Visualization parameters
    axis_size = 50.0  # Size of the coordinate axes in mm (adjust as needed)
    # You might want smaller axes if many poses overlap
    # axis_size = 30.0

    # --- Load Data ---
    # Load JSON
    with open(json_file_path, 'r') as f:
        grasping_data_list = json.load(f) # Load the whole list of poses

    # --- Load Mesh ---
    print(f"Loading mesh: {obj_file_path}")
    mesh = o3d.io.read_triangle_mesh(obj_file_path, enable_post_processing=True)
    # Optional: Assign a uniform color if textures/materials don't load properly
    # mesh.paint_uniform_color([0.7, 0.7, 0.7])
    mesh.compute_vertex_normals() # Helps with shading
    print(f"Successfully loaded mesh.")

    # --- Create Geometries (Mesh + All Pose Frames) ---
    geometries_to_draw = [mesh] # Start the list with the mesh

    print(f"Processing {len(grasping_data_list)} pose entries from {os.path.basename(json_file_path)}...")

    # Iterate through each pose dictionary in the loaded list
    for i, pose_info in enumerate(grasping_data_list):
        pose_name = pose_info.get("grabbing_pose_name", f"Pose {i+1}") # Get name or use index
        print(f"  Adding frames for: {pose_name}")

        # Extract and convert transformation matrices to numpy arrays
        T_object_ee_grasp = np.array(pose_info["T_object_ee_grasp"], dtype=np.float64)
        T_object_ee_before_grasp = np.array(pose_info["T_object_ee_before_grasp"], dtype=np.float64)

        # Create a frame for the grasp pose
        grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=axis_size, origin=[0, 0, 0]
        )
        grasp_frame.transform(T_object_ee_grasp) # Apply the transformation
        geometries_to_draw.append(grasp_frame) # Add to the list

        # Create a frame for the pre-grasp pose
        before_grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=axis_size, origin=[0, 0, 0]
        )
        # Slightly offset pre-grasp frame for better visibility if needed (optional)
        # Example: Move it slightly along its own Z-axis before applying the main transform
        # offset_transform = np.identity(4)
        # offset_transform[2, 3] = -axis_size * 0.2 # Move back 20% of axis size
        # combined_transform = T_object_ee_before_grasp @ offset_transform
        # before_grasp_frame.transform(combined_transform)
        before_grasp_frame.transform(T_object_ee_before_grasp) # Apply the transformation
        geometries_to_draw.append(before_grasp_frame) # Add to the list

    # Optional: Create a frame for the object's origin itself
    # object_origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=axis_size * 0.75, origin=[0, 0, 0] # Slightly larger/different size
    # )
    # geometries_to_draw.append(object_origin_frame)

    # --- Visualize ---
    print(f"\nVisualizing mesh: {os.path.basename(obj_file_path)}")
    print(f"Displaying {len(grasping_data_list)*2} pose frames (grasp + pre-grasp for each entry).")

    o3d.visualization.draw_geometries(
        geometries_to_draw, # Pass the list containing mesh + all frames
        window_name=f"Grasp Visualization ({os.path.basename(obj_file_path)})",
        width=1280,
        height=960,
        mesh_show_back_face=True
    )
    print("Visualization window closed.")



