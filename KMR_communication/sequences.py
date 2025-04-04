# sequences.py
import time
import json
import numpy as np
import cv2
import os
import winsound

from . import config
from . import kuka_api as api
from . import utils
from .camera_handler import CameraHandler # If used directly
from . import image_processing_client as ipc

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
    bounding_boxes = ipc.send_image_to_server(image_bytes, item_name)

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
    first_bbox = bounding_boxes[0]
    pose_result_raw = ipc.send_for_pose_estimation(image_bytes, first_bbox, item_name)

    if not pose_result_raw or 'poses' not in pose_result_raw or not pose_result_raw['poses']:
        print("6D Pose estimation failed or returned no poses.")
        return None

    # Extract 4x4 pose matrix (assuming server returns it correctly)
    # Pose from server is T_cam_obj (Camera -> Object)
    T_cam_obj_list = pose_result_raw['poses'][0]['pose']
    T_cam_obj = np.array(T_cam_obj_list) # Convert list from JSON back to numpy array

    # IMPORTANT: Verify units from pose estimation server. Assume meters needs conversion.
    # If server provides meters, convert translation to mm. If already mm, skip.
    if np.max(np.abs(T_cam_obj[:3, 3])) < 10.0: # Heuristic: if max translation < 10, assume meters
        print("Converting pose estimation translation from meters to mm.")
        T_cam_obj[:3, 3] *= 1000
    else:
        print("Assuming pose estimation translation is already in mm.")

    print(f"Estimated T_cam_obj:\n{T_cam_obj}")

    # Calculate Object in World
    if T_world_cam is None:
         print("Cannot calculate object in world: T_world_cam is None.")
         return None

    T_world_obj = utils.calculate_object_in_world(T_world_cam, T_cam_obj)
    print(f"Calculated T_world_obj:\n{T_world_obj}")

    return T_world_obj # Return the object pose in world coordinates


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


def go_around_positions(camera_handler: CameraHandler, **kwargs):
    """Moves the robot through predefined poses from a file and captures data."""
    output_folder = kwargs.get("output_folder", config.DEFAULT_GO_AROUND_OUTPUT_FOLDER)
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
             joint_dict["speed"] = pose_entry.get("speed", config.DEFAULT_ARM_SPEED) # Use default speed if not specified
             Poses.append(joint_dict)
        else:
             print(f"Warning: Skipping invalid pose entry in {pose_file}: {pose_entry}")

    if not Poses:
        print(f"No valid joint poses found in {pose_file}")
        return

    os.makedirs(output_folder, exist_ok=True)

    # Optional: Check current position to decide start direction (more complex logic)
    # current_position = api.get_iiwa_joint_position()
    # if current_position: ... decide order ...

    print(f"Starting GoAround sequence using {len(Poses)} poses from {pose_file}")
    for i, pose in enumerate(Poses):
        print(f"Moving to pose {i+1}/{len(Poses)}: { {k: round(v, 3) for k, v in pose.items() if k != 'speed'} }")
        response = api.goto_joint(
            pose["A1"], pose["A2"], pose["A3"], pose["A4"],
            pose["A5"], pose["A6"], pose["A7"], speed=pose["speed"]
        )
        time.sleep(0.5) # Give time for command acknowledgement

        if response and response.ok and response.text.strip() == "OK":
            print("Move successful. Capturing data...")
            # Call the data capture function
            get_and_save_image_data(
                camera_handler,
                output_folder=output_folder,
                do_detection=do_detection,
                do_6d_estimation=do_6d_estimation,
                detection_item=detection_item,
                json_filename=json_filename,
                delay_before_capture=1.5 # Allow extra time for settling after move
            )
        else:
            print(f"Error moving to pose {i+1} or response not OK: {response.text if response else 'No response'}")
            # Decide whether to stop or continue
            # break
            print("Continuing to next pose...")

    print("GoAround sequence finished.")


def execute_sequence(camera_handler: CameraHandler, **kwargs):
    """Executes a sequence involving moving the KMR and performing GoAround."""
    output_folder = kwargs.get("output_folder", config.DEFAULT_GO_AROUND_OUTPUT_FOLDER)
    do_detection = kwargs.get("do_detection", True) # Default to True
    do_6d_estimation = kwargs.get("do_6d_estimation", True) # Default to True
    detection_item = kwargs.get("detection_item", "mustard bottle")
    clean_folder = kwargs.get("clean_folder", False) # Renamed from clean_fodler
    json_filename = kwargs.get("json_filename", "sequence_data.json") # Use a different default filename

    if clean_folder:
        utils.clean_directory(output_folder)

    os.makedirs(output_folder, exist_ok=True) # Ensure directory exists

    locations = kwargs.get("locations", [1, 2]) # Use predefined locations 1 and 2

    for i, location in enumerate(locations):
        print(f"\n--- Moving to Location {location} ---")
        response = api.move_to_location(location)
        # Add check for move success
        if response and response.ok and response.text.strip() == "OK":
            print(f"Successfully arrived at location {location}. Waiting for system to settle...")
            time.sleep(5) # Increased wait time after KMR move

            print(f"\n--- Starting GoAround at Location {location} ---")
            # Pass parameters down to go_around_positions
            go_around_positions(
                camera_handler,
                output_folder=output_folder,
                do_detection=do_detection,
                do_6d_estimation=do_6d_estimation,
                detection_item=detection_item,
                # Use a location-specific json filename or append to the main one
                json_filename=f"location_{location}_{json_filename}"
            )
        else:
            print(f"Failed to move to location {location}. Skipping GoAround.")
            # Optional: Add error handling or stop the sequence

    print("\n=== Full Sequence Execution Finished ===")


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

    # Prepare data for JSON
    data = {
        "image_filename": image_filename,
        "timestamp": timestamp,
        "joints": joint_positions,
        "ee_pose_in_base": end_pose,
        "kmr_pose_in_world": kmr_pose
        # Consider adding T_world_camera here as well if useful for calibration
    }

    # Append data to the JSON file
    print(f"Appending calibration data to {json_file_path}")
    utils.append_to_json_list(json_file_path, data)
    print(f"Saved calibration image and data point.")


def move_to_hand_poses_and_capture(camera_handler: CameraHandler, num_sets: int = 5):
    """Moves through predefined hand poses multiple times, capturing calibration data."""
    input_file = config.GO_AROUND_HAND_POSES_FILE # Use the same poses for this example

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


def just_pick_it_step2_align_A1(T_world_obj: np.ndarray):
    """Aligns the robot's A1 joint towards the object's world position."""
    if T_world_obj is None:
        print("Error: T_world_obj is None, cannot align A1.")
        return

    print("--- Starting Step 2: Align A1 ---")
    # 1. Get current robot state (KMR pose and IIWA joints)
    kmr_pose = api.get_pose()
    joints = api.get_iiwa_joint_position()
    if not kmr_pose or not joints:
        print("Error: Failed to get robot state for A1 alignment.")
        return

    # 2. Calculate IIWA base position in world
    iiwa_base_in_world_mm = utils.get_iiwa_base_in_world([kmr_pose['x'], kmr_pose['y'], kmr_pose['theta']])

    # 3. Calculate vector from IIWA base to Object in world XY plane
    obj_pos_world_mm = T_world_obj[:3, 3]
    vector_base_to_obj = obj_pos_world_mm[:2] - iiwa_base_in_world_mm[:2] # Only consider X, Y

    # 4. Calculate the angle of this vector in the world frame
    # angle = atan2(y, x)
    world_angle_to_object_rad = np.arctan2(vector_base_to_obj[1], vector_base_to_obj[0])

    # 5. Calculate the desired A1 angle
    # The world angle corresponds to KMR_theta + A1 (+ any base offset like pi/2)
    # So, A1_desired = world_angle - KMR_theta (- offset)
    # Let's assume A1 = 0 points along KMR's X axis (needs verification based on setup)
    # If A1=0 aligns with KMR X, then world_angle = KMR_theta + A1
    kmr_theta_rad = kmr_pose['theta']
    a1_desired_rad = world_angle_to_object_rad - kmr_theta_rad

    # Normalize angle to be within IIWA limits (approx -170 to +170 deg, or -2.96 to 2.96 rad)
    a1_desired_rad = (a1_desired_rad + np.pi) % (2 * np.pi) - np.pi # Normalize to [-pi, pi]

    # Clamp to physical joint limits (adjust these values if needed)
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
        speed=0.3 # Use a moderate speed for alignment
    )

    if response and response.ok and response.text.strip() == "OK":
        print("A1 alignment move successful.")
    else:
        print("Error during A1 alignment move.")


def just_pick_it_full_sequence(camera_handler: CameraHandler):
    """Runs the two steps of the JustPickIt sequence."""
    # Step 1: Calculate object pose in world from estimated camera pose
    T_world_obj = just_pick_it_step1_calculate_world_pose(camera_handler)

    print("JustPickIt Step 1 completed. T_world_obj:", T_world_obj)

    if T_world_obj is not None:
        # Add a delay or check before starting step 2
        time.sleep(2)
        # Step 2: Align A1 joint towards the calculated object pose
        just_pick_it_step2_align_A1(T_world_obj)
    else:
        print("JustPickIt sequence failed at Step 1.")