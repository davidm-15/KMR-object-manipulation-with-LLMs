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
from communication.client import send_image_to_server
from communication.client import send_for_pose_estimation
from scipy.spatial.transform import Rotation as R

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
        # if i < 11:
        #     continue
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
    detection_item = kwargs.get("detection_item", "mustard bottle")
    clean_folder = kwargs.get("clean_folder", False) # Renamed from clean_fodler
    json_filename = kwargs.get("json_filename", "sequence_data.json") # Use a different default filename

    if clean_folder:
        utils.clean_directory(output_folder)

    os.makedirs(output_folder, exist_ok=True) # Ensure directory exists

    locations = kwargs.get("locations", [8]) # Use predefined locations 1 and 2

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



    drive_to_object(T_world_obj)



    if T_world_obj_list is not None:
        # align_A1_with_object(np.array(T_world_obj_list))

        # Calculate object position in IIWA base coordinates
        print("Calculating object position in IIWA base coordinates...")
                
        kmr_pose = api.get_pose()
        # Get IIWA base position in world coordinates
        iiwa_base_in_world_mm = utils.get_iiwa_base_in_world([kmr_pose['x'], kmr_pose['y'], kmr_pose['theta']])
                
        # Calculate object position relative to IIWA base (only X,Y,Z)
        object_position = np.array(T_world_obj[:3, 3])  # Extract object position from transformation matrix
        object_relative_to_base = object_position - iiwa_base_in_world_mm
            
        print(f"Object position in world: {object_position}")
        print(f"IIWA base position in world: {iiwa_base_in_world_mm}")
        print(f"Object position relative to IIWA base: {object_relative_to_base}")

        # Extract rotation matrix from transformation matrix
        rotation_matrix = T_world_obj[:3, :3]

        # Convert 3x3 rotation matrix to Euler angles in ZYX (extrinsic) convention
        r = R.from_matrix(rotation_matrix)
        angles_rad = r.as_euler('zyx')
        angles_deg = np.degrees(angles_rad)
        
        print(f"Euler angles (ZYX convention):")
        print(f"A (Z rotation/yaw): {angles_rad[0]:.4f} rad ({angles_deg[0]:.2f}°)")
        print(f"B (Y rotation/pitch): {angles_rad[1]:.4f} rad ({angles_deg[1]:.2f}°)")
        print(f"C (X rotation/roll): {angles_rad[2]:.4f} rad ({angles_deg[2]:.2f}°)")
        

        # Define gripper offset in z direction (mm)
        gripper_z_offset = 210  # mm

        # Calculate needed end-effector position for proper gripper positioning
        print("\nCalculating end-effector position with gripper offset consideration:")
        # The gripper is offset along the z-axis of the end-effector frame
        # We need to move the end-effector backward by this offset to position the gripper correctly

        # Convert gripper offset from end-effector frame to world frame
        # We need to use the rotation matrix to transform the offset vector [0, 0, -gripper_z_offset]
        # (negative because we're moving backward from the object position)
        offset_vector_ee = np.array([0, 0, -gripper_z_offset, 1])  # Homogeneous coordinates
        offset_vector_world = np.dot(T_world_obj, offset_vector_ee)

        # Calculate the target end-effector position
        target_ee_position = offset_vector_world[:3]

        print(f"Object position in world: {object_position}")
        print(f"Target end-effector position (accounting for gripper offset): {target_ee_position}")
        print(f"World frame offset vector from object to end-effector: {target_ee_position - object_position}")

        # Calculate distance from object to target position as a sanity check
        distance = np.linalg.norm(target_ee_position - object_position)
        print(f"Distance from object to target position: {distance:.2f} mm (should be close to {gripper_z_offset} mm)")

        # Convert rotation matrix to A,B,C Euler angles (same convention as IIWA uses)
        # Note: These angles are in radians, need to be converted to degrees for the API call
        r = R.from_matrix(rotation_matrix)
        angles_rad = r.as_euler('zyx')  # ZYX convention = A,B,C for IIWA

        # Extract target position and orientation for the end effector
        target_x = float(target_ee_position[0] - iiwa_base_in_world_mm[0])  # Convert to IIWA base coordinates
        target_y = float(target_ee_position[1] - iiwa_base_in_world_mm[1])
        target_z = float(target_ee_position[2] - iiwa_base_in_world_mm[2])

        # Extract orientation angles in radians
        a_rad = float(angles_rad[0])  # Rotation around Z axis
        b_rad = float(angles_rad[1])  # Rotation around Y axis
        c_rad = float(angles_rad[2])  # Rotation around X axis

        # Convert radians to degrees for the API
        a_deg = float(np.degrees(a_rad))
        b_deg = float(np.degrees(b_rad))
        c_deg = float(np.degrees(c_rad))

        print("\nMoving end effector to target position with calculated orientation:")
        print(f"Position (IIWA base coords): X={target_x:.2f}, Y={target_y:.2f}, Z={target_z:.2f} mm")
        print(f"Orientation (degrees): A={a_deg:.2f}, B={b_deg:.2f}, C={c_deg:.2f}")

        # First align A1 with object for better approach

        # Then move to the target position 
        print("Moving to target position...")
        response = api.goto_position(
            x=target_x, 
            y=target_y, 
            z=target_z, 
            a=a_deg, 
            b=b_deg, 
            c=c_deg,
            speed=0.2,  # Lower speed for safety
            motion_type="ptp"  # Point-to-point motion
        )

        if response and response.ok and response.text.strip() == "OK":
            print("Successfully moved to target position.")
        else:
            print(f"Error moving to target position: {response.text if response else 'No response'}")



    return T_world_obj_list

def drive_to_object(T_world_obj):
    Object_position = np.array(T_world_obj[:3, 3])  



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
        [-0.0319, 0.024, 0.9992, 12685.1941],
        [-0.9895, 0.1401, -0.035, 15230.1932],
        [-0.1409, -0.9898, 0.0193, 869.069],
        [0.0, 0.0, 0.0, 1.0]
    ])


    # Get the current KMR pose to calculate the IIWA base position in world
    kmr_pose = api.get_pose()
    if not kmr_pose:
        print("Error: Failed to get KMR pose.")
        return

    # Calculate IIWA base position in world coordinates
    iiwa_base_in_world_mm = utils.get_iiwa_base_in_world([kmr_pose['x'], kmr_pose['y'], kmr_pose['theta']])
    
    # Extract object position from transformation matrix
    object_position = object_in_world[:3, 3]
    
    # Subtract IIWA base position from object position (only X,Y,Z)
    object_relative_to_base = object_position - iiwa_base_in_world_mm
    
    print(f"Object position in world: {object_position}")
    print(f"IIWA base position in world: {iiwa_base_in_world_mm}")
    print(f"Object position relative to IIWA base: {object_relative_to_base}")


    align_A1_with_object(object_in_world)


def Object_to_world():
    object_pose_in_camera = [
        [-0.9996992349624634, -0.020981529727578163, -0.012695626355707645, -0.01995106227695942],
        [-0.020352913066744804, 0.9986512660980225, -0.04776711016893387, 0.017018599435687065],
        [0.013680730015039444, -0.047494351863861084, -0.9987779855728149, 0.39274975657463074],
        [0.0, 0.0, 0.0, 1.0]
    ]

    T_cam_obj = np.array(object_pose_in_camera)
    
    # Check if the units need to be converted (meters to mm)
    if np.max(np.abs(T_cam_obj[:3, 3])) < 10.0:  # Heuristic: if max translation < 10, assume meters
        print("Converting pose translation from meters to mm.")
        T_cam_obj[:3, 3] *= 1000
    else:
        print("Assuming pose translation is already in mm.")
    
    # Get current robot state
    kmr_pose = api.get_pose()
    iiwa_pos = api.get_iiwa_position()
    
    if not all([kmr_pose, iiwa_pos]):
        print("Failed to get complete robot state.")
        return None
    
    # Calculate T_world_cam
    T_world_cam = utils.calculate_camera_in_world(kmr_pose, iiwa_pos)
    if T_world_cam is None:
        print("Failed to calculate camera pose in world.")
        return None
    
    # Calculate object in world coordinates
    T_world_obj = utils.calculate_object_in_world(T_world_cam, T_cam_obj)
    print(f"Object pose in world coordinates:\n{T_world_obj}")
    
    return T_world_obj




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