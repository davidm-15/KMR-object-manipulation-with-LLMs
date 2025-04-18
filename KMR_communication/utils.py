# utils.py
import json
import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation as R
from . import config # 


def load_json_data(filepath: str) -> list | dict:
    """Loads data from a JSON file, handling file not found and decode errors."""
    try:
        with open(filepath, "r") as file:
            try:
                data = json.load(file)
                # Basic check if it should be a list (common case in the original code)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                     return data
                else: # If it's something else, maybe return default based on expectation?
                    print(f"Warning: Unexpected data type {type(data)} in {filepath}, returning empty list.")
                    return [] # Defaulting to list based on original code patterns
            except json.JSONDecodeError:
                print(f"Warning: Error decoding JSON from {filepath}. Returning empty list.")
                return [] # Defaulting to list
    except FileNotFoundError:
        print(f"Info: File not found: {filepath}. Returning empty list.")
        return [] # Defaulting to list

def append_to_json_list(filepath: str, data_to_append: dict):
    """Loads a JSON file (expecting a list), appends data, and saves it back."""
    existing_data = load_json_data(filepath)
    if not isinstance(existing_data, list):
        print(f"Error: Expected a list in {filepath} but found {type(existing_data)}. Cannot append.")
        # Or initialize as a list if appropriate: existing_data = []
        return

    existing_data.append(data_to_append)

    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as file:
            json.dump(existing_data, file, indent=4)
        print(f"Appended data to {filepath}")
    except IOError as e:
        print(f"Error writing to {filepath}: {e}")

def save_json_data(filepath: str, data: dict | list):
    """Saves data (dict or list) to a JSON file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as file:
            json.dump(data, file, indent=4)
        print(f"Saved data to {filepath}")
    except IOError as e:
        print(f"Error writing to {filepath}: {e}")


def ndarray_to_bytes(image: np.ndarray, format: str = ".png") -> bytes | None:
    """Converts a NumPy ndarray image to bytes."""
    success, buffer = cv2.imencode(format, image)
    if success:
        return buffer.tobytes()
    else:
        print("Error encoding image to bytes.")
        return None

def get_rotation_matrix_2D(angle: float = 0.0) -> np.ndarray:
    """Calculates a 2D rotation matrix for a given angle in radians."""
    c, s = np.cos(angle), np.sin(angle)
    R_mat = np.array([[c, -s],
                      [s,  c]])
    return R_mat

def get_iiwa_base_in_world(kmr_pose_m_rad: list | np.ndarray) -> np.ndarray:
    """
    Calculates the position of the iiwa base in the world coordinate system.

    Args:
        kmr_pose_m_rad (list | np.ndarray): The pose [x, y, theta] of the KMR base
                                           in the world frame (meters, radians).

    Returns:
        np.ndarray: The 3D position [x, y, z] of the iiwa base in the world
                    coordinate system (in mm).
    """
    kmr_x_m, kmr_y_m, kmr_theta_rad = kmr_pose_m_rad
    kmr_x_mm = kmr_x_m * 1000
    kmr_y_mm = kmr_y_m * 1000

    # Use offsets from config
    x_offset = config.IIWA_BASE_X_OFFSET
    y_offset = config.IIWA_BASE_Y_OFFSET
    z_offset = config.IIWA_BASE_Z_OFFSET # Fixed height

    # Create a 2D rotation matrix for the KMR's orientation
    rotation_matrix_2d = get_rotation_matrix_2D(kmr_theta_rad)

    # Apply the rotation matrix to the offsets
    offset_vector = np.array([x_offset, y_offset])
    rotated_offset = rotation_matrix_2d @ offset_vector

    # Calculate the final iiwa base position in world (mm)
    iiwa_base_position_mm = np.array([
        kmr_x_mm + rotated_offset[0],
        kmr_y_mm + rotated_offset[1],
        z_offset
    ])

    return iiwa_base_position_mm

def get_T_world_iiwabase(kmr_pose_m_rad: list | np.ndarray) -> np.ndarray:
    """
    Calculates the 4x4 transformation matrix of the iiwa base pose
    in the world coordinate system.
    """
    kmr_theta_rad = kmr_pose_m_rad["theta"] + np.pi/2*3

    # --- Rotation Part (R_world_iiwabase) ---
    # Assumes iiwa base rotation is only around World Z axis, same as KMR
    
    print(f"kmr_theta_rad: {kmr_theta_rad}")

    cos_t = np.cos(kmr_theta_rad)
    sin_t = np.sin(kmr_theta_rad)
    R_world_iiwabase = np.array([
        [cos_t, -sin_t, 0.0],
        [sin_t,  cos_t, 0.0],
        [0.0,    0.0,   1.0]
    ])

    # --- Translation Part (p_world_iiwabase) ---
    p_world_iiwabase_mm = get_iiwa_base_in_world([kmr_pose_m_rad['x'], kmr_pose_m_rad['y'], kmr_pose_m_rad['theta']])

    # --- Construct the 4x4 Matrix ---
    T = np.identity(4)
    T[0:3, 0:3] = R_world_iiwabase # np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    T[0:3, 3] = p_world_iiwabase_mm # Assign the 1D array directly

    return T

# Helper function to extract Euler angles (check API's expected convention!)
def rotation_matrix_to_euler_zyx(R_matrix):
    """Converts a 3x3 rotation matrix to ZYX Euler angles (roll, pitch, yaw) in radians."""
    # Ensure input is a NumPy array
    R_matrix = np.asarray(R_matrix)
    r = R.from_matrix(R_matrix)
    # Common convention: 'zyx' means apply Yaw (Z), then Pitch (Y), then Roll (X)
    # The resulting angles are often called roll, pitch, yaw respectively.
    # Adjust 'zyx' if your API uses a different convention (e.g., 'xyz', 'zyz')
    euler_angles = r.as_euler('xyz', degrees=False) # Get radians
    return euler_angles # Returns [roll, pitch, yaw] or [a, b, c] depending on context


def calculate_end_effector_in_world(kmr_pose_m_rad: dict, iiwa_pose_mm_rad: dict) -> np.ndarray | None:
    """
    Calculates the 4x4 transformation matrix of the end-effector in the world frame.

    Args:
        kmr_pose_m_rad (dict): KMR pose {'x': float (m), 'y': float (m), 'theta': float (rad)}.
        iiwa_pose_mm_rad (dict): IIWA end-effector pose {'x': float (mm), 'y': float (mm), 'z': float (mm),
                                                         'A': float (rad), 'B': float (rad), 'C': float (rad)}.

    Returns:
        np.ndarray | None: The 4x4 transformation matrix (end-effector frame to world frame) or None if error.
    """

    try:
        iiwa_base_pos_mm = get_iiwa_base_in_world([kmr_pose_m_rad['x'], kmr_pose_m_rad['y'], kmr_pose_m_rad['theta']])
        kmr_theta_rad = kmr_pose_m_rad['theta']

        angle = np.pi/2*3 + kmr_theta_rad 

        rot_z = R.from_euler('z', angle, degrees=False).as_matrix()
        T_world_iiwaBase = np.eye(4)
        T_world_iiwaBase[:3, :3] = rot_z
        T_world_iiwaBase[:3, 3] = iiwa_base_pos_mm

        ee_pos_mm = np.array([iiwa_pose_mm_rad["x"], iiwa_pose_mm_rad["y"], iiwa_pose_mm_rad["z"]])

        ee_orient_rad = np.array([iiwa_pose_mm_rad["A"], iiwa_pose_mm_rad["B"], iiwa_pose_mm_rad["C"]])
        ee_rot_matrix = R.from_euler('xyz', ee_orient_rad, degrees=False).as_matrix() # Check convention! KUKA might be different. Often it's ZYX extrinsic.

        T_iiwaBase_ee = np.eye(4)
        T_iiwaBase_ee[:3, :3] = ee_rot_matrix
        T_iiwaBase_ee[:3, 3] = ee_pos_mm

        T_world_ee = T_world_iiwaBase @ T_iiwaBase_ee

        return T_world_ee

    except KeyError as e:
        print(f"Error: Missing key in input dictionaries: {e}")
        return None
    except Exception as e:
        print(f"Error calculating end effector in world: {e}")
        return None


def calculate_camera_in_world(kmr_pose_m_rad: dict, iiwa_pose_mm_rad: dict) -> np.ndarray | None:
    """
    Calculates the 4x4 transformation matrix of the camera in the world frame.

    Args:
        kmr_pose_m_rad (dict): KMR pose {'x': float (m), 'y': float (m), 'theta': float (rad)}.
        iiwa_pose_mm_rad (dict): IIWA end-effector pose {'x': float (mm), 'y': float (mm), 'z': float (mm),
                                                         'A': float (rad), 'B': float (rad), 'C': float (rad)}.

    Returns:
        np.ndarray | None: The 4x4 transformation matrix (camera frame to world frame) or None if error.
    """
    try:
        # First, get the end effector in world frame
        T_world_ee = calculate_end_effector_in_world(kmr_pose_m_rad, iiwa_pose_mm_rad)
        if T_world_ee is None:
            return None
        
        # Load the camera extrinsic calibration data
        extrinsic_data = load_json_data(config.CAMERA_EXTRINSIC_FILE)
        if not extrinsic_data or "transformation_matrix" not in extrinsic_data:
             print(f"Error: Could not load or parse {config.CAMERA_EXTRINSIC_FILE}")
             return None
        
        # Transform from end effector to camera frame
        T_ee_cam = np.array(extrinsic_data["transformation_matrix"])
        
        # Calculate camera in world frame
        T_world_cam = T_world_ee @ T_ee_cam

        return T_world_cam

    except Exception as e:
        print(f"Error calculating camera in world: {e}")
        return None

def calculate_object_in_world(T_world_cam: np.ndarray, T_cam_obj: np.ndarray) -> np.ndarray:
    """
    Calculates the 4x4 transformation matrix of the object in the world frame.

    Args:
        T_world_cam (np.ndarray): 4x4 Transform from World frame to Camera frame.
        T_cam_obj (np.ndarray): 4x4 Transform from Camera frame to Object frame.
                                 (Ensure translation is in mm).

    Returns:
        np.ndarray: The 4x4 transformation matrix (object frame to world frame).
    """

    T_world_obj = T_world_cam @ T_cam_obj
    return T_world_obj

def inverse_homogeneous_transform(T: np.ndarray) -> np.ndarray:
    """
    Computes the inverse of a 4x4 homogeneous transformation matrix.

    Args:
        T (np.ndarray): 4x4 transformation matrix.

    Returns:
        np.ndarray: Inverse of the transformation matrix.
    """
    R = T[:3, :3]
    t = T[:3, 3]

    # Inverse of rotation
    R_inv = R.T

    # Inverse of translation
    t_inv = -R_inv @ t

    # Construct the inverse transformation matrix
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv

    return T_inv

def clean_directory(directory_path: str):
    """Removes all files and subdirectories within a given directory."""
    if not os.path.isdir(directory_path):
        print(f"Directory not found, cannot clean: {directory_path}")
        return

    print(f"Cleaning directory: {directory_path}")
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                # Use shutil.rmtree to remove non-empty directories if needed
                # os.rmdir(file_path) # Only removes empty directories
                import shutil
                shutil.rmtree(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

# Add other potential utils if needed