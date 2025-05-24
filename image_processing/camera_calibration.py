import numpy as np
import cv2
import glob
import json
import os
import cv2.aruco as aruco
from image_processing.basler_camera import BaslerCamera
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from utils import utils
from utils.image_utils import get_rotation_matrix_x, get_rotation_matrix_y, get_rotation_matrix_z
import argparse
import json
from scipy.spatial.transform import Rotation as R
import numpy as np
import os

# Run me with python -m image_processing.camera_calibration intrinsic
# python -m image_processing.camera_calibration end_effector

def main():
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(description="Camera Calibration Utility")
    parser.add_argument("function", choices=["intrinsic", "extrinsic", "capture", "visualize_box", "camera_pose", "go_around", "recalculate", "end_effector"], help="Function to execute")
    parser.add_argument("--images_path", type=str, help="Path to the folder containing images")
    parser.add_argument("--display_images", action="store_true", help="Display images while processing")
    parser.add_argument("--output_path", type=str, help="Path to save calibration data")
    parser.add_argument("--chessboard_size", type=int, nargs=2, default=[5, 8], help="Size of the chessboard (columns, rows)")

    args = parser.parse_args()

    match args.function:
        case "intrinsic":
            intrinsic_calibration("images\intrinsic_calibration", display_images=False, output_path="image_processing\calibration_data")

        case "extrinsic":
            extrinsic_calibration("images\extrinsic_calibration", intrinsic_path="image_processing\calibration_data\camera_intrinsics.json", output_path="image_processing\calibration_data")

        case "capture":
            result = capture_images("images\intrinsic_calibration")

        case "visualize_box":
            visualize_box()

        case "camera_pose":
            camera_pose()

        case "go_around":
            visualise_go_around()

        case "recalculate":
            Recalculate_world_position()

        case "end_effector":
            estimate_T_ee_g()

def intrinsic_calibration(images_path: str, **kwargs: dict) -> dict:
    """
    Calibrates the camera using a set of chessboard images.

    Args:
        images_path (str): Path to the folder containing the chessboard images.
                           Images should be in PNG format (or other formats OpenCV can read).
        **kwargs (dict): Optional keyword arguments:
            display_images (bool): If True, displays images with detected corners. Default: False.
            output_path (str): Path to save the calibration data. Default: same as images_path.
            chessboard_size (tuple): (columns, rows) of internal corners. Default: (5, 8).
                                     Example: A 6x9 chessboard has (5,8) internal corners.
            image_format (str): Image file extension (e.g., "png", "jpg"). Default: "png".

    Raises:
        RuntimeError: If no chessboard patterns are detected in the images or calibration fails.
        FileNotFoundError: If images_path does not exist or no images are found.

    Returns:
        dict: Camera calibration data containing the camera matrix, distortion coefficients,
              image resolution, and mean reprojection error.

    Notes:
        This function creates a file named 'camera_intrinsics.json' in the output_path.
        The file contains the camera calibration matrix, distortion coefficients,
        image resolution, and mean reprojection error.
    """
    display_images = kwargs.get("display_images", False)
    output_path = kwargs.get("output_path", images_path)
    chessboard_size = kwargs.get("chessboard_size", (5, 8)) # (cols, rows) of internal corners
    image_format = kwargs.get("image_format", "png")

    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Images path '{images_path}' does not exist.")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")

    images = glob.glob(os.path.join(images_path, f"*.{image_format}"))

    if not images:
        raise FileNotFoundError(f"No '{image_format}' images found in '{images_path}'.")

    cols_corners, rows_corners = chessboard_size
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(rows-1,cols-1,0)
    # These are the 3D coordinates of the chessboard corners in its own coordinate system.
    # The Z-coordinate is 0 because we assume the chessboard is planar.
    objp = np.zeros((cols_corners * rows_corners, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols_corners, 0:rows_corners].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    img_shape = None # To store image shape for calibration

    print(f"Processing {len(images)} images for calibration...")
    found_count = 0
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"Warning: Could not read image {fname}. Skipping.")
            continue

        if img_shape is None:
            img_shape = img.shape[:2] # (height, width)
        elif img_shape != img.shape[:2]:
            print(f"Warning: Image {fname} has a different resolution {img.shape[:2]} "
                  f"than the first image {img_shape}. Skipping. "
                  "All images for calibration should have the same resolution.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (cols_corners, rows_corners), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            found_count += 1

            if display_images:
                cv2.drawChessboardCorners(img, (cols_corners, rows_corners), corners2, ret)
                # Resize for display if too large
                h, w = img.shape[:2]
                scale = min(640/w, 480/h, 1.0) # Ensure it fits, don't upscale
                display_img = cv2.resize(img, (int(w*scale), int(h*scale)))
                cv2.imshow(f'Chessboard Detection - {os.path.basename(fname)}', display_img)
                cv2.waitKey(100) # ms delay
        else:
            print(f"Chessboard not found in {os.path.basename(fname)}")

    if display_images:
        cv2.destroyAllWindows()

    print(f"\nNumber of images where chessboard was successfully detected: {found_count} / {len(images)}")

    if found_count == 0:
        raise RuntimeError(
            "No chessboard patterns detected in any image. Check images, chessboard_size, "
            f"or image_format ('{image_format}'). Expected {chessboard_size} internal corners."
        )
    if found_count < 5: # OpenCV recommends at least 10-20 images, but 5 is a bare minimum
        print(f"Warning: Only {found_count} images with detected patterns. "
              "Calibration quality might be low. Consider adding more diverse images.")

    # gray.shape[::-1] gives (width, height)
    # img_shape is (height, width), so img_shape[::-1] is (width, height)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape[::-1], None, None)

    if not ret:
        raise RuntimeError("Camera calibration failed. Ensure chessboard images are suitable and diverse.")

    # --- Reprojection Error Calculation ---
    total_error = 0
    for i in range(len(objpoints)):
        # Project the 3D object points to 2D image points using the
        # calibrated camera matrix (mtx), distortion coefficients (dist),
        # and the per-image rotation (rvecs[i]) and translation (tvecs[i]) vectors.
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

        # Calculate the L2 norm (Euclidean distance) between the reprojected points
        # and the originally detected image points.
        # This error is then averaged over all points in the current image.
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

    mean_reprojection_error = total_error / len(objpoints)
    # A good calibration typically has a mean reprojection error less than 1.0 pixels.
    # Values between 0.1 and 0.5 are common for high-quality calibrations.

    camera_data = {
        "K": mtx.tolist(),
        "resolution": [int(img_shape[0]), int(img_shape[1])], # height, width
        "dist_coeff": dist.tolist(),
        "mean_reprojection_error_pixels": float(f"{mean_reprojection_error:.4f}")
    }

    print("\n--- Calibration Results ---")
    print(f"Camera Matrix (K):\n{mtx}")
    print(f"\nDistortion Coefficients (k1, k2, p1, p2, k3):\n{dist}")
    print(f"\nImage Resolution (height, width): {camera_data['resolution']}")
    print(f"\nMean Reprojection Error: {mean_reprojection_error:.4f} pixels")
    if mean_reprojection_error > 1.0:
        print("Warning: Mean reprojection error is high (> 1.0 pixels). "
              "Calibration might be suboptimal. Consider re-capturing images with more variation "
              "in chessboard pose and ensuring good lighting and sharp focus.")

    out_file = os.path.join(output_path, "camera_intrinsics.json")
    with open(out_file, "w") as json_file:
        json.dump(camera_data, json_file, indent=4)
    print(f"\nCamera calibration data saved to '{out_file}'.")

    return camera_data

def capture_images(output_folder: str) -> None:
    """
        Captures images from a Basler camera and saves them to a specified folder. Use 'Space' to capture an image, 'q' to quit.

        Args:
            output_folder (str): Path to the folder where images will be saved.
    """
    camera = BaslerCamera()
    camera.stream_and_capture(output_folder)
    camera.close()
    pass


def get_robot_transformation(image_path: str) -> np.array:
    """
    Retrieves the robot transformation matrix from a JSON file based on the image filename. It is expected that there is a JSON file named 'calibration_data.json' in the same folder as the image. The JSON file should contain a list of objects, each with an 'image' key and a 'pose' key. The 'pose' key should contain the robot pose data (x, y, z, A, B, C).

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.array: 4x4 transformation matrix representing the robot pose.

    Raises:
        ValueError: If no transformation is found for the given image.
    
    """


    folder_path = os.path.dirname(image_path)
    with open(os.path.join(folder_path, "calibration_data.json"), "r") as json_file:
        calibration_data = json.load(json_file)
    
    for data in calibration_data:
        if data["image"] == os.path.basename(image_path):
            pose = data["pose"]
            A = np.eye(4)

            xrot = get_rotation_matrix_x(pose["C"])
            yrot = get_rotation_matrix_y(pose["B"])
            zrot = get_rotation_matrix_z(pose["A"])

            full_rot = zrot @ yrot @ xrot

            A[:3, :3] = full_rot[:3, :3]

            A[:3, 3] = np.array([pose["x"], pose["y"], pose["z"]])
            return A
    
    raise ValueError(f"No transformation found for image {image_path}")

def extrinsic_calibration(file_path: str, **kwargs: dict) -> dict:
    """
    Calibrates the camera extrinsics using ArUco markers and robot pose data.
    Includes reprojection error calculation.

    Args:
        file_path (str): Path to the folder containing ArUco marker images and robot pose data.
        **kwargs: Additional keyword arguments:
            intrinsic_path (str): Path to camera_intrinsics.json.
                                  Default: "image_processing/calibration_data/camera_intrinsics.json"
            output_path (str): Path to save camera_extrinsic.json. Default: same as file_path.
            marker_length (float): Side length of the ArUco marker (in same units as robot poses
                                   and tvecs from estimatePoseSingleMarkers). Default: 120.0.
            aruco_dict_name (int): OpenCV ArUco dictionary ID (e.g., aruco.DICT_6X6_1000).
                                   Default: aruco.DICT_6X6_1000.
            hand_eye_method (int): OpenCV hand-eye calibration method.
                                   Default: cv2.CALIB_HAND_EYE_TSAI.
            image_format (str): Image file extension (e.g., "png", "jpg"). Default: "png".

    Raises:
        RuntimeError: If not enough valid marker detections are found for calibration or if calibration fails.
        FileNotFoundError: If critical files (intrinsics, images) are missing.

    Returns:
        dict: Extrinsic calibration data including transformation matrix and reprojection error.
    """

    intrinsic_path = kwargs.get("intrinsic_path", os.path.join("image_processing", "calibration_data", "camera_intrinsics.json"))
    output_path = kwargs.get("output_path", file_path)
    # Marker length used for estimatePoseSingleMarkers AND for defining 3D object points for reprojection
    marker_length_val = float(kwargs.get("marker_length", 120.0))
    aruco_dict_id = kwargs.get("aruco_dict_name", aruco.DICT_6X6_1000)
    hand_eye_method = kwargs.get("hand_eye_method", cv2.CALIB_HAND_EYE_TSAI)
    image_format = kwargs.get("image_format", "png")


    if not os.path.exists(intrinsic_path):
        raise FileNotFoundError(f"Intrinsic calibration file not found: {intrinsic_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image data path not found: {file_path}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(intrinsic_path, "r") as json_file:
        camera_data = json.load(json_file)

    camera_matrix = np.array(camera_data["K"], dtype=np.float64)
    dist_coeffs = np.array(camera_data["dist_coeff"], dtype=np.float64)
    if dist_coeffs.ndim == 1: # Ensure shape (1,N) or (N,1) for OpenCV
        dist_coeffs = dist_coeffs.reshape(1, -1)

    aruco_images = sorted(glob.glob(os.path.join(file_path, f"*.{image_format}")))
    if not aruco_images:
        raise FileNotFoundError(f"No '{image_format}' images found in '{file_path}'.")

    aruco_dict = aruco.getPredefinedDictionary(aruco_dict_id)
    if hasattr(aruco, 'DetectorParameters_create'): # OpenCV 3.x, 4.0-4.6
       parameters = aruco.DetectorParameters_create()
    else: # OpenCV 4.7+
       parameters = aruco.DetectorParameters()
    
    # For cv2.calibrateHandEye
    A_matrices_list = [] # List of T_base2ee (robot pose)
    B_matrices_list = [] # List of T_marker2cam (marker pose in camera)
    
    # For reprojection error
    original_detected_corners_list = [] # List of detected 2D corners from images

    print(f"Processing {len(aruco_images)} images for hand-eye calibration...")
    for image_path in aruco_images:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # corners: list of (1, 4, 2) arrays for each detected marker
        # ids: array of marker IDs
        corners_all_markers, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
  
        if ids is not None and len(ids) > 0:
            # Use the first detected marker. If specific marker ID is needed, filter here.
            # We need its specific corners array which is corners_all_markers[0]
            current_marker_corners = corners_all_markers[0] # Shape (1,4,2)

            # rvecs, tvecs are arrays of vectors, one for each marker
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(current_marker_corners, marker_length_val, camera_matrix, dist_coeffs)

            try:
                A_i = get_robot_transformation(image_path) # T_base2ee (robot pose)
                print("FFF"*15)
                print(f"{A_i=}")
                
                B_i = np.eye(4) # T_marker2cam (marker pose in camera frame)
                B_i[:3, :3], _ = cv2.Rodrigues(rvecs[0]) # rvecs[0] is for the first (and only, in this case) marker
                B_i[:3, 3] = tvecs[0].flatten()
                
                A_matrices_list.append(A_i)
                B_matrices_list.append(B_i)
                original_detected_corners_list.append(current_marker_corners) # Store (1,4,2)

            except ValueError as e:
                print(f"Skipping {os.path.basename(image_path)} due to error: {e}")
                continue
        else:
            print(f"No ArUco markers detected in {os.path.basename(image_path)}")
    
    num_valid_poses = len(A_matrices_list)
    print(f"\nNumber of valid image-pose pairs for calibration: {num_valid_poses}")

    if num_valid_poses < 3: # Hand-eye generally needs at least 3, preferably more diverse poses
        raise RuntimeError(f"Not enough valid marker/robot pose pairs ({num_valid_poses}) for calibration. Need at least 3.")
    
    # Prepare inputs for calibrateHandEye
    # R_gripper2base, t_gripper2base (from robot poses)
    # R_target2cam, t_target2cam (from marker detections)
    # The interpretation of "gripper" and "target" depends on the specific hand-eye setup (eye-in-hand vs eye-to-hand)
    # and the method used. Assuming eye-in-hand, T_cam2ee is sought.
    # User's original code implies A_list are T_base2ee, B_list are T_marker2cam.
    # The call calibrateHandEye(A_rotations, A_translations, B_rotations, B_translations)
    # with Tsai typically solves T_base2ee_i * X = Y * T_marker2cam_i for X=T_ee2cam (if Y is T_base2marker)
    # or a similar formulation resulting in T_cam2ee.
    
    A_rotations = [A[:3, :3] for A in A_matrices_list]
    A_translations = [A[:3, 3].reshape(3,1) for A in A_matrices_list] # Ensure column vector
    B_rotations = [B[:3, :3] for B in B_matrices_list]
    B_translations = [B[:3, 3].reshape(3,1) for B in B_matrices_list] # Ensure column vector
    
    # R_cam2ee, t_cam2ee should be T_camera_to_endeffector (X in our notation)
    R_cam2ee, t_cam2ee_col_vec = cv2.calibrateHandEye(
        A_rotations, A_translations, 
        B_rotations, B_translations, 
        method=hand_eye_method
    )
    
    T_cam2ee = np.eye(4) # This is X_calib = T_camera_to_endeffector
    T_cam2ee[:3, :3] = R_cam2ee
    T_cam2ee[:3, 3] = t_cam2ee_col_vec.flatten()
    
    print("\n--- Hand-Eye Calibration Result (T_camera_to_endeffector) ---")
    print(f"Rotation matrix (R_cam2ee):\n{R_cam2ee}")
    print(f"\nTranslation vector (t_cam2ee) [units consistent with marker_length & robot poses]:\n{t_cam2ee_col_vec.flatten()}")
    print(f"\nTransformation matrix (T_cam2ee):\n{T_cam2ee}")

    # --- Reprojection Error Calculation ---
    # Y_static = T_base2marker. Should be constant.
    # Y_i = A_i * X_calib * inv(B_i)
    Y_candidates_T_base2marker = []
    for i in range(num_valid_poses):
        A_i = A_matrices_list[i]
        B_i = B_matrices_list[i]
        try:
            inv_B_i = np.linalg.inv(B_i)
            Y_i = A_i @ T_cam2ee @ inv_B_i
            Y_candidates_T_base2marker.append(Y_i)
        except np.linalg.LinAlgError:
            print(f"Warning: Singular matrix B_i for pose {i} during Y_static estimation. Skipping this Y_i.")
            continue
            
    mean_reprojection_error_pixels = -1.0 # Default / error value
    
    if not Y_candidates_T_base2marker:
        print("\nWarning: Could not estimate any T_base2marker (Y_static) candidates. Reprojection error cannot be calculated.")
    else:
        # Averaging Y_static (T_base2marker_avg)
        # Simple averaging: average translations, use first valid rotation for simplicity.
        # More robust rotation averaging is complex and beyond "nothing else".
        Y_translations = [Y[:3,3] for Y in Y_candidates_T_base2marker]
        avg_Y_translation = np.mean(np.array(Y_translations), axis=0)
        avg_Y_rotation = Y_candidates_T_base2marker[0][:3,:3] # Use rotation from the first candidate
        
        T_base2marker_avg = np.eye(4)
        T_base2marker_avg[:3,:3] = avg_Y_rotation
        T_base2marker_avg[:3,3] = avg_Y_translation
        print(f"\nEstimated average static T_base2marker (Y_avg):\n{T_base2marker_avg}")

        try:
            inv_T_base2marker_avg = np.linalg.inv(T_base2marker_avg)
        except np.linalg.LinAlgError:
            print("Error: Average T_base2marker_avg is singular. Cannot compute reprojection error.")
            inv_T_base2marker_avg = None

        if inv_T_base2marker_avg is not None:
            total_error_sum_pixels = 0
            total_points_projected = 0

            # Define 3D object points for the marker (origin at center, Z=0 plane)
            half_L = marker_length_val / 2.0
            obj_points_marker_3d = np.array([
                [-half_L,  half_L, 0.0], [ half_L,  half_L, 0.0],
                [ half_L, -half_L, 0.0], [-half_L, -half_L, 0.0]
            ], dtype=np.float32) # Shape (4,3)

            for i in range(num_valid_poses):
                A_i = A_matrices_list[i] # T_base2ee_i
                detected_corners_2d = original_detected_corners_list[i][0].reshape(-1, 2) # Shape (4,2)

                # Calculate B_reprojected_i = inv(Y_avg) * A_i * X_calib
                # This is the expected T_marker2cam_i based on calibration
                try:
                    T_marker2cam_reprojected = inv_T_base2marker_avg @ A_i @ T_cam2ee
                except np.linalg.LinAlgError: # Should not happen if T_cam2ee, A_i, inv_Y are fine
                    print(f"Warning: LinAlgError during B_reprojected calculation for pose {i}. Skipping.")
                    continue

                R_m2c_reco = T_marker2cam_reprojected[:3,:3]
                t_m2c_reco = T_marker2cam_reprojected[:3,3]
                rvec_m2c_reco, _ = cv2.Rodrigues(R_m2c_reco) # Convert rotation matrix to rvec

                # Project the 3D marker points to 2D image plane
                imgpoints_reprojected, _ = cv2.projectPoints(obj_points_marker_3d,
                                                             rvec_m2c_reco, t_m2c_reco,
                                                             camera_matrix, dist_coeffs)
                
                error_per_marker = cv2.norm(detected_corners_2d, imgpoints_reprojected.reshape(-1,2), cv2.NORM_L2)
                total_error_sum_pixels += error_per_marker
                total_points_projected += len(obj_points_marker_3d) # typically 4 points per marker

            if total_points_projected > 0:
                mean_reprojection_error_pixels = total_error_sum_pixels / total_points_projected
                print(f"\nMean Reprojection Error: {mean_reprojection_error_pixels:.4f} pixels per point")
            else:
                print("\nWarning: No points were projected for error calculation.")
    
    extrinsic_data = {
        "rotation_matrix_cam2ee": R_cam2ee.tolist(),
        "translation_vector_cam2ee": t_cam2ee_col_vec.flatten().tolist(),
        "transformation_matrix_cam2ee": T_cam2ee.tolist(),
        "mean_reprojection_error_pixels": float(f"{mean_reprojection_error_pixels:.4f}") if mean_reprojection_error_pixels >=0 else None,
        "num_poses_used_for_calibration": num_valid_poses,
        "num_Y_candidates_for_reprojection": len(Y_candidates_T_base2marker),
        "marker_length_used": marker_length_val,
        "hand_eye_method_used": hand_eye_method
    }
    
    output_json_path = os.path.join(output_path, "camera_extrinsic.json")
    with open(output_json_path, "w") as f:
        json.dump(extrinsic_data, f, indent=4)
    print(f"\nExtrinsic hand-eye calibration data saved to '{output_json_path}'.")

    return extrinsic_data



def estimate_T_ee_g():
    """
    Estimate the transformation from EE to Gripper using multiple end-effector poses.

    :param T_b_ee_list: List of 4x4 transformation matrices (base to EE).
    :return: Estimated 4x4 transformation matrix T_ee_g
    """

    # Load JSON file
    with open("image_processing/calibration_data/end_efector_poses.json", "r") as f:
        data = json.load(f)

    # Extract transformation matrices (camera_in_world)
    T_b_ee_list = [np.array(entry["camera_in_world"]) for entry in data]


    N = len(T_b_ee_list)
    assert N >= 2, "Need at least two different EE poses."

    # Solve for translation (assuming constant gripper position)
    A = []
    b = []
    
    for i in range(N - 1):
        T1 = T_b_ee_list[i]
        T2 = T_b_ee_list[i + 1]

        R1, t1 = T1[:3, :3], T1[:3, 3]
        R2, t2 = T2[:3, :3], T2[:3, 3]

        A.append(R1 - R2)
        b.append(t2 - t1)

    A = np.vstack(A)
    b = np.hstack(b)

    # Solve for the gripper translation in EE frame
    t_ee_g = np.linalg.lstsq(A, b, rcond=None)[0]

    # Assume rotation is identity (if small rotation)
    T_ee_g = np.eye(4)
    T_ee_g[:3, 3] = t_ee_g

    print("Estimated T_ee_g:\n", T_ee_g)




def plotit():
    T = np.array([
        [-0.24203446, -0.12054252, -0.96275065, 0.0],
        [0.02938975, -0.99270817, 0.11690484, -87.25507252],
        [-0.96982244, 0.0, 0.2438123, -14.70566091],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # Create identity basis vectors (RGB axes)
    basis_vectors = np.eye(3)*30  # [[1,0,0], [0,1,0], [0,0,1]]

    # Transform the origin
    transformed_origin = T[:3, 3]

    # Rotate and translate basis vectors
    transformed_vectors = basis_vectors @ T[:3, :3]  # Rotate basis vectors
    transformed_vectors += transformed_origin  # Translate basis vectors

    # Debugging print
    print("Transformed Origin:\n", transformed_origin)
    print("Transformed Vectors:\n", transformed_vectors)

    # Plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define original origin
    origin = np.zeros(3)

    # Original basis vectors (RGB)
    colors = ['r', 'g', 'b']
    for i in range(3):
        ax.quiver(*origin, *basis_vectors[i], color=colors[i], linestyle='dashed', label=f'Original {colors[i].upper()}')

    # Transformed basis vectors (start from transformed origin)
    for i in range(3):
        ax.quiver(*transformed_origin, *(transformed_vectors[i] - transformed_origin), color=colors[i], label=f'Transformed {colors[i].upper()}')

    # Labels and limits
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])
    ax.set_zlim([-100, 100])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Coordinate Basis Transformation")
    ax.legend()

    plt.show()

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
    ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def my_quiver(ax, matrix):
    """
    Plot the 3D axes of a given transformation matrix.

    Args:
        ax: The matplotlib 3D axis to plot on.
        matrix: A 3x4 or 4x4 matrix. Each column represents a 3D point.
    """

    colors = ['r', 'g', 'b']
    for i in range(3):
        ax.plot(
        [matrix[0, 3], matrix[0, i]],
        [matrix[1, 3], matrix[1, i]],
        [matrix[2, 3], matrix[2, i]],
        color=colors[i], label=f'Axis {colors[i].upper()}'
        )
        
def visualize_box():

    # Load the calibration matrix from the camera_extrinsic.json file
    with open("image_processing/calibration_data/camera_extrinsic.json", "r") as f:
        extrinsic_data = json.load(f)
    calib_matrix = np.array(extrinsic_data["transformation_matrix"])
    
    BOX_HEIGHT = 700
    BOX_LENGTH = 1080
    BOX_WIDTH = 630

    BOX_CENTER_X = 14000
    BOX_CENTER_Y = 15000
    BOX_MIN_Z = 0

    height = BOX_HEIGHT
    length = BOX_LENGTH
    width = BOX_WIDTH
    center_x = BOX_CENTER_X
    center_y = BOX_CENTER_Y
    min_z = BOX_MIN_Z

    iiwa_base = utils.get_iiwa_base_in_world(np.array([center_x, center_y, 0]))

    # Calculate the offset for the center position
    offset_x = center_x - length / 2
    offset_y = center_y - width / 2

    # Define the 8 vertices of the box
    vertices = np.array([
        [offset_x, offset_y, min_z],  # Bottom-front-left
        [offset_x + length, offset_y, min_z],  # Bottom-front-right
        [offset_x + length, offset_y + width, min_z],  # Bottom-back-right
        [offset_x, offset_y + width, min_z],  # Bottom-back-left
        [offset_x, offset_y, min_z + height],  # Top-front-left
        [offset_x + length, offset_y, min_z + height],  # Top-front-right
        [offset_x + length, offset_y + width, min_z + height],  # Top-back-right
        [offset_x, offset_y + width, min_z + height]  # Top-back-left
    ])
    print(vertices)
    # Define the edges connecting the vertices
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    # Plot the box
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])  # Make axes equal

    for edge in edges:
        start, end = edge
        ax.plot(
            [vertices[start][0], vertices[end][0]],
            [vertices[start][1], vertices[end][1]],
            [vertices[start][2], vertices[end][2]],
            'b'
        )


    # Define the position and orientation
    position = np.array([-0.06084885817435056, 8.390708723339504, 806.0509324443722])
    orientation = np.array([3.1308282468610966, -0.8097527279037569, 3.049039185227726])  # Roll, Pitch, Yaw
    # orientation[0], orientation[2] = orientation[2], orientation[0]  # Swap roll and yaw
    position = position
        
    # Compute the rotation matrix
    rotation_matrix = R.from_euler('xyz', orientation, degrees=False).as_matrix()

    base_to_ee = np.hstack((rotation_matrix, position.reshape(3, 1)))
    base_to_ee = np.vstack((base_to_ee, np.zeros((1, 4))))
    base_to_ee[3, 3] = 1

    # Define the arrow directions (unit vectors)
    arrows = np.eye(3) * 350  # Scale by 100
    arrows = np.vstack((arrows, np.ones(3)))
    arrows = np.hstack((arrows, np.zeros((4, 1)))) 
    arrows[3, 3] = 1

    center_x = 14000
    center_y = 15000
    min_z = 0



    transformation_matrix = np.vstack((np.hstack((np.eye(3), utils.get_iiwa_base_in_world(np.array([center_x, center_y, 0])).reshape(3, 1))), np.zeros((1, 4))))
    transformation_matrix[3, 3] = 1




    iiwa_base = transformation_matrix @ arrows

    my_quiver(ax, iiwa_base)


    
    
    print("Base to EE:", base_to_ee)
    print("Arrows:", arrows)
    print("Base to EE @ Arrows:", base_to_ee @ arrows)
    print("Base to EE @ Arrows @ Transformation:", base_to_ee @ arrows @ transformation_matrix)

    iiwa_ee = transformation_matrix @ base_to_ee @ arrows
    my_quiver(ax, iiwa_ee)
    

    camera_pos = transformation_matrix @ base_to_ee @ calib_matrix @ arrows
    my_quiver(ax, camera_pos)





    set_axes_equal(ax)
    plt.show()


def camera_pose():
    # Load the calibration matrix from the camera_extrinsic.json file
    with open("image_processing/calibration_data/camera_extrinsic.json", "r") as f:
        extrinsic_data = json.load(f)
    calib_matrix = np.array(extrinsic_data["transformation_matrix"])

    # Define the position and orientation in the world frame
    position = np.array([-0.06084885817435056, 8.390708723339504, 806.0509324443722])
    orientation = np.array([3.1308282468610966, -0.8097527279037569, 3.049039185227726])  # Roll, Pitch, Yaw

    # Compute the rotation matrix
    rotation_matrix = R.from_euler('xyz', orientation, degrees=False).as_matrix()

    # Create the transformation matrix for the end effector (world to EE)
    world_to_ee = np.hstack((rotation_matrix, position.reshape(3, 1)))
    world_to_ee = np.vstack((world_to_ee, np.array([0, 0, 0, 1])))

    # Compute the end effector position in the world frame
    ee_position = world_to_ee[:3, 3]

    print("End Effector Position in World Frame:", ee_position)
    return ee_position

def visualise_go_around():
    # Load the JSON file

    json_file = "image_processing\calibration_data\GoAroundHandPoses.json"
    # json_file = "communication/HandPoses.json"
    # json_file = "images/GoAround/new_camera_in_world.json"
    with open(json_file, "r") as f:
        data = json.load(f)

    # Extract camera_in_world matrices
    camera_matrices = [np.array(item["camera_in_world"]) for item in data]

    # Plot the camera positions and orientations
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for matrix in camera_matrices:
        # Extract rotation and translation
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]

        # Define the axes (unit vectors) in the local frame
        axes = np.eye(3)

        # Transform the axes to the world frame
        transformed_axes = rotation @ axes

        # Plot the axes using quiver
        colors = ['r', 'g', 'b']
        scale = 50
        for i in range(3):
            ax.quiver(
            translation[0], translation[1], translation[2],  # Origin
            transformed_axes[0, i] * scale, transformed_axes[1, i] * scale, transformed_axes[2, i] * scale,  # Direction (scaled)
            color=colors[i], label=f'Axis {colors[i].upper()}'
            )

    # Plot the global X, Y, Z axes
    ax.quiver(14000, 14000, 0, 4000, 0, 0, color='r', linestyle='dashed', label='Global X')
    ax.quiver(14000, 14000, 0, 0, 4000, 0, color='g', linestyle='dashed', label='Global Y')
    ax.quiver(14000, 14000, 0, 0, 0, 4000, color='b', linestyle='dashed', label='Global Z')
    set_axes_equal(ax)
    ax.set_title("Camera Positions and Orientations")
    plt.show()


def Recalculate_world_position():
    # Load the JSON file
    input_file = "images/GoAround/data.json"
    output_file = "images/GoAround/new_camera_in_world.json"

    with open(input_file, "r") as f:
        data = json.load(f)

    new_data = []

    for item in data:
        pose = item["pose"]
        position = item["position"]

        # Extract position and orientation

        camera_in_world = utils.calculate_camera_in_world(pose, position)
        # Save the new camera_in_world
        new_data.append({
            "image": item["image"],
            "pose": pose,
            "position": position,
            "camera_in_world": camera_in_world.tolist()
        })

    # Save the new JSON file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(new_data, f, indent=4)


if __name__ == "__main__":
    main()



    pass