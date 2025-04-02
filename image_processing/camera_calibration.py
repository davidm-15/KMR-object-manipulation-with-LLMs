import numpy as np
import cv2
import glob
import json
import os
import cv2.aruco as aruco
from image_processing.basler_camera import BaslerCamera
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from communication.KMR_communication import get_iiwa_base_in_world
from communication.KMR_communication import GetCameraInWorld
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
        images_path (str): Path to the folder containing the chessboard images. Images should be in PNG format.

    Raises:
        RuntimeError: If no chessboard patterns are detected in the images or calibration fails.

    Returns:
        dict: Camera calibration data containing the camera matrix, distortion coefficients, and image resolution.

    Notes:
        This function creates a file named 'camera_data.json' in the same folder as input images. 
        The file contains the camera calibration matrix, distortion coefficients, and image resolution.
    """
    display_images = kwargs.get("display_images", False)
    output_path = kwargs.get("output_path", images_path)
    chessboard_size = kwargs.get("chessboard_size", (5, 8))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    images = glob.glob(os.path.join(images_path, "*.png"))

    if not images:
        raise RuntimeError(f"No PNG images found in '{images_path}'.")

    xssize, yssize = chessboard_size
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((xssize * yssize, 3), np.float32)
    objp[:, :2] = np.mgrid[0:xssize, 0:yssize].T.reshape(-1, 2)

    objpoints, imgpoints = [], []
    found = 0
    gray = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Warning: Could not read image {fname}. Skipping.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (xssize, yssize), None)

        if ret and corners is not None:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            cv2.drawChessboardCorners(img, (xssize, yssize), corners2, ret)
            found += 1
            if display_images:
                cv2.imshow('Chessboard Detection', cv2.resize(img, (640, 480)))
                cv2.waitKey(100)

    print(f"Number of images used for calibration: {found}")

    if display_images:
        cv2.destroyAllWindows()

    if found == 0:
        raise RuntimeError("No chessboard patterns detected. Check images or pattern size. Are the images in PNG format?")

    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if not ret:
        raise RuntimeError("Camera calibration failed. Ensure chessboard images are suitable.")

    camera_data = {
        "K": mtx.tolist(),
        "resolution": [gray.shape[0], gray.shape[1]],
        "dist_coeff": dist.tolist()
    }
    print("Camera Matrix (K):")
    print(mtx)
    print("\nDistortion Coefficients:")
    print(dist)
    print("\nImage Resolution:")
    print(gray.shape[::-1])

    out_file = os.path.join(output_path, "camera_intrinsics.json")
    with open(out_file, "w") as json_file:
        json.dump(camera_data, json_file, indent=4)
        print(f"Camera calibration data saved to '{out_file}'.")

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

def extrinsic_calibration(file_path: str, **kwargs) -> None:
    """
    Calibrates the camera extrinsics using ArUco markers and robot pose data.

    Args:
        file_path (str): Path to the folder containing ArUco marker images and robot pose data.
        **kwargs: Additional keyword arguments.

    Raises:
        RuntimeError: If not enough valid marker detections are found for calibration.  


    """

    intrinsic_path = kwargs.get("intrinsic_path", "image_processing\calibration_data\camera_intrinsics.json")
    output_path = kwargs.get("output_path", file_path)

    with open(intrinsic_path, "r") as json_file:
        camera_data = json.load(json_file)

    camera_matrix = np.array(camera_data["K"])
    dist_coeffs = np.array(camera_data["dist_coeff"])
    
    aruco_images = glob.glob(file_path + "/*.png")
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
    parameters = aruco.DetectorParameters()
    
    A_list, B_list = [], []
    
    for image_path in aruco_images:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
  
        if ids is not None:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 120, camera_matrix, dist_coeffs)

            try:
                A_i = get_robot_transformation(image_path)
                B_i = np.eye(4)
                B_i[:3, :3] = cv2.Rodrigues(rvecs[0])[0]
                B_i[:3, 3] = tvecs[0][0]
                A_list.append(A_i)
                B_list.append(B_i)

            except ValueError as e:
                print(e)
                continue
    
    print(f"Number of valid marker detections: {len(A_list)}")

    if not A_list or not B_list:
        raise RuntimeError("Not enough valid marker detections for calibration.")
    
    A_rotations = [A[:3, :3] for A in A_list]
    A_translations = [A[:3, 3] for A in A_list]
    B_rotations = [B[:3, :3] for B in B_list]
    B_translations = [B[:3, 3] for B in B_list]
    
    R_cam2ee, t_cam2ee = cv2.calibrateHandEye(A_rotations, A_translations, B_rotations, B_translations, method=cv2.CALIB_HAND_EYE_TSAI)
    T_cam2ee = np.eye(4)
    T_cam2ee[:3, :3] = R_cam2ee
    T_cam2ee[:3, 3] = t_cam2ee.flatten()
    
    print("\n Rotation matrix:\n", R_cam2ee)
    print("\n Translation vector:\n", t_cam2ee)
    print("\n Transformation matrix:\n", T_cam2ee)
    print("\n")


    extrinsic_data = {
        "rotation_matrix": R_cam2ee.tolist(),
        "translation_vector": t_cam2ee.tolist(),
        "transformation_matrix": T_cam2ee.tolist()
    }
    
    with open(os.path.join(output_path, "camera_extrinsic.json"), "w") as f:
        json.dump(extrinsic_data, f)



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

    iiwa_base = get_iiwa_base_in_world(np.array([center_x, center_y, 0]))

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



    transformation_matrix = np.vstack((np.hstack((np.eye(3), get_iiwa_base_in_world(np.array([center_x, center_y, 0])).reshape(3, 1))), np.zeros((1, 4))))
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

    json_file = "images/GoAround/data.json"
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



        camera_in_world = GetCameraInWorld(pose, position)
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