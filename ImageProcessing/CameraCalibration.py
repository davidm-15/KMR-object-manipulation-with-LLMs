import numpy as np
import cv2
import glob
import yaml
import json
import os
import cv2.aruco as aruco
from ImageProcessing.basler_camera import BaslerCamera
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from communication.KMR_communication import get_iiwa_base_in_world



def get_rotation_matrix_x(angle: float = 0.0) -> np.array:
        Rx = np.array([[1, 0, 0, 0],
                       [0, np.cos(angle), -np.sin(angle), 0],
                       [0, np.sin(angle), np.cos(angle), 0],
                       [0, 0, 0, 1]])

        return Rx


def get_rotation_matrix_y(angle: float = 0.0) -> np.array:
    Ry = np.array([[np.cos(angle), 0, np.sin(angle), 0],
                    [0, 1, 0, 0],
                    [-np.sin(angle), 0, np.cos(angle), 0],
                    [0, 0, 0, 1]])

    return Ry


def get_rotation_matrix_z(angle: float = 0.0) -> np.array:
    Rz = np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                    [np.sin(angle), np.cos(angle), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    return Rz




def intrinsic_calibration(images_path):
    images = glob.glob(os.path.join(images_path, "*.png"))
    
    xssize, yssize = 5, 8
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((xssize * yssize, 3), np.float32)
    objp[:, :2] = np.mgrid[0:xssize, 0:yssize].T.reshape(-1, 2)
    
    objpoints, imgpoints = [], []
    found = 0
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (xssize, yssize), None)
        
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            img = cv2.drawChessboardCorners(img, (xssize, yssize), corners2, ret)
            found += 1
            cv2.imshow('img', cv2.resize(img, (640, 480)))
            output_path = os.path.join(images_path, "WithLines", os.path.basename(fname))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, img)
            cv2.waitKey(100)
    
    print(f"Number of images used for calibration: {found}")
    cv2.destroyAllWindows()
    
    if found == 0:
        raise RuntimeError("No chessboard patterns detected. Check images or pattern size.")
    
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    calibration_data = {'camera_matrix': mtx.tolist(), 'dist_coeff': dist.tolist()}
    
    with open(os.path.join(images_path, "calibration_matrix.yaml"), "w") as f:
        yaml.dump(calibration_data, f)
    
    with open(os.path.join(images_path, "camera_data.json"), "w") as json_file:
        json.dump({"K": mtx.tolist(), "resolution": [gray.shape[0], gray.shape[1]], "dist_coeff": dist.tolist()}, json_file)

def capture_images(output_folder):
    camera = BaslerCamera()
    camera.stream_and_capture(output_folder)
    camera.close()

def get_robot_transformation(image_path):
    folder_path = os.path.dirname(image_path)
    with open(os.path.join(folder_path, "calibration_data.json"), "r") as json_file:
        calibration_data = json.load(json_file)
    
    for data in calibration_data:
        if data["image"] == os.path.basename(image_path):
            pose = data["pose"]
            A = np.eye(4)
            # A[:3, :3] = cv2.Rodrigues(np.array([pose["A"], pose["B"], pose["C"]]))[0]

            xrot = get_rotation_matrix_x(pose["C"])
            yrot = get_rotation_matrix_y(pose["B"])
            zrot = get_rotation_matrix_z(pose["A"])

            brainrot = zrot @ yrot @ xrot

            A[:3, :3] = brainrot[:3, :3]

            

            A[:3, 3] = np.array([pose["x"], pose["y"], pose["z"]])
            return A
    
    raise ValueError(f"No transformation found for image {image_path}")

def extrinsic_calibration(file_path):
    with open("ImageProcessing/images/BaslerCalibration/camera_data.json", "r") as json_file:
        camera_data = json.load(json_file)
        camera_matrix = np.array(camera_data["K"])
        np.set_printoptions(suppress=True)
        # print(f"Camera matrix:\n{camera_matrix}")
    
    dist_coeffs = np.zeros((5, 1))  # Adjust if needed
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
            # print(f"{tvecs=}")

            # aruco.drawDetectedMarkers(image, corners, ids)
            # for i in range(len(ids)):
            #     cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 3)
            # cv2.imshow('Aruco Detection', cv2.resize(image, (640, 480)))
            # cv2.waitKey(1000)

            # print(f"{rvecs=}")
            # print(f"{tvecs=}")
            try:
                A_i = get_robot_transformation(image_path)
                B_i = np.eye(4)
                B_i[:3, :3] = cv2.Rodrigues(rvecs[0])[0]
                B_i[:3, 3] = tvecs[0][0]
                A_list.append(A_i)
                B_list.append(B_i)

                # print(f"{A_i=}")
                # print(f"{B_i=}")


                # print("B Translation Vector:", B_i[:3, 3])
                # print("A Translation Vector:", A_i[:3, 3])

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
    
    methods = [
        cv2.CALIB_HAND_EYE_HORAUD,
        cv2.CALIB_HAND_EYE_ANDREFF,
        cv2.CALIB_HAND_EYE_DANIILIDIS,
        cv2.CALIB_HAND_EYE_TSAI,
        cv2.CALIB_HAND_EYE_PARK
    ]
    
    method_names = [
        "CALIB_HAND_EYE_TSAI",
        "CALIB_HAND_EYE_HORAUD",
        "CALIB_HAND_EYE_ANDREFF",
        "CALIB_HAND_EYE_DANIILIDIS",
        "CALIB_HAND_EYE_PARK"
    ]
    
    for method, name in zip(methods, method_names):

        debug_info = {
            "grip2base_all_R": [mat.tolist() for mat in A_rotations],
            "grip2base_all_t": [vec.tolist() for vec in A_translations],
            "tar2cam_all_R": [mat.tolist() for mat in B_rotations],
            "tar2cam_all_t": [vec.tolist() for vec in B_translations]
        }

        with open("debug_hand_eye_calibration.json", "w") as debug_file:
            json.dump(debug_info, debug_file, indent=4)
        R_cam2ee, t_cam2ee = cv2.calibrateHandEye(A_rotations, A_translations, B_rotations, B_translations, method=3)
        T_cam2ee = np.eye(4)
        T_cam2ee[:3, :3] = R_cam2ee
        T_cam2ee[:3, 3] = t_cam2ee.flatten()
        
        print(f"Hand-eye calibration result using {name}:")
        print("Rotation matrix:\n", R_cam2ee)
        print("Translation vector:\n", t_cam2ee)
        print("Transformation matrix:\n", T_cam2ee)
        print("\n")
        break


    extrinsic_data = {
        "rotation_matrix": R_cam2ee.tolist(),
        "translation_vector": t_cam2ee.tolist(),
        "transformation_matrix": T_cam2ee.tolist()
    }
    
    with open(os.path.join(file_path, "extrinsic.json"), "w") as f:
        json.dump(extrinsic_data, f)




def pose_to_matrix(position, rpy):
    """
    Converts a 6D pose (XYZ position + roll-pitch-yaw angles) into a 4x4 transformation matrix.
    """
    R_mat = R.from_euler('xyz', rpy, degrees=False).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = position
    return T

def matrix_to_pose(T):
    """
    Converts a 4x4 transformation matrix into a 6D pose (XYZ position + roll-pitch-yaw angles).
    """
    position = T[:3, 3]
    rpy = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
    return np.hstack((position, rpy))

def compute_ee_pose(camera_pose, T_ee_to_cam):
    """
    Computes the 6D pose of the end-effector given the desired 6D pose of the camera
    and the transformation matrix from EE to camera.
    """
    T_world_to_cam = pose_to_matrix(camera_pose[:3], camera_pose[3:])

    # T_ee_to_cam[0:3, 0:3] = np.eye(3)

    T_cam_to_ee = np.linalg.inv(T_ee_to_cam)
    T_world_to_ee = T_world_to_cam @ T_cam_to_ee
    return matrix_to_pose(T_world_to_ee)

def load_transformation_matrix(filepath):
    """
    Loads a transformation matrix from a JSON file.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return np.array(data["transformation_matrix"])


def TEST():
    image_path = "C:/Users/siram/OneDrive/Plocha/ARUCO.png"
    camera_data_path = "ImageProcessing/images/BaslerCalibration/camera_data.json"

    # Load camera data
    with open(camera_data_path, "r") as json_file:
        camera_data = json.load(json_file)
        camera_matrix = np.array(camera_data["K"])

    dist_coeffs = np.zeros((5, 1))  # Adjust if needed
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
    parameters = aruco.DetectorParameters()

    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect Aruco markers
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 12, camera_matrix, dist_coeffs)
        print(f"Translation vector: {tvecs[0][0]}")
        print(f"Rotation vector: {rvecs[0][0]}")
    else:
        print("No Aruco markers detected.")






def convert():
    # Get all images in sorted order
    image_files = sorted(glob.glob("ImageProcessing/images/BaslerCalibration/*.png"))

    # Read the first image to get dimensions
    frame = cv2.imread(image_files[0])
    h, w, layers = frame.shape

    # Define the AVI video writer
    out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), 10, (w, h))

    # Add frames to the video
    for img in image_files:
        frame = cv2.imread(img)
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()


def create_individual_jsons(input_json_path, output_folder):
    """
    Creates individual JSON files for each image's data from a given JSON file.
    """
    with open(input_json_path, "r") as f:
        calibration_data = json.load(f)

    os.makedirs(output_folder, exist_ok=True)

    for entry in calibration_data:
        image_name = entry["image"]
        output_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}.json")
        entry = entry["pose"]
        entry["X"] = entry.pop("C")
        entry["Y"] = entry.pop("B")
        entry["Z"] = entry.pop("A")
        with open(output_path, "w") as out_file:
            json.dump(entry, out_file, indent=4)

    print(f"Individual JSON files created in {output_folder}")




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


if __name__ == "__main__":

    # create_individual_jsons("communication\images\HandPosesImages\calibration_data.json", "ImageProcessing\\images\\calibb\\calibration_data_marker\\robot_poses")


    # capture_images("ImageProcessing/images/OneAxis")
    # intrinsic_calibration("ImageProcessing/images/BaslerCalibration")
    # intrinsic_calibration("ImageProcessing/images/BaslerCalibration/images1")


    # plotit()
    # extrinsic_calibration("communication/images/CalibrationImages")
    # extrinsic_calibration("communication/images/CalibrationImages/KMR_1")
    # extrinsic_calibration("communication/images/CalibrationImages/Ground")

    # TEST()
    # convert()


    # calib_matrix = np.array([[-0.00279883, -0.99995403,  0.00917078, 96.90320424],
    #                         [ 0.99959602, -0.00305698, -0.02825706, -4.95372373],
    #                         [ 0.02828379,  0.00908799,  0.99955862, 63.4821385 ],
    #                         [ 0.0,         0.0,         0.0,         1.0]])
    
    
    # pos = np.array([0, 0, 808])
    # rot = np.array([2.949606435, -0.886801792, -3.118379774])
    # T = pose_to_matrix(pos, rot)

    # # Transform the position using the calibration matrix
    # transformed_position = calib_matrix @ T @ np.array([0, 0, 0, 1])

    # print("Transformed Position:", transformed_position[:3])
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

    def visualize_box():
        # Box dimensions in mm
        height = 700
        length = 1080
        width = 630

        # Center position in 2D (x, y) and z position for the bottom face
        center_x = 14000
        center_y = 15000
        min_z = 0

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
        # Plot point O at (0, 0, 0)
        # ax.scatter(0, 0, 0, color='r', s=100, label='Point O (0, 0, 0)')
        # ax.legend()



        # Define the position and orientation
        position = np.array([-0.06084885817435056, 8.390708723339504, 806.0509324443722])
        orientation = np.array([3.1308282468610966, -0.8097527279037569, 3.049039185227726])  # Roll, Pitch, Yaw
        position = position + iiwa_base
            
        # Compute the rotation matrix
        rotation_matrix = R.from_euler('xyz', orientation, degrees=False).as_matrix()

        # Define the arrow directions (unit vectors)
        arrows = np.eye(3) * 100  # Scale by 100
        arrows = np.vstack((arrows, np.zeros(3)))  # Add row of ones for translation
        arrows = np.hstack((arrows, np.zeros((4, 1))))  # Add column of zeros for translation




        # Rotate the arrows
        rotated_arrows = arrows @ rotation_matrix

        # Plot the arrows
        colors = ['r', 'g', 'b']
        for i in range(3):
            ax.quiver(
            position[0], position[1], position[2],  # Starting point
            rotated_arrows[0, i], rotated_arrows[1, i], rotated_arrows[2, i],  # Direction
            color=colors[i], label=f'Axis {colors[i].upper()}'
            )


        # Plot the iiwa base
        ax.scatter(iiwa_base[0], iiwa_base[1], iiwa_base[2], color='g', s=100, label='iiwa Base')
        ax.legend()

        transform = np.array([[-0.00279883, -0.99995403,  0.00917078, 96.90320424],
                    [ 0.99959602, -0.00305698, -0.02825706, -4.95372373],
                    [ 0.02828379,  0.00908799,  0.99955862, 63.4821385 ],
                    [ 0.0,         0.0,         0.0,         1.0]])

        # Transform the arrows into the new coordinate frame
        rotated_arrows_transformed =  rotated_arrows @ transform[:3, :3]

        for i in range(3):
            ax.quiver(
            position[0] + transform[0, 3], position[1] + transform[1, 3], position[2] + transform[2, 3],  # Starting point
            rotated_arrows_transformed[0, i], rotated_arrows_transformed[1, i], rotated_arrows_transformed[2, i],  # Direction
            color=colors[i], linestyle='dashed', label=f'Transformed Axis {colors[i].upper()}'
            )
        


        # Set labels and limits
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        # ax.set_xlim([offset_x - 100, offset_x + length + 100])
        # ax.set_ylim([offset_y - 100, offset_y + width + 100])
        # ax.set_zlim([min_z, min_z + height + 100])
        ax.set_title("3D Visualization of a Box")
        set_axes_equal(ax)

        plt.show()
        

    visualize_box()


    pass