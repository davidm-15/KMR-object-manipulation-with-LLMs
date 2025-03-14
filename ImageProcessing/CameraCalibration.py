import numpy as np
import cv2
import glob
import yaml
import json
import os
import cv2.aruco as aruco
from ImageProcessing.basler_camera import BaslerCamera
from scipy.spatial.transform import Rotation as R

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
        json.dump({"K": mtx.tolist(), "resolution": [gray.shape[0], gray.shape[1]]}, json_file)

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
            A[:3, :3] = cv2.Rodrigues(np.array([pose["A"], pose["B"], pose["C"]]))[0]
            A[:3, 3] = np.array([pose["x"], pose["y"], pose["z"]])
            return A
    
    raise ValueError(f"No transformation found for image {image_path}")

def extrinsic_calibration(file_path):
    with open("ImageProcessing/images/BaslerCalibration/camera_data.json", "r") as json_file:
        camera_data = json.load(json_file)
        camera_matrix = np.array(camera_data["K"])
    
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
            print(f"{tvecs=}")

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



                print("B Translation Vector:", B_i[:3, 3])
                print("A Translation Vector:", A_i[:3, 3])

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
        cv2.CALIB_HAND_EYE_TSAI,
        cv2.CALIB_HAND_EYE_PARK,
        cv2.CALIB_HAND_EYE_HORAUD,
        cv2.CALIB_HAND_EYE_ANDREFF,
        cv2.CALIB_HAND_EYE_DANIILIDIS
    ]
    
    method_names = [
        "CALIB_HAND_EYE_TSAI",
        "CALIB_HAND_EYE_PARK",
        "CALIB_HAND_EYE_HORAUD",
        "CALIB_HAND_EYE_ANDREFF",
        "CALIB_HAND_EYE_DANIILIDIS"
    ]
    
    for method, name in zip(methods, method_names):
        R_cam2ee, t_cam2ee = cv2.calibrateHandEye(A_rotations, A_translations, B_rotations, B_translations, method=method)
        T_cam2ee = np.eye(4)
        T_cam2ee[:3, :3] = R_cam2ee
        T_cam2ee[:3, 3] = t_cam2ee.flatten()
        
        print(f"Hand-eye calibration result using {name}:")
        print("Rotation matrix:\n", R_cam2ee)
        print("Translation vector:\n", t_cam2ee)
        print("Transformation matrix:\n", T_cam2ee)
        print("\n")


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
    R_mat = R.from_euler('xyz', rpy, degrees=True).as_matrix()
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

if __name__ == "__main__":
    # capture_images("ImageProcessing/images/BaslerCalibration")
    # intrinsic_calibration("ImageProcessing/images/BaslerCalibration")
    extrinsic_calibration("communication/images/CalibrationImages")
    extrinsic_calibration("communication/images/CalibrationImages/Ground")

    # TEST()



    # # Load transformation from End-Effector to Camera
    # T_ee_to_cam = load_transformation_matrix("communication/images/CalibrationImages/extrinsic.json")
    
    # # Desired Camera Pose in World Frame
    # camera_pose = np.array([0, 0, 690, 105, -49, -177])  # [x, y, z, roll, pitch, yaw]
    
    # # Compute End-Effector Pose
    # ee_pose = compute_ee_pose(camera_pose, T_ee_to_cam)
    
    # print("Computed End-Effector Pose (XYZ + RPY):", ee_pose)
    # # Convert angles to radians
    # ee_pose_radians = ee_pose.copy()
    # ee_pose_radians[3:] = np.radians(ee_pose[3:])

    # np.set_printoptions(suppress=True)
    # print("Computed End-Effector Pose (XYZ + RPY in radians):", ee_pose_radians)




    # world_coordinates = np.array([0, 0, 0, 0, 0, 0])
    # robot_base = np.array([29.9818, 15.5825, 0, -1.351, 0, 0])

    # z = 700
    # y = 363
    # x = 184

