import requests
import tkinter as tk
import time
from image_processing.basler_camera import BaslerCamera
from image_processing.realsense_camera import RealSenseCamera
from communication.client import send_image_to_server, send_for_pose_estimation
from scipy.spatial.transform import Rotation as R
import json
import numpy as np
import cv2
import os
import winsound
import tkinter.ttk as ttk


# Run me with python -m communication.KMR_communication

# Change this to your KUKA robot's IP address
# ROBOT_IP = "172.31.1.10"
ROBOT_IP = "10.35.129.5"
PORT = 30000

# Initialize the Basler camera
# camera = RealSenseCamera()
if __name__ == "__main__":
    try:
        camera = BaslerCamera()
    except Exception as e:
        print("Failed to initialize the camera:", e)
        camera = None
        print("Camera not initialized!")


def get_rotation_matrix_2D(angle: float = 0.0) -> np.array:
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    return R

def call_endpoint(endpoint, params=None):
    url = f"http://{ROBOT_IP}:{PORT}/{endpoint}"
    if params:
        url += "?" + "&".join([f"{key}={value}" for key, value in params.items()])

    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"Response from {endpoint}:", response.text)
        else:
            print(f"Failed to get a valid response from {endpoint}. Status Code:", response.status_code)
        return response
    except Exception as e:
        print(f"Error calling {endpoint}:", e)
        return None

def move(x, y, theta):
    params = {"x": x, "y": y, "Theta": theta}
    call_endpoint("ArrowsMove", params)

def goto_position(x, y, z, a, b, c):
    params = {"x": x, "y": y, "z": z, "a": a, "b": b, "c": c, "Speed": 0.2, "Motion": "ptp"}
    call_endpoint("GotoPosition", params)

def goto_joint(a1, a2, a3, a4, a5, a6, a7, speed=0.2):
    speed = 0.2
    params = {"A1": a1, "A2": a2, "A3": a3, "A4": a4, "A5": a5, "A6": a6, "A7": a7, "speed": speed}
    response = call_endpoint("GotoJoint", params)
    return response

def CloseGripper(force=1):
    params = {"force": force}
    call_endpoint("CloseGripper", params)

def OpenGripper():
    call_endpoint("OpenGripper")

def InitGripper():
    call_endpoint("InitGripper")

def ReleaseObject():
    call_endpoint("ReleaseObject")

def GetGripperState():
    response = call_endpoint("GetGripperState")
    if response:
        print("Gripper State:", response.text)

def SetLED(color):
    params = {"color": color}
    call_endpoint("SetLED", params)

def get_iiwa_base_in_world(position):
    """ Function for calculating the position of the iiwa base in the world coordinate system.

    Args:
        position (list): The position of the iiwa end effector in the world coordinate system. Must be in mm!!!

    Returns:
        np.array: The position of the iiwa base in the world coordinate system. 
    
    """

    # Define the offsets for the iiwa base position (in mm)
    x_offset = 363
    y_offset = -184
    z_offset = 700

    # Create a rotation matrix for the given angle (position[2])
    rotation_matrix = get_rotation_matrix_2D(position[2])

    # Apply the rotation matrix to the offsets
    offset_vector = np.array([x_offset, y_offset])
    rotated_offset = rotation_matrix @ offset_vector

    # Calculate the final position
    iiwa_base_position = np.array([position[0] + rotated_offset[0], position[1] + rotated_offset[1], z_offset])
    
    return iiwa_base_position


def Get_joints_Write_to_file():
    joints = call_endpoint("GetIIWAJointsPosition")
    position = call_endpoint("GetIIWAposition")
    pose = call_endpoint("GetPose")
    data = {}
    output_folder = "communication/"
    output_file = "HandPoses.json"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if joints and position:
        joints_data = joints.json()
        position_data = position.json()
        pose_data = pose.json()
        print(joints_data)
        print(position_data)
        data["joints"] = {key: joints_data[key] for key in sorted(joints_data.keys())}
        data["position"] = position_data
        data["camera_in_world"] = GetCameraInWorld(pose_data, position_data).tolist()
        try:
            with open(f"{output_folder}{output_file}", "r") as file:
                try:
                    existing_data = json.load(file)
                except json.JSONDecodeError:
                    existing_data = []
        except FileNotFoundError:
            existing_data = []

        existing_data.append(data)

        with open(f"{output_folder}{output_file}", "w") as file:
            json.dump(existing_data, file, indent=4)

def MoveToLocation(TargetNumber):
    params = {"TargetNumber": TargetNumber}
    response = call_endpoint("MoveToLocation", params)
    return response

def GetJointsPosition():
    print("Getting joints position")
    response = call_endpoint("GetIIWAJointsPosition")
    return response.json()

def GetEndPose():
    response = call_endpoint("GetIIWAposition")
    return response.json()

def adjust_joint(joint, delta):
        current_position = GetJointsPosition()
        if current_position:
            new_position = current_position
            new_position[joint] += delta
            goto_joint(new_position["A1"], new_position["A2"], new_position["A3"], new_position["A4"], new_position["A5"], new_position["A6"], new_position["A7"], speed=0.3)

def GoAroundRepeat():
    
    start1 = {
        "A1": -2.967,
        "A2": 1.2384342821018501,
        "A3": 6.669219412797257e-05,
        "A4": 1.9050366590496899,
        "A5": -6.922080533164632e-05,
        "A6": -1.0786968479657064,
        "A7": 2.432792627739246e-05,
        "speed": 0.4
    }

    start2 = {
        "A1": -2.967,
        "A2": 1.0185616252358491,
        "A3": 6.519415621437019e-05,
        "A4": 1.6939632994100207,
        "A5": -6.912498675571182e-05,
        "A6": -1.6807948533628259,
        "A7": 2.432792627739246e-05,
        "speed": 0.4
    }

    starts = [start1, start2]

    for i in range(0, 2):
        Start = starts[i]
        End = Start.copy()
        End["A1"] = -Start["A1"]
        GoAround(Start, End)
    


def GetCameraInWorld(pose_data, position_data):
    with open("image_processing/calibration_data/camera_extrinsic.json", "r") as f:
        extrinsic_data = json.load(f)
    calib_matrix = np.array(extrinsic_data["transformation_matrix"])
    iiwa_base_in_world = get_iiwa_base_in_world([pose_data["x"]*1000, pose_data["y"]*1000, pose_data["theta"]])

    angle = np.pi/2*3 + pose_data["theta"]
    if angle > 2 * np.pi:
        angle -= 2 * np.pi
    elif angle < 0:
        angle += 2 * np.pi
    
    print("Angle", angle)
    rotation2D = get_rotation_matrix_2D(angle)
    print("Rotation2D", rotation2D)
    BaseRotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    BaseRotation[0:2, 0:2] = rotation2D
    print("BaseRotation", BaseRotation)

    iiwa_base_in_world = np.vstack((np.hstack((BaseRotation, iiwa_base_in_world.reshape(3, 1))), np.zeros((1, 4))))
    iiwa_base_in_world[3, 3] = 1

    print("iiwa_base_in_world: \n", iiwa_base_in_world)

    position = np.array([position_data["x"], position_data["y"], position_data["z"]])

    orientation = np.array([position_data["C"], position_data["B"], position_data["A"]])
    
    rotation_matrix = R.from_euler('xyz', orientation, degrees=False).as_matrix() 
    base_to_ee = np.hstack((rotation_matrix, position.reshape(3, 1)))
    base_to_ee = np.vstack((base_to_ee, np.zeros((1, 4))))
    base_to_ee[3, 3] = 1

    camera_in_world = iiwa_base_in_world @ base_to_ee @ calib_matrix

    return camera_in_world

def ObjectInWorld(camera_in_world, object_in_camera):
    """ Function for calculating the position of the object in the world coordinate system."
    """
    object_in_world = camera_in_world @ object_in_camera

    return object_in_world



def GetAndSaveImage(**kwargs):
    output_folder = kwargs.get("output_folder", "images/GoAround/")
    do_detection = kwargs.get("do_detection", False)
    do_6d_estimation = kwargs.get("do_6d_estimation", False)
    detection_item = kwargs.get("detection_item", "mustard bottle")

    time.sleep(2)
    timestamp = int(time.time())
    if camera != None:
        image = camera.capture_image()
        camera.save_image(image, f"{output_folder}image_{timestamp}.png")
        if do_detection:
            print("Sending image to server for detection...")
            image_data = ndarray_to_bytes(image)
            bounding_boxes = send_image_to_server(image_data, detection_item)
            if bounding_boxes:
                print("\n"*3)
                print("OBJECT DETECTED!!!")
                print("\n"*3)
                print("Bounding boxes:", bounding_boxes)
                for (x1, y1, x2, y2) in bounding_boxes:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                camera.save_image(image, f"{output_folder}image_{timestamp}_detected.png")
                if do_6d_estimation:
                    # Call the 6D estimation function here
                    pose_result = send_for_pose_estimation(image_data, bounding_boxes[0], detection_item)["poses"][0]["pose"]
                    if pose_result:
                        print(f"6D Pose: \n{pose_result}")
                        object_in_camera = np.array(pose_result)
                        object_in_camera[:3, 3] = object_in_camera[:3, 3] * 1000  # Convert to mm

                        pose_data = call_endpoint("GetPose")
                        position_data = call_endpoint("GetIIWAposition")
                        pose_data = pose_data.json()
                        position_data = position_data.json()
                        camera_in_world = GetCameraInWorld(pose_data, position_data)
                        object_in_world = ObjectInWorld(camera_in_world, object_in_camera)
                        
                        print("Object in world coordinates: \n", object_in_world)


                    



    pose_response = call_endpoint("GetPose")
    position_response = call_endpoint("GetIIWAposition")
    joints_response = call_endpoint("GetIIWAJointsPosition")
    if joints_response:
        joints_data = joints_response.json()
        sorted_joints = {key: joints_data[key] for key in sorted(joints_data.keys())}
        joints_response = sorted_joints
    if pose_response and position_response:
        pose_data = pose_response.json()
        position_data = position_response.json()

        camera_in_world = GetCameraInWorld(pose_data, position_data)

        data = {
            "image": f"image_{timestamp}.png",
            "pose": pose_data,
            "position": position_data,
            "joints": joints_response,
            "camera_in_world": camera_in_world.tolist()
        }
        try:
            with open(f"{output_folder}data.json", "r") as json_file:
                try:
                    images_data = json.load(json_file)
                except json.JSONDecodeError:
                    images_data = []
        except FileNotFoundError:
            images_data = []

        images_data.append(data)

        with open(f"{output_folder}data.json", "w") as json_file:
            json.dump(images_data, json_file, indent=4)
    time.sleep(1)

def GoAroundPositions(**kwargs):
    output_folder = kwargs.get("output_folder", "images/GoAround/")
    do_detection = kwargs.get("do_detection", False)
    do_6d_estimation = kwargs.get("do_6d_estimation", False)
    detection_item = kwargs.get("detection_item", "foam brick")

    try:
        with open("image_processing/calibration_data/GoAroundHandPoses.json", "r") as file:
            poses_data = json.load(file)
            Poses = [
                {
                    "A1": pose["joints"]["A1"],
                    "A2": pose["joints"]["A2"],
                    "A3": pose["joints"]["A3"],
                    "A4": pose["joints"]["A4"],
                    "A5": pose["joints"]["A5"],
                    "A6": pose["joints"]["A6"],
                    "A7": pose["joints"]["A7"],
                    "speed": 0.3
                }
                for pose in poses_data
            ]
    except FileNotFoundError:
        print("GoAroundHandPoses.json file not found.")
        return
    except json.JSONDecodeError:
        print("Error decoding GoAroundHandPoses.json.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    current_position = GetJointsPosition()
    time.sleep(1)

    if current_position:
        start_distance = sum(abs(Poses[0][joint] - current_position[joint]) for joint in list(Poses[0].keys())[:-1])
        end_distance = sum(abs(Poses[-1][joint] - current_position[joint]) for joint in list(Poses[-1].keys())[:-1])
        print(f"Start distance: {start_distance}")
        print(f"End distance: {end_distance}")
        if end_distance < start_distance:
            Poses = Poses[::-1]

    for i, pose in enumerate(Poses):
        print(f"Going to pose {pose['A1'], pose['A2'], pose['A3'], pose['A4'], pose['A5'], pose['A6'], pose['A7']}")
        response = goto_joint(pose["A1"], pose["A2"], pose["A3"], pose["A4"], pose["A5"], pose["A6"], pose["A7"], pose["speed"])
        time.sleep(0.5)
        if response and response.text.strip() == "OK":
            GetAndSaveImage(output_folder=output_folder, do_detection=do_detection, do_6d_estimation=do_6d_estimation, detection_item=detection_item)

    


def GoAround(Start, End, **kwargs):

    output_folder = kwargs.get("output_folder", "images/GoAround/")
    do_detection = kwargs.get("do_detection", False)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    current_position = GetJointsPosition()
    print(current_position)
    if current_position:
        start_distance  = abs(Start["A1"]-current_position["A1"])
        end_distance = abs(End["A1"]-current_position["A1"])
        print(f"Start distance: {start_distance}")
        print(f"End distance: {end_distance}")
        if end_distance < start_distance:
            Start, End = End, Start


    steps = 9
    for i in range(steps + 1):
        params = {key: Start[key] + (End[key] - Start[key]) * i / steps for key in Start}
        response = goto_joint(params["A1"], params["A2"], params["A3"], params["A4"], params["A5"], params["A6"], params["A7"], params["speed"])
        if response and response.text.strip() == "OK":
            GetAndSaveImage(output_folder=output_folder, do_detection=do_detection)

def ExecuteSequence(**kwargs):
    do_detection = kwargs.get("do_detection", False)
    do_6d_estimation = kwargs.get("do_6d_estimation", False)
    output_folder = kwargs.get("output_folder", "images/GoAround/")
    detection_item = kwargs.get("detection_item", "mustard bottle")
    clean_fodler = kwargs.get("clean_fodler", False)

    if clean_fodler:
        if os.path.exists(output_folder):
            for file in os.listdir(output_folder):
                file_path = os.path.join(output_folder, file)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        os.rmdir(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")

    locations = [1, 2]
    for location in locations:
        response = MoveToLocation(location)
        if response and response.text.strip() == "OK":
            time.sleep(3)
            # GoAroundRepeat()
            GoAroundPositions(output_folder=output_folder, do_detection=do_detection, do_6d_estimation=do_6d_estimation, detection_item=detection_item)


def go_home():
    call_endpoint("GoHome")


def save_calibration_image(out_path: str):
        json_file_path = out_path + "calibration_data.json"
        timestamp = int(time.time())
        if camera is not None:
            image = camera.capture_image()
            image_path = f"{out_path}image_{timestamp}.png"
            camera.save_image(image, image_path)
            joint_positions = GetJointsPosition()
            end_pose = GetEndPose()
            if joint_positions:
                data = {
                    "image": f"image_{timestamp}.png",
                    "joints": joint_positions,
                    "pose": end_pose
                }
                try:
                    with open(json_file_path, "r") as json_file:
                        try:
                            calibration_data = json.load(json_file)
                        except json.JSONDecodeError:
                            calibration_data = []
                except FileNotFoundError:
                    calibration_data = []

                calibration_data.append(data)

                with open(json_file_path, "w") as json_file:
                    json.dump(calibration_data, json_file, indent=4)
            print(f"Saved calibration image and joint positions to {image_path}")


def move_to_hand_poses():
        for i in range(1, 6):
            print(f"Moving to hand poses for ScanAround_{i}...")

            path = f"images/ScanAround_{i}/"
            input_file = "image_processing\calibration_data\GoAroundHandPoses.json"
            try:
                with open(input_file, "r") as json_file:
                    hand_poses = json.load(json_file)
            except FileNotFoundError:
                print("HandPoses.json file not found.")
                return
            except json.JSONDecodeError:
                print("Error decoding HandPoses.json.")
                return

            for pose in hand_poses:
                joints = pose["joints"]
                # radians_joints = {key: np.deg2rad(value) for key, value in joints.items()}
                radians_joints = joints
                response = goto_joint(radians_joints["A1"], radians_joints["A2"], radians_joints["A3"], radians_joints["A4"], radians_joints["A5"], radians_joints["A6"], radians_joints["A7"], speed=0.5)
                if response and response.text.strip() == "OK":
                    time.sleep(1.5)
                    save_calibration_image(path)
                    time.sleep(0.5)

            print("DONE!!!")
            time.sleep(30)
            winsound.Beep(200, 500)  # Frequency: 1000 Hz, Duration: 500 ms
            print("Starting in 30 seconds...")
            time.sleep(15)
            winsound.Beep(200, 500)
            print("Starting in 15 seconds...")
            time.sleep(10)
            winsound.Beep(200, 500)
            print("Starting in 5 seconds...")
            time.sleep(5)
            winsound.Beep(200, 750)

            print("Starting now!")
            

def create_gui():
    root = tk.Tk()
    root.title("KUKA KMR IIWA Controller")
    root.geometry("600x700")

    notebook = ttk.Notebook(root)  # Create a tabbed interface
    notebook.pack(expand=True, fill="both")

    # Movement Page
    movement_frame = tk.Frame(notebook)
    notebook.add(movement_frame, text="Movement")

    control_frame = tk.Frame(movement_frame)
    control_frame.pack(pady=10)

    tk.Button(control_frame, text="↑", command=lambda: move(0.1, 0, 0)).grid(row=0, column=1)
    tk.Button(control_frame, text="←", command=lambda: move(0, 0.1, 0)).grid(row=1, column=0)
    tk.Button(control_frame, text="→", command=lambda: move(0, -0.1, 0)).grid(row=1, column=2)
    tk.Button(control_frame, text="↓", command=lambda: move(-0.1, 0, 0)).grid(row=2, column=1)
    tk.Button(control_frame, text="↺", command=lambda: move(0, 0, 0.1)).grid(row=1, column=3)
    tk.Button(control_frame, text="↻", command=lambda: move(0, 0, -0.1)).grid(row=1, column=4)

    # Actions Page
    action_frame = tk.Frame(notebook)
    notebook.add(action_frame, text="Actions")

    tk.Button(action_frame, text="Capture Image", command=lambda: call_endpoint("CaptureImage")).pack(pady=5)
    tk.Button(action_frame, text="Get Pose", command=lambda: call_endpoint("GetPose")).pack(pady=5)
    tk.Button(action_frame, text="Honk On", command=lambda: call_endpoint("HonkOn")).pack(pady=5)
    tk.Button(action_frame, text="Honk Off", command=lambda: call_endpoint("HonkOff")).pack(pady=5)

    # --- Main Frame for "IIWA Control" ---
    goto_frame = tk.Frame(notebook)
    notebook.add(goto_frame, text="IIWA control")

    # === Go To Position Section ===
    labels = ["X", "Y", "Z", "A", "B", "C"]
    entries = []
    for i, label in enumerate(labels):
        tk.Label(goto_frame, text=label+":").grid(row=0, column=i*2)
        entry = tk.Entry(goto_frame, width=5)
        entry.grid(row=0, column=i*2+1)
        entries.append(entry)

    degree_mode_position = tk.BooleanVar(value=True)
    tk.Checkbutton(goto_frame, text="Degrees", variable=degree_mode_position).grid(row=0, column=13)

    def convert_abc_to_radians_if_needed(a, b, c):
        return (np.deg2rad(a), np.deg2rad(b), np.deg2rad(c)) if degree_mode_position.get() else (a, b, c)

    tk.Button(goto_frame, text="Go To Position", command=lambda: goto_position(
        float(entries[0].get()), float(entries[1].get()), float(entries[2].get()),
        *convert_abc_to_radians_if_needed(
            float(entries[3].get()), float(entries[4].get()), float(entries[5].get())
        )
    )).grid(row=0, column=12, pady=5)

    # === Go To Joint Position Section ===
    joint_frame = tk.Frame(goto_frame)
    joint_frame.grid(row=1, column=0, columnspan=14, pady=10)

    joint_labels = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
    joint_entries = []
    for i, label in enumerate(joint_labels):
        tk.Label(joint_frame, text=label+":").grid(row=0, column=i*2)
        entry = tk.Entry(joint_frame, width=5)
        entry.grid(row=0, column=i*2+1)
        joint_entries.append(entry)

    degree_mode = tk.BooleanVar(value=True)
    tk.Checkbutton(joint_frame, text="Degrees", variable=degree_mode).grid(row=1, column=14)

    def convert_to_radians_if_needed(value):
        return np.deg2rad(value) if degree_mode.get() else value

    tk.Button(joint_frame, text="Go To Joint Position", command=lambda: goto_joint(
        *[convert_to_radians_if_needed(float(entry.get())) for entry in joint_entries]
    )).grid(row=0, column=14, pady=5)

    # === Joint Control Section ===
    joint_control_frame = tk.Frame(goto_frame)
    joint_control_frame.grid(row=2, column=0, columnspan=14, pady=10)

    tk.Label(joint_control_frame, text="Joint Control").grid(row=0, column=0, columnspan=14, pady=5)

    for i, label in enumerate(joint_labels):
        tk.Label(joint_control_frame, text=label+":").grid(row=i+1, column=0, padx=5, pady=5)
        tk.Button(joint_control_frame, text="+", command=lambda l=label: adjust_joint(l, 1*np.pi/180)).grid(row=i+1, column=1, padx=5, pady=5)
        tk.Button(joint_control_frame, text="-", command=lambda l=label: adjust_joint(l, -1*np.pi/180)).grid(row=i+1, column=2, padx=5, pady=5)


    # Other Actions Page
    other_frame = tk.Frame(notebook)
    notebook.add(other_frame, text="Other")

    tk.Button(other_frame, text="Go Around", command=lambda: GoAroundRepeat()).pack(pady=5)
    tk.Button(other_frame, text="Go Around Positions", command=lambda: GoAroundPositions()).pack(pady=5)
    tk.Button(other_frame, text="Execute sequence", command=lambda: ExecuteSequence()).pack(pady=5)
    tk.Button(other_frame, text="Move to Hand Poses", command=move_to_hand_poses).pack(pady=5)
    tk.Button(other_frame, text="Get Joints Write to File", command=Get_joints_Write_to_file).pack(pady=5)



    # Gripper Control Page
    gripper_frame = tk.Frame(notebook)
    notebook.add(gripper_frame, text="Gripper Control")

    tk.Button(gripper_frame, text="Close Gripper", command=lambda: CloseGripper(force=1)).pack(pady=5)
    tk.Button(gripper_frame, text="Open Gripper", command=OpenGripper).pack(pady=5)
    tk.Button(gripper_frame, text="Initialize Gripper", command=InitGripper).pack(pady=5)
    tk.Button(gripper_frame, text="Release Object", command=ReleaseObject).pack(pady=5)
    tk.Button(gripper_frame, text="Get Gripper State", command=GetGripperState).pack(pady=5)

    led_frame = tk.Frame(gripper_frame)
    led_frame.pack(pady=5)
    tk.Label(led_frame, text="Set LED Color:").pack(pady=5)
    tk.Button(led_frame, text="Red", command=lambda: SetLED("red")).pack(side=tk.LEFT, padx=5)
    tk.Button(led_frame, text="Green", command=lambda: SetLED("green")).pack(side=tk.LEFT, padx=5)
    tk.Button(led_frame, text="Blue", command=lambda: SetLED("blue")).pack(side=tk.LEFT, padx=5)


    root.mainloop()


    
def ndarray_to_bytes(image: np.ndarray, format: str = ".png") -> bytes:
    _, buffer = cv2.imencode(format, image)
    return buffer.tobytes()


def JustPickIt():
    # Load pose data from the specified JSON file
    pose_file_path = "images/JustPickIt/pose.json"
    try:
        with open(pose_file_path, "r") as file:
            pose_data = json.load(file)
            object_in_camera = np.array(pose_data["camera_pose"])
            print("Loaded object pose from file:", object_in_camera)
    except FileNotFoundError:
        print(f"File not found: {pose_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {pose_file_path}")
        return
    
    pose_response = call_endpoint("GetPose")
    position_response = call_endpoint("GetIIWAposition")
    pose_data = pose_response.json()
    position_data = position_response.json()

    object_in_camera[:3, 3] = object_in_camera[:3, 3] * 1000
    camera_in_word = GetCameraInWorld(pose_data, position_data)

    object_in_world = ObjectInWorld(camera_in_word, object_in_camera)
    
    print("Object in camera:\n", object_in_camera)
    print("Camera in world:\n", camera_in_word)
    print("Object in world:\n", object_in_world)

    # Save object_in_world and camera_pose back to the JSON file
    save_data = {
        "camera_pose": object_in_camera.tolist(),
        "object_in_world": object_in_world.tolist()
    }

    output_folder = "images/JustPickIt/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file_path = os.path.join(output_folder, "pose.json")
    with open(output_file_path, "w") as file:
        json.dump(save_data, file, indent=4)

    print(f"Saved updated pose data to {output_file_path}")

def JustPickIt2():
    # Load pose data from the specified JSON file
    pose_file_path = "images/JustPickIt/pose.json"
    try:
        with open(pose_file_path, "r") as file:
            pose_data = json.load(file)
            object_in_camera = np.array(pose_data["camera_pose"])
            print("Loaded object pose from file:", object_in_camera)
    except FileNotFoundError:
        print(f"File not found: {pose_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {pose_file_path}")
        return

    object_in_world = np.array(pose_data["object_in_world"])
    joints = call_endpoint("GetIIWAJointsPosition")
    joints = joints.json()
    pose_response = call_endpoint("GetPose")
    position_response = call_endpoint("GetIIWAposition")
    pose_data = pose_response.json()
    position_data = position_response.json()
    camera_in_word = GetCameraInWorld(pose_data, position_data)

    iiwa_base_in_world = get_iiwa_base_in_world([pose_data["x"]*1000, pose_data["y"]*1000, pose_data["theta"]])
    object_iiwa_angle = np.arctan2(object_in_world[1, 3] - iiwa_base_in_world[1], object_in_world[0, 3] - iiwa_base_in_world[0])
    camera_iiwa_angle = np.arctan2(camera_in_word[1, 3] - iiwa_base_in_world[1], camera_in_word[0, 3] - iiwa_base_in_world[0])
    joint_A1_KMR_angle = joints["A1"] + pose_data["theta"]
    A1_in_world = np.array([np.cos(joint_A1_KMR_angle), np.sin(joint_A1_KMR_angle)])
    
    print("Object IIWA angle in world:", object_iiwa_angle)
    print("Camera IIWA angle in world:", camera_iiwa_angle)
    print("Joint A1 angle:", joints["A1"])


    angle = np.pi/2 + pose_data["theta"] + joints["A1"]
    
    if angle > 2 * np.pi:
        angle -= 2 * np.pi
    elif angle > np.pi:
        angle -= 2 * np.pi



    print("A1 world angle:", angle)
    print("Needed rotation:", object_iiwa_angle - angle)
    print(joints["A1"] + (object_iiwa_angle - angle))
    final_angle = object_iiwa_angle - pose_data["theta"] - np.pi/2
    while final_angle > np.pi:
        final_angle -= 2 * np.pi
    while final_angle < -np.pi:
        final_angle += 2 * np.pi

    if final_angle > 170*np.pi/180:
        final_angle = 169*np.pi/180
    elif final_angle < -170*np.pi/180:
        final_angle = -169*np.pi/180



    # Calculate the new joint positions with the updated A1 angle
    new_A1_angle = final_angle
    goto_joint(new_A1_angle, joints["A2"], joints["A3"], joints["A4"], joints["A5"], joints["A6"], joints["A7"], speed=0.3)



if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    # while True:
    #     JustPickIt2()
    #     time.sleep(5)


    # JustPickIt()
    # create_gui()

    ExecuteSequence(do_detection=True, do_6d_estimation=True, output_folder="images/GoAround/", detection_item="plug-in outlet expander", clean_fodler=True)

