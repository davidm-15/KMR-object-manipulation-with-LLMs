import requests
import tkinter as tk
import time
from image_processing.basler_camera import BaslerCamera
from image_processing.realsense_camera import RealSenseCamera
from communication.client import send_image_to_server
import json
import numpy as np
import cv2
import os


# Change this to your KUKA robot's IP address
# ROBOT_IP = "172.31.1.10"
ROBOT_IP = "10.35.129.5"
PORT = 30000

# Initialize the Basler camera
# camera = RealSenseCamera()
# camera = BaslerCamera()
camera = None


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
    params = {"x": x, "y": y, "z": z, "a": a, "b": b, "c": c, "Speed": 1, "Motion": "ptp"}
    call_endpoint("GotoPosition", params)

def goto_joint(a1, a2, a3, a4, a5, a6, a7, speed=0.2):
    speed = 0.2
    params = {"A1": a1, "A2": a2, "A3": a3, "A4": a4, "A5": a5, "A6": a6, "A7": a7, "speed": speed}
    response = call_endpoint("GotoJoint", params)
    return response

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
    pose = call_endpoint("GetIIWAposition")
    data = {}
    output_folder = "communication/images/OneAxis/"
    output_file = "HandPoses.json"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if joints and pose:
        joints_data = joints.json()
        pose_data = pose.json()
        data["joints"] = joints_data
        data["pose"] = pose_data
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
            goto_joint(new_position["A0"], new_position["A1"], new_position["A2"], new_position["A3"], new_position["A4"], new_position["A5"], new_position["A6"], speed=0.3)

def GoAroundRepeat():

    start3 = {"A1": -87.31, "A2": -73.03, "A3": 1.53, "A4": 111.96, "A5": -4.88, "A6": -75.58, "A7": -1.92, "speed": 0.2}

    start3 = {"A1": -2.967, "A2": -1.274, "A3": 0.026, "A4": 1.954, "A5": -0.085, "A6": -1.319, "A7": -0.033, "speed": 0.2}
    
    start1 = {
        "A1": -2.967,
        "A2": -0.106,
        "A3": -0.068,
        "A4": 1.551,
        "A5": 1.522,
        "A6": 0.001,
        "A7": -0.813,
        "speed": 0.2
    }

    start2 = {
        "A1": -2.967,
        "A2": -0.106,
        "A3": -0.068,
        "A4": 2.015,
        "A5": 1.522,
        "A6": 0.001,
        "A7": -0.813,
        "speed": 0.2
    }

    starts = [start1, start2, start3]

    for i in range(2, 3):
        Start = starts[i]
        End = Start.copy()
        End["A1"] = -Start["A1"]
        GoAround(Start, End)
    



def GoAround(Start, End):
    # Start = {"A1": -2.9, "A2": -0.106, "A3": -0.068, "A4": 2.094, "A5": 1.643, "A6": 1.381, "A7": 0.150, "speed": 0.8}
    # End = {"A1": 2.9, "A2": -0.106, "A3": -0.068, "A4": 2.094, "A5": 1.643, "A6": 1.381, "A7": 0.150, "speed": 0.8}


    current_position = GetJointsPosition()
    print(current_position)
    if current_position:
        start_distance  = abs(Start["A1"]-current_position["A0"])
        end_distance = abs(End["A1"]-current_position["A0"])
        print(f"Start distance: {start_distance}")
        print(f"End distance: {end_distance}")
        if end_distance < start_distance:
            Start, End = End, Start


    steps = 9
    for i in range(steps + 1):
        params = {key: Start[key] + (End[key] - Start[key]) * i / steps for key in Start}
        response = goto_joint(params["A1"], params["A2"], params["A3"], params["A4"], params["A5"], params["A6"], params["A7"], params["speed"])
        if response and response.text.strip() == "OK":
            time.sleep(1)
            timestamp = int(time.time())
            if camera != None:
                image = camera.capture_image()
                camera.save_image(image, f"communication/images/BaslerImages/image_{timestamp}.png")
                image_data = ndarray_to_bytes(image)
                bounding_boxes = send_image_to_server(image_data, "coca cola can")
                if bounding_boxes:
                    print("\n"*3)
                    print("OBJECT DETECTED!!!")
                    print("\n"*3)
                    print("Bounding boxes:", bounding_boxes)
                    for (x1, y1, x2, y2) in bounding_boxes:
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    camera.save_image(image, f"communication/images/BaslerImages/image_bbox_{timestamp}.png")




            pose_response = call_endpoint("GetPose")
            position_response = call_endpoint("GetIIWAposition")
            if pose_response and position_response:
                pose_data = {}
                try:
                    pose_text = pose_response.text.strip()
                    pose_parts = pose_text.split()
                    for part in pose_parts:
                        key, value = part.split(":")
                        pose_data[key] = float(value)
                except Exception as e:
                    print("Failed to parse pose data:", e)
                position_data = position_response.json()
                data = {
                    "image": f"image_{timestamp}.png",
                    "pose": pose_data,
                    "position": position_data
                }
                try:
                    with open("communication/images/BaslerImages/images_data.json", "r") as json_file:
                        try:
                            images_data = json.load(json_file)
                        except json.JSONDecodeError:
                            images_data = []
                except FileNotFoundError:
                    images_data = []

                images_data.append(data)

                with open("communication/images/BaslerImages/images_data.json", "w") as json_file:
                    json.dump(images_data, json_file, indent=4)
            time.sleep(1)

def ExecuteSequence():
    locations = [79, 80, 81, 82]
    for location in locations:
        response = MoveToLocation(location)
        if response and response.text.strip() == "OK":
            time.sleep(3)
            GoAroundRepeat()


def go_home():
    call_endpoint("GoHome")


def save_calibration_image():
        timestamp = int(time.time())
        if camera is not None:
            image = camera.capture_image()
            image_path = f"communication/images/CalibrationImages/image_{timestamp}.png"
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
                    with open("communication/images/CalibrationImages/calibration_data.json", "r") as json_file:
                        try:
                            calibration_data = json.load(json_file)
                        except json.JSONDecodeError:
                            calibration_data = []
                except FileNotFoundError:
                    calibration_data = []

                calibration_data.append(data)

                with open("communication/images/CalibrationImages/calibration_data.json", "w") as json_file:
                    json.dump(calibration_data, json_file, indent=4)
            print(f"Saved calibration image and joint positions to {image_path}")


def move_to_hand_poses():
        try:
            with open("communication/images/CalibrationImages/HandPoses2.json", "r") as json_file:
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
            response = goto_joint(radians_joints["A0"], radians_joints["A1"], radians_joints["A2"], radians_joints["A3"], radians_joints["A4"], radians_joints["A5"], radians_joints["A6"], speed=0.3)
            if response and response.text.strip() == "OK":
                time.sleep(2.5)
                save_calibration_image()
                time.sleep(1)

def create_gui():
    root = tk.Tk()
    root.title("KUKA KMR IIWA Controller")
    root.geometry("600x700")
    
    control_frame = tk.Frame(root)
    control_frame.pack(pady=10)
    
    tk.Button(control_frame, text="↑", command=lambda: move(0.1, 0, 0)).grid(row=0, column=1)
    tk.Button(control_frame, text="←", command=lambda: move(0, 0.1, 0)).grid(row=1, column=0)
    tk.Button(control_frame, text="→", command=lambda: move(0, -0.1, 0)).grid(row=1, column=2)
    tk.Button(control_frame, text="↓", command=lambda: move(-0.1, 0, 0)).grid(row=2, column=1)
    tk.Button(control_frame, text="↺", command=lambda: move(0, 0, 0.1)).grid(row=1, column=3)
    tk.Button(control_frame, text="↻", command=lambda: move(0, 0, -0.1)).grid(row=1, column=4)
    
    action_frame = tk.Frame(root)
    action_frame.pack(pady=10)
    
    tk.Button(action_frame, text="Capture Image", command=lambda: call_endpoint("CaptureImage")).grid(row=0, column=0)
    tk.Button(action_frame, text="Get Pose", command=lambda: call_endpoint("GetPose")).grid(row=0, column=1)
    tk.Button(action_frame, text="Honk On", command=lambda: call_endpoint("HonkOn")).grid(row=1, column=0)
    tk.Button(action_frame, text="Honk Off", command=lambda: call_endpoint("HonkOff")).grid(row=1, column=1)
    tk.Button(action_frame, text="Get end pose", command=lambda: call_endpoint("GetIIWAposition")).grid(row=2, column=0)
    tk.Button(action_frame, text="Get joints pose", command=lambda: call_endpoint("GetIIWAJointsPosition")).grid(row=2, column=1)
    
    goto_frame = tk.Frame(root)
    goto_frame.pack(pady=10)
    
    labels = ["X", "Y", "Z", "A", "B", "C"]
    entries = []
    for i, label in enumerate(labels):
        tk.Label(goto_frame, text=label+":").grid(row=0, column=i*2)
        entry = tk.Entry(goto_frame, width=5)
        entry.grid(row=0, column=i*2+1)
        entries.append(entry)
    
    tk.Button(goto_frame, text="Go To Position", command=lambda: goto_position(
        float(entries[0].get()), float(entries[1].get()), float(entries[2].get()),
        float(entries[3].get()), float(entries[4].get()), float(entries[5].get())
    )).grid(row=0, column=12)
    
    joint_frame = tk.Frame(root)
    joint_frame.pack(pady=10)
    
    joint_labels = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
    joint_entries = []
    for i, label in enumerate(joint_labels):
        tk.Label(joint_frame, text=label+":").grid(row=0, column=i*2)
        entry = tk.Entry(joint_frame, width=5)
        entry.grid(row=0, column=i*2+1)
        joint_entries.append(entry)
    
    tk.Button(joint_frame, text="Go To Joint Position", command=lambda: goto_joint(
        float(joint_entries[0].get()), float(joint_entries[1].get()), float(joint_entries[2].get()),
        float(joint_entries[3].get()), float(joint_entries[4].get()), float(joint_entries[5].get()), float(joint_entries[6].get())
    )).grid(row=0, column=14)
    
    tk.Button(root, text="Go Around", command=lambda: GoAroundRepeat()).pack(pady=5)
    tk.Button(root, text="Go Home", command=lambda: go_home()).pack(pady=5)
    
    move_to_location_frame = tk.Frame(root)
    move_to_location_frame.pack(pady=10)
    
    tk.Label(move_to_location_frame, text="Target Number:").pack(side=tk.LEFT)
    target_number_entry = tk.Entry(move_to_location_frame, width=5)
    target_number_entry.pack(side=tk.LEFT)
    
    tk.Button(move_to_location_frame, text="MoveToLocation", command=lambda: MoveToLocation(int(target_number_entry.get()))).pack(side=tk.LEFT)
    tk.Button(root, text="Execute sequence", command=lambda: ExecuteSequence()).pack(pady=5)
    
    adjust_frame = tk.Frame(root)
    adjust_frame.pack(pady=10)
    
    joint_labels = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    for i, label in enumerate(joint_labels):
        tk.Label(adjust_frame, text=label+":").grid(row=i, column=0)
        tk.Button(adjust_frame, text="+", command=lambda l=label: adjust_joint(l, 1*np.pi/180)).grid(row=i, column=1)
        tk.Button(adjust_frame, text="-", command=lambda l=label: adjust_joint(l, -1*np.pi/180)).grid(row=i, column=2)
    



    tk.Button(root, text="Save Calibration Image", command=save_calibration_image).pack(pady=5)

    tk.Button(root, text="Move to Hand Poses", command=move_to_hand_poses).pack(pady=5)

    tk.Button(root, text="Get Joints Write to File", command=Get_joints_Write_to_file).pack(pady=5)

    root.mainloop()


    
def ndarray_to_bytes(image: np.ndarray, format: str = ".png") -> bytes:
    _, buffer = cv2.imencode(format, image)
    return buffer.tobytes()

if __name__ == "__main__":
    create_gui()
