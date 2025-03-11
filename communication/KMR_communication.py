import requests
import tkinter as tk
import time
from ImageProcessing.basler_camera import BaslerCamera
from ImageProcessing.realsense_camera import RealSenseCamera
import json


# Change this to your KUKA robot's IP address
# ROBOT_IP = "172.31.1.10"
ROBOT_IP = "10.35.129.5"
PORT = 30000

# Initialize the Basler camera
# camera = RealSenseCamera()
camera = None

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

def goto_joint(a1, a2, a3, a4, a5, a6, a7, speed=1):
    params = {"A1": a1, "A2": a2, "A3": a3, "A4": a4, "A5": a5, "A6": a6, "A7": a7, "speed": speed}
    response = call_endpoint("GotoJoint", params)
    return response

def MoveToLocation(TargetNumber):
    params = {"TargetNumber": TargetNumber}
    response = call_endpoint("MoveToLocation", params)
    return response

def GetJointsPosition():
    response = call_endpoint("GetIIWAJointsPosition")
    return response.json()

def GoAroundRepeat():
    start1 = {
        "A1": -2.967,
        "A2": -0.106,
        "A3": -0.068,
        "A4": 1.551,
        "A5": 1.522,
        "A6": 0.001,
        "A7": -0.813,
        "speed": 0.8
    }

    start2 = {
        "A1": -2.967,
        "A2": -0.106,
        "A3": -0.068,
        "A4": 2.015,
        "A5": 1.522,
        "A6": 0.001,
        "A7": -0.813,
        "speed": 0.8
    }

    for i in range(2):
        Start = start1 if i % 2 == 0 else start2
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
    locations = [73, 74, 75, 76]
    for location in locations:
        response = MoveToLocation(location)
        if response and response.text.strip() == "OK":
            time.sleep(3)
            GoAround()


def go_home():
    call_endpoint("GoHome")

def create_gui():
    root = tk.Tk()
    root.title("KUKA KMR IIWA Controller")
    root.geometry("500x600")
    
    tk.Button(root, text="↑", command=lambda: move(0.1, 0, 0)).pack()
    
    frame = tk.Frame(root)
    frame.pack()
    tk.Button(frame, text="←", command=lambda: move(0, 0.1, 0)).pack(side=tk.LEFT)
    tk.Button(frame, text="→", command=lambda: move(0, -0.1, 0)).pack(side=tk.RIGHT)
    
    tk.Button(root, text="↓", command=lambda: move(-0.1, 0, 0)).pack()
    
    tk.Button(root, text="Rotate Left", command=lambda: move(0, 0, 0.1)).pack()
    tk.Button(root, text="Rotate Right", command=lambda: move(0, 0, -0.1)).pack()
    tk.Button(root, text="Capture Image", command=lambda: call_endpoint("CaptureImage")).pack()
    tk.Button(root, text="Get Pose", command=lambda: call_endpoint("GetPose")).pack()
    tk.Button(root, text="Set Pose", command=lambda: call_endpoint("SetPose")).pack()
    tk.Button(root, text="Honk On", command=lambda: call_endpoint("HonkOn")).pack()
    tk.Button(root, text="Honk Off", command=lambda: call_endpoint("HonkOff")).pack()
    tk.Button(root, text="Get end pose", command=lambda: call_endpoint("GetIIWAposition")).pack()
    tk.Button(root, text="Get joints pose", command=lambda: call_endpoint("GetIIWAJointsPosition")).pack()
    
    # Goto Position Section
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
    
    # Goto Joint Position Section
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
    

    tk.Button(root, text="Go Around", command=lambda: GoAroundRepeat()).pack()

    tk.Button(root, text="Go Home", command=lambda: go_home()).pack()

    # MoveToLocation Section
    move_to_location_frame = tk.Frame(root)
    move_to_location_frame.pack(pady=10)
    
    tk.Label(move_to_location_frame, text="Target Number:").pack(side=tk.LEFT)
    target_number_entry = tk.Entry(move_to_location_frame, width=5)
    target_number_entry.pack(side=tk.LEFT)
    
    tk.Button(move_to_location_frame, text="MoveToLocation", command=lambda: MoveToLocation(int(target_number_entry.get()))).pack(side=tk.LEFT)

    tk.Button(root, text="Execute sequence", command=lambda: ExecuteSequence()).pack()





    root.mainloop()


    


if __name__ == "__main__":
    create_gui()
