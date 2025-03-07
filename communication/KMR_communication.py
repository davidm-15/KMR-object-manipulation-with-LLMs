import requests
import tkinter as tk
import time

# Change this to your KUKA robot's IP address
# ROBOT_IP = "172.31.1.10"
ROBOT_IP = "10.35.129.5"
PORT = 30000

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

def goto_joint(a1, a2, a3, a4, a5, a6, a7):
    params = {"A1": a1, "A2": a2, "A3": a3, "A4": a4, "A5": a5, "A6": a6, "A7": a7, "Speed": 1}
    response = call_endpoint("GotoJoint", params)
    return response

def GoAround():
    Start = {"A1": -2.9, "A2": -0.106, "A3": -0.068, "A4": 2.094, "A5": 1.643, "A6": 1.381, "A7": 0.150, "Speed": 0.2}
    End = {"A1": 2.9, "A2": -0.106, "A3": -0.068, "A4": 2.094, "A5": 1.643, "A6": 1.381, "A7": 0.150, "Speed": 0.2}

    steps = 8
    for i in range(steps + 1):
        params = {key: Start[key] + (End[key] - Start[key]) * i / steps for key in Start}
        response = goto_joint(params["A1"], params["A2"], params["A3"], params["A4"], params["A5"], params["A6"], params["A7"])
        if response and response.text.strip() == "OK":
            time.sleep(1)
        
            time.sleep(1)

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
    

    tk.Button(root, text="Go Around", command=lambda: GoAround()).pack()

    tk.Button(root, text="Go Home", command=lambda: go_home()).pack()

    root.mainloop()

if __name__ == "__main__":
    create_gui()
