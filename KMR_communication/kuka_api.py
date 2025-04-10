# kuka_api.py
import requests
import numpy as np
from . import config  # Import the configuration

def call_endpoint(endpoint: str, params: dict = None, method: str = "GET", **kwargs) -> requests.Response | None:
    """
    Sends a request to a specific endpoint of the KUKA robot API.

    Args:
        endpoint (str): The API endpoint name (e.g., "GetPose").
        params (dict, optional): Dictionary of query parameters. Defaults to None.
        method (str, optional): HTTP method ('GET' or 'POST'). Defaults to "GET".

    Returns:
        requests.Response | None: The response object or None if an error occurred.
    """
    timeout = kwargs.get('timeout', 10)  # Default timeout for requests

    url = f"{config.BASE_URL}/{endpoint}"
    response = None
    try:
        if method.upper() == "GET":
            response = requests.get(url, params=params, timeout=timeout) # Added timeout

        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        print(f"Response from {endpoint} ({response.status_code}): {response.text[:100]}...") # Print truncated response
        return response

    except requests.exceptions.RequestException as e:
        print(f"Error calling {endpoint}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred calling {endpoint}: {e}")
        return None

# --- KMR Base Movement ---
def move(x: float, y: float, theta: float, timeout: int = 200):
    """Moves the KMR base relative to its current position."""
    params = {"x": x, "y": y, "Theta": theta}
    call_endpoint("ArrowsMove", params, timeout=timeout)

def move_to_location(target_number: int, timeout: int = 200):
    """Moves the KMR to a predefined location number."""
    params = {"TargetNumber": target_number}
    response = call_endpoint("MoveToLocation", params, timeout=timeout)
    return response # Return response for checking success ("OK")

# --- IIWA Arm Movement ---
def goto_position(x: float, y: float, z: float, a: float, b: float, c: float,
                  speed: float = config.DEFAULT_ARM_SPEED, motion_type: str = "ptp"):
    """Moves the IIWA arm to a Cartesian position."""
    params = {"x": x, "y": y, "z": z, "a": a, "b": b, "c": c, "Speed": speed, "Motion": motion_type}
    call_endpoint("GotoPosition", params)

def goto_joint(a1: float, a2: float, a3: float, a4: float, a5: float, a6: float, a7: float,
               speed: float = config.DEFAULT_ARM_SPEED):
    """Moves the IIWA arm to a specific joint configuration (in radians)."""
    params = {"A1": a1, "A2": a2, "A3": a3, "A4": a4, "A5": a5, "A6": a6, "A7": a7, "speed": speed}
    response = call_endpoint("GotoJoint", params)
    return response # Return response for checking success ("OK")

def go_home():
    """Sends the IIWA arm to its home position."""
    call_endpoint("GoHome")

# --- Gripper Control ---
def close_gripper(force: int = 1):
    """Closes the gripper."""
    params = {"force": force}
    call_endpoint("CloseGripper", params)

def open_gripper():
    """Opens the gripper."""
    call_endpoint("OpenGripper")

def init_gripper():
    """Initializes the gripper."""
    call_endpoint("InitGripper")

def release_object():
    """Releases the object (specific gripper action)."""
    call_endpoint("ReleaseObject")

# --- Status Information ---
def get_pose() -> dict | None:
    """Gets the current pose of the KMR base."""
    response = call_endpoint("GetPose")
    return response.json() if response and response.ok else None

def get_iiwa_position() -> dict | None:
    """Gets the current Cartesian position of the IIWA end-effector."""
    response = call_endpoint("GetIIWAposition")
    print("IIWA Position:", response.text)
    return response.json() if response and response.ok else None

def get_iiwa_joint_position() -> dict | None:
    """Gets the current joint positions of the IIWA arm."""
    response = call_endpoint("GetIIWAJointsPosition")
    # Sort joints for consistency if needed
    if response and response.ok:
        joints_data = response.json()
        return {key: joints_data[key] for key in sorted(joints_data.keys())}
    return None

def get_gripper_state() -> str | None:
    """Gets the current state of the gripper."""
    response = call_endpoint("GetGripperState")
    if response and response.ok:
        print("Gripper State:", response.text)
        return response.text
    return None

def IsPositionInZone(x: float, y: float, orientation: float, zone: int) -> bool:
    """Checks if a position is within a specified zone."""

    # Convert orientation from degrees to radians
    orientation_rad = np.radians(orientation)

    # Create the rotation matrix
    rot_mat = np.array([
        [np.cos(orientation_rad), -np.sin(orientation_rad)],
        [np.sin(orientation_rad), np.cos(orientation_rad)]
    ])

    print("Rotation Matrix: ", rot_mat)

    Zone_length = config.LONGEST_LENGTH_KMR + 420
    Zone_width = config.LONGEST_WIDTH_KMR + 420

    pos1 = rot_mat @ np.array([Zone_length/2, Zone_width/2])
    pos2 = rot_mat @ np.array([Zone_length/2, -Zone_width/2])
    pos3 = rot_mat @ np.array([-Zone_length/2, -Zone_width/2])
    pos4 = rot_mat @ np.array([-Zone_length/2, Zone_width/2])
    pos5 = rot_mat @ np.array([Zone_length/2, 0])
    pos6 = rot_mat @ np.array([-Zone_length/2, 0])
    pos7 = rot_mat @ np.array([0, Zone_width/2])
    pos8 = rot_mat @ np.array([0, -Zone_width/2])
    pos9 = rot_mat @ np.array([0, 0])
    

    poses = [pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9]


    for pos in poses:
        params = {
            "x": x + pos[0]/1000,
            "y": y + pos[1]/1000,
            "orientation": orientation,
            "id": zone
        }

        print("x: ", params["x"], " \n y: ", params["y"])

        response = call_endpoint("IsPositionInZone", params)

        if response and response.ok:
            if response.text == "true" or response.text == "false":
                print("IsPositionInZone:", response.text)
                if response.text == "true":
                    return True
            else:
                print("Unexpected response from IsPositionInZone:", response.text)
                return None
            
        else:
            print("Error calling IsPositionInZone:", response)
            return None
              
    return False

# --- Other Actions ---
def set_led(color: str):
    """Sets the color of an LED."""
    params = {"color": color}
    call_endpoint("SetLED", params)

def capture_image_api():
    """Triggers image capture via API (if available)."""
    # Note: This might not be the same as capturing from Basler/RealSense directly
    call_endpoint("CaptureImage")

def honk_on():
    """Turns the honk on."""
    call_endpoint("HonkOn")

def honk_off():
    """Turns the honk off."""
    call_endpoint("HonkOff")