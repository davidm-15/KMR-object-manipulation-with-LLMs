# kuka_api.py
import requests
import config  # Import the configuration

def call_endpoint(endpoint: str, params: dict = None, method: str = "GET") -> requests.Response | None:
    """
    Sends a request to a specific endpoint of the KUKA robot API.

    Args:
        endpoint (str): The API endpoint name (e.g., "GetPose").
        params (dict, optional): Dictionary of query parameters. Defaults to None.
        method (str, optional): HTTP method ('GET' or 'POST'). Defaults to "GET".

    Returns:
        requests.Response | None: The response object or None if an error occurred.
    """
    url = f"{config.BASE_URL}/{endpoint}"
    response = None
    try:
        if method.upper() == "GET":
            response = requests.get(url, params=params, timeout=10) # Added timeout
        elif method.upper() == "POST":
            response = requests.post(url, params=params, timeout=10) # Example for POST
        else:
            print(f"Unsupported HTTP method: {method}")
            return None

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
def move(x: float, y: float, theta: float):
    """Moves the KMR base relative to its current position."""
    params = {"x": x, "y": y, "Theta": theta}
    call_endpoint("ArrowsMove", params)

def move_to_location(target_number: int):
    """Moves the KMR to a predefined location number."""
    params = {"TargetNumber": target_number}
    response = call_endpoint("MoveToLocation", params)
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