import requests
import keyboard  # You'll need to install this: pip install keyboard
import time #Import time module

# Change this to your KUKA robot's IP address
ROBOT_IP = "172.31.1.10"
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
    except Exception as e:
        print(f"Error calling {endpoint}:", e)

def count_with_1_2():
    params = {"num1": 1, "num2": 2}
    call_endpoint("Count", params)

def count_with_5_7():
    params = {"num1": 5, "num2": 7}
    call_endpoint("Count", params)



def move_up():
    params = {"x": 0.1, "y": 0, "Theta": 0}
    call_endpoint("ArrowsMove", params)

def move_down():
    params = {"x": -0.1, "y": 0, "Theta": 0}
    call_endpoint("ArrowsMove", params)

def move_left():
    params = {"x": 0, "y": 0.1, "Theta": 0}
    call_endpoint("ArrowsMove", params)

def move_right():
    params = {"x": 0, "y": -0.1, "Theta": 0}
    call_endpoint("ArrowsMove", params)

def rotate_left():
    params = {"x": 0, "y": 0, "Theta": 0.1}
    call_endpoint("ArrowsMove", params)

def rotate_right():
    params = {"x": 0, "y": 0, "Theta": -0.1}
    call_endpoint("ArrowsMove", params)

def main():
    print("Press the up arrow to call Count with 5 and 7.")
    print("Press the down arrow to call Count with 1 and 2.")
    print("Press ESC to exit.")

    while True:
        key = keyboard.read_event().name

        match key:
            case "down":
                move_down()
                time.sleep(0.2) # Add a small delay

            case "up":
                move_up()
                time.sleep(0.2) # Add a small delay

            case "left":
                move_left()
                time.sleep(0.2)

            case "right":
                move_right()
                time.sleep(0.2)

            case "a":
                rotate_left()
                time.sleep(0.2)

            case "d":
                rotate_right()
                time.sleep(0.2)

            case "esc":
                print("Exiting...")
                break

        
        time.sleep(0.01) #small delay for CPU usage

if __name__ == "__main__":
    main()