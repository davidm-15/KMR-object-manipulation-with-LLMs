# camera_handler.py
# from image_processing.realsense_camera import RealSenseCamera # Keep if needed
import numpy as np
import os
import time
import sys

try:
    # This assumes image_processing is a sibling directory to KMR_communication
    from image_processing.basler_camera import BaslerCamera
    # from image_processing.realsense_camera import RealSenseCamera # If needed
except ModuleNotFoundError:
    print("Could not find camera modules in standard location.")
    # Try adding the parent directory (project root) to sys.path
    # This makes imports like 'image_processing.basler_camera' work
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        print(f"Adding project root to sys.path: {_PROJECT_ROOT}")
        sys.path.insert(0, _PROJECT_ROOT) # Insert at beginning

    # Retry the import
    try:
         from image_processing.basler_camera import BaslerCamera
         # from image_processing.realsense_camera import RealSenseCamera
    except ModuleNotFoundError as e:
         print(f"ERROR: Still could not find camera modules after path adjustment: {e}")
         # Define dummy classes or raise the error, depending on desired behavior
         BaslerCamera = None # Or raise ImportError("BaslerCamera not found")


class CameraHandler:
    def __init__(self, camera_type='basler'):
        self.camera = None
        self.camera_type = camera_type
        self._initialize_camera()

    def _initialize_camera(self):
        """Initializes the selected camera."""
        try:
            if self.camera_type.lower() == 'basler':
                self.camera = BaslerCamera()
                print("Basler camera initialized successfully.")
            # elif self.camera_type.lower() == 'realsense':
            #     self.camera = RealSenseCamera()
            #     print("RealSense camera initialized successfully.")
            else:
                print(f"Error: Unsupported camera type '{self.camera_type}'")
                self.camera = None
        except Exception as e:
            print(f"Failed to initialize {self.camera_type} camera: {e}")
            self.camera = None

    def is_ready(self) -> bool:
        """Checks if the camera is initialized."""
        return self.camera is not None

    def capture_image(self) -> np.ndarray | None:
        """Captures an image using the initialized camera."""
        if not self.is_ready():
            print("Error: Camera not initialized.")
            return None
        try:
            # Add a small delay before capture if needed
            # time.sleep(0.1)
            image = self.camera.capture_image()
            if image is None:
                 print("Warning: Camera capture returned None.")
            return image
        except Exception as e:
            print(f"Error capturing image: {e}")
            return None

    def save_image(self, image: np.ndarray, filepath: str):
        """Saves the captured image to a file."""
        if not self.is_ready():
            print("Error: Camera not initialized.")
            return
        if image is None:
             print("Error: Cannot save None image.")
             return
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.camera.save_image(image, filepath)
            print(f"Image saved to {filepath}")
        except Exception as e:
            print(f"Error saving image to {filepath}: {e}")

    def close(self):
        """Closes the camera connection if necessary."""
        if self.camera and hasattr(self.camera, 'close'):
             try:
                 self.camera.close()
                 print(f"{self.camera_type} camera closed.")
             except Exception as e:
                 print(f"Error closing camera: {e}")
        self.camera = None

# Example Usage (optional, for testing)
if __name__ == "__main__":
    cam_handler = CameraHandler(camera_type='basler')
    if cam_handler.is_ready():
        img = cam_handler.capture_image()
        if img is not None:
            cam_handler.save_image(img, "test_capture.png")
        cam_handler.close()
    else:
        print("Camera could not be used.")