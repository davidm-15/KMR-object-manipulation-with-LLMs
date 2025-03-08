import time
import cv2
import numpy as np

# Base class for cameras
class CameraBase:
    def __init__(self):
        """Initialize the camera (to be implemented in subclasses)."""
        self.camera = None

    def capture_image(self):
        """Capture an image (to be implemented in subclasses)."""
        raise NotImplementedError("Subclasses must implement capture_image()")

    def save_image(self, image, filename=None):
        """Save an image to a file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"images/{self.__class__.__name__}_image_{timestamp}.png"
        
        cv2.imwrite(filename, image)
        print(f"Image saved as {filename}")

    def close(self):
        """Close the camera (to be implemented in subclasses)."""
        raise NotImplementedError("Subclasses must implement close()")


# Basler Camera
from pypylon import pylon

class BaslerCamera(CameraBase):
    def __init__(self):
        """Initialize the Basler camera."""
        super().__init__()
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed

        print("Basler camera initialized.")

    def capture_image(self):
        """Capture an image from the Basler camera."""
        grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grab_result.GrabSucceeded():
            image = self.converter.Convert(grab_result).GetArray()
            grab_result.Release()
            return image
        else:
            grab_result.Release()
            return None

    def close(self):
        """Close the Basler camera."""
        self.camera.StopGrabbing()
        self.camera.Close()
        print("Basler camera closed.")


# RealSense Camera
import pyrealsense2 as rs

class RealSenseCamera(CameraBase):
    def __init__(self):
        """Initialize the RealSense camera."""
        super().__init__()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

        print("RealSense camera initialized.")

    def capture_image(self):
        """Capture an image from the RealSense camera."""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())

    def close(self):
        """Close the RealSense camera."""
        self.pipeline.stop()
        print("RealSense camera closed.")
