from pypylon import pylon
import numpy as np
from pathlib import Path
from typing import Optional
from .camera_base import CameraBase

class BaslerCamera(CameraBase):
    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize the Basler camera with optional config."""
        super().__init__(config_path)
        factory = pylon.TlFactory.GetInstance()
        ptl = factory.CreateTl('BaslerGigE')
        empty_camera_info = ptl.CreateDeviceInfo()
        empty_camera_info.SetPropertyValue('IpAddress', '172.31.10.20')
        camera_device = ptl.CreateDevice(empty_camera_info)
        self.camera = pylon.InstantCamera(camera_device)
        self.camera.Open()
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed

        print("Basler camera initialized.")

    def capture_image(self) -> Optional[np.ndarray]:
        """Capture an image from the Basler camera."""
        grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grab_result.GrabSucceeded():
            image = self.converter.Convert(grab_result).GetArray()
            grab_result.Release()
            return image

        grab_result.Release()
        return None

    def close(self) -> None:
        """Close the Basler camera."""
        self.camera.StopGrabbing()
        self.camera.Close()
        print("Basler camera closed.")
