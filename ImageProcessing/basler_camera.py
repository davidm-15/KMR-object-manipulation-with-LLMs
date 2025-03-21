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
        camera_info = ptl.CreateDeviceInfo()



        cable = True
        if cable:
            camera_info.SetPortNr('3956')
            camera_info.SetIpAddress('172.31.1.20')
        else:
            camera_info.SetPortNr('3956')
            camera_info.SetIpAddress('10.35.129.5')


        print("Available properties:")
        keys = camera_info.GetPropertyNames()[1]
        print(keys)
        for key in keys:
            print(f"{key}: {camera_info.GetPropertyValue(key)}")



        camera_device = ptl.CreateDevice(camera_info)
        print(f"Camera device: {str(camera_device)}")

        
        


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
