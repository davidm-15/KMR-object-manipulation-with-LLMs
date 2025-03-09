import pyrealsense2 as rs
import numpy as np
from pathlib import Path
from typing import Optional
from .camera_base import CameraBase

class RealSenseCamera(CameraBase):
    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize the RealSense camera with optional config."""
        super().__init__(config_path)
        self.pipeline = rs.pipeline()
        self.config_rs = rs.config()
        self.config_rs.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config_rs)

        print("RealSense camera initialized.")

    def capture_image(self) -> Optional[np.ndarray]:
        """Capture an image from the RealSense camera."""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())

    def close(self) -> None:
        """Close the RealSense camera."""
        self.pipeline.stop()
        print("RealSense camera closed.")
