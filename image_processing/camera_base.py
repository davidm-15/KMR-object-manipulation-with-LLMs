import time
import json
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np
from numpy.typing import NDArray


class CameraBase:
    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize the camera with optional configuration."""
        self.camera: Optional[object] = None
        self.config: Dict[str, Any] = {}

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: Path) -> None:
        """Load camera parameters from a JSON file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file {config_path} not found.")
        
        with config_path.open("r", encoding="utf-8") as file:
            self.config = json.load(file)
        print(f"Loaded config: {self.config}")

    def capture_image(self) -> Optional[NDArray[np.uint8]]:
        """Capture an image (to be implemented in subclasses)."""
        raise NotImplementedError("Subclasses must implement capture_image()")

    def save_image(self, image: NDArray[np.uint8], filename: Optional[Path] = None) -> None:
        """Save an image to a file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = Path(f"images/{self.__class__.__name__}_image_{timestamp}.png")
        
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        if filename.suffix.lower() not in valid_extensions:
            filename = filename.with_suffix('.png')
        cv2.imwrite(str(filename), image)
        print(f"Image saved as {filename}")

    def close(self) -> None:
        """Close the camera (to be implemented in subclasses)."""
        raise NotImplementedError("Subclasses must implement close()")

    def stream_and_capture(self, out_folder: Optional[Path] = None) -> None:
        """Stream the camera feed and save an image when spacebar is pressed."""
        if self.camera is None:
            print("Camera not initialized.")
            return

        print("Press 'Space' to capture an image, 'q' to quit.")
        while True:
            image = self.capture_image()
            if image is None:
                print("Failed to capture image.")
                continue

            resized_image = cv2.resize(image, (640, 480))  # Resize to 640x480 for better visibility
            cv2.imshow("Camera Stream", resized_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Spacebar to save image
                filename = time.strftime(out_folder + "/%Y%m%d_%H%M%S.png")
                self.save_image(image, filename)
            elif key == ord('q'):  # 'q' to quit
                break

        cv2.destroyAllWindows()
