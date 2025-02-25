import pyrealsense2 as rs
import numpy as np
import cv2
import time
import keyboard

def capture_image(color_image):
    # Generate timestamp-based filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"images/RealSense_image_{timestamp}.png"
    
    # Save image
    cv2.imwrite(filename, color_image)
    print(f"Image saved as {filename}")

def main():
    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    pipeline.start(config)
    
    try:
        print("Press 'space' to capture an image, or 'q' to quit.")
        while True:
            # Wait for a frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            # Convert image to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            
            # Show the stream
            cv2.imshow("RealSense Stream", color_image)
            
            if keyboard.is_pressed('space'):
                capture_image(color_image)
                time.sleep(0.5)  # Prevent multiple captures from a single press
            elif keyboard.is_pressed('q'):
                print("Exiting...")
                break
            
            # Exit when 'q' is pressed in OpenCV window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
