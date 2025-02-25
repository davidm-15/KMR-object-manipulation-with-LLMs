from pypylon import pylon
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import keyboard

def capture_images(basler_image, realsense_image):
    # Generate a synchronized timestamp-based filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    basler_filename = f"images/basler_image_{timestamp}.png"
    realsense_filename = f"images/RealSense_image_{timestamp}.png"
    
    # Save images
    cv2.imwrite(basler_filename, basler_image)
    cv2.imwrite(realsense_filename, realsense_image)
    
    print(f"Images saved as {basler_filename} and {realsense_filename}")

def main():
    # Initialize Basler camera
    basler_camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    basler_camera.Open()
    basler_camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    basler_converter = pylon.ImageFormatConverter()
    basler_converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    
    # Initialize RealSense pipeline
    realsense_pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    realsense_pipeline.start(config)
    
    last_capture_time = 0
    capture_delay = 0.5  # Prevent rapid multiple captures
    
    try:
        print("Press 'space' to capture images, or 'q' to quit.")
        while basler_camera.IsGrabbing():
            # Capture Basler frame
            basler_result = basler_camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if not basler_result.GrabSucceeded():
                basler_result.Release()
                continue
            basler_image = basler_converter.Convert(basler_result).GetArray()
            basler_result.Release()
            
            # Capture RealSense frame
            frames = realsense_pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            realsense_image = np.asanyarray(color_frame.get_data())
            
            # Resize images for display
            basler_display = cv2.resize(basler_image, (640, 480))
            realsense_display = cv2.resize(realsense_image, (640, 480))
            
            # Show streams
            cv2.imshow("Basler Camera Stream", basler_display)
            cv2.imshow("RealSense Stream", realsense_display)
            
            current_time = time.time()
            if keyboard.is_pressed('space') and (current_time - last_capture_time) > capture_delay:
                capture_images(basler_image, realsense_image)
                last_capture_time = current_time
            
            if keyboard.is_pressed('q') or cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break
    
    finally:
        # Cleanup
        basler_camera.StopGrabbing()
        basler_camera.Close()
        realsense_pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
