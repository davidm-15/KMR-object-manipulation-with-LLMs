from pypylon import pylon
import numpy as np
import cv2
import time
import keyboard

def FieldOfView():
    pass

def capture_image(image):
    # Generate timestamp-based filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"images/basler_image_{timestamp}.png"
    
    # Save image
    cv2.imwrite(filename, image)
    print(f"Image saved as {filename}")

def main():
    # Initialize Basler camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    
    # Set camera parameters if needed
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    
    last_capture_time = 0
    capture_delay = 0.5  # Prevent rapid multiple captures
    
    try:
        print("Press 'space' to capture an image, or 'q' to quit.")
        while camera.IsGrabbing():
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            
            if grab_result.GrabSucceeded():
                # Convert image to OpenCV format
                image = converter.Convert(grab_result).GetArray()
                
                # Resize image for display
                display_image = cv2.resize(image, (640, 480))
                
                # Show the stream
                cv2.imshow("Basler Camera Stream", display_image)
                
                current_time = time.time()
                if keyboard.is_pressed('space') and (current_time - last_capture_time) > capture_delay:
                    capture_image(image)
                    last_capture_time = current_time
                
                if keyboard.is_pressed('q') or cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting...")
                    break
            
            grab_result.Release()
    
    finally:
        # Stop camera and close OpenCV window
        camera.StopGrabbing()
        camera.Close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
