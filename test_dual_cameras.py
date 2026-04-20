#!/usr/bin/env python3
from picamera2 import Picamera2
import time
import cv2
import numpy as np

def test_camera_port(camera_id, camera_name):
    """Test individual camera port"""
    try:
        print(f"\n=== Testing {camera_name} (Camera {camera_id}) ===")
        
        # Initialize camera
        picam = Picamera2(camera_id)
        
        # Get camera info
        camera_info = picam.camera_info
        print(f"Camera Info: {camera_info}")
        
        # Configure camera
        config = picam.create_preview_configuration(main={"size": (640, 480)})
        print(f"Configuration: {config}")
        
        picam.configure(config)
        picam.start()
        
        # Let camera stabilize
        time.sleep(2)
        
        # Capture image
        image_path = f"/home/pi/{camera_name.lower()}_test.jpg"
        picam.capture_file(image_path)
        print(f"✓ Image saved: {image_path}")
        
        # Capture array for analysis
        image_array = picam.capture_array()
        print(f"✓ Image shape: {image_array.shape}")
        
        # Basic image stats
        if len(image_array.shape) == 3:
            mean_vals = np.mean(image_array, axis=(0,1))
            print(f"✓ Channel means: {mean_vals}")
        
        picam.stop()
        return True, image_path
        
    except Exception as e:
        print(f"✗ {camera_name} failed: {e}")
        return False, str(e)

def main():
    print("Testing Raspberry Pi 5 Dual Camera Setup")
    print("Based on Hoang's NDVI System Configuration")
    
    results = {}
    
    # Test Camera 0 (should be first CSI port)
    success, result = test_camera_port(0, "Camera_0")
    results["camera_0"] = (success, result)
    
    # Test Camera 1 (should be second CSI port)
    success, result = test_camera_port(1, "Camera_1")
    results["camera_1"] = (success, result)
    
    # Summary
    print("\n" + "="*50)
    print("CAMERA TEST SUMMARY")
    print("="*50)
    
    working_cameras = []
    for cam_name, (success, result) in results.items():
        status = "✓ WORKING" if success else "✗ FAILED"
        print(f"{cam_name.upper()}: {status}")
        if success:
            working_cameras.append(cam_name)
            print(f"  Image: {result}")
    
    print(f"\nWorking cameras: {len(working_cameras)}")
    
    if len(working_cameras) == 2:
        print("\n🎉 SUCCESS: Both cameras detected!")
        print("Ready for NDVI dual-camera setup")
    elif len(working_cameras) == 1:
        print("\n⚠️  WARNING: Only one camera working")
        print("Check second camera connection")
    else:
        print("\n❌ ERROR: No cameras working")
        print("Check camera connections and enable camera interface")

if __name__ == "__main__":
    main()
