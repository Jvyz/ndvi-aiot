#!/usr/bin/env python3
import cv2
import sys

def test_camera(device_id):
    """Test if camera device is accessible"""
    try:
        cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            return False, "Cannot open camera"
        
        ret, frame = cap.read()
        if not ret:
            return False, "Cannot read frame"
        
        height, width = frame.shape[:2]
        cap.release()
        return True, f"Resolution: {width}x{height}"
    
    except Exception as e:
        return False, str(e)

def main():
    print("Testing camera devices...")
    working_cameras = []
    
    # Test first 10 video devices
    for i in range(10):
        device_path = f"/dev/video{i}"
        success, info = test_camera(i)
        
        if success:
            print(f"✓ {device_path}: WORKING - {info}")
            working_cameras.append(i)
        else:
            print(f"✗ {device_path}: {info}")
    
    print(f"\nFound {len(working_cameras)} working cameras: {working_cameras}")
    return working_cameras

if __name__ == "__main__":
    working_cameras = main()
