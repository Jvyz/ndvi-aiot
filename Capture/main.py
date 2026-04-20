#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated NDVI-GPS-DataHub Pipeline for AIoT 2025 Competition
VGU Agritech Team - Multispectral Imaging for Crop Health Monitoring

This pipeline integrates:
1. Continuous RGB/NIR image capture and alignment
2. NDVI calculation
3. GPS coordinate acquisition  
4. Automatic data upload to Advantech DataHub

Requirements:
- Stereo calibration files: rgb_mapx.npy, rgb_mapy.npy, ir_mapx.npy, ir_mapy.npy
- GPS module connected via serial
- Network connectivity for DataHub
"""

import os
import time
import datetime
import threading
import json
import argparse
import numpy as np
import cv2 as cv
from picamera2 import Picamera2

# Import your existing modules
from opencv_registration import OpenCVAligner, alignment_quality
from ndvi_utils import compute_ndvi_from_rgb_ir, colorize_ndvi
import L76X

# Import Advantech DataHub SDK
from wisepaasdatahubedgesdk.EdgeAgent import EdgeAgent
import wisepaasdatahubedgesdk.Common.Constants as constant
from wisepaasdatahubedgesdk.Model.Edge import (
    EdgeAgentOptions, DCCSOptions, EdgeData, EdgeTag, 
    EdgeConfig, NodeConfig, DeviceConfig, AnalogTagConfig
)

class NDVIGPSDataLogger:
    """
    Main pipeline class that integrates camera capture, GPS, and DataHub upload
    """
    
    def __init__(self, config):
        self.config = config
        self.running = False
        
        # Initialize components
        self.rgb_cam = None
        self.ir_cam = None
        self.gps = None
        self.aligner = None
        self.edge_agent = None
        
        # Data storage
        self.latest_data = {
            'ndvi_mean': 0.0,
            'ndvi_vegetation_mean': 0.0,
            'vegetation_percentage': 0.0,
            'latitude': 0.0,
            'longitude': 0.0,
            'gps_accuracy': 0.0,
            'timestamp': None,
            'gps_valid': False,
            'alignment_quality': 0.0
        }
        
        self.data_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'images_captured': 0,
            'successful_alignments': 0,
            'gps_fixes': 0,
            'datahub_uploads': 0
        }
        
    def setup_cameras(self):
        """Initialize RGB and NIR cameras"""
        print("[Camera] Initializing cameras...")
        
        try:
            # RGB Camera (Camera 0)
            self.rgb_cam = Picamera2(camera_num=self.config['cameras']['rgb_index'])
            rgb_cfg = self.rgb_cam.create_video_configuration(
                main={"size": self.config['cameras']['resolution'], "format": "RGB888"},
                buffer_count=4
            )
            self.rgb_cam.configure(rgb_cfg)
            self.rgb_cam.start()
            time.sleep(0.5)
            
            # Apply RGB camera settings
            rgb_controls = {
                "AeEnable": self.config['cameras']['rgb_auto_exposure'],
                "AwbEnable": self.config['cameras']['rgb_auto_wb'],
                "ExposureTime": self.config['cameras']['rgb_exposure_us'],
                "AnalogueGain": self.config['cameras']['rgb_gain']
            }
            self.rgb_cam.set_controls(rgb_controls)
            
            # NIR Camera (Camera 1)
            self.ir_cam = Picamera2(camera_num=self.config['cameras']['ir_index'])
            ir_cfg = self.ir_cam.create_video_configuration(
                main={"size": self.config['cameras']['resolution'], "format": "RGB888"},
                buffer_count=4
            )
            self.ir_cam.configure(ir_cfg)
            self.ir_cam.start()
            time.sleep(0.5)
            
            # Apply NIR camera settings
            ir_controls = {
                "AeEnable": self.config['cameras']['ir_auto_exposure'],
                "AwbEnable": False,  # Disable AWB for NIR
                "ExposureTime": self.config['cameras']['ir_exposure_us'],
                "AnalogueGain": self.config['cameras']['ir_gain']
            }
            self.ir_cam.set_controls(ir_controls)
            
            print("[Camera] ✓ Cameras initialized successfully")
            return True
            
        except Exception as e:
            print(f"[Camera] ✗ Camera initialization failed: {e}")
            return False
    
    def setup_aligner(self):
        """Load calibration and initialize stereo aligner"""
        print("[Aligner] Loading stereo calibration...")
        
        try:
            # Load rectification maps from calibration
            rgb_mapx = np.load("rgb_mapx.npy")
            rgb_mapy = np.load("rgb_mapy.npy") 
            ir_mapx = np.load("ir_mapx.npy")
            ir_mapy = np.load("ir_mapy.npy")
            
            # Initialize aligner with rectification maps
            self.aligner = OpenCVAligner(
                rectify_maps=(rgb_mapx, rgb_mapy, ir_mapx, ir_mapy),
                grid_step=self.config['processing']['grid_step'],
                down_for_H=self.config['processing']['down_for_H']
            )
            
            print("[Aligner] ✓ Stereo aligner initialized")
            return True
            
        except Exception as e:
            print(f"[Aligner] ✗ Failed to load calibration: {e}")
            return False
    
    def setup_gps(self):
        """Initialize GPS module"""
        print("[GPS] Initializing GPS module...")
        
        try:
            self.gps = L76X.L76X()
            
            # Configure GPS for optimal performance
            print("[GPS] Configuring GPS settings...")
            self.gps.L76X_Set_Baudrate(115200)
            time.sleep(1)
            
            self.gps.L76X_Send_Command(self.gps.SET_POS_FIX_400MS)
            time.sleep(1)
            
            self.gps.L76X_Send_Command(self.gps.SET_NMEA_OUTPUT)
            time.sleep(1)
            
            self.gps.L76X_Exit_BackupMode()
            time.sleep(2)
            
            print("[GPS] ✓ GPS module initialized")
            return True
            
        except Exception as e:
            print(f"[GPS] ✗ GPS initialization failed: {e}")
            return False
    
    def setup_datahub(self):
        """Initialize connection to Advantech DataHub"""
        print("[DataHub] Connecting to Advantech DataHub...")
        
        try:
            # Create edge agent options
            edge_options = EdgeAgentOptions(nodeId=self.config['datahub']['node_id'])
            edge_options.connectType = constant.ConnectType['DCCS']
            
            # Configure DCCS connection
            dccs_options = DCCSOptions(
                apiUrl=self.config['datahub']['api_url'],
                credentialKey=self.config['datahub']['credential_key']
            )
            edge_options.DCCS = dccs_options
            
            # Create edge agent
            self.edge_agent = EdgeAgent(edge_options)
            self.edge_agent.on_connected = self._on_datahub_connected
            self.edge_agent.on_disconnected = self._on_datahub_disconnected
            
            # Connect to DataHub
            self.edge_agent.connect()
            
            print("[DataHub] ✓ Connection initiated")
            return True
            
        except Exception as e:
            print(f"[DataHub] ✗ Connection failed: {e}")
            return False
    
    def _on_datahub_connected(self, agent, connected):
        """DataHub connection callback"""
        if connected:
            print("[DataHub] ✓ Connected to DataHub")
            # Upload device configuration
            config = self._generate_device_config()
            self.edge_agent.uploadConfig(
                action=constant.ActionType['Create'], 
                edgeConfig=config
            )
        else:
            print("[DataHub] ✗ Connection failed")
    
    def _on_datahub_disconnected(self, agent, disconnected):
        """DataHub disconnection callback"""
        if disconnected:
            print("[DataHub] ✗ Disconnected from DataHub")
    
    def _generate_device_config(self):
        """Generate device configuration for DataHub"""
        config = EdgeConfig()
        device_config = DeviceConfig(
            id='NDVIDevice001',
            name='NDVI GPS Agricultural Sensor',
            description='Multispectral imaging device for crop health monitoring',
            deviceType='Agricultural Sensor',
            retentionPolicyName=''
        )
        
        # Define analog tags for our measurements
        tags = [
            ('NDVI_Mean', 'Mean NDVI value of captured area', -1.0, 1.0),
            ('NDVI_Vegetation_Mean', 'Mean NDVI of vegetation pixels only', -1.0, 1.0),
            ('Vegetation_Percentage', 'Percentage of vegetation pixels', 0.0, 100.0),
            ('GPS_Latitude', 'GPS Latitude coordinate', -90.0, 90.0),
            ('GPS_Longitude', 'GPS Longitude coordinate', -180.0, 180.0),
            ('GPS_Accuracy_Meters', 'GPS accuracy in meters', 0.0, 1000.0),
            ('Alignment_Quality', 'Image alignment quality (NCC)', -1.0, 1.0)
        ]
        
        for tag_name, description, span_low, span_high in tags:
            analog_tag = AnalogTagConfig(
                name=tag_name,
                description=description,
                readOnly=True,
                arraySize=0,
                spanHigh=span_high,
                spanLow=span_low,
                engineerUnit='',
                integerDisplayFormat=4,
                fractionDisplayFormat=4
            )
            device_config.analogTagList.append(analog_tag)
        
        config.node.deviceList.append(device_config)
        return config
    
    def capture_frame_pair(self):
        """Capture synchronized RGB and NIR frames"""
        try:
            # Capture RGB frame
            rgb_array = self.rgb_cam.capture_array("main")
            rgb_bgr = cv.cvtColor(rgb_array, cv.COLOR_RGB2BGR)
            
            # Capture NIR frame  
            ir_array = self.ir_cam.capture_array("main")
            ir_gray = cv.cvtColor(ir_array, cv.COLOR_RGB2GRAY)
            
            self.stats['images_captured'] += 1
            return rgb_bgr, ir_gray
            
        except Exception as e:
            print(f"[Camera] Frame capture failed: {e}")
            return None, None
    
    def process_images(self, rgb_bgr, ir_gray):
        """Process image pair to compute NDVI"""
        try:
            # Align IR to RGB
            ir_aligned, align_info = self.aligner.align_ir_to_rgb(rgb_bgr, ir_gray)
            
            if ir_aligned is None:
                print("[Processing] Image alignment failed")
                return None
            
            self.stats['successful_alignments'] += 1
            
            # Calculate alignment quality
            quality = alignment_quality(rgb_bgr, ir_aligned)
            
            # Compute NDVI
            ndvi = compute_ndvi_from_rgb_ir(
                rgb_bgr, ir_aligned, 
                normalize=self.config['processing']['normalize_ndvi']
            )
            
            # Calculate NDVI statistics
            ndvi_mean = float(np.nanmean(ndvi))
            
            # Calculate vegetation metrics
            veg_threshold = self.config['processing']['vegetation_threshold']
            veg_mask = ndvi > veg_threshold
            veg_count = int(np.count_nonzero(veg_mask))
            veg_percentage = (veg_count / ndvi.size) * 100.0
            veg_ndvi_mean = float(np.nanmean(ndvi[veg_mask])) if veg_count > 0 else 0.0
            
            return {
                'ndvi_mean': ndvi_mean,
                'ndvi_vegetation_mean': veg_ndvi_mean,
                'vegetation_percentage': veg_percentage,
                'alignment_quality': quality,
                'ndvi_array': ndvi,
                'ir_aligned': ir_aligned
            }
            
        except Exception as e:
            print(f"[Processing] Image processing failed: {e}")
            return None
    
    def get_gps_data(self):
        """Get current GPS coordinates"""
        try:
            # Read GPS data
            self.gps.L76X_Gat_GNRMC()
            
            if self.gps.Status == 1:  # Valid GPS fix
                # Validate coordinates for Vietnam region
                if self.gps.validate_coordinates():
                    self.stats['gps_fixes'] += 1
                    
                    # Calculate accuracy (distance from actual HCMC position)
                    actual_lat = 11.106594492815729
                    actual_lon = 106.61450678350253
                    
                    # Simple distance calculation
                    lat_diff = self.gps.Lat - actual_lat
                    lon_diff = self.gps.Lon - actual_lon
                    accuracy = np.sqrt(lat_diff**2 + lon_diff**2) * 111000  # Rough meters
                    
                    return {
                        'latitude': self.gps.Lat,
                        'longitude': self.gps.Lon,
                        'accuracy': accuracy,
                        'valid': True,
                        'satellites': self.gps.satellites_in_use,
                        'hdop': self.gps.hdop
                    }
            
            return {
                'latitude': 0.0,
                'longitude': 0.0, 
                'accuracy': 999.0,
                'valid': False,
                'satellites': self.gps.satellites_in_use if hasattr(self.gps, 'satellites_in_use') else 0,
                'hdop': 99.9
            }
            
        except Exception as e:
            print(f"[GPS] GPS reading failed: {e}")
            return {'latitude': 0.0, 'longitude': 0.0, 'accuracy': 999.0, 'valid': False}
    
    def upload_to_datahub(self, data):
        """Upload data to Advantech DataHub"""
        try:
            if not self.edge_agent or not self.edge_agent.isConnected:
                print("[DataHub] Not connected, skipping upload")
                return False
            
            # Create edge data
            edge_data = EdgeData()
            edge_data.timestamp = datetime.datetime.now()
            
            # Add tags with current measurements
            device_id = 'NDVIDevice001'
            tags = [
                ('NDVI_Mean', data['ndvi_mean']),
                ('NDVI_Vegetation_Mean', data['ndvi_vegetation_mean']),
                ('Vegetation_Percentage', data['vegetation_percentage']),
                ('GPS_Latitude', data['latitude']),
                ('GPS_Longitude', data['longitude']),
                ('GPS_Accuracy_Meters', data['gps_accuracy']),
                ('Alignment_Quality', data['alignment_quality'])
            ]
            
            for tag_name, value in tags:
                tag = EdgeTag(device_id, tag_name, value)
                edge_data.tagList.append(tag)
            
            # Send data
            self.edge_agent.sendData(edge_data)
            self.stats['datahub_uploads'] += 1
            
            print(f"[DataHub] ✓ Data uploaded - NDVI: {data['ndvi_mean']:.3f}, "
                  f"GPS: ({data['latitude']:.6f}, {data['longitude']:.6f})")
            return True
            
        except Exception as e:
            print(f"[DataHub] Upload failed: {e}")
            return False
    
    def save_local_data(self, data, rgb_bgr, ir_aligned, ndvi_array):
        """Save data locally for backup/analysis"""
        try:
            timestamp = datetime.datetime.now()
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            
            # Create output directory
            output_dir = self.config['output']['directory']
            os.makedirs(output_dir, exist_ok=True)
            
            if self.config['output']['save_images']:
                # Save images
                cv.imwrite(os.path.join(output_dir, f"rgb_{timestamp_str}.png"), rgb_bgr)
                cv.imwrite(os.path.join(output_dir, f"ir_{timestamp_str}.png"), ir_aligned)
                
                # Save NDVI visualization
                ndvi_vis = colorize_ndvi(ndvi_array)
                cv.imwrite(os.path.join(output_dir, f"ndvi_{timestamp_str}.png"), ndvi_vis)
            
            if self.config['output']['save_data']:
                # Save measurement data as JSON
                data_entry = {
                    'timestamp': timestamp.isoformat(),
                    'measurements': data,
                    'statistics': self.stats.copy()
                }
                
                data_file = os.path.join(output_dir, f"data_{timestamp_str}.json")
                with open(data_file, 'w') as f:
                    json.dump(data_entry, f, indent=2)
            
        except Exception as e:
            print(f"[Storage] Local save failed: {e}")
    
    def run_pipeline(self):
        """Main pipeline execution loop"""
        print("\n=== NDVI-GPS-DataHub Pipeline Started ===")
        print(f"[Config] Capture interval: {self.config['timing']['capture_interval_seconds']}s")
        print(f"[Config] Upload interval: {self.config['timing']['upload_interval_seconds']}s")
        
        self.running = True
        last_upload_time = 0
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Capture image pair
                rgb_bgr, ir_gray = self.capture_frame_pair()
                if rgb_bgr is None or ir_gray is None:
                    time.sleep(1)
                    continue
                
                # Process images
                ndvi_result = self.process_images(rgb_bgr, ir_gray)
                if ndvi_result is None:
                    time.sleep(1)
                    continue
                
                # Get GPS data
                gps_data = self.get_gps_data()
                
                # Combine all data
                current_data = {
                    'ndvi_mean': ndvi_result['ndvi_mean'],
                    'ndvi_vegetation_mean': ndvi_result['ndvi_vegetation_mean'],
                    'vegetation_percentage': ndvi_result['vegetation_percentage'],
                    'latitude': gps_data['latitude'],
                    'longitude': gps_data['longitude'],
                    'gps_accuracy': gps_data['accuracy'],
                    'gps_valid': gps_data['valid'],
                    'alignment_quality': ndvi_result['alignment_quality'],
                    'timestamp': datetime.datetime.now()
                }
                
                # Update shared data
                with self.data_lock:
                    self.latest_data.update(current_data)
                
                # Print status
                print(f"\n[Status] Images: {self.stats['images_captured']}, "
                      f"Alignments: {self.stats['successful_alignments']}, "
                      f"GPS fixes: {self.stats['gps_fixes']}, "
                      f"Uploads: {self.stats['datahub_uploads']}")
                print(f"[NDVI] Mean: {current_data['ndvi_mean']:.3f}, "
                      f"Vegetation: {current_data['ndvi_vegetation_mean']:.3f} "
                      f"({current_data['vegetation_percentage']:.1f}%)")
                print(f"[GPS] Lat: {current_data['latitude']:.6f}, "
                      f"Lon: {current_data['longitude']:.6f}, "
                      f"Accuracy: {current_data['gps_accuracy']:.1f}m, "
                      f"Valid: {current_data['gps_valid']}")
                
                # Upload to DataHub if interval reached
                current_time = time.time()
                if (current_time - last_upload_time) >= self.config['timing']['upload_interval_seconds']:
                    if self.upload_to_datahub(current_data):
                        last_upload_time = current_time
                
                # Save local data if enabled
                if self.config['output']['save_local']:
                    self.save_local_data(
                        current_data, rgb_bgr, 
                        ndvi_result['ir_aligned'], 
                        ndvi_result['ndvi_array']
                    )
                
                # Wait for next capture
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.config['timing']['capture_interval_seconds'] - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n[Pipeline] Stopping pipeline...")
        except Exception as e:
            print(f"\n[Pipeline] ✗ Pipeline error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("[Cleanup] Stopping pipeline...")
        self.running = False
        
        try:
            if self.rgb_cam:
                self.rgb_cam.stop()
            if self.ir_cam:
                self.ir_cam.stop()
            if self.gps:
                self.gps.config.cleanup()
            if self.edge_agent and self.edge_agent.isConnected:
                self.edge_agent.disconnect()
        except Exception as e:
            print(f"[Cleanup] Error during cleanup: {e}")
        
        print("[Cleanup] ✓ Pipeline stopped")

def load_config(config_file):
    """Load configuration from JSON file"""
    default_config = {
        "cameras": {
            "rgb_index": 0,
            "ir_index": 1,
            "resolution": [1920, 1080],
            "rgb_exposure_us": 6000,
            "ir_exposure_us": 8000,
            "rgb_gain": 2.0,
            "ir_gain": 4.0,
            "rgb_auto_exposure": False,
            "ir_auto_exposure": False,
            "rgb_auto_wb": True
        },
        "processing": {
            "grid_step": 24,
            "down_for_H": 0.5,
            "normalize_ndvi": True,
            "vegetation_threshold": 0.2
        },
        "datahub": {
            "node_id": "2b3e7a1c-2970-4ecd-81bc-33de9b7eda3d",
            "api_url": "https://api-dccs-ensaas.sa.wise-paas.com/",
            "credential_key": "ae13a5d125a904eaa5468733c53134yl"
        },
        "timing": {
            "capture_interval_seconds": 5,
            "upload_interval_seconds": 30
        },
        "output": {
            "save_local": True,
            "save_images": True,
            "save_data": True,
            "directory": "ndvi_gps_output"
        }
    }
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
            # Merge with defaults
            for section in user_config:
                if section in default_config:
                    default_config[section].update(user_config[section])
        except Exception as e:
            print(f"[Config] Error loading {config_file}: {e}")
            print("[Config] Using default configuration")
    else:
        print(f"[Config] Creating default config file: {config_file}")
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
    
    return default_config

def main():
    parser = argparse.ArgumentParser(description="NDVI-GPS-DataHub Pipeline")
    parser.add_argument("--config", default="pipeline_config.json", 
                       help="Configuration file path")
    parser.add_argument("--test-components", action="store_true",
                       help="Test individual components without running full pipeline")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"[Config] Loaded configuration from {args.config}")
    
    # Create pipeline
    pipeline = NDVIGPSDataLogger(config)
    
    # Test mode - check components individually
    if args.test_components:
        print("\n=== Component Testing Mode ===")
        
        success_count = 0
        
        if pipeline.setup_cameras():
            success_count += 1
        if pipeline.setup_aligner():
            success_count += 1
        if pipeline.setup_gps():
            success_count += 1
        if pipeline.setup_datahub():
            success_count += 1
        
        print(f"\n[Test] {success_count}/4 components initialized successfully")
        
        if success_count == 4:
            print("[Test] All components ready! You can run the full pipeline.")
        else:
            print("[Test] Some components failed. Check the error messages above.")
        
        pipeline.cleanup()
        return
    
    # Full pipeline mode
    print("\n=== Full Pipeline Mode ===")
    
    # Initialize all components
    if not pipeline.setup_cameras():
        print("[Pipeline] ✗ Camera setup failed")
        return
    
    if not pipeline.setup_aligner():
        print("[Pipeline] ✗ Aligner setup failed")
        return
    
    if not pipeline.setup_gps():
        print("[Pipeline] ✗ GPS setup failed")
        return
    
    if not pipeline.setup_datahub():
        print("[Pipeline] ✗ DataHub setup failed")
        return
    
    print("[Pipeline] ✓ All components initialized successfully")
    
    # Wait a moment for DataHub connection
    print("[Pipeline] Waiting for DataHub connection...")
    time.sleep(5)
    
    # Run the main pipeline
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
