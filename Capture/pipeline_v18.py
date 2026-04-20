#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete AIoT Plant Health Assessment System
AIoT 2025 Competition - VGU
Integrates: RGB+NIR capture, alignment, NDVI, AI inference, GPS, WISE-IoT DataHub

*** REVISED (v13) - DataHub SDK Fix ***
1.  Replaced the entire 'DataHubConnector' with the official Advantech SDK
    (wisepaasdatahubedgesdk) from 'main.py'.
2.  Updated DataHub config settings in SystemConfig to match 'main.py'.
3.  The '/capture' route now uses the new SDK-based send_data() method.
4.  Keeps the responsive background worker thread from v12.
"""

import os
import sys
import time
import threading
import json
import uuid
from datetime import datetime
from pathlib import Path
import logging
import socket
import netifaces # Prerequisite: pip install netifaces

# Core libraries
import numpy as np
import cv2 as cv
import pandas as pd
from flask import Flask, render_template_string, request, jsonify, Response
import joblib
from sklearn.impute import SimpleImputer
# import paho.mqtt.client as mqtt # --- REMOVED Paho-MQTT ---

# Hardware modules
try:
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except ImportError:
    print("[Warning] PiCamera2 not available - using simulation mode")
    CAMERA_AVAILABLE = False

# Local modules
from v2_preprocessing import ZhangCorePreprocessor, create_zhang_calibration_data
from L76X import L76X
from ndvi_utils import compute_ndvi_from_rgb_ir, colorize_ndvi
from segmentation_pipeline_kmeans import create_hsv_nir_mask, segment_leaves_from_mask, segment_leaf_structure

# --- NEW: Import Advantech DataHub SDK ---
try:
    from wisepaasdatahubedgesdk.EdgeAgent import EdgeAgent
    import wisepaasdatahubedgesdk.Common.Constants as constant
    from wisepaasdatahubedgesdk.Model.Edge import (
        EdgeAgentOptions, DCCSOptions, EdgeData, EdgeTag, 
        EdgeConfig, NodeConfig, DeviceConfig, AnalogTagConfig
    )
    DATAHUB_SDK_AVAILABLE = True
except ImportError:
    print("[Error] Advantech DataHub SDK not found.")
    print("Please install it: pip install wisepaas-datahub-edgesdk")
    DATAHUB_SDK_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ====================== CONFIGURATION ======================
class SystemConfig:
    # Hardware
    RGB_CAMERA_INDEX = 0
    NIR_CAMERA_INDEX = 1
    IMAGE_SIZE = (1920, 1080)
    FPS = 5
    
    # Processing
    CALIB_MAPS_PATH = "calib_maps"
    MODEL_PATH = "models/rf_model.joblib"
    IMPUTER_PATH = "models/imputer.joblib"
    OUTPUT_DIR = "system_output"
    
    # --- UPDATED: WISE-IoT DataHub Config (from main.py) ---
    DATAHUB_HOST = "https://api-dccs-ensaas.sa.wise-paas.com/" # Correct URL from main.py
    DATAHUB_NODE_ID = "2b3e7a1c-2970-4ecd-81bc-33de9b7eda3d"
    DATAHUB_API_KEY = "ae13a5d125a904eaa5468733c53134yl"
    DATAHUB_DEVICE_ID = "NDVIDevice001" # Device ID as defined in main.py
    
    # GPS
    GPS_UPDATE_INTERVAL = 5.0  # seconds
    
    # Web Interface
    WEB_HOST = "0.0.0.0"
    WEB_PORT = 8001
    STREAM_SIZE = (640, 480) 

    # AI
    AI_CONFIDENCE_THRESHOLD = 0.80

# ====================== SHARED STATE ======================
class SystemState:
    def __init__(self):
        self.lock = threading.Lock()
        
        # --- Live Data (from worker thread) ---
        self.live_aligned_rgb = None
        self.live_aligned_nir = None
        self.live_vi_data = None
        
        # JPEG streams (6 panels)
        self.rgb_jpg = None
        self.nir_jpg = None
        self.ai_prediction_jpg = None
        self.ndvi_heatmap_jpg = None
        self.rdvi_heatmap_jpg = None
        self.segmentation_jpg = None

        # --- Data for UI / DataHub ---
        self.current_ndvi_value = 0.0
        self.vegetation_indices_stats = {}
        self.ai_prediction = None
        self.confidence = 0.0
        self.alignment_quality = 0.0 # New: for datahub

        # GPS data
        self.gps_data = {
            'latitude': 0.0, 'longitude': 0.0, 'altitude': 0.0,
            'status': 'No Fix', 'satellites': 0, 'hdop': 99.9
        }
        
        # System status
        self.last_capture_time = None
        self.processing_status = "Ready"
        self.error_message = ""

state = SystemState()

# ====================== CAMERA MANAGEMENT ======================
class CameraManager:
    # (This class is unchanged from v12)
    def __init__(self):
        self.rgb_cam = None
        self.nir_cam = None
        self.running = False
        
    def initialize_cameras(self):
        if not CAMERA_AVAILABLE:
            logger.warning("Camera hardware not available - using simulation")
            return True
        try:
            self.rgb_cam = Picamera2(camera_num=SystemConfig.RGB_CAMERA_INDEX)
            rgb_config = self.rgb_cam.create_video_configuration(
                main={"size": SystemConfig.IMAGE_SIZE, "format": "RGB888"},
                buffer_count=4
            )
            self.rgb_cam.configure(rgb_config)
            self.rgb_cam.start()
            time.sleep(0.5)
            self.nir_cam = Picamera2(camera_num=SystemConfig.NIR_CAMERA_INDEX)
            nir_config = self.nir_cam.create_video_configuration(
                main={"size": SystemConfig.IMAGE_SIZE, "format": "RGB888"},
                buffer_count=4
            )
            self.nir_cam.configure(nir_config)
            self.nir_cam.start()
            time.sleep(0.5)
            frame_period_us = int(1_000_000 / SystemConfig.FPS)
            self.rgb_cam.set_controls({
                "AeEnable": True, "AwbEnable": True,
                "FrameDurationLimits": (frame_period_us, frame_period_us),
            })
            logger.info("RGB Camera (CAM0) set to AE and AWB")
            self.nir_cam.set_controls({
                "AeEnable": True, "AwbEnable": False,
                "FrameDurationLimits": (frame_period_us, frame_period_us),
            })
            logger.info("NIR Camera (CAM1) set to AE (AWB Disabled)")
            logger.info("Cameras initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def capture_frame_pair(self):
        if not CAMERA_AVAILABLE:
            h, w = SystemConfig.IMAGE_SIZE[1], SystemConfig.IMAGE_SIZE[0]
            rgb_frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            nir_frame = np.random.randint(0, 255, (h, w), dtype=np.uint8)
            return rgb_frame, nir_frame
        try:
            rgb_array = self.rgb_cam.capture_array("main")
            rgb_frame = cv.cvtColor(rgb_array, cv.COLOR_RGB2BGR)
            nir_array = self.nir_cam.capture_array("main")
            nir_frame = cv.cvtColor(nir_array, cv.COLOR_RGB2GRAY)
            return rgb_frame, nir_frame
        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            return None, None
    
    def stop(self):
        if CAMERA_AVAILABLE:
            try:
                if self.rgb_cam: self.rgb_cam.stop()
                if self.nir_cam: self.nir_cam.stop()
            except: pass

# ====================== PROCESSING ENGINE ======================
class ProcessingEngine:
    # (This class is unchanged from v12)
    def __init__(self):
        self.preprocessor = None
        self.rf_model = None
        self.imputer = None
        self.feature_columns = [
            'pixel_count_total',
            'edge_ndvi_mean', 'edge_ndvi_std', 'edge_rdvi_mean', 'edge_rdvi_std',
            'edge_sr_mean', 'edge_sr_std', 'edge_pixel_count',
            'inter_ndvi_mean', 'inter_ndvi_std', 'inter_rdvi_mean', 'inter_rdvi_std',
            'inter_sr_mean', 'inter_sr_std', 'inter_pixel_count'
        ]
        
    def initialize(self):
        try:
            calib_data = create_zhang_calibration_data()
            self.preprocessor = ZhangCorePreprocessor(SystemConfig.CALIB_MAPS_PATH, calib_data)
            if os.path.exists(SystemConfig.MODEL_PATH):
                self.rf_model = joblib.load(SystemConfig.MODEL_PATH)
                logger.info("Random Forest model loaded")
            else:
                logger.warning(f"AI Model not found at {SystemConfig.MODEL_PATH}. AI inference will be disabled.")
            if os.path.exists(SystemConfig.IMPUTER_PATH):
                self.imputer = joblib.load(SystemConfig.IMPUTER_PATH)
                logger.info("Feature imputer loaded")
            else:
                if self.rf_model: 
                    logger.error(f"Imputer not found at {SystemConfig.IMPUTER_PATH}. AI will fail.")
                    self.rf_model = None
            logger.info("Processing engine initialized")
            return True
        except Exception as e:
            logger.error(f"Processing engine initialization failed: {e}")
            return False
    
    def run_ai_inference(self, aligned_rgb, aligned_nir, red_refl_01, nir_refl_01):
        if not self.rf_model or not self.imputer:
            logger.warning("AI analysis skipped: Model not loaded.")
            return aligned_rgb, None, 0.0, 0, None
        try:
            logger.info("Starting AI Segmentation...")
            cleaned_mask, hsv_nir_debug = create_hsv_nir_mask(aligned_rgb, aligned_nir)
            all_leaf_masks, watershed_debug_img = segment_leaves_from_mask(
                cleaned_mask, aligned_rgb, min_leaf_area=500, watershed_min_distance=20
            )
            if not all_leaf_masks:
                logger.warning("AI: No leaves found by segmentation.")
                return watershed_debug_img, None, 0.0, 0, watershed_debug_img
            logger.info(f"AI: Segmentation found {len(all_leaf_masks)} leaf regions. Running model...")
            ai_prediction_image = aligned_rgb.copy()
            all_predictions = []
            all_confidences = []
            for i, (leaf_id, leaf_mask) in enumerate(all_leaf_masks):
                leaf_features = {col: np.nan for col in self.feature_columns}
                leaf_features['pixel_count_total'] = int(np.sum(leaf_mask))
                edge_mask, interveinal_mask, structure_debug = segment_leaf_structure(leaf_mask)
                if edge_mask is not None:
                    edge_mask_bool = edge_mask.astype(bool)
                    if np.any(edge_mask_bool):
                        edge_red_refl = red_refl_01[edge_mask_bool]
                        edge_nir_refl = nir_refl_01[edge_mask_bool]
                        if edge_red_refl.size > 10:
                            vi_data_edge = self.preprocessor.calculate_zhang_vegetation_indices(
                                edge_red_refl, edge_nir_refl)
                            leaf_features.update({
                                'edge_ndvi_mean': vi_data_edge.mean_ndvi, 'edge_ndvi_std': vi_data_edge.std_ndvi,
                                'edge_rdvi_mean': vi_data_edge.mean_rdvi, 'edge_rdvi_std': vi_data_edge.std_rdvi,
                                'edge_sr_mean': vi_data_edge.mean_sr, 'edge_sr_std': vi_data_edge.std_sr,
                                'edge_pixel_count': int(edge_red_refl.size)
                            })
                if interveinal_mask is not None:
                    inter_mask_bool = interveinal_mask.astype(bool)
                    if np.any(inter_mask_bool):
                        inter_red_refl = red_refl_01[inter_mask_bool]
                        inter_nir_refl = nir_refl_01[inter_mask_bool]
                        if inter_red_refl.size > 10:
                            vi_data_inter = self.preprocessor.calculate_zhang_vegetation_indices(
                                inter_red_refl, inter_nir_refl)
                            leaf_features.update({
                                'inter_ndvi_mean': vi_data_inter.mean_ndvi, 'inter_ndvi_std': vi_data_inter.std_ndvi,
                                'inter_rdvi_mean': vi_data_inter.mean_rdvi, 'inter_rdvi_std': vi_data_inter.std_rdvi,
                                'inter_sr_mean': vi_data_inter.mean_sr, 'inter_sr_std': vi_data_inter.std_sr,
                                'inter_pixel_count': int(inter_red_refl.size)
                            })
                feature_vector = {col: leaf_features.get(col, np.nan) for col in self.feature_columns}
                X = pd.DataFrame([feature_vector])[self.feature_columns]
                X_imputed = self.imputer.transform(X)
                leaf_prediction = self.rf_model.predict(X_imputed)[0]
                leaf_probabilities = self.rf_model.predict_proba(X_imputed)[0]
                leaf_confidence = float(np.max(leaf_probabilities))
                all_predictions.append(leaf_prediction)
                all_confidences.append(leaf_confidence)
                class_colors = {0: (0, 255, 0), 1: (0, 0, 255)}
                class_names = {0: 'Healthy', 1: 'Diseased'}
                color = class_colors.get(leaf_prediction, (255, 255, 255))
                contours, _ = cv.findContours(leaf_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(ai_prediction_image, contours, -1, color, 3)
                if contours:
                    M = cv.moments(contours[0])
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        label_text = f"{leaf_id}: {class_names[leaf_prediction]} ({leaf_confidence:.2f})"
                        cv.putText(ai_prediction_image, label_text, (cX-50, cY), 
                                 cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            ai_prediction = None
            confidence = 0.0
            if all_predictions:
                high_confidence_diseased_leaves = [
                    conf for pred, conf in zip(all_predictions, all_confidences) 
                    if pred == 1 and conf >= SystemConfig.AI_CONFIDENCE_THRESHOLD
                ]
                if high_confidence_diseased_leaves:
                    ai_prediction = 'Diseased'
                    confidence = float(np.mean(high_confidence_diseased_leaves))
                else:
                    ai_prediction = 'Healthy'
                    healthy_confidences = [conf for pred, conf in zip(all_predictions, all_confidences) if pred == 0]
                    if healthy_confidences:
                        confidence = float(np.mean(healthy_confidences))
                    else:
                        confidence = float(np.mean(all_confidences))
            logger.info(f"AI analysis complete: {ai_prediction} (confidence: {confidence:.3f})")
            return ai_prediction_image, ai_prediction, confidence, len(all_leaf_masks), watershed_debug_img
        except Exception as e:
            logger.error(f"AI inference failed: {e}")
            return aligned_rgb, None, 0.0, 0, hsv_nir_debug

# ====================== GPS MANAGER ======================
class GPSManager:
    # (This class is unchanged from v12)
    def __init__(self):
        self.gps = None
        self.running = False
        
    def initialize(self):
        try:
            self.gps = L76X.L76X()
            logger.info("GPS module initialized and configured to 115200 baud.")
            return True
        except Exception as e:
            logger.warning(f"GPS initialization failed: {e}")
            return False
    
    def update_loop(self):
        while self.running:
            try:
                if self.gps:
                    self.gps.L76X_Gat_GNRMC() 
                    with state.lock:
                        state.gps_data.update({
                            'latitude': self.gps.Lat,
                            'longitude': self.gps.Lon,
                            'status': 'Fix' if self.gps.Status else 'No Fix',
                            'satellites': self.gps.satellites_in_use,
                            'hdop': self.gps.hdop
                        })
                time.sleep(SystemConfig.GPS_UPDATE_INTERVAL) 
            except Exception as e:
                logger.warning(f"GPS update error: {e}")
                time.sleep(5)
    
    def start(self):
        if self.initialize():
            self.running = True
            gps_thread = threading.Thread(target=self.update_loop, daemon=True)
            gps_thread.start()
            logger.info("GPS monitoring started")

# ====================== DATAHUB CONNECTOR (FROM MAIN.PY) ======================
class DataHubConnector:
    def __init__(self):
        self.edge_agent = None
        self.connected = False
        
    def initialize(self):
        """Initialize connection to Advantech DataHub using the SDK"""
        if not DATAHUB_SDK_AVAILABLE:
            logger.error("DataHub SDK not installed. Cannot initialize connector.")
            return False
            
        print("[DataHub] Connecting to Advantech DataHub...")
        try:
            edge_options = EdgeAgentOptions(nodeId=SystemConfig.DATAHUB_NODE_ID)
            edge_options.connectType = constant.ConnectType['DCCS']
            dccs_options = DCCSOptions(
                apiUrl=SystemConfig.DATAHUB_HOST,
                credentialKey=SystemConfig.DATAHUB_API_KEY
            )
            edge_options.DCCS = dccs_options
            
            self.edge_agent = EdgeAgent(edge_options)
            self.edge_agent.on_connected = self._on_datahub_connected
            self.edge_agent.on_disconnected = self._on_datahub_disconnected
            
            self.edge_agent.connect()
            print("[DataHub] ✓ Connection initiated")
            return True
            
        except Exception as e:
            print(f"[DataHub] ✗ Connection failed: {e}")
            return False
    
    def _on_datahub_connected(self, agent, connected):
        if connected:
            self.connected = True
            print("[DataHub] ✓ Connected to DataHub")
            # Upload device configuration
            config = self._generate_device_config()
            self.edge_agent.uploadConfig(
                action=constant.ActionType['Create'], 
                edgeConfig=config
            )
        else:
            self.connected = False
            print("[DataHub] ✗ Connection failed")
    
    def _on_datahub_disconnected(self, agent, disconnected):
        if disconnected:
            self.connected = False
            print("[DataHub] ✗ Disconnected from DataHub")
    
    def _generate_device_config(self):
        """Generate device configuration for DataHub"""
        config = EdgeConfig()
        device_config = DeviceConfig(
            id=SystemConfig.DATAHUB_DEVICE_ID,
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
            ('Alignment_Quality', 'Image alignment quality (NCC)', -1.0, 1.0)
            # Note: main.py had 'GPS_Accuracy_Meters', we can add it if needed
        ]
        
        for tag_name, description, span_low, span_high in tags:
            analog_tag = AnalogTagConfig(
                name=tag_name, description=description, readOnly=True, arraySize=0,
                spanHigh=span_high, spanLow=span_low, engineerUnit='',
                integerDisplayFormat=4, fractionDisplayFormat=4
            )
            device_config.analogTagList.append(analog_tag)
        
        config.node.deviceList.append(device_config)
        return config
    
    def send_data(self, data):
        """Upload data to Advantech DataHub"""
        if not self.edge_agent or not self.connected:
            logger.warning("DataHub not connected, skipping upload")
            return False
            
        try:
            edge_data = EdgeData()
            edge_data.timestamp = datetime.now()
            
            device_id = SystemConfig.DATAHUB_DEVICE_ID
            tags_to_send = [
                ('NDVI_Mean', data.get('ndvi_mean', 0.0)),
                ('NDVI_Vegetation_Mean', data.get('ndvi_vegetation_mean', 0.0)),
                ('Vegetation_Percentage', data.get('vegetation_percentage', 0.0)),
                ('GPS_Latitude', data.get('latitude', 0.0)),
                ('GPS_Longitude', data.get('longitude', 0.0)),
                ('Alignment_Quality', data.get('alignment_quality', 0.0))
            ]
            
            for tag_name, value in tags_to_send:
                # Ensure value is a basic type (float/int), not numpy type
                tag = EdgeTag(device_id, tag_name, float(value))
                edge_data.tagList.append(tag)
            
            self.edge_agent.sendData(edge_data)
            logger.info(f"[DataHub] ✓ Data uploaded - NDVI: {data.get('ndvi_mean', 0.0):.3f}, "
                      f"GPS: ({data.get('latitude', 0.0):.6f}, {data.get('longitude', 0.0):.6f})")
            return True
            
        except Exception as e:
            print(f"[DataHub] Upload failed: {e}")
            return False

# ====================== UTILITY FUNCTIONS ======================
def get_local_ip():
    # (Unchanged)
    try:
        for interface in netifaces.interfaces():
            if interface in ('eth0', 'wlan0'):
                addresses = netifaces.ifaddresses(interface)
                if netifaces.AF_INET in addresses:
                    return addresses[netifaces.AF_INET][0]['addr']
    except Exception: pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
        s.close()
        return IP
    except Exception:
        return '127.0.0.1'

def encode_stream_image(frame, stream_name=""):
    # (Unchanged)
    if frame is None:
        frame = np.zeros((SystemConfig.STREAM_SIZE[1], SystemConfig.STREAM_SIZE[0], 3), dtype=np.uint8)
        cv.putText(frame, f"{stream_name.upper()} unavailable", (100, 240), 
                 cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    resized_frame = cv.resize(frame, SystemConfig.STREAM_SIZE)
    _, buffer = cv.imencode('.jpg', resized_frame, [int(cv.IMWRITE_JPEG_QUALITY), 75])
    return buffer.tobytes()

# ====================== BACKGROUND PROCESSING WORKER ======================
def processing_worker_thread(camera_manager, preprocessor):
    """
    Background thread to continuously capture, align, and process
    the non-AI data (RGB, NIR, VIs, Mask).
    """
    logger.info("🚀 Processing worker thread started.")
    
    while True:
        try:
            # 1. Capture
            rgb_frame, nir_frame = camera_manager.capture_frame_pair()
            if rgb_frame is None or nir_frame is None:
                logger.warning("Worker: Frame capture failed, skipping cycle.")
                time.sleep(0.5)
                continue

            # 2. Align
            aligned_rgb, aligned_nir = preprocessor.rectify_image_pair(rgb_frame, nir_frame)
            if aligned_rgb is None or aligned_nir is None:
                logger.warning("Worker: Alignment failed, skipping cycle.")
                time.sleep(0.5)
                continue
            
            # 2b. Get Alignment Quality
            quality = alignment_quality(aligned_rgb, aligned_nir)

            # 3. Calculate VIs
            red_refl_01 = (aligned_rgb[:,:,2].astype(np.float32) / 255.0)
            nir_refl_01 = (aligned_nir.astype(np.float32) / 255.0)
            vi_data = preprocessor.calculate_zhang_vegetation_indices(
                red_refl_01, nir_refl_01
            )
            
            # 4. Create simple mask
            cleaned_mask, hsv_nir_debug = create_hsv_nir_mask(aligned_rgb, aligned_nir)

            # 5. Create heatmap visualizations
            ndvi_heatmap = colorize_ndvi(vi_data.ndvi)
            rdvi_heatmap = colorize_ndvi(vi_data.rdvi) 

            # 6. Prepare data for UI stats
            vi_stats = {
                'ndvi_mean': vi_data.mean_ndvi, 'ndvi_std': vi_data.std_ndvi,
                'rdvi_mean': vi_data.mean_rdvi, 'rdvi_std': vi_data.std_rdvi,
                'sr_mean': vi_data.mean_sr, 'sr_std': vi_data.std_sr,
                'vegetation_percentage': float(np.count_nonzero(cleaned_mask) / cleaned_mask.size) * 100.0
            }
            
            # 7. Update shared state with all live data
            with state.lock:
                state.live_aligned_rgb = aligned_rgb
                state.live_aligned_nir = aligned_nir
                state.live_vi_data = vi_data
                state.current_ndvi_value = vi_data.mean_ndvi
                state.vegetation_indices_stats = vi_stats
                state.alignment_quality = float(quality)
                state.processing_status = "Live"
                
                state.rgb_jpg = encode_stream_image(aligned_rgb, "RGB")
                state.nir_jpg = encode_stream_image(cv.cvtColor(aligned_nir, cv.COLOR_GRAY2BGR), "NIR")
                state.ndvi_heatmap_jpg = encode_stream_image(ndvi_heatmap, "NDVI")
                state.rdvi_heatmap_jpg = encode_stream_image(rdvi_heatmap, "RDVI")
                state.segmentation_jpg = encode_stream_image(hsv_nir_debug, "Segment")
                
                if state.ai_prediction_jpg is None:
                    state.ai_prediction_jpg = encode_stream_image(None, "AI (Not Run)")
            
            time.sleep(0.05) 

        except Exception as e:
            logger.error(f"Error in processing worker thread: {e}")
            time.sleep(1)

# ====================== WEB INTERFACE (HTML Unchanged) ======================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>AIoT Plant Health Monitor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header {
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            text-align: center;
        }
        .header h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #7f8c8d;
            font-size: 1.1em;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .camera-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
            margin-bottom: 15px;
        }
        .camera-view {
            background: #f8f9fa;
            border-radius: 10px;
            overflow: hidden;
            border: 2px solid #e9ecef;
        }
        .camera-view img {
            width: 100%;
            height: 250px;
            object-fit: contain;
            display: block;
            background: #333;
        }
        .camera-label {
            background: #2c3e50;
            color: white;
            padding: 10px;
            text-align: center;
            font-weight: 600;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        .btn {
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            margin: 0 10px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4);
        }
        .btn:active {
            transform: translateY(0);
        }
        .btn.inference {
            background: linear-gradient(135deg, #6f42c1, #e83e8c);
            box-shadow: 0 4px 15px rgba(111, 66, 193, 0.3);
        }
        .btn.inference:hover {
            box-shadow: 0 6px 20px rgba(111, 66, 193, 0.4);
        }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .status-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #007bff;
        }
        .status-label {
            font-size: 0.9em;
            color: #6c757d;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .status-value {
            font-size: 1.4em;
            font-weight: 700;
            color: #2c3e50;
        }
        .health-status {
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            font-size: 1.5em;
            font-weight: 700;
        }
        .health-healthy { background: #d4edda; color: #155724; }
        .health-warning { background: #fff3cd; color: #856404; }
        .health-danger { background: #f8d7da; color: #721c24; }
        .results-section {
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .results-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 15px;
        }
        .vi-list { list-style: none; }
        .vi-list li {
            padding: 8px 0;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
        }
        .message {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }
        @media (max-width: 1200px) { .camera-grid { grid-template-columns: 1fr 1fr; } }
        @media (max-width: 768px) {
            .grid, .camera-grid, .results-grid { grid-template-columns: 1fr; }
            .header h1 { font-size: 2em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌱 AIoT Plant Health Monitor</h1>
            <p class="subtitle">Real-time NDVI Analysis & Disease Detection | VGU AIoT 2025</p>
        </div>

        <div class="camera-grid">
            <div class="camera-view">
                <div class="camera-label">📸 LIVE Aligned RGB</div>
                <img src="/stream/rgb" alt="Aligned RGB Feed" id="rgb-stream">
            </div>
            <div class="camera-view">
                <div class="camera-label">🔴 LIVE Aligned NIR</div>
                <img src="/stream/nir" alt="Aligned NIR Feed" id="nir-stream">
            </div>
            <div class="camera-view">
                <div class="camera-label">🔬 AI Prediction (On-Demand)</div>
                <img src="/stream/ai_prediction" alt="AI Prediction Result" id="ai_prediction-stream">
            </div>
        </div>
        
        <div class="camera-grid">
            <div class="camera-view">
                <div class="camera-label">🟢 LIVE NDVI Heatmap</div>
                <img src="/stream/ndvi" alt="NDVI Heatmap" id="ndvi-stream">
            </div>
            <div class="camera-view">
                <div class="camera-label">🟡 LIVE RDVI Heatmap</div>
                <img src="/stream/rdvi" alt="RDVI Heatmap" id="rdvi-stream">
            </div>
            <div class="camera-view">
                <div class="camera-label">🎨 LIVE Segmentation Mask</div>
                <img src="/stream/segmentation" alt="Segmentation Output" id="segmentation-stream">
            </div>
        </div>
        <div class="controls">
            <button class="btn" onclick="captureAndSend()" id="capture-btn">
                ☁️ Capture & Send to DataHub
            </button>
            <button class="btn inference" onclick="runInference()" id="inference-btn">
                🤖 Run AI Analysis
            </button>
        </div>

        <div class="grid">
            <div class="card">
                <h3>📊 System Status</h3>
                <div class="status-grid">
                    <div class="status-item">
                        <div class="status-label">Processing</div>
                        <div class="status-value" id="processing-status">Initializing...</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">Last Capture</div>
                        <div class="status-value" id="last-capture">None</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">Live NDVI</div>
                        <div class="status-value" id="ndvi-value">--</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">AI Confidence</div>
                        <div class="status-value" id="confidence">--</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>🛰️ GPS Status</h3>
                <div class="status-grid">
                    <div class="status-item">
                        <div class="status-label">Status</div>
                        <div class="status-value" id="gps-status">No Fix</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">Latitude</div>
                        <div class="status-value" id="gps-lat">--</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">Longitude</div>
                        <div class="status-value" id="gps-lon">--</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">Satellites</div>
                        <div class="status-value" id="gps-sats">--</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="health-status" id="health-status" style="display: none;">
            Plant Status: Unknown
        </div>

        <div class="results-section" id="results-section" style="display: none;">
            <h3>📈 Analysis Results</h3>
            <div class="results-grid">
                <div>
                    <h4>Live Vegetation Indices</h4>
                    <ul class="vi-list" id="vi-list">
                        </ul>
                </div>
                <div>
                    <h4>AI Prediction</h4>
                    <div id="ai-results">
                        <p>Press 'Run AI Analysis' to get results.</p>
                    </div>
                </div>
            </div>
        </div>

        <div class="message" id="message" style="display: none;">
            System ready for plant health monitoring
        </div>
    </div>

    <script>
        async function updateStatus() {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                
                document.getElementById('processing-status').textContent = data.processing_status;
                document.getElementById('ndvi-value').textContent = 
                    data.current_ndvi !== undefined ? data.current_ndvi.toFixed(4) : '--';
                document.getElementById('confidence').textContent = 
                    data.confidence ? (data.confidence * 100).toFixed(1) + '%' : '--';
                
                if (data.last_capture_time) {
                    document.getElementById('last-capture').textContent = 
                        new Date(data.last_capture_time).toLocaleTimeString();
                }
                
                document.getElementById('gps-status').textContent = data.gps_status;
                document.getElementById('gps-lat').textContent = data.gps_lat ? data.gps_lat.toFixed(6) : '--';
                document.getElementById('gps-lon').textContent = data.gps_lon ? data.gps_lon.toFixed(6) : '--';
                document.getElementById('gps-sats').textContent = data.gps_satellites || '--';
                
                const healthDiv = document.getElementById('health-status');
                if (data.ai_prediction) {
                    healthDiv.style.display = 'block';
                    healthDiv.textContent = `Plant Status: ${data.ai_prediction}`;
                    healthDiv.className = 'health-status'; // Reset classes
                    if (data.ai_prediction === 'Healthy') {
                        healthDiv.classList.add('health-healthy');
                    } else if (data.ai_prediction === 'Diseased') { 
                        healthDiv.classList.add('health-danger'); 
                    } else {
                        healthDiv.classList.add('health-warning');
                    }
                } else {
                    healthDiv.style.display = 'none';
                }
                
                const viList = document.getElementById('vi-list');
                if (data.vegetation_indices_stats) {
                    viList.innerHTML = '';
                    for (const [key, value] of Object.entries(data.vegetation_indices_stats)) {
                        const li = document.createElement('li');
                        li.innerHTML = `<span>${key.toUpperCase().replace('_', ' ')}</span><span>${value.toFixed(4)}</span>`;
                        viList.appendChild(li);
                    }
                    document.getElementById('results-section').style.display = 'block';
                }
                
                const aiResults = document.getElementById('ai-results');
                if (data.ai_prediction) {
                    aiResults.innerHTML = `
                        <p><strong>Prediction:</strong> ${data.ai_prediction}</p>
                        <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%</p>
                    `;
                }
                
                if (data.error_message) {
                    showMessage(data.error_message, 'error');
                }
                
            } catch (error) {
                console.error('Status update failed:', error);
            }
        }
        
        async function captureAndSend() {
            const btn = document.getElementById('capture-btn');
            btn.disabled = true;
            btn.textContent = '☁️ Sending...';
            
            try {
                const response = await fetch('/capture', { method: 'POST' });
                const data = await response.json();
                
                if (data.success) {
                    showMessage(data.message || 'Data sent to DataHub!', 'success');
                } else {
                    showMessage(data.error || 'DataHub send failed', 'error');
                }
            } catch (error) {
                showMessage('Request failed: ' + error.message, 'error');
            } finally {
                btn.disabled = false;
                btn.textContent = '☁️ Capture & Send to DataHub';
            }
        }
        
        async function runInference() {
            const btn = document.getElementById('inference-btn');
            btn.disabled = true;
            btn.textContent = '🤖 Analyzing...';
            
            try {
                const response = await fetch('/inference', { method: 'POST' });
                const data = await response.json();
                
                if (data.success) {
                    showMessage('AI analysis completed!', 'success');
                    // Force refresh ONLY the AI-related streams
                    refreshStreams(['ai_prediction', 'segmentation']);
                } else {
                    showMessage(data.error || 'AI analysis failed', 'error');
                }
            } catch (error) {
                    showMessage('AI analysis request failed: ' + error.message, 'error');
            } finally {
                btn.disabled = false;
                btn.textContent = '🤖 Run AI Analysis';
            }
        }
        
        function refreshStreams(stream_ids) {
            const timestamp = new Date().getTime();
            stream_ids.forEach(function(stream) {
                const elementId = stream + '-stream';
                const imgElement = document.getElementById(elementId);
                
                if (imgElement) {
                    imgElement.src = ''; 
                    setTimeout(function() {
                        imgElement.src = '/stream/' + stream + '?' + timestamp;
                    }, 50);
                }
            });
        }
        
        function showMessage(text, type = 'info') {
            const messageDiv = document.getElementById('message');
            messageDiv.textContent = text;
            messageDiv.style.display = 'block';
            
            if (type === 'success') {
                messageDiv.style.background = '#d4edda';
                messageDiv.style.borderColor = '#28a745';
                messageDiv.style.color = '#155724';
            } else if (type === 'error') {
                messageDiv.style.background = '#f8d7da';
                messageDiv.style.borderColor = '#dc3545';
                messageDiv.style.color = '#721c24';
            } else {
                messageDiv.style.background = '#e3f2fd';
                messageDiv.style.borderColor = '#2196f3';
                messageDiv.style.color = '#0d47a1';
            }
            
            setTimeout(() => {
                messageDiv.style.display = 'none';
            }, 5000);
        }
        
        setInterval(updateStatus, 2000);
        
        window.onload = () => {
            updateStatus();
            document.getElementById('ai-results').innerHTML = `<p>Press 'Run AI Analysis' to get results.</p>`;
        };
    </script>
</body>
</html>
"""

def create_web_app():
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)
    
    @app.route('/stream/<stream_type>')
    def video_stream(stream_type):
        def generate_stream():
            while True:
                frame_data = None
                with state.lock:
                    if stream_type == 'rgb': frame_data = state.rgb_jpg
                    elif stream_type == 'nir': frame_data = state.nir_jpg
                    elif stream_type == 'ai_prediction': frame_data = state.ai_prediction_jpg
                    elif stream_type == 'ndvi': frame_data = state.ndvi_heatmap_jpg
                    elif stream_type == 'rdvi': frame_data = state.rdvi_heatmap_jpg
                    elif stream_type == 'segmentation': frame_data = state.segmentation_jpg
                
                if frame_data is None:
                    frame_data = encode_stream_image(None, f"{stream_type} initializing...")
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame_data)).encode() + b'\r\n\r\n' +
                       frame_data + b'\r\n')
                time.sleep(0.05)
        
        return Response(generate_stream(), 
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    
    @app.route('/status')
    def get_status():
        with state.lock:
            return jsonify({
                'processing_status': state.processing_status,
                'current_ndvi': state.current_ndvi_value,
                'vegetation_indices_stats': state.vegetation_indices_stats,
                'ai_prediction': state.ai_prediction,
                'confidence': state.confidence,
                'last_capture_time': state.last_capture_time.isoformat() if state.last_capture_time else None,
                'gps_status': state.gps_data['status'],
                'gps_lat': state.gps_data['latitude'],
                'gps_lon': state.gps_data['longitude'],
                'gps_satellites': state.gps_data['satellites'],
                'error_message': state.error_message
            })
    
    # --- *** MODIFIED: /capture uses the NEW DataHubConnector *** ---
    @app.route('/capture', methods=['POST'])
    def capture_and_send():
        try:
            logger.info("Capture button pressed: Sending latest data to DataHub...")
            
            with state.lock:
                if state.live_vi_data is None:
                    return jsonify({'success': False, 'error': 'No live data available yet. Worker is initializing.'})
                
                # Create the data payload for the SDK
                payload = {
                    'ndvi_mean': state.current_ndvi_value,
                    'ndvi_vegetation_mean': state.vegetation_indices_stats.get('ndvi_mean', 0.0), # Assuming veg_mean is what we want
                    'vegetation_percentage': state.vegetation_indices_stats.get('vegetation_percentage', 0.0),
                    'latitude': state.gps_data['latitude'],
                    'longitude': state.gps_data['longitude'],
                    'alignment_quality': state.alignment_quality,
                    'timestamp': datetime.now()
                }
                state.last_capture_time = datetime.now()
            
            # Send to DataHub using the SDK
            datahub_success = datahub_connector.send_data(payload)
            
            if datahub_success:
                return jsonify({'success': True, 'message': 'Latest data sent to DataHub!'})
            else:
                return jsonify({'success': False, 'error': 'DataHub send failed. Check connection.'})
                
        except Exception as e:
            logger.error(f"DataHub endpoint error: {e}")
            return jsonify({'success': False, 'error': str(e)})
    
    # --- *** MODIFIED: /inference (Unchanged, but now works) *** ---
    @app.route('/inference', methods=['POST'])
    def run_inference():
        try:
            logger.info("Inference button pressed: Starting heavy AI processing...")
            with state.lock:
                if state.live_aligned_rgb is None:
                    return jsonify({'success': False, 'error': 'No live image available to analyze.'})
                rgb_to_process = state.live_aligned_rgb.copy()
                nir_to_process = state.live_aligned_nir.copy()
                red_refl_01 = (rgb_to_process[:,:,2].astype(np.float32) / 255.0)
                nir_refl_01 = (nir_to_process.astype(np.float32) / 255.0)
            
            ai_image, prediction, confidence, leaves_found, seg_image = \
                processing_engine.run_ai_inference(
                    rgb_to_process, nir_to_process, red_refl_01, nir_refl_01
                )
            
            with state.lock:
                state.ai_prediction = prediction
                state.confidence = confidence
                state.vegetation_indices_stats['leaves_detected'] = leaves_found
                state.ai_prediction_jpg = encode_stream_image(ai_image, "AI Result")
                state.segmentation_jpg = encode_stream_image(seg_image, "AI Segment")
            
            logger.info("AI inference complete.")
            return jsonify({'success': True, 'message': 'AI analysis completed'})
                
        except Exception as e:
            logger.error(f"Inference endpoint error: {e}")
            return jsonify({'success': False, 'error': str(e)})
    
    return app

# ====================== MAIN SYSTEM ======================
def main():
    global camera_manager, processing_engine, gps_manager, datahub_connector
    
    print("=" * 60)
    print("🌱 AIoT Plant Health Monitor System (v13 - SDK Merged)")
    print("VGU AIoT 2025 Competition")
    print("=" * 60)
    
    os.makedirs(SystemConfig.OUTPUT_DIR, exist_ok=True)
    
    # Check for SDK early
    if not DATAHUB_SDK_AVAILABLE:
        print("Please fix SDK installation and restart.")
        return

    # Initialize components
    camera_manager = CameraManager()
    processing_engine = ProcessingEngine()
    gps_manager = GPSManager()
    datahub_connector = DataHubConnector() # Uses new SDK class
    
    print("\n🔧 Initializing components...")
    
    if not camera_manager.initialize_cameras():
        logger.warning("⚠️ Camera initialization failed - using simulation mode")
    else:
        logger.info("✅ Cameras initialized")
    
    if not processing_engine.initialize():
        logger.error("❌ Processing engine initialization failed")
        return
    else:
        logger.info("✅ Processing engine initialized")
    
    gps_manager.start()
    
    # Initialize the new SDK-based connector
    if datahub_connector.initialize():
        logger.info("✅ DataHub connection sequence initiated...")
    else:
        logger.warning("⚠️ DataHub initialization failed - continuing without cloud sync")
    
    # Start the background processing thread
    worker_thread = threading.Thread(
        target=processing_worker_thread,
        args=(camera_manager, processing_engine.preprocessor),
        daemon=True
    )
    worker_thread.start()
    
    local_ip = get_local_ip()
    hostname = socket.gethostname()

    print("\n🌐 Starting web interface...")
    print(f"   Access via IP: http://{local_ip}:{SystemConfig.WEB_PORT}/")
    print(f"   Access via Hostname: http://{hostname}.local:{SystemConfig.WEB_PORT}/")
    print("\n🚀 **Access the Web UI Now:**")
    print(f"   ➡️  http://{hostname}.local:{SystemConfig.WEB_PORT}/")
    print("=" * 60)
    
    try:
        app = create_web_app()
        app.run(host=SystemConfig.WEB_HOST, port=SystemConfig.WEB_PORT, 
                debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down system...")
    finally:
        camera_manager.stop()
        if datahub_connector.edge_agent:
            datahub_connector.edge_agent.disconnect()

if __name__ == '__main__':
    main()
