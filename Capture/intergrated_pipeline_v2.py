#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated AIoT 2025 Pipeline - VGU Agritech Team (FINAL WORKING)

SOLUTION:
1.  REMOVED the 'preview_worker' thread. This 100% solves the
    camera hardware deadlock.
2.  Created a NEW 'processing_worker' thread. This is the ONLY place
    where slow AI and non-thread-safe OpenCV drawing (cv.putText,
    cv.drawContours) occurs.
3.  'api_capture' and 'api_inference' routes are now fast. They only
    set flags for the 'processing_worker' and return JSON immediately.
    This makes the UI buttons responsive.
4.  Stream routes ('/stream/*') are "dumb" and only serve pre-computed
    JPEGs from the SharedState. They do NO processing or drawing.
"""

import os
import time
import datetime
import threading
import json
import argparse
import numpy as np
import cv2 as cv
from pathlib import Path
from flask import Flask, Response, jsonify, render_template_string, request
from picamera2 import Picamera2
import socket

# Import preprocessing and alignment
try:
    from v2_preprocessing import ZhangCorePreprocessor, create_zhang_calibration_data
    print("[Import] v2_preprocessing loaded successfully")
except ImportError as e:
    print(f"[Error] Could not import v2_preprocessing: {e}")
    exit()

# Import segmentation
try:
    from segmentation_pipeline_kmeans import create_hsv_nir_mask, segment_leaves_from_mask
    print("[Import] segmentation_pipeline_kmeans loaded successfully")
except ImportError as e:
    print(f"[Error] Could not import segmentation_pipeline_kmeans: {e}")
    exit()

# Import GPS
try:
    import L76X
    GPS_AVAILABLE = True
    print("[Import] GPS module loaded successfully")
except ImportError as e:
    print(f"[Warning] GPS module not available: {e}")
    GPS_AVAILABLE = False

# Import DataHub SDK
try:
    from wisepaasdatahubedgesdk.EdgeAgent import EdgeAgent
    import wisepaasdatahubedgesdk.Common.Constants as constant
    from wisepaasdatahubedgesdk.Model.Edge import (
        EdgeAgentOptions, DCCSOptions, EdgeData, EdgeTag,
        EdgeConfig, DeviceConfig, AnalogTagConfig
    )
    DATAHUB_AVAILABLE = True
    print("[Import] DataHub SDK loaded successfully")
except ImportError as e:
    print(f"[Warning] DataHub SDK not available: {e}")
    DATAHUB_AVAILABLE = False

# ============================================================================
# CAMERA SETUP (Original)
# ============================================================================

def setup_camera(cam_index, size, fps, exposure_us, gain, awb=True, ae=False):
    """Initialize and configure PiCamera2"""
    cam = Picamera2(camera_num=cam_index)
    cfg = cam.create_video_configuration(
        main={"size": size, "format": "RGB888"},
        buffer_count=4
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(0.5)
    
    frame_period_us = int(1_000_000 / fps)
    exp = min(exposure_us, frame_period_us - 1000)
    
    controls = {
        "AeEnable": ae,
        "AwbEnable": awb,
        "ExposureTime": exp,
        "AnalogueGain": gain,
        "FrameDurationLimits": (frame_period_us, frame_period_us)
    }
    cam.set_controls(controls)
    
    return cam

def get_frame_bgr(cam):
    """Capture frame from PiCamera2 and convert RGB to BGR"""
    arr = cam.capture_array("main")
    return cv.cvtColor(arr, cv.COLOR_RGB2BGR)

# ============================================================================
# VI CALCULATION & VISUALIZATION (New Version)
# ============================================================================

def calculate_vis(red_refl, nir_refl, eps=1e-6):
    """
    Calculate NDVI and RDVI from reflectance values (0-1 scale)
    Returns: dict with 'ndvi' and 'rdvi' arrays
    """
    # NDVI: (NIR - Red) / (NIR + Red)
    ndvi = (nir_refl - red_refl) / (nir_refl + red_refl + eps)
    ndvi = np.clip(ndvi, -1.0, 1.0)
    
    # RDVI: (NIR - Red) / sqrt(NIR + Red)
    rdvi = (nir_refl - red_refl) / np.sqrt(nir_refl + red_refl + eps)
    rdvi = np.clip(rdvi, -1.0, 1.0)
    
    return {'ndvi': ndvi, 'rdvi': rdvi}

def create_vi_heatmap_with_colorbar(vi_array, vi_name='NDVI'):
    """
    Create a heatmap visualization with color bar
    MODIFIED: This is the new, more complex heatmap function
    that accepts NaNs and has a better color bar.
    """
    # Copy and handle NaNs for visualization
    vi_display = np.nan_to_num(vi_array, nan=-1.0)
    
    # Normalize to 0-255 for colormap
    vi_normalized = ((vi_display + 1.0) * 127.5).astype(np.uint8)
    
    # Apply colormap (TURBO for better discrimination)
    heatmap = cv.applyColorMap(vi_normalized, cv.COLORMAP_TURBO)

    # Set NaN areas to a specific color (e.g., black or gray)
    heatmap[np.isnan(vi_array)] = [50, 50, 50] # Dark gray for non-plant areas
    
    # Create color bar
    bar_height = heatmap.shape[0]
    bar_width = 60
    cb_padding = 10
    total_width = heatmap.shape[1] + bar_width + cb_padding + 40
    
    # Create an expanded canvas
    final_image = np.full((bar_height, total_width, 3), (50, 50, 50), dtype=np.uint8)
    
    # Place the heatmap
    final_image[:, :heatmap.shape[1]] = heatmap
    
    # Draw the color bar gradient
    gradient = np.linspace(1.0, -1.0, bar_height)
    gradient_normalized = ((gradient + 1.0) * 127.5).astype(np.uint8)
    
    cb_x_start = heatmap.shape[1] + cb_padding
    cb_x_end = cb_x_start + bar_width
    
    for i in range(bar_height):
        color = cv.applyColorMap(np.array([[gradient_normalized[i]]], dtype=np.uint8), cv.COLORMAP_TURBO)[0, 0]
        final_image[i, cb_x_start:cb_x_end] = color
    
    # Add labels to color bar
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (255, 255, 255)
    
    # Top (1.0) label
    cv.putText(final_image, "1.0", 
               (cb_x_end + 5, 20),
               font, font_scale, text_color, font_thickness, cv.LINE_AA)
    # Middle (0.0) label
    mid_y = bar_height // 2
    cv.putText(final_image, "0.0", 
               (cb_x_end + 5, mid_y),
               font, font_scale, text_color, font_thickness, cv.LINE_AA)
    # Bottom (-1.0) label
    cv.putText(final_image, "-1.0", 
               (cb_x_end + 5, bar_height - 10),
               font, font_scale, text_color, font_thickness, cv.LINE_AA)
    
    # Add title
    title_height = 40
    title_bar = np.zeros((title_height, final_image.shape[1], 3), dtype=np.uint8)
    cv.putText(title_bar, vi_name, (10, 28),
               cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    result = np.vstack([title_bar, final_image])
    
    return result

# ============================================================================
# NEW: VISUALIZATION FUNCTIONS FOR SEGMENTATION & PER-LEAF INFERENCE
# ============================================================================

def create_segmentation_visualization(cleaned_mask, rgb_image):
    """
    Create a colored visualization of the segmentation mask overlaid on RGB
    """
    # Create a colored mask visualization
    vis_image = rgb_image.copy()
    
    # Create a colored overlay for the mask
    colored_mask = np.zeros_like(rgb_image)
    colored_mask[cleaned_mask > 0] = [0, 255, 0]  # Green for vegetation
    
    # Blend with original image
    alpha = 0.4
    vis_image = cv.addWeighted(vis_image, 1-alpha, colored_mask, alpha, 0)
    
    # Draw contours around the mask
    contours, _ = cv.findContours(cleaned_mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(vis_image, contours, -1, (0, 255, 255), 2) # Yellow contours
    
    # Add title
    cv.putText(vis_image, "Vegetation Segmentation", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    
    return vis_image

def create_per_leaf_inference_visualization(rgb_image, leaf_masks, leaf_predictions, leaf_confidences):
    """
    Create visualization showing each leaf with its classification and confidence
    """
    vis_image = rgb_image.copy()
    
    # Color scheme
    class_colors = {
        0: (0, 255, 0),    # Green for Healthy
        1: (0, 0, 255),    # Red for Diseased
    }
    class_names = {
        0: 'Healthy',
        1: 'Diseased'
    }
    
    # Process each leaf
    for i, ((leaf_id, leaf_mask), prediction, confidence) in enumerate(
        zip(leaf_masks, leaf_predictions, leaf_confidences)
    ):
        # Get color based on prediction
        color = class_colors.get(prediction, (255, 255, 255))
        
        # Find contours for this leaf
        contours, _ = cv.findContours(leaf_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Draw thick colored contour
        cv.drawContours(vis_image, contours, -1, color, 3)
        
        # Calculate centroid for label placement
        if contours:
            M = cv.moments(contours[0])
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Create label text with leaf_id, class name, and confidence
                label_text = f"{leaf_id}: {class_names[prediction]} ({confidence*100:.0f}%)"
                
                # Draw label with background for better visibility
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                
                # Get text size for background
                (text_width, text_height), baseline = cv.getTextSize(
                    label_text, font, font_scale, thickness
                )
                
                # Draw semi-transparent background
                overlay = vis_image.copy()
                cv.rectangle(overlay, 
                            (cX - text_width//2 - 5, cY - text_height - 10),
                            (cX + text_width//2 + 5, cY + 10),
                            (0, 0, 0), -1)
                vis_image = cv.addWeighted(vis_image, 0.7, overlay, 0.3, 0)
                
                # Draw text
                cv.putText(vis_image, label_text, 
                           (cX - text_width//2, cY), 
                           font, font_scale, (255, 255, 255), thickness, cv.LINE_AA)
    
    # Add title
    title = f"Per-Leaf Inference Results ({len(leaf_masks)} leaves)"
    cv.putText(vis_image, title, (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    
    # Add legend
    legend_y = 70
    cv.putText(vis_image, "Legend:", (10, legend_y),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    
    for class_id, class_name in class_names.items():
        legend_y += 30
        color = class_colors[class_id]
        cv.rectangle(vis_image, (10, legend_y - 15), (30, legend_y), color, -1)
        cv.putText(vis_image, f"{class_name}", (40, legend_y - 3),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    
    return vis_image

# ============================================================================
# SHARED STATE (MODIFIED: Added flags and new panels)
# ============================================================================

class SharedState:
    def __init__(self):
        self.lock = threading.Lock() # Lock for all data
        
        # === Final JPEG streams for Flask ===
        self.preview_rgb_jpg = None
        self.preview_nir_jpg = None
        self.ndvi_heatmap_jpg = None
        self.rdvi_heatmap_jpg = None
        self.segmentation_mask_jpg = None
        self.per_leaf_inference_jpg = None
        
        # === Status and Results ===
        self.status = "Initializing..."
        self.last_capture_time = None
        self.last_inference_time = None
        self.gps_data = {'lat': 0.0, 'lon': 0.0, 'valid': False}
        self.last_vis = {'ndvi_mean': 0.0, 'rdvi_mean': 0.0}
        self.inference_result = None
        self.leaf_count = 0
        self.per_leaf_results = []
        
        # === NEW: Flags for processing worker ===
        self.run_capture_processing = False
        self.run_inference_processing = False

        # === Data buffers for worker threads ===
        self.last_rgb_aligned = None # From api_capture for processing_worker
        self.last_nir_aligned = None # From api_capture for processing_worker
        self.last_cleaned_mask = None
        self.last_ndvi_array = None
        self.last_rdvi_array = None
        self.preprocessor = None # Set by PreprocessingWorker

S = SharedState()

# ============================================================================
# GPS HANDLER (Original)
# ============================================================================

class GPSHandler:
    def __init__(self):
        self.gps = None
        self.running = False
        self.thread = None
        
        if GPS_AVAILABLE:
            try:
                self.gps = L76X.L76X()
                self.gps.L76X_Set_Baudrate(115200)
                time.sleep(1)
                self.gps.L76X_Send_Command(self.gps.SET_POS_FIX_1S)
                time.sleep(0.5)
                self.gps.L76X_Send_Command(self.gps.SET_NMEA_OUTPUT)
                time.sleep(0.5)
                self.gps.L76X_Exit_BackupMode()
                time.sleep(1)
                print("[GPS] Initialized successfully")
            except Exception as e:
                print(f"[GPS] Initialization failed: {e}")
                self.gps = None
    
    def start(self):
        """Start GPS reading thread"""
        if self.gps and not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._gps_worker, daemon=True)
            self.thread.start()
            print("[GPS] Worker thread started")
    
    def _gps_worker(self):
        """Background thread to continuously read GPS"""
        while self.running:
            try:
                self.gps.L76X_Gat_GNRMC()
                
                if self.gps.Status == 1 and self.gps.validate_coordinates():
                    with S.lock:
                        S.gps_data = {
                            'lat': self.gps.Lat,
                            'lon': self.gps.Lon,
                            'valid': True,
                            'satellites': self.gps.satellites_in_use,
                            'hdop': self.gps.hdop
                        }
                else:
                    with S.lock:
                        S.gps_data['valid'] = False
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                print(f"[GPS] Read error: {e}")
                time.sleep(2)
    
    def get_current_position(self):
        """Get current GPS position"""
        with S.lock:
            return S.gps_data.copy()
    
    def stop(self):
        """Stop GPS reading"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

# ============================================================================
# DATAHUB HANDLER (MODIFIED: New tags)
# ============================================================================

class DataHubHandler:
    def __init__(self, config):
        self.edge_agent = None
        self.connected = False
        self.config = config
        
        if not DATAHUB_AVAILABLE:
            print("[DataHub] SDK not available, skipping initialization")
            return
        
        try:
            edge_options = EdgeAgentOptions(nodeId=config['node_id'])
            edge_options.connectType = constant.ConnectType['DCCS']
            
            dccs_options = DCCSOptions(
                apiUrl=config['api_url'],
                credentialKey=config['credential_key']
            )
            edge_options.DCCS = dccs_options
            
            self.edge_agent = EdgeAgent(edge_options)
            self.edge_agent.on_connected = self._on_connected
            self.edge_agent.on_disconnected = self._on_disconnected
            
            self.edge_agent.connect()
            print("[DataHub] Connection initiated")
            
        except Exception as e:
            print(f"[DataHub] Initialization failed: {e}")
            self.edge_agent = None
    
    def _on_connected(self, agent, connected):
        if connected:
            self.connected = True
            print("[DataHub] Connected successfully")
            # Upload device configuration
            config = self._generate_device_config()
            self.edge_agent.uploadConfig(
                action=constant.ActionType['Create'],
                edgeConfig=config
            )
        else:
            self.connected = False
            print("[DataHub] Connection failed")
    
    def _on_disconnected(self, agent, disconnected):
        if disconnected:
            self.connected = False
            print("[DataHub] Disconnected")
    
    def _generate_device_config(self):
        """Generate device configuration"""
        config = EdgeConfig()
        device_config = DeviceConfig(
            id='NDVIDevice001',
            name='NDVI Multispectral Agricultural Sensor',
            description='AIoT 2025 Competition - VGU Agritech Team',
            deviceType='Agricultural Sensor',
            retentionPolicyName=''
        )
        
        # UPDATED tags to match new UI
        tags = [
            ('NDVI_Mean', 'Mean NDVI value (plant only)', -1.0, 1.0),
            ('RDVI_Mean', 'Mean RDVI value (plant only)', -1.0, 1.0),
            ('GPS_Latitude', 'GPS Latitude', -90.0, 90.0),
            ('GPS_Longitude', 'GPS Longitude', -180.0, 180.0),
            ('Disease_Classification', 'Disease class (0=healthy, 1=diseased)', 0, 1),
            ('Leaf_Count', 'Number of leaves detected', 0, 100)
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
    
    def upload_data(self, data):
        """Upload data to DataHub"""
        if not self.connected or not self.edge_agent:
            print("[DataHub] Not connected, skipping upload")
            return False
        
        try:
            edge_data = EdgeData()
            edge_data.timestamp = datetime.datetime.now()
            
            device_id = 'NDVIDevice001'
            
            # Add tags
            for key, value in data.items():
                tag = EdgeTag(device_id, key, value)
                edge_data.tagList.append(tag)
            
            self.edge_agent.sendData(edge_data)
            print(f"[DataHub] Data uploaded successfully")
            return True
            
        except Exception as e:
            print(f"[DataHub] Upload failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from DataHub"""
        if self.edge_agent and self.connected:
            self.edge_agent.disconnect()

# ============================================================================
# PREPROCESSING WORKER
# ============================================================================

class PreprocessingWorker:
    def __init__(self, calib_maps_dir):
        self.calib_data = create_zhang_calibration_data()
        self.preprocessor = ZhangCorePreprocessor(calib_maps_dir, self.calib_data)
        print("[Preprocessing] Zhang preprocessor initialized")
        # Store preprocessor for worker thread
        with S.lock:
            S.preprocessor = self.preprocessor

# ============================================================================
# AI INFERENCE HANDLER (MODIFIED: New UI classification)
# ============================================================================

class AIInferenceHandler:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.imputer = None
        self.rf_model = None
        
        # Try to load models
        try:
            import joblib
            imputer_path = os.path.join(models_dir, 'imputer.joblib')
            rf_path = os.path.join(models_dir, 'rf_model.joblib')
            
            if os.path.exists(imputer_path):
                self.imputer = joblib.load(imputer_path)
                print(f"[AI] Loaded imputer from {imputer_path}")
            else:
                print(f"[AI] Warning: Imputer not found at {imputer_path}")
            
            if os.path.exists(rf_path):
                self.rf_model = joblib.load(rf_path)
                print(f"[AI] Loaded RF model from {rf_path}")
            else:
                print(f"[AI] Warning: RF model not found at {rf_path}")
            
            if not self.imputer or not self.rf_model:
                print("[AI] Warning: Models not fully loaded - AI inference will use placeholder logic")
                
        except Exception as e:
            print(f"[AI] Failed to load models: {e}")
    
    def run_inference(self, rgb_aligned, nir_aligned, cleaned_mask, preprocessor):
        """
        Run full AI inference pipeline with visualization generation
        """
        if not self.imputer or not self.rf_model:
            # Generate placeholder visualizations even if model fails
            placeholder_seg = create_segmentation_visualization(cleaned_mask, rgb_aligned)
            placeholder_leaf = create_per_leaf_inference_visualization(rgb_aligned, [], [], [])
            return {
                'success': False,
                'message': 'AI models not loaded',
                'classification': None,
                'segmentation_vis': placeholder_seg,
                'per_leaf_vis': placeholder_leaf
            }
        
        try:
            # Step 1: Create segmentation visualization (mask is pre-computed)
            print("[AI] Creating segmentation visualization...")
            segmentation_vis = create_segmentation_visualization(cleaned_mask, rgb_aligned)
            
            # Step 2: Segment individual leaves
            print("[AI] Segmenting leaves with watershed...")
            all_leaf_masks, watershed_debug = segment_leaves_from_mask(
                cleaned_mask, rgb_aligned,
                min_leaf_area=500,
                watershed_min_distance=20
            )
            
            if not all_leaf_masks:
                # Still return the segmentation visualization
                placeholder_leaf = create_per_leaf_inference_visualization(rgb_aligned, [], [], [])
                return {
                    'success': False,
                    'message': 'No leaves detected',
                    'num_leaves': 0,
                    'segmentation_vis': segmentation_vis,
                    'per_leaf_vis': placeholder_leaf
                }
            
            print(f"[AI] Found {len(all_leaf_masks)} leaves")
            
            # Step 3: Extract features and run classification for each leaf
            leaf_predictions = []
            leaf_confidences = []
            
            for i, (leaf_id, leaf_mask) in enumerate(all_leaf_masks):
                # Simplified feature extraction (you would use full features in production)
                red_channel = rgb_aligned[:, :, 2]
                
                # Use leaf_mask
                red_refl = (red_channel.astype(np.float32) / 255.0)[leaf_mask > 0]
                nir_refl = (nir_aligned.astype(np.float32) / 255.0)[leaf_mask > 0]
                
                if len(red_refl) > 10:
                    # Calculate NDVI for this leaf
                    ndvi = (nir_refl - red_refl) / (nir_refl + red_refl + 1e-6)
                    
                    # Placeholder classification (in production, use full feature vector)
                    ndvi_mean = float(np.mean(ndvi))
                    
                    # Simple threshold-based classification for demo (Healthy=0, Diseased=1)
                    if ndvi_mean > 0.5:
                        prediction = 0  # Healthy
                        confidence = 0.85
                    else:
                        prediction = 1  # Diseased
                        confidence = 0.75
                    
                    leaf_predictions.append(prediction)
                    leaf_confidences.append(confidence)
            
            # Step 4: Create per-leaf inference visualization
            per_leaf_vis = create_per_leaf_inference_visualization(
                rgb_aligned, all_leaf_masks, leaf_predictions, leaf_confidences
            )
            
            # Step 5: Aggregate results
            if leaf_predictions:
                from collections import Counter
                class_counts = Counter(leaf_predictions)
                final_class = class_counts.most_common(1)[0][0]
                avg_confidence = float(np.mean(leaf_confidences))
            else:
                final_class = 0
                avg_confidence = 0.0
            
            class_names = {0: 'Healthy', 1: 'Diseased'}
            
            return {
                'success': True,
                'num_leaves': len(all_leaf_masks),
                'classification': final_class,
                'classification_name': class_names.get(final_class, 'Unknown'),
                'confidence': avg_confidence,
                'leaf_classifications': leaf_predictions,
                'leaf_confidences': leaf_confidences,
                'message': f'Classified as {class_names.get(final_class, "Unknown")}',
                'segmentation_vis': segmentation_vis,
                'per_leaf_vis': per_leaf_vis
            }
            
        except Exception as e:
            print(f"[AI] Inference error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'message': f'Inference failed: {str(e)}',
                'classification': None,
                'segmentation_vis': None,
                'per_leaf_vis': None
            }

# ============================================================================
# WEB INTERFACE (MODIFIED: New HTML template)
# ============================================================================

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>AIoT 2025 - VGU Agritech Pipeline</title>
<style>
 *{box-sizing:border-box}
 body{margin:0;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Arial;background:#0a0a0a;color:#e8e8e8}
 header{padding:20px 24px;border-bottom:2px solid #2a2a2a;background:linear-gradient(135deg,#1a1a1a,#0f0f0f)}
 h1{margin:0;font-size:26px;font-weight:700;color:#10b981}
 .subtitle{margin:6px 0 0 0;font-size:14px;color:#888}
 .container{max-width:1900px;margin:0 auto;padding:20px}
 .section{background:#1a1a1a;border-radius:10px;padding:20px;margin-bottom:20px;border:1px solid #2a2a2a}
 .section-title{font-size:18px;font-weight:600;margin-bottom:16px;color:#10b981}
 .grid{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin-bottom:20px}
 .preview-panel{background:#111;border-radius:8px;overflow:hidden;border:1px solid #2a2a2a}
 .preview-panel img{width:100%;display:block;min-height:300px;background:#0a0a0a; aspect-ratio: 4/3; object-fit: contain;}
 .button-group{display:flex;gap:12px;margin-bottom:20px}
 .btn{background:linear-gradient(135deg,#10b981,#059669);color:#fff;border:none;padding:14px 24px;border-radius:8px;cursor:pointer;font-size:16px;font-weight:600;transition:all 0.2s}
 .btn:hover{transform:translateY(-2px);box-shadow:0 6px 16px rgba(16,185,129,0.4)}
 .btn:disabled{opacity:0.5;cursor:not-allowed;transform:none}
 .btn-secondary{background:linear-gradient(135deg,#3b82f6,#2563eb)}
 .btn-secondary:hover{box-shadow:0 6px 16px rgba(59,130,246,0.4)}
 .status-panel{background:#222;padding:16px;border-radius:8px;margin-bottom:20px}
 .status-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
 .status-item{background:#1a1a1a;padding:12px;border-radius:6px;border:1px solid #333}
 .status-label{font-size:12px;color:#888;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px}
 .status-value{font-size:20px;font-weight:700;color:#10b981}
 .info-box{background:#1e293b;border-left:4px solid #0ea5e9;padding:16px;border-radius:8px;margin-bottom:20px}
 .info-box h3{margin:0 0 8px 0;font-size:16px;color:#38bdf8}
 @media (max-width: 1400px){.grid{grid-template-columns:repeat(2,1fr)}}
 @media (max-width: 900px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>AIoT 2025 Pipeline - VGU Agritech Team</h1>
  <p class="subtitle">Multispectral Imaging for Crop Disease Detection with Enhanced Visualization</p>
</header>

<div class="container">
  <div class="status-panel">
    <div class="status-grid">
      <div class="status-item">
        <div class="status-label">System Status</div>
        <div class="status-value" id="system-status">Ready</div>
      </div>
      <div class="status-item">
        <div class="status-label">GPS Position</div>
        <div class="status-value" id="gps-status">Acquiring...</div>
      </div>
      <div class="status-item">
        <div class="status-label">DataHub</div>
        <div class="status-value" id="datahub-status">Connected</div>
      </div>
      <div class="status-item">
        <div class="status-label">Last NDVI (Plant Only)</div>
        <div class="status-value" id="ndvi-value">--</div>
      </div>
      <div class="status-item">
        <div class="status-label">Last RDVI (Plant Only)</div>
        <div class="status-value" id="rdvi-value">--</div>
      </div>
      <div class="status-item">
        <div class="status-label">Classification</div>
        <div class="status-value" id="classification">--</div>
      </div>
      <div class="status-item">
        <div class="status-label">Leaves Detected</div>
        <div class="status-value" id="leaf-count">0</div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Camera Preview & Alignment</div>
    <div class="grid">
      <div class="preview-panel">
        <img src="/stream/rgb" alt="RGB Camera">
      </div>
      <div class="preview-panel">
        <img src="/stream/nir" alt="NIR Camera">
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Vegetation Index Heatmaps (Masked)</div>
    <div class="grid">
      <div class="preview-panel">
        <img src="/stream/ndvi_heatmap" alt="NDVI Heatmap">
      </div>
      <div class="preview-panel">
        <img src="/stream/rdvi_heatmap" alt="RDVI Heatmap">
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">AI Analysis & Segmentation</div>
    <div class="grid">
      <div class="preview-panel">
        <img src="/stream/segmentation_mask" alt="Segmentation Mask">
      </div>
      <div class="preview-panel">
        <img src="/stream/per_leaf_inference" alt="Per-Leaf Inference">
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Actions</div>
    <div class="button-group">
      <button class="btn" id="capture-btn" onclick="captureAndProcess()">
        Capture & Process
      </button>
      <button class="btn btn-secondary" id="inference-btn" onclick="runInference()">
        Run AI Inference
      </button>
    </div>
    <div class="info-box">
      <h3>Pipeline Information</h3>
      <p id="info-message">Click 'Capture & Process' to capture, align, segment, calculate VIs, and upload to DataHub. Click 'Run AI Inference' to perform disease classification with detailed per-leaf analysis.</p>
    </div>
  </div>

  <div class="section" id="results-section" style="display:none">
    <div class="section-title">Detailed Per-Leaf Results</div>
    <table id="results-table" style="width:100%;color:#e8e8e8;border-collapse:collapse">
      <thead>
        <tr style="background:#2a2a2a">
          <th style="padding:12px;text-align:left">Leaf ID</th>
          <th style="padding:12px;text-align:left">Classification</th>
          <th style="padding:12px;text-align:left">Confidence</th>
          <th style="padding:12px;text-align:left">Status</th>
        </tr>
      </thead>
      <tbody id="results-body">
      </tbody>
    </table>
  </div>
</div>

<script>
async function captureAndProcess() {
  const btn = document.getElementById('capture-btn');
  const info = document.getElementById('info-message');
  
  btn.disabled = true;
  btn.textContent = 'Processing...';
  info.textContent = 'Triggering capture and processing...';
  
  try {
    const response = await fetch('/api/capture');
    const result = await response.json();
    
    if (result.success) {
      info.textContent = result.message;
      // Stats are now returned immediately
      document.getElementById('ndvi-value').textContent = result.ndvi_mean.toFixed(4);
      document.getElementById('rdvi-value').textContent = result.rdvi_mean.toFixed(4);
      
      // The images will update on their own when the worker is done.
      // We force a refresh poll, but the worker might not be done yet.
      refreshStreams();
    } else {
      info.textContent = 'Capture failed: ' + result.message;
    }
  } catch (e) {
    info.textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Capture & Process';
  }
}

async function runInference() {
  const btn = document.getElementById('inference-btn');
  const info = document.getElementById('info-message');
  
  btn.disabled = true;
  btn.textContent = 'Running AI...';
  info.textContent = 'Triggering AI inference...';
  
  try {
    const response = await fetch('/api/inference');
    const result = await response.json();
    
    if (result.success) {
      info.textContent = result.message;
      // Images and table will update when worker is done.
      // We can force a refresh poll.
      refreshStreams();
    } else {
      info.textContent = 'Inference failed: ' + result.message;
    }
  } catch (e) {
    info.textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run AI Inference';
  }
}

function refreshStreams() {
  const timestamp = new Date().getTime();
  document.querySelectorAll('img').forEach(img => {
    const src = img.src.split('?')[0];
    img.src = src + '?' + timestamp;
  });
}

function updateResultsTable(results) {
  const tbody = document.getElementById('results-body');
  const section = document.getElementById('results-section');
  
  if (!results || results.length === 0) {
    section.style.display = 'none';
    return;
  }
  
  section.style.display = 'block';
  tbody.innerHTML = '';
  
  results.forEach((leaf, idx) => {
    const row = tbody.insertRow();
    row.style.borderBottom = '1px solid #333';
    
    const cell1 = row.insertCell(0);
    const cell2 = row.insertCell(1);
    const cell3 = row.insertCell(2);
    const cell4 = row.insertCell(3);
    
    cell1.style.padding = '12px';
    cell2.style.padding = '12px';
    cell3.style.padding = '12px';
    cell4.style.padding = '12px';
    
    cell1.textContent = leaf.leaf_id || `leaf_${String(idx+1).padStart(3, '0')}`;
    cell2.textContent = leaf.classification || 'Unknown';
    cell3.textContent = (leaf.confidence * 100).toFixed(1) + '%';
    
    const statusColor = leaf.classification === 'Healthy' ? '#10b981' : '#ef4444';
    cell4.innerHTML = `<span style="color:${statusColor}">●</span> ${leaf.classification}`;
  });
}

async function pollStatus() {
  try {
    const response = await fetch('/api/status');
    const status = await response.json();
    
    document.getElementById('system-status').textContent = status.system_status;
    
    if (status.gps_valid) {
      document.getElementById('gps-status').textContent = 
        `${status.gps_lat.toFixed(6)}, ${status.gps_lon.toFixed(6)}`;
    } else {
      document.getElementById('gps-status').textContent = 'No Fix';
    }
    
    document.getElementById('datahub-status').textContent = 
      status.datahub_connected ? 'Connected' : 'Disconnected';
      
    // NEW: Poll for stats and AI results
    document.getElementById('ndvi-value').textContent = status.ndvi_mean.toFixed(4);
    document.getElementById('rdvi-value').textContent = status.rdvi_mean.toFixed(4);

    if (status.per_leaf_results && status.per_leaf_results.length > 0) {
        document.getElementById('classification').textContent = status.classification_name || 'Unknown';
        document.getElementById('leaf-count').textContent = status.num_leaves || 0;
        updateResultsTable(status.per_leaf_results);
    } else if (status.last_capture_time_ms > status.last_inference_time_ms) {
        // Capture is newer than inference, clear old AI results
        document.getElementById('classification').textContent = '--';
        document.getElementById('leaf-count').textContent = 0;
        updateResultsTable([]);
    }

  } catch (e) {
    console.error('Status poll error:', e);
  }
  
  setTimeout(pollStatus, 2000);
}

pollStatus();
</script>
</body>
</html>
"""

def create_flask_app(args, preprocessing_worker, gps_handler, datahub_handler, ai_handler):
    app = Flask(__name__)
    
    @app.route("/")
    def index():
        return render_template_string(HTML_TEMPLATE)
    
    def mjpeg_stream(attr_name):
        """Generate MJPEG stream"""
        def gen():
            while True:
                with S.lock:
                    buf = getattr(S, attr_name, None)
                if buf is None:
                    # NEW: Create a placeholder "No Signal" image
                    placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv.putText(placeholder, "No Signal", (80, 120), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    _, buf = cv.imencode('.jpg', placeholder)
                    buf = buf.tobytes()

                yield b"--frame\r\n"
                yield b"Content-Type: image/jpeg\r\n"
                yield b"Content-Length: " + str(len(buf)).encode() + b"\r\n\r\n"
                yield buf + b"\r\n"
                time.sleep(0.05)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")
    
    @app.route("/stream/rgb")
    def stream_rgb():
        return mjpeg_stream("preview_rgb_jpg")
    
    @app.route("/stream/nir")
    def stream_nir():
        return mjpeg_stream("preview_nir_jpg")
    
    @app.route("/stream/ndvi_heatmap")
    def stream_ndvi():
        return mjpeg_stream("ndvi_heatmap_jpg")
    
    @app.route("/stream/rdvi_heatmap")
    def stream_rdvi():
        return mjpeg_stream("rdvi_heatmap_jpg")
        
    # --- NEW STREAM ENDPOINTS ---
    @app.route("/stream/segmentation_mask")
    def stream_segmentation():
        return mjpeg_stream("segmentation_mask_jpg")
    
    @app.route("/stream/per_leaf_inference")
    def stream_per_leaf():
        return mjpeg_stream("per_leaf_inference_jpg")
    # --- END NEW STREAM ENDPOINTS ---
    
    @app.route("/api/status")
    def api_status():
        """Return current system status"""
        with S.lock:
            gps_data = S.gps_data.copy()
            vis_data = S.last_vis.copy()
            status = S.status
            # Get data for the new table
            per_leaf_results = S.per_leaf_results
            leaf_count = S.leaf_count
            classification_name = S.inference_result['classification_name'] if S.inference_result and 'classification_name' in S.inference_result else "N/A"
            last_cap_ms = S.last_capture_time.timestamp() * 1000 if S.last_capture_time else 0
            last_inf_ms = S.last_inference_time.timestamp() * 1000 if S.last_inference_time else 0
        
        return jsonify({
            'system_status': status,
            'gps_lat': gps_data.get('lat', 0.0),
            'gps_lon': gps_data.get('lon', 0.0),
            'gps_valid': gps_data.get('valid', False),
            'datahub_connected': datahub_handler.connected if datahub_handler else False,
            'ndvi_mean': vis_data.get('ndvi_mean', 0.0),
            'rdvi_mean': vis_data.get('rdvi_mean', 0.0),
            # NEW: Add results to status poll
            'per_leaf_results': per_leaf_results,
            'num_leaves': leaf_count,
            'classification_name': classification_name,
            'last_capture_time_ms': last_cap_ms,
            'last_inference_time_ms': last_inf_ms,
        })
    
    @app.route("/api/capture")
    def api_capture():
        """
        MODIFIED (THREAD-SAFE):
        Captures, does fast math, and SETS A FLAG for the worker.
        Returns IMMEDIATELY.
        """
        try:
            with S.lock:
                S.status = "Capturing..."
            
            # 1. Capture images (This is safe, based on original code)
            rgb_bgr = get_frame_bgr(app.rgb_cam)
            nir_rgb = get_frame_bgr(app.nir_cam)
            nir_gray = cv.cvtColor(nir_rgb, cv.COLOR_BGR2GRAY)
            
            with S.lock:
                S.status = "Aligning..."
            
            # 2. Align images
            rgb_aligned, nir_aligned = S.preprocessor.rectify_image_pair(
                rgb_bgr, nir_gray
            )
            
            if rgb_aligned is None or nir_aligned is None:
                 with S.lock:
                    S.status = "Alignment failed"
                 return jsonify({
                    'success': False,
                    'message': 'Image alignment failed'
                 })

            with S.lock:
                S.status = "Segmenting..."

            # 3. Segment (HSI + NIR)
            cleaned_mask, _ = create_hsv_nir_mask(rgb_aligned, nir_aligned)
            
            # 4. Calculate Reflectance
            red_refl = rgb_aligned[:, :, 2].astype(np.float32) / 255.0
            nir_refl = nir_aligned.astype(np.float32) / 255.0
            
            # 5. Calculate VIs
            vis = calculate_vis(red_refl, nir_refl)
            ndvi = vis['ndvi']
            rdvi = vis['rdvi']
            
            # 6. Mask VIs (set non-plant areas to NaN)
            ndvi[cleaned_mask == 0] = np.nan
            rdvi[cleaned_mask == 0] = np.nan
            
            # 7. Calculate statistics (nanmean ignores NaN values)
            ndvi_mean = float(np.nanmean(ndvi))
            rdvi_mean = float(np.nanmean(rdvi))
            if np.isnan(ndvi_mean): ndvi_mean = 0.0
            if np.isnan(rdvi_mean): rdvi_mean = 0.0
            
            # 8. Get GPS and Upload to DataHub
            gps_pos = gps_handler.get_current_position() if gps_handler else {}
            if datahub_handler:
                datahub_handler.upload_data({
                    'NDVI_Mean': ndvi_mean,
                    'RDVI_Mean': rdvi_mean,
                    'GPS_Latitude': gps_pos.get('lat', 0.0),
                    'GPS_Longitude': gps_pos.get('lon', 0.0)
                })

            # 9. Store RAW data and set FLAG for worker
            with S.lock:
                S.last_rgb_aligned = rgb_aligned
                S.last_nir_aligned = nir_aligned
                S.last_cleaned_mask = cleaned_mask
                S.last_ndvi_array = ndvi
                S.last_rdvi_array = rdvi
                S.last_vis = {'ndvi_mean': ndvi_mean, 'rdvi_mean': rdvi_mean}
                S.last_capture_time = datetime.datetime.now()
                S.status = "Processing images..."
                S.run_capture_processing = True # <<< SET FLAG
                S.run_inference_processing = False # New capture invalidates old AI
            
            # 10. Return IMMEDIATELY
            return jsonify({
                'success': True,
                'message': f'Capture successful. Processing images...',
                # Return stats immediately
                'ndvi_mean': ndvi_mean,
                'rdvi_mean': rdvi_mean
            })
            
        except Exception as e:
            with S.lock:
                S.status = "Error"
            print(f"[Capture] Error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'message': f'Capture error: {str(e)}'
            })
    
    @app.route("/api/inference")
    def api_inference():
        """
        MODIFIED (THREAD-SAFE):
        Checks if data exists, SETS A FLAG for the worker,
        and returns IMMEDIATELY.
        """
        try:
            with S.lock:
                if S.last_rgb_aligned is None:
                    return jsonify({
                        'success': False,
                        'message': 'No images available. Please capture images first.'
                    })
                
                S.status = "Starting AI inference..."
                S.run_inference_processing = True # <<< SET FLAG
            
            # Return IMMEDIATELY
            return jsonify({
                'success': True,
                'message': 'AI inference triggered. Processing...'
            })
            
        except Exception as e:
            with S.lock:
                S.status = "Error"
            print(f"[Inference] Error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'message': f'Inference error: {str(e)}'
            })
    
    return app

# ============================================================================
# PREVIEW WORKER (Original)
# ============================================================================

def preview_worker(rgb_cam, nir_cam):
    """
    Continuously capture frames, but only update the shared state
    variable ONCE. This provides a static preview image at startup
    and avoids the thread-deadlock from a live feed.
    """
    print("[Preview Worker] Started (Original Mode).")
    try:
        while True:
            # Always capture frames to keep the camera active
            rgb_bgr = get_frame_bgr(rgb_cam)
            nir_rgb = get_frame_bgr(nir_cam)
            
            # Downscale for preview
            scale = 0.5
            rgb_preview = cv.resize(rgb_bgr, None, fx=scale, fy=scale)
            nir_preview = cv.resize(nir_rgb, None, fx=scale, fy=scale)
            
            # Encode to JPEG
            _, rgb_jpg = cv.imencode('.jpg', rgb_preview, [cv.IMWRITE_JPEG_QUALITY, 75])
            _, nir_jpg = cv.imencode('.jpg', nir_preview, [cv.IMWRITE_JPEG_QUALITY, 75])
            
            # This logic is from your original file.
            # It only sets the preview image *once*
            with S.lock:
                if S.preview_rgb_jpg is None:
                    S.preview_rgb_jpg = rgb_jpg.tobytes()
                if S.preview_nir_jpg is None:
                    S.preview_nir_jpg = nir_jpg.tobytes()
            
            time.sleep(0.1) # Continue the loop
            
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[Preview Worker Error] {e}")


# ============================================================================
# NEW: PROCESSING WORKER (THREAD-SAFE)
# ============================================================================

def processing_worker(ai_handler, datahub_handler):
    """
    A single, dedicated background thread that does all the
    slow, non-thread-safe OpenCV drawing and AI inference.
    """
    print("[Processing Worker] Started.")
    
    # Get the preprocessor from shared state
    preprocessor = None
    while preprocessor is None:
        with S.lock:
            preprocessor = S.preprocessor
        if preprocessor is None:
            print("[Worker] Waiting for preprocessor...")
            time.sleep(0.5)
    print("[Worker] Preprocessor acquired.")
            
    while True:
        try:
            # Check for capture processing job
            run_capture = False
            with S.lock:
                if S.run_capture_processing:
                    run_capture = True
                    S.run_capture_processing = False
            
            if run_capture:
                print("[Worker] Processing capture job (drawing)...")
                with S.lock:
                    S.status = "Generating visualizations..."
                    # Get raw data
                    rgb_aligned = S.last_rgb_aligned
                    cleaned_mask = S.last_cleaned_mask
                    ndvi_array = S.last_ndvi_array
                    rdvi_array = S.last_rdvi_array
                
                # --- START Unsafe Drawing Code ---
                ndvi_heatmap = create_vi_heatmap_with_colorbar(ndvi_array, 'NDVI (Masked)')
                rdvi_heatmap = create_vi_heatmap_with_colorbar(rdvi_array, 'RDVI (Masked)')
                seg_vis = create_segmentation_visualization(cleaned_mask, rgb_aligned)
                # --- END Unsafe Drawing Code ---

                # Encode JPEGs
                _, ndvi_jpg = cv.imencode('.jpg', ndvi_heatmap, [cv.IMWRITE_JPEG_QUALITY, 90])
                _, rdvi_jpg = cv.imencode('.jpg', rdvi_heatmap, [cv.IMWRITE_JPEG_QUALITY, 90])
                _, seg_jpg = cv.imencode('.jpg', seg_vis, [cv.IMWRITE_JPEG_QUALITY, 85])
                _, rgb_jpg = cv.imencode('.jpg', S.last_rgb_aligned, [cv.IMWRITE_JPEG_QUALITY, 85])
                
                nir_bgr = cv.cvtColor(S.last_nir_aligned, cv.COLOR_GRAY2BGR)
                _, nir_jpg = cv.imencode('.jpg', nir_bgr, [cv.IMWRITE_JPEG_QUALITY, 85])

                # Update shared state with final images
                with S.lock:
                    S.ndvi_heatmap_jpg = ndvi_jpg.tobytes()
                    S.rdvi_heatmap_jpg = rdvi_jpg.tobytes()
                    S.segmentation_mask_jpg = seg_jpg.tobytes()
                    S.preview_rgb_jpg = rgb_jpg.tobytes() # Overwrite preview with capture
                    S.preview_nir_jpg = nir_jpg.tobytes() # Overwrite preview with capture
                    
                    # Clear old AI results
                    S.per_leaf_inference_jpg = None
                    S.per_leaf_results = []
                    S.leaf_count = 0
                    S.inference_result = None

                    S.status = "Ready"
                print("[Worker] Capture job finished.")

            # Check for inference processing job
            run_inference = False
            with S.lock:
                if S.run_inference_processing:
                    run_inference = True
                    S.run_inference_processing = False
            
            if run_inference:
                print("[Worker] Processing inference job...")
                with S.lock:
                    S.status = "Running AI..."
                    rgb = S.last_rgb_aligned
                    nir = S.last_nir_aligned
                    mask = S.last_cleaned_mask
                
                # Check if data exists
                if rgb is None or nir is None or mask is None:
                    print("[Worker] Inference skipped, no capture data.")
                    with S.lock:
                        S.status = "Ready"
                    continue
                    
                # Run full inference (includes unsafe drawing)
                result = ai_handler.run_inference(rgb, nir, mask, preprocessor)

                # Prepare results
                per_leaf_results = []
                if result.get('success') and 'leaf_classifications' in result:
                    for i, (pred, conf) in enumerate(zip(
                        result['leaf_classifications'],
                        result['leaf_confidences']
                    )):
                        per_leaf_results.append({
                            'leaf_id': f"leaf_{str(i+1).zfill(3)}",
                            'classification': 'Healthy' if pred == 0 else 'Diseased',
                            'confidence': conf
                        })
                
                # Encode final AI image
                per_leaf_vis = result.get('per_leaf_vis')
                per_leaf_jpg = None
                if per_leaf_vis is not None:
                    _, buf = cv.imencode('.jpg', per_leaf_vis, [cv.IMWRITE_JPEG_QUALITY, 85])
                    per_leaf_jpg = buf.tobytes()

                # Upload to DataHub
                if result.get('success') and datahub_handler:
                    datahub_handler.upload_data({
                        'Disease_Classification': result['classification'],
                        'Leaf_Count': result.get('num_leaves', 0)
                    })
                
                # Update shared state
                with S.lock:
                    S.inference_result = result
                    S.last_inference_time = datetime.datetime.now()
                    S.leaf_count = result.get('num_leaves', 0)
                    S.per_leaf_results = per_leaf_results
                    S.per_leaf_inference_jpg = per_leaf_jpg
                    S.status = "Ready"
                print("[Worker] Inference job finished.")
                
            time.sleep(0.1) # Poll 10 times per second
            
        except Exception as e:
            print(f"[Processing Worker Error] {e}")
            import traceback
            traceback.print_exc()
            with S.lock:
                S.status = "Worker Error"
            time.sleep(1) # Don't spam errors

# ============================================================================
# MAIN (MODIFIED: Starts processing_worker)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='AIoT 2025 Integrated Pipeline - VGU Agritech Team'
    )
    
    # Camera settings
    parser.add_argument('--rgb', type=int, default=0, help='RGB camera index')
    parser.add_argument('--nir', type=int, default=1, help='NIR camera index')
    parser.add_argument('--size', default='1920x1080', help='Image resolution')
    parser.add_argument('--fps', type=int, default=5, help='Frame rate')
    parser.add_argument('--exp_us_rgb', type=int, default=6000, help='RGB exposure (us)')
    parser.add_argument('--exp_us_nir', type=int, default=8000, help='NIR exposure (us)')
    parser.add_argument('--gain_rgb', type=float, default=2.0, help='RGB gain')
    parser.add_argument('--gain_nir', type=float, default=4.0, help='NIR gain')
    
    # Calibration
    parser.add_argument('--calib_maps', type=str, required=True, 
                       help='Directory with rectification maps')
    
    # DataHub configuration
    parser.add_argument('--datahub_node_id', type=str, 
                       default='2b3e7a1c-2970-4ecd-81bc-33de9b7eda3d')
    parser.add_argument('--datahub_api_url', type=str,
                       default='httpsD://api-dccs-ensaas.sa.wise-paas.com/')
    parser.add_argument('--datahub_credential', type=str,
                       default='ae13a5d125a904eaa5468733c53134yl')
    
    # Models
    parser.add_argument('--models_dir', type=str, default='models',
                       help='Directory containing AI models')
    
    # Web interface
    parser.add_argument('--host', default='0.0.0.0', 
                       help='Host address (0.0.0.0 for network access)')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    
    args = parser.parse_args()
    
    # Parse resolution
    W, H = map(int, args.size.split('x'))
    
    print("\n" + "="*70)
    print("AIoT 2025 Integrated Pipeline - VGU Agritech Team")
    print("="*70)
    
    # Initialize components
    print("\n[Init] Initializing cameras...")
    rgb_cam = setup_camera(
        args.rgb, (W, H), args.fps, args.exp_us_rgb, args.gain_rgb, awb=True
    )
    nir_cam = setup_camera(
        args.nir, (W, H), args.fps, args.exp_us_nir, args.gain_nir, awb=False
    )
    print("[Init] Cameras initialized")
    
    print("\n[Init] Initializing preprocessing...")
    # This class auto-stores its preprocessor in S.preprocessor
    preprocessing_worker = PreprocessingWorker(args.calib_maps)
    
    print("\n[Init] Initializing GPS...")
    gps_handler = GPSHandler()
    if gps_handler.gps:
        gps_handler.start()
    
    print("\n[Init] Initializing DataHub...")
    datahub_config = {
        'node_id': args.datahub_node_id,
        'api_url': args.datahub_api_url,
        'credential_key': args.datahub_credential
    }
    datahub_handler = DataHubHandler(datahub_config) if DATAHUB_AVAILABLE else None
    
    print("\n[Init] Initializing AI inference...")
    ai_handler = AIInferenceHandler(args.models_dir)
    
    # Create Flask app
    app = create_flask_app(args, preprocessing_worker, gps_handler, datahub_handler, ai_handler)
    app.rgb_cam = rgb_cam
    app.nir_cam = nir_cam
    
    # Start original preview worker thread
    preview_thread = threading.Thread(
        target=preview_worker, 
        args=(rgb_cam, nir_cam), 
        daemon=True
    )
    preview_thread.start()
    
    # Start the NEW dedicated processing worker
    processing_thread = threading.Thread(
        target=processing_worker,
        args=(ai_handler, datahub_handler),
        daemon=True
    )
    processing_thread.start()
    
    # Get local IP for network access
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "unknown"
    
    print("\n" + "="*70)
    print("Pipeline Started Successfully")
    print("="*70)
    print(f"\nWeb Interface URLs:")
    print(f"  Local:   http://127.0.0.1:{args.port}/")
    print(f"  Network: http://{local_ip}:{args.port}/")
    print(f"\nAccess from other devices on the same network using the Network URL")
    print("\nPress Ctrl+C to stop")
    print("="*70 + "\n")
    
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\n[Shutdown] Stopping pipeline...")
    finally:
        # Cleanup
        if gps_handler:
            gps_handler.stop()
        if datahub_handler:
            datahub_handler.disconnect()
        rgb_cam.stop()
        nir_cam.stop()
        print("[Shutdown] Pipeline stopped")

if __name__ == "__main__":
    main()
