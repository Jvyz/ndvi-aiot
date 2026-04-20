#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete AIoT Plant Health Assessment System
AIoT 2025 Competition - VGU
Integrates: RGB+NIR capture, alignment, NDVI, AI inference, GPS, WISE-IoT DataHub

*** MODIFIED: Segmentation logic updated.
    - Runs fast 'create_hsv_nir_mask' for heatmaps every time.
    - Runs slow 'segment_leaves_from_mask' (Watershed) ONLY if AI model is loaded.
    - AI panel now shows leaf_id (e.g., "leaf_001") for numbering.
***
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
import paho.mqtt.client as mqtt
import matplotlib.pyplot as plt # Needed for colormaps

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
# --- MODIFIED: Import new functions ---
from segmentation_pipeline_kmeans import create_hsv_nir_mask, segment_leaves_from_mask, segment_leaf_structure

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
    
    # WISE-IoT DataHub
    DATAHUB_HOST = "https://api-dccs-ensaas.education.wise-paas.com/"
    DATAHUB_PORT = 1883
    DEVICE_ID = "2b3e7a1c-2970-4ecd-81bc-33de9b7eda3d"
    API_KEY = "ae13a5d125a904eaa5468733c53134yl"
    
    # GPS
    GPS_UPDATE_INTERVAL = 5.0  # seconds
    
    # Web Interface
    WEB_HOST = "0.0.0.0"
    WEB_PORT = 8001  # Set to 8001 as requested
    
    # --- NEW: Standard size for all stream panels ---
    STREAM_SIZE = (640, 480) 

# ====================== DASHBOARD HELPER FUNCTIONS ======================

def create_heatmap_bgr(vi_array: np.ndarray, mask: np.ndarray, cmap_name: str = 'RdYlGn', vmin: float = -1.0, vmax: float = 1.0, add_colorbar: bool = True) -> np.ndarray:
    """
    Converts a VI array (float) into a BGR heatmap (uint8), applying a mask
    and optionally adding a color bar.
    """
    if vi_array is None or vi_array.size == 0:
        return np.zeros((SystemConfig.STREAM_SIZE[1], SystemConfig.STREAM_SIZE[0], 3), dtype=np.uint8) # Return black square
    
    # Ensure mask is boolean and same size
    if mask is None or mask.shape != vi_array.shape:
        logger.warning("Mask is invalid or missing, coloring entire image.")
        mask = np.ones_like(vi_array, dtype=bool)
    else:
        mask = mask.astype(bool)

    # Initialize a blank image for the heatmap
    heatmap_base = np.zeros_like(vi_array, dtype=np.float32)

    # Apply mask: only values within the mask are used for coloring
    heatmap_base[mask] = vi_array[mask]
    
    # Handle NaNs before normalization
    # Replace NaN with vmin for coloring (or a specific neutral color if desired)
    vi_array_nonan = np.nan_to_num(heatmap_base, nan=vmin) 
    
    # Normalize values within the valid range
    normalized = (vi_array_nonan - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0.0, 1.0)
    
    # Get colormap (Matplotlib colormaps give RGBA floats 0-1)
    cmap = plt.get_cmap(cmap_name)
    colored_rgba = cmap(normalized) 
    
    # Convert to BGR (0-255) and remove alpha channel
    colored_bgr_uint8 = (colored_rgba[:, :, 2::-1] * 255).astype(np.uint8)
    
    # Set areas outside the mask to a background color (e.g., light yellow/beige)
    # This simulates the paper background in your example
    background_color = [200, 230, 255] # A light yellow/beige in BGR
    colored_bgr_uint8[~mask] = background_color
    
    # --- Add Color Bar ---
    if add_colorbar:
        cb_width = 30
        cb_height = colored_bgr_uint8.shape[0]
        cb_padding = 10
        total_width = colored_bgr_uint8.shape[1] + cb_width + cb_padding + 40 # Extra padding for text
        
        # Create an expanded canvas
        final_image = np.full((cb_height, total_width, 3), background_color, dtype=np.uint8)
        
        # Place the heatmap
        final_image[:, :colored_bgr_uint8.shape[1]] = colored_bgr_uint8
        
        # Draw the color bar gradient
        gradient = np.linspace(vmax, vmin, cb_height) # Inverted for typical vertical display
        gradient_normalized = (gradient - vmin) / (vmax - vmin)
        gradient_normalized = np.clip(gradient_normalized, 0.0, 1.0)
        gradient_colors_rgba = cmap(gradient_normalized)
        gradient_colors_bgr = (gradient_colors_rgba[:, 2::-1] * 255).astype(np.uint8)
        
        cb_x_start = colored_bgr_uint8.shape[1] + cb_padding
        cb_x_end = cb_x_start + cb_width
        
        for i in range(cb_height):
            final_image[i, cb_x_start:cb_x_end] = gradient_colors_bgr[i]
            
        # Add labels to color bar
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_color = (0, 0, 0) # Black text
        
        # Top (vmax) label
        cv.putText(final_image, f"{vmax:.1f}", 
                   (cb_x_end + 5, 20),
                   font, font_scale, text_color, font_thickness, cv.LINE_AA)
        # Middle (0.0) label
        if vmin < 0.0 < vmax:
            mid_y = int(cb_height * (vmax - 0.0) / (vmax - vmin)) # Calculate position of 0.0
            cv.putText(final_image, "0.0", 
                       (cb_x_end + 5, mid_y),
                       font, font_scale, text_color, font_thickness, cv.LINE_AA)
        # Bottom (vmin) label
        cv.putText(final_image, f"{vmin:.1f}", 
                   (cb_x_end + 5, cb_height - 10),
                   font, font_scale, text_color, font_thickness, cv.LINE_AA)

        return final_image
    else:
        return colored_bgr_uint8


# ====================== SHARED STATE ======================
class SystemState:
    def __init__(self):
        self.lock = threading.Lock()
        
        # Camera data (full resolution)
        self.rgb_frame = None
        self.nir_frame = None
        self.aligned_rgb = None
        self.aligned_nir = None
        
        # Processing results
        self.current_ndvi_value = 0.0
        self.vegetation_indices = {}
        self.ai_prediction = None
        self.confidence = 0.0
        self.segmentation_result_full = None # This is the full-res AI-annotated image

        # GPS data
        self.gps_data = {
            'latitude': 0.0,
            'longitude': 0.0,
            'altitude': 0.0,
            'status': 'No Fix',
            'satellites': 0,
            'hdop': 99.9
        }
        
        # System status
        self.last_capture_time = None
        self.processing_status = "Ready"
        self.error_message = ""
        
        # --- MODIFIED: JPEG streams for all 6 panels ---
        self.rgb_jpg = None           # Aligned RGB
        self.nir_jpg = None           # Aligned NIR
        self.ai_prediction_jpg = None # AI Prediction (result stream)
        self.ndvi_heatmap_jpg = None  # NEW
        self.rdvi_heatmap_jpg = None  # NEW
        self.segmentation_jpg = None  # NEW

state = SystemState()

# ====================== CAMERA MANAGEMENT (MODIFIED) ======================
class CameraManager:
    def __init__(self):
        self.rgb_cam = None
        self.nir_cam = None
        self.running = False
        
    def initialize_cameras(self):
        """Initialize RGB and NIR cameras"""
        if not CAMERA_AVAILABLE:
            logger.warning("Camera hardware not available - using simulation")
            return True
            
        try:
            # RGB Camera
            self.rgb_cam = Picamera2(camera_num=SystemConfig.RGB_CAMERA_INDEX)
            rgb_config = self.rgb_cam.create_video_configuration(
                main={"size": SystemConfig.IMAGE_SIZE, "format": "RGB888"},
                buffer_count=4
            )
            self.rgb_cam.configure(rgb_config)
            self.rgb_cam.start()
            time.sleep(0.5)
            
            # NIR Camera (NoIR)
            self.nir_cam = Picamera2(camera_num=SystemConfig.NIR_CAMERA_INDEX)
            nir_config = self.nir_cam.create_video_configuration(
                main={"size": SystemConfig.IMAGE_SIZE, "format": "RGB888"},
                buffer_count=4
            )
            self.nir_cam.configure(nir_config)
            self.nir_cam.start()
            time.sleep(0.5)
            
            # --- *** MODIFIED: Set separate camera controls *** ---
            
            # Set camera controls
            frame_period_us = int(1_000_000 / SystemConfig.FPS)
            
            # --- RGB Camera (CAM0) Controls ---
            # Enable Auto-Exposure and Auto-White-Balance for good color
            self.rgb_cam.set_controls({
                "AeEnable": True,
                "AwbEnable": True,
                "FrameDurationLimits": (frame_period_us, frame_period_us),
            })
            logger.info("RGB Camera (CAM0) set to Auto-Exposure (AeEnable=True) and Auto-White-Balance (AwbEnable=True)")

            # --- NIR Camera (CAM1) Controls ---
            # Enable Auto-Exposure. This is CRITICAL for the 720nm filter.
            # Disable Auto-White-Balance (AWB makes no sense for NIR).
            self.nir_cam.set_controls({
                "AeEnable": True,
                "AwbEnable": False,
                "FrameDurationLimits": (frame_period_us, frame_period_us),
            })
            logger.info("NIR Camera (CAM1) set to Auto-Exposure (AeEnable=True) for 720nm filter compatibility")
            # --- *** END MODIFICATION *** ---

            logger.info("Cameras initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def capture_frame_pair(self):
        """Capture synchronized RGB and NIR frames"""
        if not CAMERA_AVAILABLE:
            # Simulation mode - create dummy frames
            h, w = SystemConfig.IMAGE_SIZE[1], SystemConfig.IMAGE_SIZE[0]
            rgb_frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            nir_frame = np.random.randint(0, 255, (h, w), dtype=np.uint8)
            return rgb_frame, nir_frame
            
        try:
            # Capture RGB frame
            rgb_array = self.rgb_cam.capture_array("main")
            rgb_frame = cv.cvtColor(rgb_array, cv.COLOR_RGB2BGR)
            
            # Capture NIR frame
            nir_array = self.nir_cam.capture_array("main")
            nir_frame = cv.cvtColor(nir_array, cv.COLOR_RGB2GRAY)
            
            return rgb_frame, nir_frame
            
        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            return None, None
    
    def stop(self):
        """Stop cameras"""
        if CAMERA_AVAILABLE:
            try:
                if self.rgb_cam:
                    self.rgb_cam.stop()
                if self.nir_cam:
                    self.nir_cam.stop()
            except:
                pass

# ====================== PROCESSING ENGINE ======================
class ProcessingEngine:
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
        """Initialize processing components"""
        try:
            # Initialize image preprocessor
            calib_data = create_zhang_calibration_data()
            self.preprocessor = ZhangCorePreprocessor(SystemConfig.CALIB_MAPS_PATH, calib_data)
            
            # Load AI models if available
            if os.path.exists(SystemConfig.MODEL_PATH):
                self.rf_model = joblib.load(SystemConfig.MODEL_PATH)
                logger.info("Random Forest model loaded")
            else:
                logger.warning(f"AI Model not found at {SystemConfig.MODEL_PATH}. AI inference will be disabled.")
            
            if os.path.exists(SystemConfig.IMPUTER_PATH):
                self.imputer = joblib.load(SystemConfig.IMPUTER_PATH)
                logger.info("Feature imputer loaded")
            else:
                if self.rf_model: # Only a problem if the model is loaded
                    logger.error(f"Imputer not found at {SystemConfig.IMPUTER_PATH}. AI will fail.")
                    self.rf_model = None # Disable AI
            
            logger.info("Processing engine initialized")
            return True
            
        except Exception as e:
            logger.error(f"Processing engine initialization failed: {e}")
            return False
    
    def process_image_pair(self, rgb_frame, nir_frame):
        """
        Process RGB+NIR frame pair, performs AI inference,
        and updates ALL 6 JPEG streams in SystemState.
        """
        try:
            with state.lock:
                state.processing_status = "Aligning images..."
            
            # 1. Image alignment
            aligned_rgb, aligned_nir = self.preprocessor.rectify_image_pair(rgb_frame, nir_frame)
            logger.info("Image alignment completed")
            
            # 2. Prepare calibrated reflectance data
            red_channel = aligned_rgb[:,:,2]
            red_refl_01 = (red_channel.astype(np.float32) / 255.0)
            nir_refl_01 = (aligned_nir.astype(np.float32) / 255.0)

            # 3. Calculate all VIs
            vi_data = self.preprocessor.calculate_zhang_vegetation_indices(
                red_refl_01, nir_refl_01
            )
            
            # Get 2D arrays for heatmaps
            ndvi_array_2d = vi_data.ndvi
            rdvi_array_2d = vi_data.rdvi
            mean_ndvi = float(vi_data.mean_ndvi)
            
            
            with state.lock:
                state.processing_status = "Creating mask..."
            
            # --- *** MODIFIED SEGMENTATION LOGIC *** ---
            
            # 4. ALWAYS run the FAST mask generation for heatmaps
            cleaned_mask, hsv_nir_debug = create_hsv_nir_mask(
                aligned_rgb, aligned_nir
            )
            logger.info("Fast HSV+NIR mask generated for heatmaps")
            
            # This is the mask used for heatmaps
            combined_leaf_mask = cleaned_mask
            
            # Initialize variables for AI-specific results
            all_leaf_masks = []
            watershed_debug_img = None
            ai_prediction_image = aligned_rgb.copy() # Default AI panel is just the RGB
            
            # 5. Advanced AI Inference with structural analysis
            ai_prediction = None
            confidence = 0.0
            
            # ONLY run expensive segmentation IF the AI model is loaded
            if self.rf_model and self.imputer:
                try:
                    with state.lock:
                        state.processing_status = "Segmenting for AI..."
                    
                    # 5a. Run the SLOW (Watershed) segmentation
                    all_leaf_masks, watershed_debug_img = segment_leaves_from_mask(
                        cleaned_mask, 
                        aligned_rgb,
                        min_leaf_area=500,
                        watershed_min_distance=20
                    )
                    
                    if not all_leaf_masks:
                        logger.warning("AI: No leaves found by segmentation. Skipping analysis.")
                    
                    else:
                        logger.info(f"AI: Segmentation found {len(all_leaf_masks)} leaf regions")
                        with state.lock:
                            state.processing_status = "Running AI analysis..."
                        
                        # Process each segmented leaf
                        all_predictions = []
                        all_confidences = []
                        
                        for i, (leaf_id, leaf_mask) in enumerate(all_leaf_masks):
                            # ... (feature extraction logic copied from old file) ...
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
                            
                            # Annotate the result image
                            class_colors = {0: (0, 255, 0), 1: (0, 0, 255)}  # BGR
                            class_names = {0: 'Healthy', 1: 'Diseased'}
                            color = class_colors.get(leaf_prediction, (255, 255, 255))
                            contours, _ = cv.findContours(leaf_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                            cv.drawContours(ai_prediction_image, contours, -1, color, 3)
                            
                            if contours:
                                M = cv.moments(contours[0])
                                if M["m00"] != 0:
                                    cX = int(M["m10"] / M["m00"])
                                    cY = int(M["m01"] / M["m00"])
                                    
                                    # --- *** MODIFIED: Added leaf_id for numbering *** ---
                                    label_text = f"{leaf_id}: {class_names[leaf_prediction]} ({leaf_confidence:.2f})"
                                    
                                    cv.putText(ai_prediction_image, label_text, (cX-50, cY), 
                                             cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # --- *** CHANGE 1: Python Verdict Logic (2-Label) *** ---
                        # Determine overall plant health
                        if all_predictions:
                            # MODIFIED: Check if any leaf was predicted as 'Diseased' (label 1)
                            diseased_leaves_present = any(p == 1 for p in all_predictions)
                            
                            if diseased_leaves_present:
                                ai_prediction = 'Diseased'
                                # Get the average confidence of only the 'Diseased' predictions
                                confidence = float(np.mean([conf for pred, conf in zip(all_predictions, all_confidences) if pred == 1]))
                            else:
                                ai_prediction = 'Healthy'
                                # Get the average confidence of all the 'Healthy' predictions
                                confidence = float(np.mean(all_confidences))
                        
                        logger.info(f"AI analysis complete: {ai_prediction} (confidence: {confidence:.3f})")
                
                except Exception as e:
                    logger.warning(f"AI inference failed: {e}")
                    # If AI fails, the AI panel will just be the base RGB image
            
            # 6. Fallback for Segmentation Panel
            # If AI is off (or failed), watershed_debug_img will be None.
            # In that case, use the simple hsv_nir_debug as the segmentation panel.
            if watershed_debug_img is None:
                logger.info("AI model not loaded; using simple mask for segmentation view.")
                watershed_debug_img = hsv_nir_debug # Use the fast mask debug
            
            # 7. Calculate overall vegetation indices for display
            vegetation_indices = {
                'ndvi_mean': vi_data.mean_ndvi,
                'ndvi_std': vi_data.std_ndvi,
                'rdvi_mean': vi_data.mean_rdvi,
                'rdvi_std': vi_data.std_rdvi,
                'sr_mean': vi_data.mean_sr,
                'sr_std': vi_data.std_sr,
                'leaves_detected': len(all_leaf_masks) # Will be 0 if AI is off
            }
            
            # --- *** NEW: Encode all 6 panels for streaming *** ---
            logger.info("Encoding all 6 stream panels...")
            target_size = SystemConfig.STREAM_SIZE # (640, 480)
            jpeg_quality = [int(cv.IMWRITE_JPEG_QUALITY), 75]

            # 1. Aligned RGB
            _, rgb_jpg_bytes = cv.imencode('.jpg', cv.resize(aligned_rgb, target_size), jpeg_quality)
            # 2. Aligned NIR (convert to BGR for streaming)
            _, nir_jpg_bytes = cv.imencode('.jpg', cv.resize(cv.cvtColor(aligned_nir, cv.COLOR_GRAY2BGR), target_size), jpeg_quality)
            # 3. AI Prediction (was 'result')
            _, ai_prediction_jpg_bytes = cv.imencode('.jpg', cv.resize(ai_prediction_image, target_size), jpeg_quality)
            
            # 4. NDVI Heatmap (with color bar and masked by combined_leaf_mask)
            panel_ndvi_bgr_with_colorbar = create_heatmap_bgr(ndvi_array_2d, combined_leaf_mask, cmap_name='RdYlGn', vmin=-1.0, vmax=1.0, add_colorbar=True)
            # Resize after adding color bar to keep it proportionally scaled
            _, ndvi_jpg_bytes = cv.imencode('.jpg', cv.resize(panel_ndvi_bgr_with_colorbar, target_size, interpolation=cv.INTER_LINEAR), jpeg_quality)
            
            # 5. RDVI Heatmap (with color bar and masked by combined_leaf_mask)
            panel_rdvi_bgr_with_colorbar = create_heatmap_bgr(rdvi_array_2d, combined_leaf_mask, cmap_name='RdYlGn', vmin=-1.5, vmax=1.5, add_colorbar=True)
            # Resize after adding color bar
            _, rdvi_jpg_bytes = cv.imencode('.jpg', cv.resize(panel_rdvi_bgr_with_colorbar, target_size, interpolation=cv.INTER_LINEAR), jpeg_quality)
            
            # 6. Segmentation (uses watershed_debug_img, which has a fallback)
            _, seg_jpg_bytes = cv.imencode('.jpg', cv.resize(watershed_debug_img, target_size, interpolation=cv.INTER_NEAREST), jpeg_quality)
            
            # Update state with ALL data
            with state.lock:
                # Store full-res images
                state.aligned_rgb = aligned_rgb
                state.aligned_nir = aligned_nir
                state.segmentation_result_full = ai_prediction_image
                
                # Store JPEGs for streaming
                state.rgb_jpg = rgb_jpg_bytes.tobytes()
                state.nir_jpg = nir_jpg_bytes.tobytes()
                state.ai_prediction_jpg = ai_prediction_jpg_bytes.tobytes() # 'result' stream is AI Prediction
                state.ndvi_heatmap_jpg = ndvi_jpg_bytes.tobytes()
                state.rdvi_heatmap_jpg = rdvi_jpg_bytes.tobytes()
                state.segmentation_jpg = seg_jpg_bytes.tobytes()

                # Store data
                state.current_ndvi_value = mean_ndvi
                state.vegetation_indices = vegetation_indices
                state.ai_prediction = ai_prediction
                state.confidence = confidence
                state.processing_status = "Complete"
                state.last_capture_time = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            with state.lock:
                state.processing_status = "Error"
                state.error_message = str(e)
            return False

# ====================== GPS MANAGER ======================
class GPSManager:
    def __init__(self):
        self.gps = None
        self.running = False
        
    def initialize(self):
        """Initialize GPS module"""
        try:
            self.gps = L76X() 
            logger.info("GPS module initialized and configured to 115200 baud.")
            return True
        except Exception as e:
            logger.warning(f"GPS initialization failed: {e}")
            return False
    
    def update_loop(self):
        """GPS update loop - runs reliably due to timeout in L76X.py"""
        while self.running:
            try:
                if self.gps:
                    self.gps.L76X_Gat_GNRMC() 
                    
                    with state.lock:
                        state.gps_data.update({
                            'latitude': self.gps.Lat,
                            'longitude': self.gps.Lon,
                            'altitude': 0.0, 
                            'status': 'Fix' if self.gps.Status else 'No Fix',
                            'satellites': self.gps.satellites_in_use,
                            'hdop': self.gps.hdop,
                            'fix_quality': self.gps.fix_quality
                        })
                
                time.sleep(SystemConfig.GPS_UPDATE_INTERVAL) 
                
            except Exception as e:
                logger.warning(f"GPS update error: {e}")
                time.sleep(5)
    
    def start(self):
        """Start GPS monitoring"""
        if self.initialize():
            self.running = True
            gps_thread = threading.Thread(target=self.update_loop, daemon=True)
            gps_thread.start()
            logger.info("GPS monitoring started")

# ====================== DATAHUB CONNECTOR ======================
class DataHubConnector:
    def __init__(self):
        self.client = None
        self.connected = False
        
    def initialize(self):
        """Initialize MQTT connection to DataHub"""
        try:
            self.client = mqtt.Client(client_id=SystemConfig.DEVICE_ID)
            self.client.username_pw_set(SystemConfig.DEVICE_ID, SystemConfig.API_KEY)
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            
            self.client.connect(SystemConfig.DATAHUB_HOST, SystemConfig.DATAHUB_PORT, 60)
            self.client.loop_start()
            
            logger.info("DataHub connection initialized")
            return True
            
        except Exception as e:
            logger.error(f"DataHub initialization failed: {e}")
            return False
    
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            logger.info("Connected to WISE-IoT DataHub")
        else:
            logger.error(f"DataHub connection failed with code {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        logger.warning("Disconnected from DataHub")
    
    def send_data(self, data):
        """Send data to DataHub"""
        if not self.connected:
            logger.warning("DataHub not connected - data not sent")
            return False
            
        try:
            topic = f"v1/devices/{SystemConfig.DEVICE_ID}/telemetry"
            payload = json.dumps(data)
            
            result = self.client.publish(topic, payload)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info("Data sent to DataHub successfully")
                return True
            else:
                logger.error(f"Failed to send data to DataHub: {result.rc}")
                return False
                
        except Exception as e:
            logger.error(f"DataHub send error: {e}")
            return False

# ====================== UTILITY FUNCTIONS ======================

def get_local_ip():
    """Dynamically find the Pi's local network IP address."""
    # Try using netifaces first for robustness
    try:
        for interface in netifaces.interfaces():
            if interface in ('eth0', 'wlan0'): # Check common interfaces
                addresses = netifaces.ifaddresses(interface)
                if netifaces.AF_INET in addresses:
                    return addresses[netifaces.AF_INET][0]['addr']
    except Exception:
        pass

    # Fallback method using socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Connect to an external host (doesn't send data) to determine the local IP
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
        s.close()
        return IP
    except Exception:
        return '127.0.0.1' # Default to localhost if no network is available

# ====================== WEB INTERFACE ======================
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
        /* --- MODIFIED: 3-Column Grid for streams --- */
        .camera-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
            margin-bottom: 15px; /* Added margin to separate rows */
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
            object-fit: contain; /* Changed from cover to contain */
            display: block;
            background: #333; /* Dark background for loading */
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
        .vi-list {
            list-style: none;
        }
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
        @media (max-width: 1200px) {
            .camera-grid {
                grid-template-columns: 1fr 1fr; /* 2 columns on medium screens */
            }
        }
        @media (max-width: 768px) {
            .grid, .camera-grid, .results-grid {
                grid-template-columns: 1fr; /* 1 column on small screens */
            }
            .header h1 {
                font-size: 2em;
            }
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
                <div class="camera-label">📸 Aligned RGB</div>
                <img src="/stream/rgb" alt="Aligned RGB Feed" id="rgb-stream">
            </div>
            <div class="camera-view">
                <div class="camera-label">🔴 Aligned NIR</div>
                <img src="/stream/nir" alt="Aligned NIR Feed" id="nir-stream">
            </div>
            <div class="camera-view">
                <div class="camera-label">🔬 AI Prediction</div>
                <img src="/stream/ai_prediction" alt="AI Prediction Result" id="ai_prediction-stream">
            </div>
        </div>
        
        <div class="camera-grid">
            <div class="camera-view">
                <div class="camera-label">🟢 NDVI Heatmap</div>
                <img src="/stream/ndvi" alt="NDVI Heatmap" id="ndvi-stream">
            </div>
            <div class="camera-view">
                <div class="camera-label">🟡 RDVI Heatmap</div>
                <img src="/stream/rdvi" alt="RDVI Heatmap" id="rdvi-stream">
            </div>
            <div class="camera-view">
                <div class="camera-label">🎨 Segmentation</div>
                <img src="/stream/segmentation" alt="Segmentation Output" id="segmentation-stream">
            </div>
        </div>
        <div class="controls">
            <button class="btn" onclick="captureImage()" id="capture-btn">
                📷 Capture & Process
            </button>
            <button class="btn inference" onclick="runInference()" id="inference-btn">
                🤖 Re-Analyze
            </button>
            <button class="btn" onclick="sendToDataHub()" id="datahub-btn">
                ☁️ Send to DataHub
            </button>
        </div>

        <div class="grid">
            <div class="card">
                <h3>📊 System Status</h3>
                <div class="status-grid">
                    <div class="status-item">
                        <div class="status-label">Processing</div>
                        <div class="status-value" id="processing-status">Ready</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">Last Capture</div>
                        <div class="status-value" id="last-capture">None</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">NDVI Value</div>
                        <div class="status-value" id="ndvi-value">--</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">Confidence</div>
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
                    <h4>Vegetation Indices</h4>
                    <ul class="vi-list" id="vi-list">
                        </ul>
                </div>
                <div>
                    <h4>AI Prediction</h4>
                    <div id="ai-results">
                        </div>
                </div>
            </div>
        </div>

        <div class="message" id="message" style="display: none;">
            System ready for plant health monitoring
        </div>
    </div>

    <script>
        // --- (JavaScript is unchanged, it just works with the status) ---
        // Update status periodically
        async function updateStatus() {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                
                // Update system status
                document.getElementById('processing-status').textContent = data.processing_status;
                document.getElementById('ndvi-value').textContent = 
                    data.current_ndvi !== undefined ? data.current_ndvi.toFixed(4) : '--';
                document.getElementById('confidence').textContent = 
                    data.confidence ? (data.confidence * 100).toFixed(1) + '%' : '--';
                
                if (data.last_capture_time) {
                    document.getElementById('last-capture').textContent = 
                        new Date(data.last_capture_time).toLocaleTimeString();
                }
                
                // Update GPS status
                document.getElementById('gps-status').textContent = data.gps_status;
                document.getElementById('gps-lat').textContent = 
                    data.gps_lat ? data.gps_lat.toFixed(6) : '--';
                document.getElementById('gps-lon').textContent = 
                    data.gps_lon ? data.gps_lon.toFixed(6) : '--';
                document.getElementById('gps-sats').textContent = data.gps_satellites || '--';
                
                // Update health status
                if (data.ai_prediction) {
                    const healthDiv = document.getElementById('health-status');
                    healthDiv.style.display = 'block';
                    healthDiv.textContent = `Plant Status: ${data.ai_prediction}`;
                    
                    // --- *** CHANGE 2: JavaScript Banner Logic (2-Label) *** ---
                    // Set appropriate color
                    healthDiv.className = 'health-status';
                    if (data.ai_prediction === 'Healthy') {
                        healthDiv.classList.add('health-healthy');
                    } else if (data.ai_prediction === 'Diseased') { // MODIFIED
                        healthDiv.classList.add('health-danger'); // MODIFIED (Red for any disease)
                    } else { // Fallback for null or other statuses
                        healthDiv.classList.add('health-warning');
                    }
                }
                
                // Update vegetation indices
                if (data.vegetation_indices) {
                    const viList = document.getElementById('vi-list');
                    viList.innerHTML = '';
                    
                    for (const [key, value] of Object.entries(data.vegetation_indices)) {
                        const li = document.createElement('li');
                        li.innerHTML = `<span>${key.toUpperCase().replace('_', ' ')}</span><span>${value.toFixed(4)}</span>`;
                        viList.appendChild(li);
                    }
                    
                    document.getElementById('results-section').style.display = 'block';
                }
                
                // Update AI results
                if (data.ai_prediction) {
                    document.getElementById('ai-results').innerHTML = `
                        <p><strong>Prediction:</strong> ${data.ai_prediction}</p>
                        <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%</p>
                    `;
                }
                
                // Show messages
                if (data.error_message) {
                    showMessage(data.error_message, 'error');
                }
                
            } catch (error) {
                console.error('Status update failed:', error);
            }
        }
        
        async function captureImage() {
            const btn = document.getElementById('capture-btn');
            btn.disabled = true;
            btn.textContent = '⏳ Processing...';
            
            try {
                const response = await fetch('/capture', { method: 'POST' });
                const data = await response.json();
                
                if (data.success) {
                    showMessage('Image captured and processed successfully!', 'success');
                    // --- NEW: Force refresh streams ---
                    refreshStreams();
                } else {
                    showMessage(data.error || 'Capture failed', 'error');
                }
            } catch (error) {
                showMessage('Capture request failed: ' + error.message, 'error');
            } finally {
                btn.disabled = false;
                btn.textContent = '📷 Capture & Process';
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
                    // --- NEW: Force refresh streams ---
                    refreshStreams();
                } else {
                    showMessage(data.error || 'AI analysis failed', 'error');
                }
            } catch (error) {
                    showMessage('AI analysis request failed: ' + error.message, 'error');
            } finally {
                btn.disabled = false;
                btn.textContent = '🤖 Re-Analyze';
            }
        }
        
        // --- NEW: Function to force-refresh streams ---
        function refreshStreams() {
            // Add a cache-busting query parameter
            const timestamp = new Date().getTime();
            document.getElementById('rgb-stream').src = '/stream/rgb?' + timestamp;
            document.getElementById('nir-stream').src = '/stream/nir?' + timestamp;
            document.getElementById('ai_prediction-stream').src = '/stream/ai_prediction?' + timestamp; // Corrected ID
            document.getElementById('ndvi-stream').src = '/stream/ndvi?' + timestamp;
            document.getElementById('rdvi-stream').src = '/stream/rdvi?' + timestamp;
            document.getElementById('segmentation-stream').src = '/stream/segmentation?' + timestamp;
        }

        async function sendToDataHub() {
            const btn = document.getElementById('datahub-btn');
            btn.disabled = true;
            btn.textContent = '☁️ Sending...';
            
            try {
                const response = await fetch('/send_datahub', { method: 'POST' });
                const data = await response.json();
                
                if (data.success) {
                    showMessage('Data sent to WISE-IoT DataHub successfully!', 'success');
                } else {
                    showMessage(data.error || 'DataHub send failed', 'error');
                }
            } catch (error) {
                showMessage('DataHub request failed: ' + error.message, 'error');
            } finally {
                btn.disabled = false;
                btn.textContent = '☁️ Send to DataHub';
            }
        }
        
        function showMessage(text, type = 'info') {
            const messageDiv = document.getElementById('message');
            messageDiv.textContent = text;
            messageDiv.style.display = 'block';
            
            // Set color based on type
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
            
            // Hide after 5 seconds
            setTimeout(() => {
                messageDiv.style.display = 'none';
            }, 5000);
        }
        
        // Update status every 2 seconds
        setInterval(updateStatus, 2000);
        updateStatus(); // Initial update
    </script>
</body>
</html>
"""

def create_web_app():
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)
    
    # --- MODIFIED: This single endpoint now handles all 6 streams ---
    @app.route('/stream/<stream_type>')
    def video_stream(stream_type):
        def generate_stream():
            while True:
                frame_data = None
                with state.lock:
                    if stream_type == 'rgb' and state.rgb_jpg is not None:
                        frame_data = state.rgb_jpg
                    elif stream_type == 'nir' and state.nir_jpg is not None:
                        frame_data = state.nir_jpg
                    # --- Renamed from result_jpg to ai_prediction_jpg in state ---
                    elif stream_type == 'ai_prediction' and state.ai_prediction_jpg is not None:
                        frame_data = state.ai_prediction_jpg
                    elif stream_type == 'ndvi' and state.ndvi_heatmap_jpg is not None:
                        frame_data = state.ndvi_heatmap_jpg
                    elif stream_type == 'rdvi' and state.rdvi_heatmap_jpg is not None:
                        frame_data = state.rdvi_heatmap_jpg
                    elif stream_type == 'segmentation' and state.segmentation_jpg is not None:
                        frame_data = state.segmentation_jpg
                
                if frame_data is None:
                    # Generate placeholder image if data is not ready
                    placeholder = np.zeros((SystemConfig.STREAM_SIZE[1], SystemConfig.STREAM_SIZE[0], 3), dtype=np.uint8)
                    cv.putText(placeholder, f"{stream_type.upper()} loading...", (180, 240), 
                             cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    _, frame_data_bytes = cv.imencode('.jpg', placeholder)
                    frame_data = frame_data_bytes.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame_data)).encode() + b'\r\n\r\n' +
                       frame_data + b'\r\n')
                time.sleep(0.1)
        
        return Response(generate_stream(), 
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    
    @app.route('/status')
    def get_status():
        with state.lock:
            return jsonify({
                'processing_status': state.processing_status,
                'current_ndvi': state.current_ndvi_value,
                'vegetation_indices': state.vegetation_indices,
                'ai_prediction': state.ai_prediction,
                'confidence': state.confidence,
                'last_capture_time': state.last_capture_time.isoformat() if state.last_capture_time else None,
                'gps_status': state.gps_data['status'],
                'gps_lat': state.gps_data['latitude'],
                'gps_lon': state.gps_data['longitude'],
                'gps_satellites': state.gps_data['satellites'],
                'error_message': state.error_message
            })
    
    # --- MODIFIED: Capture route now just captures and processes ---
    @app.route('/capture', methods=['POST'])
    def capture_image():
        try:
            # Capture frame pair
            rgb_frame, nir_frame = camera_manager.capture_frame_pair()
            
            if rgb_frame is None or nir_frame is None:
                return jsonify({'success': False, 'error': 'Frame capture failed'})
            
            # Store raw frames in state (ProcessingEngine will use them)
            with state.lock:
                state.rgb_frame = rgb_frame
                state.nir_frame = nir_frame
            
            # Process the images (this function now updates all 6 JPEGs)
            success = processing_engine.process_image_pair(rgb_frame, nir_frame)
            
            if success:
                return jsonify({'success': True, 'message': 'Image processed successfully'})
            else:
                return jsonify({'success': False, 'error': 'Image processing failed'})
                
        except Exception as e:
            logger.error(f"Capture endpoint error: {e}")
            return jsonify({'success': False, 'error': str(e)})
    
    # --- MODIFIED: Inference route re-processes existing frames ---
    @app.route('/inference', methods=['POST'])
    def run_inference():
        try:
            with state.lock:
                if state.rgb_frame is None or state.nir_frame is None:
                    return jsonify({'success': False, 'error': 'No captured images available. Please capture first.'})
                # Get the last captured frames
                rgb_frame = state.rgb_frame.copy()
                nir_frame = state.nir_frame.copy()
            
            # Re-run processing (this updates all 6 JPEGs)
            success = processing_engine.process_image_pair(rgb_frame, nir_frame)
            
            if success:
                return jsonify({'success': True, 'message': 'AI inference completed'})
            else:
                return jsonify({'success': False, 'error': 'AI inference failed'})
                
        except Exception as e:
            logger.error(f"Inference endpoint error: {e}")
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/send_datahub', methods=['POST'])
    def send_to_datahub():
        try:
            # Prepare data payload
            with state.lock:
                payload = {
                    'timestamp': datetime.now().isoformat(),
                    'device_id': SystemConfig.DEVICE_ID,
                    'ndvi_value': state.current_ndvi_value,
                    'vegetation_indices': state.vegetation_indices,
                    'ai_prediction': state.ai_prediction,
                    'confidence': state.confidence,
                    'gps_data': state.gps_data.copy(),
                    'session_id': str(uuid.uuid4())
                }
            
            # Send to DataHub
            success = datahub_connector.send_data(payload)
            
            if success:
                return jsonify({'success': True, 'message': 'Data sent to DataHub'})
            else:
                return jsonify({'success': False, 'error': 'DataHub send failed'})
                
        except Exception as e:
            logger.error(f"DataHub endpoint error: {e}")
            return jsonify({'success': False, 'error': str(e)})
    
    return app

# ====================== MAIN SYSTEM ======================
def main():
    global camera_manager, processing_engine, gps_manager, datahub_connector
    
    print("=" * 60)
    print("🌱 AIoT Plant Health Monitor System")
    print("VGU AIoT 2025 Competition")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(SystemConfig.OUTPUT_DIR, exist_ok=True)
    
    # Initialize components
    camera_manager = CameraManager()
    processing_engine = ProcessingEngine()
    gps_manager = GPSManager()
    datahub_connector = DataHubConnector()
    
    # Initialize hardware and services
    print("\n🔧 Initializing components...")
    
    # Camera initialization
    if camera_manager.initialize_cameras():
        logger.info("✅ Cameras initialized")
    else:
        logger.warning("⚠️ Camera initialization failed - using simulation mode")
    
    # Processing engine
    if processing_engine.initialize():
        logger.info("✅ Processing engine initialized")
    else:
        logger.error("❌ Processing engine initialization failed")
        return
    
    # GPS (optional)
    gps_manager.start()
    
    # DataHub (optional)
    if datahub_connector.initialize():
        logger.info("✅ DataHub connected")
    else:
        logger.warning("⚠️ DataHub connection failed - continuing without cloud sync")
    
    # --- START WEB INTERFACE & PRINT LINK ---
    local_ip = get_local_ip()
    hostname = socket.gethostname() # Should return 'agritech'

    print("\n🌐 Starting web interface...")
    
    # Print the specific access links
    print(f"   Access via IP: http://{local_ip}:{SystemConfig.WEB_PORT}/")
    print(f"   Access via Hostname: http://{hostname}.local:{SystemConfig.WEB_PORT}/")
    
    # --- OUTPUT THE CLICKABLE LINK (MAIN OUTPUT) ---
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
        # Cleanup
        camera_manager.stop()
        if datahub_connector.client:
            datahub_connector.client.loop_stop()
            datahub_connector.client.disconnect()

if __name__ == '__main__':
    main()
