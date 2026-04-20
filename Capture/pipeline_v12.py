#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete AIoT Plant Health Assessment System
AIoT 2025 Competition - VGU
Integrates: RGB+NIR capture, alignment, NDVI, AI inference, GPS, WISE-IoT DataHub

*** MODIFIED v12: Added continuous capture thread for live streaming
    - Background thread continuously updates RGB/NIR preview streams
    - Heavy AI processing only runs when "Capture & Process" is clicked
    - All devices on network can see live preview
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
    WEB_PORT = 8080
    
    # Stream settings
    STREAM_SIZE = (640, 480)
    AI_CONFIDENCE_THRESHOLD = 0.80

# ====================== DASHBOARD HELPER FUNCTIONS ======================

def create_heatmap_bgr(vi_array: np.ndarray, mask: np.ndarray, cmap_name: str = 'RdYlGn', vmin: float = -1.0, vmax: float = 1.0, add_colorbar: bool = True) -> np.ndarray:
    """
    Converts a VI array (float) into a BGR heatmap (uint8), applying a mask
    and optionally adding a color bar.
    """
    if vi_array is None or vi_array.size == 0:
        return np.zeros((SystemConfig.STREAM_SIZE[1], SystemConfig.STREAM_SIZE[0], 3), dtype=np.uint8)
    
    # Ensure mask is boolean and same size
    if mask is None or mask.shape != vi_array.shape:
        logger.warning("Mask is invalid or missing, coloring entire image.")
        mask = np.ones_like(vi_array, dtype=bool)
    else:
        mask = mask.astype(bool)

    # Initialize a blank image for the heatmap
    heatmap_base = np.zeros_like(vi_array, dtype=np.float32)
    heatmap_base[mask] = vi_array[mask]
    
    # Handle NaNs before normalization
    vi_array_nonan = np.nan_to_num(heatmap_base, nan=vmin) 
    
    # Normalize values within the valid range
    normalized = (vi_array_nonan - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0.0, 1.0)
    
    # Get colormap
    cmap = plt.get_cmap(cmap_name)
    colored_rgba = cmap(normalized) 
    
    # Convert to BGR (0-255) and remove alpha channel
    colored_bgr_uint8 = (colored_rgba[:, :, 2::-1] * 255).astype(np.uint8)
    
    # Set areas outside the mask to a background color
    background_color = [200, 230, 255]
    colored_bgr_uint8[~mask] = background_color
    
    # Add Color Bar
    if add_colorbar:
        cb_width = 30
        cb_height = colored_bgr_uint8.shape[0]
        cb_padding = 10
        total_width = colored_bgr_uint8.shape[1] + cb_width + cb_padding + 40
        
        final_image = np.full((cb_height, total_width, 3), background_color, dtype=np.uint8)
        final_image[:, :colored_bgr_uint8.shape[1]] = colored_bgr_uint8
        
        gradient = np.linspace(vmax, vmin, cb_height)
        gradient_normalized = (gradient - vmin) / (vmax - vmin)
        gradient_normalized = np.clip(gradient_normalized, 0.0, 1.0)
        gradient_colors_rgba = cmap(gradient_normalized)
        gradient_colors_bgr = (gradient_colors_rgba[:, 2::-1] * 255).astype(np.uint8)
        
        cb_x_start = colored_bgr_uint8.shape[1] + cb_padding
        cb_x_end = cb_x_start + cb_width
        
        for i in range(cb_height):
            final_image[i, cb_x_start:cb_x_end] = gradient_colors_bgr[i]
            
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_color = (0, 0, 0)
        
        cv.putText(final_image, f"{vmax:.1f}", 
                   (cb_x_end + 5, 20),
                   font, font_scale, text_color, font_thickness, cv.LINE_AA)
        if vmin < 0.0 < vmax:
            mid_y = int(cb_height * (vmax - 0.0) / (vmax - vmin))
            cv.putText(final_image, "0.0", 
                       (cb_x_end + 5, mid_y),
                       font, font_scale, text_color, font_thickness, cv.LINE_AA)
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
        self.segmentation_result_full = None

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
        
        # JPEG streams for all 6 panels
        self.rgb_jpg = None
        self.nir_jpg = None
        self.ai_prediction_jpg = None
        self.ndvi_heatmap_jpg = None
        self.rdvi_heatmap_jpg = None
        self.segmentation_jpg = None

state = SystemState()

# ====================== CAMERA MANAGEMENT ======================
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
            
            # Set camera controls
            frame_period_us = int(1_000_000 / SystemConfig.FPS)
            
            # RGB Camera Controls
            self.rgb_cam.set_controls({
                "AeEnable": True,
                "AwbEnable": True,
                "FrameDurationLimits": (frame_period_us, frame_period_us),
            })
            logger.info("RGB Camera (CAM0) set to Auto-Exposure and Auto-White-Balance")

            # NIR Camera Controls
            self.nir_cam.set_controls({
                "AeEnable": True,
                "AwbEnable": False,
                "FrameDurationLimits": (frame_period_us, frame_period_us),
            })
            logger.info("NIR Camera (CAM1) set to Auto-Exposure for 720nm filter")

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

# ====================== CONTINUOUS CAPTURE THREAD ======================
class ContinuousCaptureThread:
    """Background thread for continuous camera preview streaming"""
    def __init__(self, camera_manager):
        self.camera_manager = camera_manager
        self.running = False
        self.thread = None
        
    def start(self):
        """Start the continuous capture thread"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info("Continuous capture thread started")
    
    def stop(self):
        """Stop the continuous capture thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def _capture_loop(self):
        """Continuously capture and encode frames for streaming"""
        target_size = SystemConfig.STREAM_SIZE
        jpeg_quality = [int(cv.IMWRITE_JPEG_QUALITY), 75]
        
        while self.running:
            try:
                # Capture frame pair
                rgb_frame, nir_frame = self.camera_manager.capture_frame_pair()
                
                if rgb_frame is None or nir_frame is None:
                    time.sleep(0.1)
                    continue
                
                # Encode preview frames (without heavy processing)
                rgb_resized = cv.resize(rgb_frame, target_size)
                nir_resized = cv.resize(nir_frame, target_size)
                nir_bgr = cv.cvtColor(nir_resized, cv.COLOR_GRAY2BGR)
                
                _, rgb_jpg = cv.imencode('.jpg', rgb_resized, jpeg_quality)
                _, nir_jpg = cv.imencode('.jpg', nir_bgr, jpeg_quality)
                
                # Update state with preview frames
                with state.lock:
                    state.rgb_jpg = rgb_jpg.tobytes()
                    state.nir_jpg = nir_jpg.tobytes()
                    
                    # Store raw frames for later processing
                    state.rgb_frame = rgb_frame
                    state.nir_frame = nir_frame
                
                # Limit frame rate to ~10 FPS for preview
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Continuous capture error: {e}")
                time.sleep(0.5)

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
                if self.rf_model:
                    logger.error(f"Imputer not found at {SystemConfig.IMPUTER_PATH}. AI will fail.")
                    self.rf_model = None
            
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
            ai_prediction_image = aligned_rgb.copy()
            
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
                        
                        # Determine overall plant health
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
                                healthy_confidences = [
                                    conf for pred, conf in zip(all_predictions, all_confidences) 
                                    if pred == 0
                                ]
                                if healthy_confidences:
                                    confidence = float(np.mean(healthy_confidences))
                                else:
                                    confidence = float(np.mean(all_confidences))
                        
                        logger.info(f"AI analysis complete: {ai_prediction} (confidence: {confidence:.3f})")
                
                except Exception as e:
                    logger.warning(f"AI inference failed: {e}")
            
            # 6. Fallback for Segmentation Panel
            if watershed_debug_img is None:
                logger.info("AI model not loaded; using simple mask for segmentation view.")
                watershed_debug_img = hsv_nir_debug
            
            # 7. Calculate overall vegetation indices for display
            vegetation_indices = {
                'ndvi_mean': vi_data.mean_ndvi,
                'ndvi_std': vi_data.std_ndvi,
                'rdvi_mean': vi_data.mean_rdvi,
                'rdvi_std': vi_data.std_rdvi,
                'sr_mean': vi_data.mean_sr,
                'sr_std': vi_data.std_sr,
                'leaves_detected': len(all_leaf_masks)
            }
            
            # 8. Encode all 6 panels for streaming
            logger.info("Encoding all 6 stream panels...")
            target_size = SystemConfig.STREAM_SIZE
            jpeg_quality = [int(cv.IMWRITE_JPEG_QUALITY), 75]

            _, rgb_jpg_bytes = cv.imencode('.jpg', cv.resize(aligned_rgb, target_size), jpeg_quality)
            _, nir_jpg_bytes = cv.imencode('.jpg', cv.resize(cv.cvtColor(aligned_nir, cv.COLOR_GRAY2BGR), target_size), jpeg_quality)
            _, ai_prediction_jpg_bytes = cv.imencode('.jpg', cv.resize(ai_prediction_image, target_size), jpeg_quality)
            
            panel_ndvi_bgr_with_colorbar = create_heatmap_bgr(ndvi_array_2d, combined_leaf_mask, cmap_name='RdYlGn', vmin=-1.0, vmax=1.0, add_colorbar=True)
            _, ndvi_jpg_bytes = cv.imencode('.jpg', cv.resize(panel_ndvi_bgr_with_colorbar, target_size, interpolation=cv.INTER_LINEAR), jpeg_quality)
            
            panel_rdvi_bgr_with_colorbar = create_heatmap_bgr(rdvi_array_2d, combined_leaf_mask, cmap_name='RdYlGn', vmin=-1.5, vmax=1.5, add_colorbar=True)
            _, rdvi_jpg_bytes = cv.imencode('.jpg', cv.resize(panel_rdvi_bgr_with_colorbar, target_size, interpolation=cv.INTER_LINEAR), jpeg_quality)
            
            _, seg_jpg_bytes = cv.imencode('.jpg', cv.resize(watershed_debug_img, target_size, interpolation=cv.INTER_NEAREST), jpeg_quality)
            
            # Update state with ALL data
            with state.lock:
                state.aligned_rgb = aligned_rgb
                state.aligned_nir = aligned_nir
                state.segmentation_result_full = ai_prediction_image
                
                state.rgb_jpg = rgb_jpg_bytes.tobytes()
                state.nir_jpg = nir_jpg_bytes.tobytes()
                state.ai_prediction_jpg = ai_prediction_jpg_bytes.tobytes()
                state.ndvi_heatmap_jpg = ndvi_jpg_bytes.tobytes()
                state.rdvi_heatmap_jpg = rdvi_jpg_bytes.tobytes()
                state.segmentation_jpg = seg_jpg_bytes.tobytes()

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
        """GPS update loop"""
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
    try:
        for interface in netifaces.interfaces():
            if interface in ('eth0', 'wlan0'):
                addresses = netifaces.ifaddresses(interface)
                if netifaces.AF_INET in addresses:
                    return addresses[netifaces.AF_INET][0]['addr']
    except Exception:
        pass

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
        s.close()
        return IP
    except Exception:
        return '127.0.0.1'

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
                grid-template-columns: 1fr 1fr;
            }
        }
        @media (max-width: 768px) {
            .grid, .camera-grid, .results-grid {
                grid-template-columns: 1fr;
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
                document.getElementById('gps-lat').textContent = 
                    data.gps_lat ? data.gps_lat.toFixed(6) : '--';
                document.getElementById('gps-lon').textContent = 
                    data.gps_lon ? data.gps_lon.toFixed(6) : '--';
                document.getElementById('gps-sats').textContent = data.gps_satellites || '--';
                
                if (data.ai_prediction) {
                    const healthDiv = document.getElementById('health-status');
                    healthDiv.style.display = 'block';
                    healthDiv.textContent = `Plant Status: ${data.ai_prediction}`;
                    
                    healthDiv.className = 'health-status';
                    if (data.ai_prediction === 'Healthy') {
                        healthDiv.classList.add('health-healthy');
                    } else if (data.ai_prediction === 'Diseased') { 
                        healthDiv.classList.add('health-danger'); 
                    } else {
                        healthDiv.classList.add('health-warning');
                    }
                }
                
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
                
                if (data.ai_prediction) {
                    document.getElementById('ai-results').innerHTML = `
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
        
        async function captureImage() {
            const btn = document.getElementById('capture-btn');
            btn.disabled = true;
            btn.textContent = '⏳ Processing...';
            
            try {
                const response = await fetch('/capture', { method: 'POST' });
                const data = await response.json();
                
                if (data.success) {
                    showMessage('Image captured and processed successfully!', 'success');
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
        
        function refreshStreams() {
            const timestamp = new Date().getTime();
            document.getElementById('rgb-stream').src = '/stream/rgb?' + timestamp;
            document.getElementById('nir-stream').src = '/stream/nir?' + timestamp;
            document.getElementById('ai_prediction-stream').src = '/stream/ai_prediction?' + timestamp;
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
        updateStatus();
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
                    if stream_type == 'rgb' and state.rgb_jpg is not None:
                        frame_data = state.rgb_jpg
                    elif stream_type == 'nir' and state.nir_jpg is not None:
                        frame_data = state.nir_jpg
                    elif stream_type == 'ai_prediction' and state.ai_prediction_jpg is not None:
                        frame_data = state.ai_prediction_jpg
                    elif stream_type == 'ndvi' and state.ndvi_heatmap_jpg is not None:
                        frame_data = state.ndvi_heatmap_jpg
                    elif stream_type == 'rdvi' and state.rdvi_heatmap_jpg is not None:
                        frame_data = state.rdvi_heatmap_jpg
                    elif stream_type == 'segmentation' and state.segmentation_jpg is not None:
                        frame_data = state.segmentation_jpg
                
                if frame_data is None:
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
    
    @app.route('/capture', methods=['POST'])
    def capture_image():
        try:
            rgb_frame, nir_frame = camera_manager.capture_frame_pair()
            
            if rgb_frame is None or nir_frame is None:
                return jsonify({'success': False, 'error': 'Frame capture failed'})
            
            with state.lock:
                state.rgb_frame = rgb_frame
                state.nir_frame = nir_frame
            
            success = processing_engine.process_image_pair(rgb_frame, nir_frame)
            
            if success:
                return jsonify({'success': True, 'message': 'Image processed successfully'})
            else:
                return jsonify({'success': False, 'error': 'Image processing failed'})
                
        except Exception as e:
            logger.error(f"Capture endpoint error: {e}")
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/inference', methods=['POST'])
    def run_inference():
        try:
            with state.lock:
                if state.rgb_frame is None or state.nir_frame is None:
                    return jsonify({'success': False, 'error': 'No captured images available. Please capture first.'})
                rgb_frame = state.rgb_frame.copy()
                nir_frame = state.nir_frame.copy()
            
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
    global camera_manager, processing_engine, gps_manager, datahub_connector, continuous_capture
    
    print("=" * 60)
    print("🌱 AIoT Plant Health Monitor System")
    print("VGU AIoT 2025 Competition")
    print("=" * 60)
    
    os.makedirs(SystemConfig.OUTPUT_DIR, exist_ok=True)
    
    camera_manager = CameraManager()
    processing_engine = ProcessingEngine()
    gps_manager = GPSManager()
    datahub_connector = DataHubConnector()
    continuous_capture = None
    
    print("\n🔧 Initializing components...")
    
    # Camera initialization
    if camera_manager.initialize_cameras():
        logger.info("✅ Cameras initialized")
        continuous_capture = ContinuousCaptureThread(camera_manager)
        continuous_capture.start()
        logger.info("✅ Live preview streaming started")
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
    
    # Start web interface
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
        if continuous_capture:
            continuous_capture.stop()
        camera_manager.stop()
        if datahub_connector.client:
            datahub_connector.client.loop_stop()
            datahub_connector.client.disconnect()

if __name__ == '__main__':
    main()
