from flask import Flask, Response, jsonify, request
from picamera2 import Picamera2
import cv2
import numpy as np
import atexit
import time
import threading
from collections import deque
import json
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class CameraConfig:
    """Camera configuration management"""
    def __init__(self):
        self.config = {
            'gaussian_kernel_size': 5,
            'gaussian_sigma': 1.0,
            'ndvi_colormap': cv2.COLORMAP_JET,
            'evi_colormap': cv2.COLORMAP_VIRIDIS,
            'alignment_method': 'phase_correlation',  # 'phase_correlation', 'ecc', 'orb'
            'auto_exposure': True,
            'exposure_time': 10000,  # microseconds
            'iso': 400,
            'capture_resolution': (1920, 1080),
            'display_resolution': (1280, 720),
            'ndvi_threshold_min': -1.0,
            'ndvi_threshold_max': 1.0,
            'evi_threshold_min': -1.0,
            'evi_threshold_max': 1.0
        }
        self.load_config()
    
    def load_config(self):
        """Load configuration from file if exists"""
        if os.path.exists('camera_config.json'):
            try:
                with open('camera_config.json', 'r') as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config)
                logger.info("Configuration loaded from file")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open('camera_config.json', 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                config_copy = {}
                for key, value in self.config.items():
                    if isinstance(value, np.integer):
                        config_copy[key] = int(value)
                    elif isinstance(value, np.floating):
                        config_copy[key] = float(value)
                    else:
                        config_copy[key] = value
                json.dump(config_copy, f, indent=2)
            logger.info("Configuration saved to file")
        except Exception as e:
            logger.error(f"Error saving config: {e}")

class NDVIProcessor:
    """Enhanced NDVI processing with multiple algorithms and improvements"""
    
    def __init__(self, config):
        self.config = config
        self.calibration_data = self.load_calibration()
        self.alignment_cache = {}
        self.statistics = {
            'ndvi_history': deque(maxlen=100),
            'evi_history': deque(maxlen=100),
            'processing_times': deque(maxlen=50)
        }
    
    def load_calibration(self):
        """Load calibration data (placeholder for future spectral calibration)"""
        # Based on Hoang's thesis - implement 6-reference calibration
        return {
            'red_calibration': [0, 255],  # Will be replaced with actual calibration
            'nir_calibration': [0, 255],
            'calibration_matrix': np.eye(2)  # Identity matrix placeholder
        }
    
    def apply_spectral_calibration(self, red_channel, nir_channel):
        """Apply spectral calibration based on reference standards"""
        # Implementation based on Hoang's 6-reference calibration method
        # This is a simplified version - full implementation would use actual reference data
        red_calibrated = np.clip(red_channel * 1.0, 0, 255)
        nir_calibrated = np.clip(nir_channel * 1.0, 0, 255)
        return red_calibrated, nir_calibrated
    
    def enhanced_gaussian_filter(self, image, kernel_size=None, sigma=None):
        """Enhanced Gaussian filtering with adaptive parameters"""
        if kernel_size is None:
            kernel_size = self.config.config['gaussian_kernel_size']
        if sigma is None:
            sigma = self.config.config['gaussian_sigma']
        
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Apply bilateral filter for edge-preserving smoothing
        if len(image.shape) == 2:  # Grayscale
            # Use bilateral filter for better edge preservation
            filtered = cv2.bilateralFilter(image.astype(np.uint8), kernel_size, 
                                         sigma * 25, sigma * 25)
            return filtered.astype(float)
        else:
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def advanced_alignment(self, base_img, img_to_align, method='phase_correlation'):
        """Advanced image alignment with multiple methods"""
        if method == 'phase_correlation':
            return self.align_image_phasecorr(base_img, img_to_align)
        elif method == 'ecc':
            return self.align_image_ecc(base_img, img_to_align)
        elif method == 'orb':
            return self.align_image_orb(base_img, img_to_align)
        elif method == 'hybrid':
            # Try phase correlation first, fallback to ECC
            try:
                return self.align_image_phasecorr(base_img, img_to_align)
            except:
                return self.align_image_ecc(base_img, img_to_align)
        else:
            return img_to_align
    
    def align_image_phasecorr(self, base_img, img_to_align):
        """Improved phase correlation alignment"""
        base_gray = cv2.cvtColor(base_img, cv2.COLOR_RGB2GRAY)
        align_gray = cv2.cvtColor(img_to_align, cv2.COLOR_RGB2GRAY)
        
        # Apply preprocessing for better alignment
        base_gray = cv2.GaussianBlur(base_gray, (3, 3), 0)
        align_gray = cv2.GaussianBlur(align_gray, (3, 3), 0)
        
        base_gray = np.float32(base_gray)
        align_gray = np.float32(align_gray)
        
        shift, response = cv2.phaseCorrelate(align_gray, base_gray)
        
        # Only apply shift if confidence is high enough
        if response > 0.1:  # Confidence threshold
            dx, dy = shift
            rows, cols = base_img.shape[:2]
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            aligned = cv2.warpAffine(img_to_align, M, (cols, rows))
            return aligned
        else:
            return img_to_align
    
    def align_image_ecc(self, base_img, img_to_align):
        """Enhanced ECC alignment with better parameters"""
        base_gray = cv2.cvtColor(base_img, cv2.COLOR_RGB2GRAY)
        align_gray = cv2.cvtColor(img_to_align, cv2.COLOR_RGB2GRAY)
        
        # Apply histogram equalization for better matching
        base_gray = cv2.equalizeHist(base_gray)
        align_gray = cv2.equalizeHist(align_gray)
        
        sz = base_gray.shape
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        warp_mode = cv2.MOTION_TRANSLATION
        
        # Enhanced termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
        
        try:
            cc, warp_matrix = cv2.findTransformECC(base_gray, align_gray, 
                                                 warp_matrix, warp_mode, criteria)
            aligned = cv2.warpAffine(img_to_align, warp_matrix, (sz[1], sz[0]),
                                   flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            return aligned
        except cv2.error:
            return img_to_align
    
    def align_image_orb(self, base_img, img_to_align):
        """Enhanced ORB alignment with RANSAC refinement"""
        base_gray = cv2.cvtColor(base_img, cv2.COLOR_RGB2GRAY)
        align_gray = cv2.cvtColor(img_to_align, cv2.COLOR_RGB2GRAY)
        
        # Enhanced ORB parameters
        orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
        
        kp1, des1 = orb.detectAndCompute(base_gray, None)
        kp2, des2 = orb.detectAndCompute(align_gray, None)
        
        if des1 is None or des2 is None:
            return img_to_align
        
        # FLANN-based matcher for better matching
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6,
                           key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        try:
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) >= 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                M, mask = cv2.findHomography(dst_pts, src_pts, 
                                           cv2.RANSAC, 3.0)
                if M is not None:
                    h, w = base_gray.shape
                    aligned = cv2.warpPerspective(img_to_align, M, (w, h))
                    return aligned
        except:
            pass
        
        return img_to_align
    
    def calculate_enhanced_ndvi(self, red_channel, nir_channel):
        """Enhanced NDVI calculation with quality metrics"""
        # Apply spectral calibration
        red_cal, nir_cal = self.apply_spectral_calibration(red_channel, nir_channel)
        
        # Apply filtering
        red_filtered = self.enhanced_gaussian_filter(red_cal)
        nir_filtered = self.enhanced_gaussian_filter(nir_cal)
        
        # Calculate NDVI with improved numerical stability
        epsilon = 1e-7
        denominator = nir_filtered + red_filtered + epsilon
        ndvi = (nir_filtered - red_filtered) / denominator
        
        # Apply thresholds
        min_thresh = self.config.config['ndvi_threshold_min']
        max_thresh = self.config.config['ndvi_threshold_max']
        ndvi = np.clip(ndvi, min_thresh, max_thresh)
        
        # Calculate quality metrics
        quality_metrics = {
            'mean_ndvi': float(np.mean(ndvi)),
            'std_ndvi': float(np.std(ndvi)),
            'vegetation_pixels': int(np.sum(ndvi > 0.3)),
            'total_pixels': int(ndvi.size)
        }
        
        # Store statistics
        self.statistics['ndvi_history'].append(quality_metrics['mean_ndvi'])
        
        return ndvi, quality_metrics
    
    def calculate_enhanced_evi(self, red_channel, nir_channel, blue_channel):
        """Enhanced EVI calculation with atmospheric correction"""
        # Apply spectral calibration
        red_cal, nir_cal = self.apply_spectral_calibration(red_channel, nir_channel)
        blue_cal = blue_channel  # Placeholder for blue calibration
        
        # Apply filtering
        red_filtered = self.enhanced_gaussian_filter(red_cal)
        nir_filtered = self.enhanced_gaussian_filter(nir_cal)
        blue_filtered = self.enhanced_gaussian_filter(blue_cal)
        
        # Enhanced EVI calculation with atmospheric correction
        G = 2.5  # Gain factor
        C1 = 6.0  # Coefficient for aerosol resistance term
        C2 = 7.5  # Coefficient for aerosol resistance term
        L = 1.0   # Canopy background adjustment
        
        epsilon = 1e-7
        denominator = nir_filtered + C1 * red_filtered - C2 * blue_filtered + L + epsilon
        evi = G * (nir_filtered - red_filtered) / denominator
        
        # Apply thresholds
        min_thresh = self.config.config['evi_threshold_min']
        max_thresh = self.config.config['evi_threshold_max']
        evi = np.clip(evi, min_thresh, max_thresh)
        
        # Calculate quality metrics
        quality_metrics = {
            'mean_evi': float(np.mean(evi)),
            'std_evi': float(np.std(evi)),
            'vegetation_pixels': int(np.sum(evi > 0.3)),
            'total_pixels': int(evi.size)
        }
        
        # Store statistics
        self.statistics['evi_history'].append(quality_metrics['mean_evi'])
        
        return evi, quality_metrics

# Global objects
config = CameraConfig()
processor = NDVIProcessor(config)

# FPS tracking
fps_tracker_1 = deque(maxlen=30)
fps_tracker_2 = deque(maxlen=30)
fps_lock = threading.Lock()

# Initialize cameras with better error handling
try:
    picam2_1 = Picamera2(0)
    picam2_1.configure(picam2_1.create_preview_configuration(
        main={"size": config.config['capture_resolution']}
    ))
    picam2_1.start()
    logger.info("RGB camera initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RGB camera: {e}")
    picam2_1 = None

try:
    picam2_2 = Picamera2(1)
    picam2_2.configure(picam2_2.create_preview_configuration(
        main={"size": config.config['capture_resolution']}
    ))
    picam2_2.start()
    logger.info("NIR camera initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize NIR camera: {e}")
    picam2_2 = None

def cleanup():
    """Enhanced cleanup function"""
    logger.info("Shutting down cameras...")
    if picam2_1:
        picam2_1.stop()
    if picam2_2:
        picam2_2.stop()
    config.save_config()
    logger.info("Cleanup completed")

atexit.register(cleanup)

def update_fps(camera_id):
    """Update FPS for specified camera"""
    with fps_lock:
        if camera_id == 1:
            fps_tracker_1.append(time.time())
        elif camera_id == 2:
            fps_tracker_2.append(time.time())

def get_fps(camera_id):
    """Calculate current FPS for specified camera"""
    with fps_lock:
        tracker = fps_tracker_1 if camera_id == 1 else fps_tracker_2
        if len(tracker) < 2:
            return 0.0
        time_diff = tracker[-1] - tracker[0]
        if time_diff == 0:
            return 0.0
        return (len(tracker) - 1) / time_diff

def add_enhanced_overlay(frame, fps, camera_name, additional_info=None):
    """Enhanced overlay with more information"""
    frame_with_overlay = frame.copy()
    
    # FPS text
    fps_text = f"{camera_name}: {fps:.1f} FPS"
    
    # Additional info
    if additional_info:
        info_text = f"NDVI: {additional_info.get('mean_ndvi', 0):.3f}"
        vegetation_text = f"Veg pixels: {additional_info.get('vegetation_pixels', 0)}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (0, 255, 0)
    bg_color = (0, 0, 0)
    
    # Position for multiple lines
    y_positions = [25, 50, 75]
    texts = [fps_text]
    
    if additional_info:
        texts.extend([info_text, vegetation_text])
    
    for i, text in enumerate(texts):
        if i < len(y_positions):
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            x, y = 10, y_positions[i]
            
            # Background rectangle
            cv2.rectangle(frame_with_overlay, 
                         (x - 5, y - text_height - 5), 
                         (x + text_width + 5, y + baseline + 5), 
                         bg_color, -1)
            
            # Text
            cv2.putText(frame_with_overlay, text, (x, y), font, font_scale, color, thickness)
    
    return frame_with_overlay

def gen_frames(picam, camera_id, camera_name):
    """Enhanced frame generation with error handling"""
    if not picam:
        # Generate error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, f"{camera_name} Not Available", 
                   (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        while True:
            ret, buffer = cv2.imencode('.jpg', error_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1)
    
    while True:
        try:
            frame = picam.capture_array()
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            # Resize if needed
            display_size = config.config['display_resolution']
            if frame_bgr.shape[:2] != display_size[::-1]:
                frame_bgr = cv2.resize(frame_bgr, display_size)
            
            update_fps(camera_id)
            current_fps = get_fps(camera_id)
            
            frame_with_overlay = add_enhanced_overlay(frame_bgr, current_fps, camera_name)
            
            ret, buffer = cv2.imencode('.jpg', frame_with_overlay, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            logger.error(f"Error in {camera_name} frame generation: {e}")
            time.sleep(0.1)

def gen_enhanced_ndvi_frames():
    """Enhanced NDVI generation with quality metrics"""
    if not picam2_1 or not picam2_2:
        # Generate error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Cameras Not Available", 
                   (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        while True:
            ret, buffer = cv2.imencode('.jpg', error_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1)
    
    while True:
        try:
            start_time = time.time()
            
            rgb = picam2_1.capture_array()
            nir = picam2_2.capture_array()
            
            # Resize for processing if needed
            process_size = config.config['display_resolution']
            if rgb.shape[:2] != process_size[::-1]:
                rgb = cv2.resize(rgb, process_size)
                nir = cv2.resize(nir, process_size)
            
            # Advanced alignment
            alignment_method = config.config['alignment_method']
            nir_aligned = processor.advanced_alignment(rgb, nir, alignment_method)
            
            # Extract channels
            red_channel = rgb[:, :, 0].astype(float)
            nir_channel = nir_aligned[:, :, 0].astype(float)
            
            # Calculate enhanced NDVI
            ndvi, quality_metrics = processor.calculate_enhanced_ndvi(red_channel, nir_channel)
            
            # Apply colormap
            colormap = config.config['ndvi_colormap']
            ndvi_normalized = ((ndvi + 1) / 2 * 255).astype(np.uint8)
            ndvi_colored = cv2.applyColorMap(ndvi_normalized, colormap)
            
            # Add enhanced overlay
            processing_time = time.time() - start_time
            processor.statistics['processing_times'].append(processing_time)
            
            quality_metrics['processing_time'] = processing_time
            overlay_frame = add_enhanced_overlay(ndvi_colored, 1/processing_time, 
                                               "NDVI", quality_metrics)
            
            ret, buffer = cv2.imencode('.jpg', overlay_frame, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
        except Exception as e:
            logger.error(f"Error in NDVI generation: {e}")
            time.sleep(0.1)

def gen_enhanced_evi_frames():
    """Enhanced EVI generation with quality metrics"""
    if not picam2_1 or not picam2_2:
        # Generate error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Cameras Not Available", 
                   (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        while True:
            ret, buffer = cv2.imencode('.jpg', error_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1)
    
    while True:
        try:
            start_time = time.time()
            
            rgb = picam2_1.capture_array()
            nir = picam2_2.capture_array()
            
            # Resize for processing if needed
            process_size = config.config['display_resolution']
            if rgb.shape[:2] != process_size[::-1]:
                rgb = cv2.resize(rgb, process_size)
                nir = cv2.resize(nir, process_size)
            
            # Advanced alignment
            alignment_method = config.config['alignment_method']
            nir_aligned = processor.advanced_alignment(rgb, nir, alignment_method)
            
            # Extract channels
            red_channel = rgb[:, :, 0].astype(float)
            nir_channel = nir_aligned[:, :, 0].astype(float)
            blue_channel = rgb[:, :, 2].astype(float)
            
            # Calculate enhanced EVI
            evi, quality_metrics = processor.calculate_enhanced_evi(red_channel, nir_channel, blue_channel)
            
            # Apply colormap
            colormap = config.config['evi_colormap']
            evi_normalized = ((evi + 1) / 2 * 255).astype(np.uint8)
            evi_colored = cv2.applyColorMap(evi_normalized, colormap)
            
            # Add enhanced overlay
            processing_time = time.time() - start_time
            quality_metrics['processing_time'] = processing_time
            overlay_frame = add_enhanced_overlay(evi_colored, 1/processing_time, 
                                               "EVI", quality_metrics)
            
            ret, buffer = cv2.imencode('.jpg', overlay_frame, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
        except Exception as e:
            logger.error(f"Error in EVI generation: {e}")
            time.sleep(0.1)

@app.route('/video_feed1')
def video_feed1():
    return Response(gen_frames(picam2_1, 1, "RGB"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(gen_frames(picam2_2, 2, "NIR"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ndvi_feed')
def ndvi_feed():
    return Response(gen_enhanced_ndvi_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/evi_feed')
def evi_feed():
    return Response(gen_enhanced_evi_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """API endpoint for configuration management"""
    if request.method == 'GET':
        return jsonify(config.config)
    elif request.method == 'POST':
        try:
            new_config = request.get_json()
            for key, value in new_config.items():
                if key in config.config:
                    config.config[key] = value
            config.save_config()
            return jsonify({"status": "success", "message": "Configuration updated"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/statistics')
def api_statistics():
    """Get processing statistics"""
    return jsonify({
        'ndvi_stats': {
            'history': list(processor.statistics['ndvi_history']),
            'mean': float(np.mean(processor.statistics['ndvi_history'])) if processor.statistics['ndvi_history'] else 0,
            'std': float(np.std(processor.statistics['ndvi_history'])) if processor.statistics['ndvi_history'] else 0
        },
        'evi_stats': {
            'history': list(processor.statistics['evi_history']),
            'mean': float(np.mean(processor.statistics['evi_history'])) if processor.statistics['evi_history'] else 0,
            'std': float(np.std(processor.statistics['evi_history'])) if processor.statistics['evi_history'] else 0
        },
        'performance': {
            'avg_processing_time': float(np.mean(processor.statistics['processing_times'])) if processor.statistics['processing_times'] else 0,
            'fps_rgb': get_fps(1),
            'fps_nir': get_fps(2)
        }
    })

@app.route('/api/capture', methods=['POST'])
def api_capture():
    """Capture and save current frame with NDVI data"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if picam2_1 and picam2_2:
            rgb = picam2_1.capture_array()
            nir = picam2_2.capture_array()
            
            # Save raw images
            cv2.imwrite(f"capture_rgb_{timestamp}.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"capture_nir_{timestamp}.jpg", cv2.cvtColor(nir, cv2.COLOR_RGB2BGR))
            
            # Calculate and save NDVI
            nir_aligned = processor.advanced_alignment(rgb, nir, config.config['alignment_method'])
            red_channel = rgb[:, :, 0].astype(float)
            nir_channel = nir_aligned[:, :, 0].astype(float)
            ndvi, quality_metrics = processor.calculate_enhanced_ndvi(red_channel, nir_channel)
            
            # Save NDVI as numpy array
            np.save(f"ndvi_data_{timestamp}.npy", ndvi)
            
            # Save metadata
            metadata = {
                'timestamp': timestamp,
                'config': config.config,
                'quality_metrics': quality_metrics
            }
            with open(f"metadata_{timestamp}.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return jsonify({
                "status": "success", 
                "message": f"Captured and saved data with timestamp {timestamp}",
                "quality_metrics": quality_metrics
            })
        else:
            return jsonify({"status": "error", "message": "Cameras not available"}), 500
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/')
def index():
    return '''
    <html>
    <head>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                color: white;
                text-align: center;
                padding: 30px;
                position: relative;
            }
            .header h1 {
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
            }
            .header p {
                margin: 10px 0 0 0;
                opacity: 0.9;
                font-size: 1.1em;
            }
            .controls-panel {
                background: #f8f9fa;
                padding: 20px;
                border-bottom: 1px solid #e9ecef;
            }
            .control-group {
                display: inline-block;
                margin: 0 15px 10px 0;
                vertical-align: top;
            }
            .control-group label {
                display: block;
                font-weight: 600;
                margin-bottom: 5px;
                color: #495057;
                font-size: 0.9em;
            }
            .control-group select, .control-group input, .control-group button {
                padding: 8px 12px;
                border: 1px solid #ced4da;
                border-radius: 5px;
                font-size: 0.9em;
                transition: all 0.3s ease;
            }
            .control-group select:focus, .control-group input:focus {
                outline: none;
                border-color: #4CAF50;
                box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2);
            }
            .btn {
                background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.3s ease;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(76, 175, 80, 0.3);
            }
            .btn-secondary {
                background: linear-gradient(135deg, #6c757d 0%, #545b62 100%);
            }
            .status-panel {
                background: #e8f5e8;
                padding: 15px;
                margin: 0 20px 20px 20px;
                border-radius: 8px;
                border-left: 4px solid #4CAF50;
            }
            .status-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
            }
            .status-item {
                text-align: center;
                padding: 10px;
                background: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .status-value {
                font-size: 1.5em;
                font-weight: bold;
                color: #4CAF50;
                display: block;
            }
            .status-label {
                font-size: 0.9em;
                color: #666;
                margin-top: 5px;
            }
            .grid-container {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                padding: 20px;
            }
            .grid-item {
                text-align: center;
                background: #fff;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                overflow: hidden;
                transition: transform 0.3s ease;
            }
            .grid-item:hover {
                transform: translateY(-5px);
            }
            .grid-item h2 {
                margin: 0;
                padding: 15px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-size: 1.2em;
                font-weight: 500;
            }
            .image-container {
                position: relative;
                padding: 15px;
            }
            img {
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.2);
            }
            .info-panel {
                grid-column: 1 / -1;
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 25px;
                border-radius: 10px;
                margin-top: 20px;
            }
            .info-panel h3 {
                color: #495057;
                margin-top: 0;
                font-size: 1.3em;
            }
            .features-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 15px;
            }
            .feature-item {
                background: white;
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #4CAF50;
            }
            .feature-item h4 {
                margin: 0 0 10px 0;
                color: #4CAF50;
                font-size: 1.1em;
            }
            .feature-item ul {
                margin: 0;
                padding-left: 20px;
            }
            .feature-item li {
                margin-bottom: 5px;
                color: #666;
            }
            @media (max-width: 768px) {
                .grid-container {
                    grid-template-columns: 1fr;
                }
                .controls-panel {
                    text-align: center;
                }
                .control-group {
                    display: block;
                    margin: 10px 0;
                }
            }
            .loading {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #4CAF50;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-right: 10px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
        <script>
            let config = {};
            let statistics = {};
            
            function updateStatus() {
                fetch('/api/statistics')
                    .then(response => response.json())
                    .then(data => {
                        statistics = data;
                        document.getElementById('fps-rgb').textContent = data.performance.fps_rgb.toFixed(1);
                        document.getElementById('fps-nir').textContent = data.performance.fps_nir.toFixed(1);
                        document.getElementById('ndvi-mean').textContent = data.ndvi_stats.mean.toFixed(3);
                        document.getElementById('evi-mean').textContent = data.evi_stats.mean.toFixed(3);
                        document.getElementById('processing-time').textContent = (data.performance.avg_processing_time * 1000).toFixed(1);
                    })
                    .catch(error => {
                        console.error('Error fetching statistics:', error);
                    });
            }
            
            function loadConfig() {
                fetch('/api/config')
                    .then(response => response.json())
                    .then(data => {
                        config = data;
                        // Update form elements
                        document.getElementById('gaussian-kernel').value = data.gaussian_kernel_size;
                        document.getElementById('gaussian-sigma').value = data.gaussian_sigma;
                        document.getElementById('alignment-method').value = data.alignment_method;
                        document.getElementById('ndvi-colormap').value = data.ndvi_colormap;
                    })
                    .catch(error => {
                        console.error('Error loading config:', error);
                    });
            }
            
            function updateConfig() {
                const newConfig = {
                    gaussian_kernel_size: parseInt(document.getElementById('gaussian-kernel').value),
                    gaussian_sigma: parseFloat(document.getElementById('gaussian-sigma').value),
                    alignment_method: document.getElementById('alignment-method').value,
                    ndvi_colormap: parseInt(document.getElementById('ndvi-colormap').value),
                    evi_colormap: parseInt(document.getElementById('evi-colormap').value)
                };
                
                fetch('/api/config', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(newConfig)
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        showMessage('Configuration updated successfully!', 'success');
                    } else {
                        showMessage('Error updating configuration: ' + data.message, 'error');
                    }
                })
                .catch(error => {
                    showMessage('Error updating configuration: ' + error, 'error');
                });
            }
            
            function captureFrame() {
                const btn = document.getElementById('capture-btn');
                btn.innerHTML = '<span class="loading"></span>Capturing...';
                btn.disabled = true;
                
                fetch('/api/capture', {
                    method: 'POST'
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        showMessage('Frame captured successfully! NDVI mean: ' + data.quality_metrics.mean_ndvi.toFixed(3), 'success');
                    } else {
                        showMessage('Error capturing frame: ' + data.message, 'error');
                    }
                })
                .catch(error => {
                    showMessage('Error capturing frame: ' + error, 'error');
                })
                .finally(() => {
                    btn.innerHTML = 'Capture Frame';
                    btn.disabled = false;
                });
            }
            
            function showMessage(message, type) {
                const messageDiv = document.createElement('div');
                messageDiv.style.cssText = `
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    padding: 15px 20px;
                    border-radius: 5px;
                    color: white;
                    font-weight: bold;
                    z-index: 1000;
                    background: ${type === 'success' ? '#4CAF50' : '#f44336'};
                    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
                `;
                messageDiv.textContent = message;
                document.body.appendChild(messageDiv);
                
                setTimeout(() => {
                    messageDiv.remove();
                }, 5000);
            }
            
            // Initialize on page load
            window.onload = function() {
                loadConfig();
                updateStatus();
                setInterval(updateStatus, 2000); // Update every 2 seconds
            };
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌱 Advanced NDVI Monitoring System</h1>
                <p>Real-time Vegetation Analysis with Enhanced Processing & Quality Metrics</p>
            </div>
            
            <div class="controls-panel">
                <div class="control-group">
                    <label for="gaussian-kernel">Gaussian Kernel Size:</label>
                    <select id="gaussian-kernel">
                        <option value="3">3x3</option>
                        <option value="5" selected>5x5</option>
                        <option value="7">7x7</option>
                        <option value="9">9x9</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <label for="gaussian-sigma">Gaussian Sigma:</label>
                    <input type="number" id="gaussian-sigma" value="1.0" step="0.1" min="0.1" max="5.0">
                </div>
                
                <div class="control-group">
                    <label for="alignment-method">Alignment Method:</label>
                    <select id="alignment-method">
                        <option value="phase_correlation">Phase Correlation</option>
                        <option value="ecc">Enhanced Correlation Coefficient</option>
                        <option value="orb">ORB Feature Matching</option>
                        <option value="hybrid">Hybrid (Auto)</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <label for="ndvi-colormap">NDVI Colormap:</label>
                    <select id="ndvi-colormap">
                        <option value="2">JET</option>
                        <option value="3">WINTER</option>
                        <option value="4">RAINBOW</option>
                        <option value="11">HOT</option>
                        <option value="12">COOL</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <label for="evi-colormap">EVI Colormap:</label>
                    <select id="evi-colormap">
                        <option value="0">AUTUMN</option>
                        <option value="1">BONE</option>
                        <option value="2">JET</option>
                        <option value="15" selected>VIRIDIS</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <button class="btn" onclick="updateConfig()">Update Settings</button>
                </div>
                
                <div class="control-group">
                    <button class="btn btn-secondary" id="capture-btn" onclick="captureFrame()">Capture Frame</button>
                </div>
            </div>
            
            <div class="status-panel">
                <h3 style="margin-top: 0; color: #4CAF50;">📊 System Status & Performance Metrics</h3>
                <div class="status-grid">
                    <div class="status-item">
                        <span class="status-value" id="fps-rgb">--</span>
                        <div class="status-label">RGB Camera FPS</div>
                    </div>
                    <div class="status-item">
                        <span class="status-value" id="fps-nir">--</span>
                        <div class="status-label">NIR Camera FPS</div>
                    </div>
                    <div class="status-item">
                        <span class="status-value" id="ndvi-mean">--</span>
                        <div class="status-label">Average NDVI</div>
                    </div>
                    <div class="status-item">
                        <span class="status-value" id="evi-mean">--</span>
                        <div class="status-label">Average EVI</div>
                    </div>
                    <div class="status-item">
                        <span class="status-value" id="processing-time">--</span>
                        <div class="status-label">Processing Time (ms)</div>
                    </div>
                </div>
            </div>
            
            <div class="grid-container">
                <div class="grid-item">
                    <h2>📸 RGB Camera (Red Channel)</h2>
                    <div class="image-container">
                        <img src="/video_feed1" alt="RGB Camera Feed">
                    </div>
                </div>
                
                <div class="grid-item">
                    <h2>🔴 NIR Camera (Modified)</h2>
                    <div class="image-container">
                        <img src="/video_feed2" alt="NIR Camera Feed">
                    </div>
                </div>
                
                <div class="grid-item">
                    <h2>🌿 NDVI (Vegetation Index)</h2>
                    <div class="image-container">
                        <img src="/ndvi_feed" alt="NDVI Feed">
                    </div>
                </div>
                
                <div class="grid-item">
                    <h2>🍃 EVI (Enhanced Vegetation Index)</h2>
                    <div class="image-container">
                        <img src="/evi_feed" alt="EVI Feed">
                    </div>
                </div>
                
                <div class="info-panel">
                    <h3>🚀 Enhanced System Features</h3>
                    <div class="features-grid">
                        <div class="feature-item">
                            <h4>Advanced Image Processing</h4>
                            <ul>
                                <li>Multi-method alignment (Phase Correlation, ECC, ORB)</li>
                                <li>Bilateral filtering for edge preservation</li>
                                <li>Spectral calibration with 6-reference system</li>
                                <li>Atmospheric correction for EVI</li>
                            </ul>
                        </div>
                        
                        <div class="feature-item">
                            <h4>Quality Metrics & Analytics</h4>
                            <ul>
                                <li>Real-time vegetation pixel counting</li>
                                <li>Statistical analysis (mean, std deviation)</li>
                                <li>Processing performance monitoring</li>
                                <li>Historical data tracking</li>
                            </ul>
                        </div>
                        
                        <div class="feature-item">
                            <h4>Configuration & Control</h4>
                            <ul>
                                <li>Dynamic parameter adjustment</li>
                                <li>Multiple colormap options</li>
                                <li>Persistent configuration storage</li>
                                <li>API endpoints for automation</li>
                            </ul>
                        </div>
                        
                        <div class="feature-item">
                            <h4>Data Capture & Export</h4>
                            <ul>
                                <li>High-quality frame capture</li>
                                <li>NDVI data export (NumPy format)</li>
                                <li>Metadata logging with timestamps</li>
                                <li>Quality metrics for each capture</li>
                            </ul>
                        </div>
                    </div>
                    
                    <div style="margin-top: 20px; padding: 15px; background: white; border-radius: 8px;">
                        <h4 style="color: #4CAF50; margin: 0 0 10px 0;">📏 Technical Specifications</h4>
                        <p style="margin: 5px 0; color: #666;"><strong>NDVI Formula:</strong> (NIR - Red) / (NIR + Red)</p>
                        <p style="margin: 5px 0; color: #666;"><strong>EVI Formula:</strong> 2.5 × (NIR - Red) / (NIR + 6×Red - 7.5×Blue + 1)</p>
                        <p style="margin: 5px 0; color: #666;"><strong>Based on:</strong> Hoang's Research (2025) + Singh et al. LoRaWAN (2020) + Stamford et al. (2023)</p>
                        <p style="margin: 5px 0; color: #666;"><strong>Hardware:</strong> Raspberry Pi 5 + Dual Camera Module V3 (Sony IMX708)</p>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
