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
import tensorflow as tf
from tensorflow import keras
import keyboard

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
            'alignment_method': 'phase_correlation',
            'capture_resolution': (1920, 1080),
            'display_resolution': (1280, 720),
            'model_input_size': (224, 224)
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

class ImageProcessor:
    """Image processing for NDVI/EVI calculation and AI inference"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.class_names = ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 
                           'Septoria_leaf_spot', 'Spider_mites', 'Target_Spot', 
                           'Yellow_Leaf_Curl_Virus', 'mosaic_virus', 'healthy']
        self.load_model()
        
        # Storage for latest captures
        self.latest_captures = {
            'rgb': None,
            'nir': None,
            'ndvi': None,
            'evi': None,
            'ndvi_colored': None,
            'evi_colored': None,
            'prediction': None,
            'timestamp': None
        }
        self.capture_lock = threading.Lock()
    
    def load_model(self):
        """Load the tomato disease classification model"""
        try:
            model_path = 'models/tomato_disease_model.keras'
            if os.path.exists(model_path):
                self.model = keras.models.load_model(model_path)
                logger.info("AI model loaded successfully")
            else:
                logger.error(f"Model file not found at {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def preprocess_for_ai(self, image):
        """Preprocess image for AI model"""
        # Resize to model input size
        resized = cv2.resize(image, self.config.config['model_input_size'])
        # Convert BGR to RGB if needed
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)
    
    def predict_disease(self, image):
        """Run AI inference on the image"""
        if self.model is None:
            return None, 0.0
        
        try:
            # Preprocess image
            processed = self.preprocess_for_ai(image)
            
            # Run prediction
            predictions = self.model.predict(processed, verbose=0)
            
            # Get top prediction
            top_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][top_idx])
            disease_name = self.class_names[top_idx]
            
            return disease_name, confidence
        except Exception as e:
            logger.error(f"Error in AI prediction: {e}")
            return None, 0.0
    
    def align_images(self, base_img, img_to_align):
        """Align two images using phase correlation"""
        try:
            base_gray = cv2.cvtColor(base_img, cv2.COLOR_RGB2GRAY)
            align_gray = cv2.cvtColor(img_to_align, cv2.COLOR_RGB2GRAY)
            
            base_gray = cv2.GaussianBlur(base_gray, (3, 3), 0)
            align_gray = cv2.GaussianBlur(align_gray, (3, 3), 0)
            
            base_gray = np.float32(base_gray)
            align_gray = np.float32(align_gray)
            
            shift, response = cv2.phaseCorrelate(align_gray, base_gray)
            
            if response > 0.1:
                dx, dy = shift
                rows, cols = base_img.shape[:2]
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                aligned = cv2.warpAffine(img_to_align, M, (cols, rows))
                return aligned
            else:
                return img_to_align
        except Exception as e:
            logger.error(f"Error in image alignment: {e}")
            return img_to_align
    
    def calculate_ndvi(self, red_channel, nir_channel):
        """Calculate NDVI"""
        red_float = red_channel.astype(float)
        nir_float = nir_channel.astype(float)
        
        # Apply Gaussian filter
        kernel_size = self.config.config['gaussian_kernel_size']
        sigma = self.config.config['gaussian_sigma']
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        red_filtered = cv2.GaussianBlur(red_float, (kernel_size, kernel_size), sigma)
        nir_filtered = cv2.GaussianBlur(nir_float, (kernel_size, kernel_size), sigma)
        
        # Calculate NDVI
        epsilon = 1e-7
        denominator = nir_filtered + red_filtered + epsilon
        ndvi = (nir_filtered - red_filtered) / denominator
        
        return np.clip(ndvi, -1, 1)
    
    def calculate_evi(self, red_channel, nir_channel, blue_channel):
        """Calculate EVI"""
        red_float = red_channel.astype(float)
        nir_float = nir_channel.astype(float)
        blue_float = blue_channel.astype(float)
        
        # Apply Gaussian filter
        kernel_size = self.config.config['gaussian_kernel_size']
        sigma = self.config.config['gaussian_sigma']
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        red_filtered = cv2.GaussianBlur(red_float, (kernel_size, kernel_size), sigma)
        nir_filtered = cv2.GaussianBlur(nir_float, (kernel_size, kernel_size), sigma)
        blue_filtered = cv2.GaussianBlur(blue_float, (kernel_size, kernel_size), sigma)
        
        # EVI calculation
        G = 2.5
        C1 = 6.0
        C2 = 7.5
        L = 1.0
        
        epsilon = 1e-7
        denominator = nir_filtered + C1 * red_filtered - C2 * blue_filtered + L + epsilon
        evi = G * (nir_filtered - red_filtered) / denominator
        
        return np.clip(evi, -1, 1)
    
    def process_capture(self, rgb_image, nir_image):
        """Process captured images to calculate NDVI and EVI"""
        with self.capture_lock:
            try:
                # Align images
                nir_aligned = self.align_images(rgb_image, nir_image)
                
                # Extract channels
                red_channel = rgb_image[:, :, 0]
                nir_channel = nir_aligned[:, :, 0]
                blue_channel = rgb_image[:, :, 2]
                
                # Calculate NDVI
                ndvi = self.calculate_ndvi(red_channel, nir_channel)
                
                # Calculate EVI
                evi = self.calculate_evi(red_channel, nir_channel, blue_channel)
                
                # Apply colormaps
                ndvi_normalized = ((ndvi + 1) / 2 * 255).astype(np.uint8)
                ndvi_colored = cv2.applyColorMap(ndvi_normalized, self.config.config['ndvi_colormap'])
                
                evi_normalized = ((evi + 1) / 2 * 255).astype(np.uint8)
                evi_colored = cv2.applyColorMap(evi_normalized, self.config.config['evi_colormap'])
                
                # Store results
                self.latest_captures['rgb'] = rgb_image
                self.latest_captures['nir'] = nir_image
                self.latest_captures['ndvi'] = ndvi
                self.latest_captures['evi'] = evi
                self.latest_captures['ndvi_colored'] = ndvi_colored
                self.latest_captures['evi_colored'] = evi_colored
                self.latest_captures['timestamp'] = datetime.now()
                
                # Calculate statistics
                ndvi_mean = float(np.mean(ndvi))
                ndvi_std = float(np.std(ndvi))
                evi_mean = float(np.mean(evi))
                evi_std = float(np.std(evi))
                
                return {
                    'success': True,
                    'ndvi_mean': ndvi_mean,
                    'ndvi_std': ndvi_std,
                    'evi_mean': evi_mean,
                    'evi_std': evi_std,
                    'timestamp': self.latest_captures['timestamp'].isoformat()
                }
            except Exception as e:
                logger.error(f"Error processing capture: {e}")
                return {'success': False, 'error': str(e)}
    
    def run_ai_inference(self, rgb_image):
        """Run AI inference on RGB image"""
        with self.capture_lock:
            try:
                disease, confidence = self.predict_disease(rgb_image)
                
                if disease:
                    self.latest_captures['prediction'] = {
                        'disease': disease,
                        'confidence': confidence,
                        'timestamp': datetime.now().isoformat()
                    }
                    return {
                        'success': True,
                        'disease': disease,
                        'confidence': confidence
                    }
                else:
                    return {'success': False, 'error': 'Prediction failed'}
            except Exception as e:
                logger.error(f"Error in AI inference: {e}")
                return {'success': False, 'error': str(e)}

# Global objects
config = CameraConfig()
processor = ImageProcessor(config)

# Initialize cameras
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
    """Cleanup function"""
    logger.info("Shutting down cameras...")
    if picam2_1:
        picam2_1.stop()
    if picam2_2:
        picam2_2.stop()
    config.save_config()
    logger.info("Cleanup completed")

atexit.register(cleanup)

# Keyboard event handlers
def capture_and_process():
    """Capture images and calculate NDVI/EVI"""
    if picam2_1 and picam2_2:
        logger.info("Capturing and processing images...")
        rgb = picam2_1.capture_array()
        nir = picam2_2.capture_array()
        
        # Process
        result = processor.process_capture(rgb, nir)
        logger.info(f"Processing result: {result}")

def run_ai_analysis():
    """Run AI disease detection"""
    if picam2_1:
        logger.info("Running AI analysis...")
        rgb = picam2_1.capture_array()
        result = processor.run_ai_inference(rgb)
        logger.info(f"AI result: {result}")

# Set up keyboard listeners in a separate thread
@app.route('/api/capture_ndvi', methods=['POST'])
def api_capture_ndvi():
    """API endpoint to trigger NDVI/EVI capture"""
    try:
        if picam2_1 and picam2_2:
            logger.info("API: Capturing and processing images...")
            rgb = picam2_1.capture_array()
            nir = picam2_2.capture_array()

            # Process
            result = processor.process_capture(rgb, nir)
            logger.info(f"API: Processing result: {result}")
            return jsonify(result)
        else:
            return jsonify({'success': False, 'error': 'Cameras not available'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/run_ai', methods=['POST'])
def api_run_ai():
    """API endpoint to trigger AI analysis"""
    try:
        if picam2_1:
            logger.info("API: Running AI analysis...")
            rgb = picam2_1.capture_array()
            result = processor.run_ai_inference(rgb)
            logger.info(f"API: AI result: {result}")
            return jsonify(result)
        else:
            return jsonify({'success': False, 'error': 'Camera not available'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def gen_frames(picam, camera_name):
    """Generate frames for live camera feed"""
    if not picam:
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
            
            # Resize for display
            display_size = config.config['display_resolution']
            if frame_bgr.shape[:2] != display_size[::-1]:
                frame_bgr = cv2.resize(frame_bgr, display_size)
            
            # Add camera name overlay
            cv2.putText(frame_bgr, camera_name, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame_bgr, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            logger.error(f"Error in {camera_name} frame generation: {e}")
            time.sleep(0.1)

def gen_result_frames(result_type):
    """Generate frames for processed results"""
    while True:
        try:
            with processor.capture_lock:
                if result_type == 'ndvi' and processor.latest_captures['ndvi_colored'] is not None:
                    frame = processor.latest_captures['ndvi_colored']
                elif result_type == 'evi' and processor.latest_captures['evi_colored'] is not None:
                    frame = processor.latest_captures['evi_colored']
                else:
                    # Generate placeholder
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, f"No {result_type.upper()} data yet", 
                               (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, "Press 'b' to capture", 
                               (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            logger.error(f"Error generating {result_type} frames: {e}")
        time.sleep(0.1)

# Flask routes
@app.route('/video_feed1')
def video_feed1():
    return Response(gen_frames(picam2_1, "RGB Camera"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(gen_frames(picam2_2, "NIR Camera"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ndvi_feed')
def ndvi_feed():
    return Response(gen_result_frames('ndvi'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/evi_feed')
def evi_feed():
    return Response(gen_result_frames('evi'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/latest_results')
def api_latest_results():
    """Get latest processing results"""
    with processor.capture_lock:
        results = {
            'has_ndvi': processor.latest_captures['ndvi'] is not None,
            'has_prediction': processor.latest_captures['prediction'] is not None,
            'timestamp': processor.latest_captures['timestamp'].isoformat() if processor.latest_captures['timestamp'] else None
        }
        
        if processor.latest_captures['ndvi'] is not None:
            results['ndvi_stats'] = {
                'mean': float(np.mean(processor.latest_captures['ndvi'])),
                'std': float(np.std(processor.latest_captures['ndvi']))
            }
            results['evi_stats'] = {
                'mean': float(np.mean(processor.latest_captures['evi'])),
                'std': float(np.std(processor.latest_captures['evi']))
            }
        
        if processor.latest_captures['prediction']:
            results['prediction'] = processor.latest_captures['prediction']
    
    return jsonify(results)

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
                text-align: center;
                border-bottom: 1px solid #e9ecef;
            }
            .key-info {
                display: inline-block;
                margin: 0 20px;
                padding: 10px 20px;
                background: #e8f5e8;
                border-radius: 8px;
                font-weight: 600;
            }
            .status-panel {
                background: #e8f5e8;
                padding: 15px;
                margin: 20px;
                border-radius: 8px;
                border-left: 4px solid #4CAF50;
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
            }
            .grid-item h2 {
                margin: 0;
                padding: 15px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-size: 1.2em;
            }
            img {
                max-width: 100%;
                height: auto;
                display: block;
            }
            .results-section {
                margin-top: 20px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
            }
            .result-item {
                margin: 10px 0;
                padding: 15px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .result-label {
                font-weight: 600;
                color: #4CAF50;
            }
            .prediction-box {
                background: #fff3cd;
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #ffc107;
                margin-top: 10px;
            }
            @media (max-width: 768px) {
                .grid-container {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        <script>
            function updateResults() {
                fetch('/api/latest_results')
                    .then(response => response.json())
                    .then(data => {
                        const resultsDiv = document.getElementById('results');
                        
                        if (data.has_ndvi) {
                            document.getElementById('ndvi-mean').textContent = data.ndvi_stats.mean.toFixed(3);
                            document.getElementById('ndvi-std').textContent = data.ndvi_stats.std.toFixed(3);
                            document.getElementById('evi-mean').textContent = data.evi_stats.mean.toFixed(3);
                            document.getElementById('evi-std').textContent = data.evi_stats.std.toFixed(3);
                            document.getElementById('timestamp').textContent = new Date(data.timestamp).toLocaleString();
                        }
                        
                        if (data.prediction) {
                            const predDiv = document.getElementById('prediction');
                            predDiv.innerHTML = `
                                <div class="prediction-box">
                                    <h3>🌱 AI Disease Detection Result</h3>
                                    <p><span class="result-label">Disease:</span> ${data.prediction.disease}</p>
                                    <p><span class="result-label">Confidence:</span> ${(data.prediction.confidence * 100).toFixed(1)}%</p>
                                    <p><span class="result-label">Time:</span> ${new Date(data.prediction.timestamp).toLocaleString()}</p>
                                </div>
                            `;
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching results:', error);
                    });
            }
            
            // Update results every 2 seconds
            setInterval(updateResults, 2000);
            
            // Initial update
            window.onload = function() {
                updateResults();
            };
        </script>
        <script>
    function captureNDVI() {
        fetch('/api/capture_ndvi', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('NDVI/EVI captured successfully!');
            } else {
                alert('Error: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(error => {
            alert('Error: ' + error);
        });
    }

    function runAI() {
        fetch('/api/run_ai', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('AI analysis complete! Disease: ' + data.disease +
                      ' (Confidence: ' + (data.confidence * 100).toFixed(1) + '%)');
            } else {
                alert('Error: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(error => {
            alert('Error: ' + error);
        });
    }
</script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌱 Plant Health Monitoring System</h1>
                <p>Real-time Camera Feed with On-Demand NDVI/EVI Analysis and AI Disease Detection</p>
            </div>
            <div class="controls-panel">
                <button class="control-btn" onclick="captureNDVI()">📊 Capture & Calculate NDVI/EVI</button>
                <button class="control-btn" onclick="runAI()">🤖 Run AI Disease Detection</button>
            </div> 
            
            <div class="grid-container">
                <div class="grid-item">
                    <h2>📸 RGB Camera (Live)</h2>
                    <img src="/video_feed1" alt="RGB Camera Feed">
                </div>
                
                <div class="grid-item">
                    <h2>🔴 NIR Camera (Live)</h2>
                    <img src="/video_feed2" alt="NIR Camera Feed">
                </div>
            </div>
            
            <div class="results-section">
                <h2 style="text-align: center; color: #4CAF50;">📊 Analysis Results</h2>
                
                <div class="status-panel">
                    <h3>Latest Capture Statistics</h3>
                    <div class="result-item">
                        <span class="result-label">NDVI Mean:</span> <span id="ndvi-mean">--</span>
                        <span style="margin-left: 20px;" class="result-label">Std Dev:</span> <span id="ndvi-std">--</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">EVI Mean:</span> <span id="evi-mean">--</span>
                        <span style="margin-left: 20px;" class="result-label">Std Dev:</span> <span id="evi-std">--</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Last Capture:</span> <span id="timestamp">--</span>
                    </div>
                </div>
                
                <div id="prediction"></div>
                
                <div class="grid-container">
                    <div class="grid-item">
                        <h2>🌿 NDVI Result</h2>
                        <img src="/ndvi_feed" alt="NDVI Result">
                    </div>
                    
                    <div class="grid-item">
                        <h2>🍃 EVI Result</h2>
                        <img src="/evi_feed" alt="EVI Result">
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
