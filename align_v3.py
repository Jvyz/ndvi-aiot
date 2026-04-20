from flask import Flask, render_template_string, jsonify
import cv2
import numpy as np
from picamera2 import Picamera2
import base64
import threading
from skimage.metrics import structural_similarity as ssim
import time

app = Flask(__name__)

class EnhancedAlignmentTester:
    def __init__(self):
        self.rgb_cam = Picamera2(0)
        self.nir_cam = Picamera2(1)
        self.rgb_cam.configure(self.rgb_cam.create_preview_configuration(main={"size": (640, 480)}))
        self.nir_cam.configure(self.nir_cam.create_preview_configuration(main={"size": (640, 480)}))
        self.rgb_cam.start()
        self.nir_cam.start()
        self.results = {}
        
        # Allow cameras to warm up
        time.sleep(2)
    
    def preprocess_image(self, img, is_nir=False):
        """Enhanced preprocessing with NIR-specific handling"""
        # Convert to grayscale
        if len(img.shape) == 3:
            if is_nir:
                # For NIR images, use red channel which often has best NIR response
                gray = img[:, :, 0]  # Red channel
            else:
                # For RGB, use standard conversion
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # NIR-specific preprocessing
        if is_nir:
            # NIR images often need different processing
            # Apply stronger noise reduction
            gray = cv2.medianBlur(gray, 3)
            
            # Enhance contrast using CLAHE (better than histogram equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # NIR images often benefit from gamma correction
            gamma = 1.2  # Adjust based on your NIR camera
            gray = np.power(gray / 255.0, gamma) * 255.0
            gray = gray.astype(np.uint8)
        else:
            # Standard RGB preprocessing
            # Apply histogram equalization to improve contrast
            gray = cv2.equalizeHist(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply edge enhancement (more aggressive for NIR)
        if is_nir:
            # Use Sobel for NIR (often more robust than Laplacian)
            sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            enhanced = np.uint8(np.sqrt(sobelx**2 + sobely**2))
        else:
            # Use Laplacian for RGB
            laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
            enhanced = np.uint8(np.absolute(laplacian))
        
        # Combine original and enhanced (different weights for NIR)
        if is_nir:
            combined = cv2.addWeighted(blurred, 0.8, enhanced, 0.2, 0)
        else:
            combined = cv2.addWeighted(blurred, 0.7, enhanced, 0.3, 0)
        
        return combined
    
    def phase_correlation_align(self, img1, img2):
        """Enhanced phase correlation with preprocessing"""
        # Preprocess images with NIR-specific handling
        processed1 = self.preprocess_image(img1, is_nir=False)  # RGB
        processed2 = self.preprocess_image(img2, is_nir=True)   # NIR
        
        # Apply window function to reduce edge effects
        h, w = processed1.shape
        window = np.outer(np.hanning(h), np.hanning(w))
        processed1 = (processed1 * window).astype(np.float32)
        processed2 = (processed2 * window).astype(np.float32)
        
        shift, response = cv2.phaseCorrelate(processed1, processed2)
        
        return shift, response
    
    def orb_feature_align(self, img1, img2):
        """ORB feature-based alignment"""
        # Preprocess with NIR-specific handling
        processed1 = self.preprocess_image(img1, is_nir=False)  # RGB
        processed2 = self.preprocess_image(img2, is_nir=True)   # NIR
        
        # Create ORB detector with more features for NIR
        orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8)
        
        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(processed1, None)
        kp2, des2 = orb.detectAndCompute(processed2, None)
        
        if des1 is None or des2 is None:
            return (0, 0), 0, 0
        
        # Match features with more lenient matching for RGB-NIR
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        if len(matches) < 4:
            return (0, 0), 0, len(matches)
        
        # Sort matches by distance and take best matches
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:min(50, len(matches))]  # Limit to best matches
        
        # Get matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography with more lenient parameters for RGB-NIR
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 8.0)
        
        if M is None:
            return (0, 0), 0, len(good_matches)
        
        # Extract translation
        shift = (M[0, 2], M[1, 2])
        
        # Calculate confidence based on inlier ratio
        confidence = np.sum(mask) / len(mask) if mask is not None else 0
        
        return shift, confidence, len(good_matches)
    
    def ecc_align(self, img1, img2):
        """Enhanced Correlation Coefficient alignment"""
        # Preprocess with NIR-specific handling
        processed1 = self.preprocess_image(img1, is_nir=False)  # RGB
        processed2 = self.preprocess_image(img2, is_nir=True)   # NIR
        
        # Convert to float32
        processed1 = processed1.astype(np.float32)
        processed2 = processed2.astype(np.float32)
        
        # Define motion model (translation)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # Define termination criteria (more iterations for RGB-NIR)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.0001)
        
        try:
            # Find the alignment transformation
            cc, warp_matrix = cv2.findTransformECC(processed1, processed2, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
            
            shift = (warp_matrix[0, 2], warp_matrix[1, 2])
            return shift, cc
        except cv2.error:
            return (0, 0), 0
    
    def add_debug_images(self, rgb, nir):
        """Add debug visualizations of preprocessing steps"""
        # Create debug images showing preprocessing steps
        rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        nir_red = nir[:, :, 0]  # NIR red channel
        
        rgb_processed = self.preprocess_image(rgb_gray, is_nir=False)
        nir_processed = self.preprocess_image(nir, is_nir=True)
        
        # Create side-by-side comparison
        h, w = rgb_gray.shape
        comparison = np.zeros((h*2, w*2), dtype=np.uint8)
        
        comparison[0:h, 0:w] = rgb_gray
        comparison[0:h, w:2*w] = nir_red
        comparison[h:2*h, 0:w] = rgb_processed
        comparison[h:2*h, w:2*w] = nir_processed
        
        # Convert to color for display
        comparison_color = cv2.cvtColor(comparison, cv2.COLOR_GRAY2BGR)
        
        # Add labels
        cv2.putText(comparison_color, 'RGB Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison_color, 'NIR Original', (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison_color, 'RGB Processed', (10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison_color, 'NIR Processed', (w+10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', comparison_color)
        debug_b64 = base64.b64encode(buffer).decode()
        
        return f"data:image/jpeg;base64,{debug_b64}"
    
    def capture_and_align(self):
        try:
            # Capture frames
            rgb = self.rgb_cam.capture_array()
            nir = self.nir_cam.capture_array()
            
            # Convert to grayscale for alignment
            rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            nir_gray = cv2.cvtColor(nir, cv2.COLOR_RGB2GRAY)
            
            # Test all three alignment methods
            methods = {}
            
            # 1. Enhanced Phase Correlation
            shift_pc, response_pc = self.phase_correlation_align(rgb_gray, nir_gray)
            methods['phase_correlation'] = {
                'shift': [float(shift_pc[0]), float(shift_pc[1])],
                'confidence': float(response_pc),
                'method': 'Phase Correlation'
            }
            
            # 2. ORB Feature Matching
            shift_orb, conf_orb, matches_orb = self.orb_feature_align(rgb_gray, nir_gray)
            methods['orb'] = {
                'shift': [float(shift_orb[0]), float(shift_orb[1])],
                'confidence': float(conf_orb),
                'matches': int(matches_orb),
                'method': 'ORB Features'
            }
            
            # 3. Enhanced Correlation Coefficient
            shift_ecc, conf_ecc = self.ecc_align(rgb_gray, nir_gray)
            methods['ecc'] = {
                'shift': [float(shift_ecc[0]), float(shift_ecc[1])],
                'confidence': float(conf_ecc),
                'method': 'ECC'
            }
            
            # Choose best method based on confidence
            best_method = 'phase_correlation'
            best_conf = methods['phase_correlation']['confidence']
            
            if methods['orb']['confidence'] > best_conf:
                best_method = 'orb'
                best_conf = methods['orb']['confidence']
            
            if methods['ecc']['confidence'] > best_conf:
                best_method = 'ecc'
                best_conf = methods['ecc']['confidence']
            
            # Apply best alignment
            best_shift = methods[best_method]['shift']
            M = np.float32([[1, 0, best_shift[0]], [0, 1, best_shift[1]]])
            aligned_nir = cv2.warpAffine(nir, M, (rgb.shape[1], rgb.shape[0]))
            
            # Calculate similarity
            similarity = ssim(rgb_gray, cv2.cvtColor(aligned_nir, cv2.COLOR_RGB2GRAY))
            
            # Create overlay (50% transparency)
            overlay = cv2.addWeighted(rgb, 0.5, aligned_nir, 0.5, 0)
            
            # Convert to base64 for web display
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            overlay_b64 = base64.b64encode(buffer).decode()
            
            _, rgb_buffer = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            rgb_b64 = base64.b64encode(rgb_buffer).decode()
            
            _, nir_buffer = cv2.imencode('.jpg', cv2.cvtColor(nir, cv2.COLOR_RGB2BGR))
            nir_b64 = base64.b64encode(nir_buffer).decode()
            
            # Create difference image for better visualization
            diff = cv2.absdiff(rgb_gray, cv2.cvtColor(aligned_nir, cv2.COLOR_RGB2GRAY))
            diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
            _, diff_buffer = cv2.imencode('.jpg', diff_colored)
            diff_b64 = base64.b64encode(diff_buffer).decode()
            
            # Add debug preprocessing visualization
            debug_b64 = self.add_debug_images(rgb, nir)
            
            self.results = {
                'rgb_image': f"data:image/jpeg;base64,{rgb_b64}",
                'nir_image': f"data:image/jpeg;base64,{nir_b64}",
                'overlay_image': f"data:image/jpeg;base64,{overlay_b64}",
                'diff_image': f"data:image/jpeg;base64,{diff_b64}",
                'debug_image': debug_b64,
                'methods': methods,
                'best_method': best_method,
                'shift': best_shift,
                'confidence': best_conf,
                'similarity': float(similarity),
                'success': True
            }
            
        except Exception as e:
            print(f"Error in capture_and_align: {e}")
            self.results = {
                'success': False,
                'error': str(e)
            }

tester = EnhancedAlignmentTester()

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Alignment Tester with NIR Processing</title>
    <style>
        body { font-family: Arial; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
        .images { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
        .overlay-section { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
        .debug-section { margin-top: 20px; }
        .image-box { text-align: center; background: #fafafa; padding: 15px; border-radius: 8px; }
        img { max-width: 100%; border: 2px solid #ddd; border-radius: 5px; }
        button { background: #4CAF50; color: white; padding: 15px 30px; 
                border: none; border-radius: 5px; font-size: 16px; cursor: pointer; margin: 10px; }
        button:hover { background: #45a049; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .results { background: #e8f5e8; padding: 15px; margin: 20px 0; border-radius: 5px; border-left: 4px solid #4CAF50; }
        .error { background: #f8d7da; color: #721c24; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .loading { color: #666; font-style: italic; }
        .method-results { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin: 15px 0; }
        .method-card { background: #f8f9fa; padding: 12px; border-radius: 8px; border: 2px solid #e9ecef; }
        .method-card.best { border-color: #28a745; background: #d4edda; }
        .method-card h4 { margin: 0 0 8px 0; color: #333; }
        .metric { font-size: 14px; margin: 4px 0; }
        .warning { background: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Enhanced NIR-RGB Alignment Tester</h1>
        
        <div class="warning">
            <strong>⚠️ Note:</strong> RGB and NIR cameras see different wavelengths. Low similarity is normal - focus on shift accuracy and confidence.
        </div>
        
        <button id="testBtn" onclick="testAlignment()">🎯 Test All Methods with NIR Processing</button>
        
        <div id="loading" class="loading" style="display:none;">
            ⏳ Testing alignment with NIR-specific preprocessing...
        </div>
        
        <div id="error" class="error" style="display:none;"></div>
        
        <div id="results" class="results" style="display:none;">
            <h3>📊 Overall Results:</h3>
            <p><strong>Best Method:</strong> <span id="bestMethod"></span></p>
            <p><strong>Final Shift:</strong> <span id="shift"></span> pixels</p>
            <p><strong>Confidence:</strong> <span id="confidence"></span></p>
            <p><strong>Similarity:</strong> <span id="similarity"></span> <em>(Low is normal for RGB-NIR)</em></p>
            <div id="quality"></div>
            
            <h3>🔍 Method Comparison:</h3>
            <div id="methodResults" class="method-results"></div>
        </div>
        
        <div class="images">
            <div class="image-box">
                <h3>📸 RGB Camera</h3>
                <img id="rgb" src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjQwIiBoZWlnaHQ9IjQ4MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxOCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPk5vIGltYWdlPC90ZXh0Pjwvc3ZnPg==" alt="RGB">
            </div>
            <div class="image-box">
                <h3>🔴 NIR Camera</h3>
                <img id="nir" src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjQwIiBoZWlnaHQ9IjQ4MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxOCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPk5vIGltYWdlPC90ZXh0Pjwvc3ZnPg==" alt="NIR">
            </div>
        </div>
        
        <div class="overlay-section">
            <div class="image-box">
                <h3>🎯 Aligned Overlay (50% blend)</h3>
                <img id="overlay" src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjQwIiBoZWlnaHQ9IjQ4MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxOCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPk5vIGltYWdlPC90ZXh0Pjwvc3ZnPg==" alt="Overlay">
            </div>
            <div class="image-box">
                <h3>📏 Difference Map</h3>
                <img id="diff" src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjQwIiBoZWlnaHQ9IjQ4MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxOCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPk5vIGltYWdlPC90ZXh0Pjwvc3ZnPg==" alt="Difference">
            </div>
        </div>
        
        <div class="debug-section">
            <div class="image-box">
                <h3>🔧 Preprocessing Debug (Top: Original, Bottom: Processed)</h3>
                <img id="debug" src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjQwIiBoZWlnaHQ9IjQ4MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxOCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPk5vIGltYWdlPC90ZXh0Pjwvc3ZnPg==" alt="Debug">
            </div>
        </div>
    </div>

    <script>
        function testAlignment() {
            // Show loading, hide results/errors
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            document.getElementById('testBtn').disabled = true;
            
            fetch('/test')
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('testBtn').disabled = false;
                    
                    if (data.success) {
                        // Update images
                        document.getElementById('rgb').src = data.rgb_image;
                        document.getElementById('nir').src = data.nir_image;
                        document.getElementById('overlay').src = data.overlay_image;
                        document.getElementById('diff').src = data.diff_image;
                        document.getElementById('debug').src = data.debug_image;
                        
                        // Update overall results
                        document.getElementById('bestMethod').textContent = data.methods[data.best_method].method;
                        document.getElementById('shift').textContent = 
                            `X: ${data.shift[0].toFixed(2)}, Y: ${data.shift[1].toFixed(2)}`;
                        document.getElementById('confidence').textContent = data.confidence.toFixed(3);
                        document.getElementById('similarity').textContent = data.similarity.toFixed(3);
                        
                        // Quality assessment (adjusted for RGB-NIR)
                        let quality = '';
                        if (data.confidence > 0.8 && Math.abs(data.shift[0]) < 3 && Math.abs(data.shift[1]) < 3) {
                            quality = '✅ <strong>Excellent alignment!</strong> High confidence and minimal shift.';
                        } else if (data.confidence > 0.5 && Math.abs(data.shift[0]) < 5 && Math.abs(data.shift[1]) < 5) {
                            quality = '✅ <strong>Good alignment!</strong> Acceptable confidence and shift.';
                        } else if (data.confidence > 0.3) {
                            quality = '⚠️ <strong>Fair alignment</strong> - May need better lighting or target.';
                        } else {
                            quality = '❌ <strong>Poor alignment</strong> - Try different scene or lighting.';
                        }
                        document.getElementById('quality').innerHTML = quality;
                        
                        // Update method comparison
                        const methodResults = document.getElementById('methodResults');
                        methodResults.innerHTML = '';
                        
                        Object.keys(data.methods).forEach(key => {
                            const method = data.methods[key];
                            const div = document.createElement('div');
                            div.className = 'method-card' + (key === data.best_method ? ' best' : '');
                            
                            let content = `
                                <h4>${method.method} ${key === data.best_method ? '🏆' : ''}</h4>
                                <div class="metric">Shift: ${method.shift[0].toFixed(2)}, ${method.shift[1].toFixed(2)}px</div>
                                <div class="metric">Confidence: ${method.confidence.toFixed(3)}</div>
                            `;
                            
                            if (method.matches !== undefined) {
                                content += `<div class="metric">Matches: ${method.matches}</div>`;
                            }
                            
                            div.innerHTML = content;
                            methodResults.appendChild(div);
                        });
                        
                        document.getElementById('results').style.display = 'block';
                    } else {
                        document.getElementById('error').textContent = 'Error: ' + data.error;
                        document.getElementById('error').style.display = 'block';
                    }
                })
                .catch(error => {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('testBtn').disabled = false;
                    document.getElementById('error').textContent = 'Error: ' + error.message;
                    document.getElementById('error').style.display = 'block';
                    console.error('Error:', error);
                });
        }
    </script>
</body>
</html>
    ''')

@app.route('/test')
def test():
    try:
        tester.capture_and_align()
        return jsonify(tester.results)
    except Exception as e:
        print(f"Error in /test route: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🌐 Enhanced NIR-RGB Alignment Tester")
    print("📱 Open: http://192.168.1.3:5000")
    print("🔬 Features: NIR-specific preprocessing + 3 alignment methods")
    app.run(host='0.0.0.0', port=5000, debug=False)
