from flask import Flask, render_template_string, jsonify
import cv2
import numpy as np
from picamera2 import Picamera2
import base64
import time

app = Flask(__name__)

class AlignmentTester:
    def __init__(self):
        self.rgb_cam = Picamera2(0)
        self.nir_cam = Picamera2(1)
        
        # Configure cameras
        config_rgb = self.rgb_cam.create_preview_configuration(main={"size": (640, 480)})
        config_nir = self.nir_cam.create_preview_configuration(main={"size": (640, 480)})
        self.rgb_cam.configure(config_rgb)
        self.nir_cam.configure(config_nir)
        
        self.rgb_cam.start()
        self.nir_cam.start()
        time.sleep(2)  # Camera warmup
    
    def test_phase_correlation(self, rgb_gray, nir_gray):
        """Simple phase correlation"""
        try:
            # Apply window to reduce edge effects
            h, w = rgb_gray.shape
            window = np.outer(np.hanning(h), np.hanning(w))
            
            rgb_windowed = (rgb_gray.astype(np.float32) * window)
            nir_windowed = (nir_gray.astype(np.float32) * window)
            
            shift, response = cv2.phaseCorrelate(rgb_windowed, nir_windowed)
            return shift, float(response)
        except:
            return (0, 0), 0
    
    def enhance_for_vegetation(self, gray_img):
        """Enhance image for better vegetation feature detection"""
        # Apply CLAHE for better local contrast (important for leaf details)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        enhanced = clahe.apply(gray_img)
        
        # Edge enhancement to bring out leaf veins and structure
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) # Sharpening kernel
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Combine original and sharpened (less aggressive than pure sharpening)
        result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
        
        return result
    
    def test_orb_features(self, rgb_gray, nir_gray):
        """ORB feature matching optimized for vegetation"""
        try:
            # Enhance images for vegetation features
            rgb_enhanced = self.enhance_for_vegetation(rgb_gray)
            nir_enhanced = self.enhance_for_vegetation(nir_gray)
            
            # Create ORB detector optimized for vegetation
            orb = cv2.ORB_create(
                nfeatures=2000,        # More features for complex plant structures
                scaleFactor=1.2,       # Smaller scale factor for finer details
                nlevels=12,            # More pyramid levels for multi-scale matching
                edgeThreshold=15,      # Lower threshold to catch more edge details
                firstLevel=0,          # Start from original image scale
                WTA_K=2,              # Default for BRIEF descriptor
                scoreType=cv2.ORB_HARRIS_SCORE,  # Harris corner response
                patchSize=31,          # Larger patch for more stable features
                fastThreshold=10       # Lower threshold for more keypoints
            )
            
            kp1, des1 = orb.detectAndCompute(rgb_enhanced, None)
            kp2, des2 = orb.detectAndCompute(nir_enhanced, None)
            
            if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
                return (0, 0), 0
            
            # Use FLANN matcher for better performance with many features
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                              table_number=6,
                              key_size=12,
                              multi_probe_level=1)
            search_params = dict(checks=50)
            
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Apply Lowe's ratio test for vegetation (more lenient for cross-spectral)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.8 * n.distance:  # More lenient than usual 0.7
                        good_matches.append(m)
            
            if len(good_matches) < 8:  # Need more matches for vegetation
                return (0, 0), 0
            
            # Sort by distance and take best matches
            good_matches = sorted(good_matches, key=lambda x: x.distance)
            best_matches = good_matches[:min(100, len(good_matches))]
            
            # Get point correspondences
            src_pts = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)
            
            # Use more robust homography estimation for vegetation
            M, mask = cv2.findHomography(src_pts, dst_pts, 
                                       cv2.RANSAC, 
                                       ransacReprojThreshold=8.0,  # More lenient for vegetation
                                       maxIters=3000,  # More iterations
                                       confidence=0.95)
            
            if M is None:
                return (0, 0), 0
            
            # Extract translation
            shift = (M[0, 2], M[1, 2])
            
            # Calculate confidence based on inlier ratio and match quality
            inlier_ratio = np.sum(mask) / len(mask) if mask is not None else 0
            
            # Bonus confidence for having many good matches
            match_bonus = min(len(best_matches) / 50.0, 1.0)  # Bonus up to 50 matches
            
            # Combined confidence
            confidence = inlier_ratio * 0.7 + match_bonus * 0.3
            
            return shift, float(confidence)
        except Exception as e:
            print(f"ORB error: {e}")
            return (0, 0), 0
    
    def test_ecc(self, rgb_gray, nir_gray):
        """Enhanced Correlation Coefficient with sanity checks"""
        try:
            # Convert to float32
            template = rgb_gray.astype(np.float32)
            image = nir_gray.astype(np.float32)
            
            # Initialize transformation matrix (translation only)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            
            # ECC parameters
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.0001)
            
            # Run ECC
            cc, warp_matrix = cv2.findTransformECC(template, image, warp_matrix, 
                                                  cv2.MOTION_TRANSLATION, criteria)
            
            shift = (warp_matrix[0, 2], warp_matrix[1, 2])
            
            # Sanity check: penalize large shifts
            shift_magnitude = np.sqrt(shift[0]**2 + shift[1]**2)
            if shift_magnitude > 50:  # Penalty for unrealistic shifts
                cc = cc * 0.1  # Drastically reduce confidence
            elif shift_magnitude > 20:
                cc = cc * 0.5  # Moderate penalty
            
            return shift, float(cc)
        except:
            return (0, 0), 0
    
    def create_overlay(self, rgb, nir, shift):
        """Create aligned overlay image"""
        # Apply shift to NIR image
        M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        aligned_nir = cv2.warpAffine(nir, M, (rgb.shape[1], rgb.shape[0]))
        
        # Create 50-50 overlay
        overlay = cv2.addWeighted(rgb, 0.5, aligned_nir, 0.5, 0)
        return overlay
    
    def image_to_base64(self, img):
        """Convert image to base64 for web display"""
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"
    
    def capture_and_test(self):
        try:
            # Capture synchronized frames
            rgb = self.rgb_cam.capture_array()
            nir = self.nir_cam.capture_array()
            
            # Convert to grayscale
            rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            nir_gray = cv2.cvtColor(nir, cv2.COLOR_RGB2GRAY)
            
            # Test all alignment methods
            results = {}
            
            # 1. Phase Correlation
            pc_shift, pc_conf = self.test_phase_correlation(rgb_gray, nir_gray)
            pc_overlay = self.create_overlay(rgb, nir, pc_shift)
            results['phase_correlation'] = {
                'shift': [float(pc_shift[0]), float(pc_shift[1])],
                'confidence': pc_conf,
                'overlay': self.image_to_base64(pc_overlay),
                'name': 'Phase Correlation'
            }
            
            # 2. ORB Features
            orb_shift, orb_conf = self.test_orb_features(rgb_gray, nir_gray)
            orb_overlay = self.create_overlay(rgb, nir, orb_shift)
            results['orb'] = {
                'shift': [float(orb_shift[0]), float(orb_shift[1])],
                'confidence': orb_conf,
                'overlay': self.image_to_base64(orb_overlay),
                'name': 'ORB Features'
            }
            
            # 3. ECC
            ecc_shift, ecc_conf = self.test_ecc(rgb_gray, nir_gray)
            ecc_overlay = self.create_overlay(rgb, nir, ecc_shift)
            results['ecc'] = {
                'shift': [float(ecc_shift[0]), float(ecc_shift[1])],
                'confidence': ecc_conf,
                'overlay': self.image_to_base64(ecc_overlay),
                'name': 'ECC'
            }
            
            # Find best method by confidence
            best_method = max(results.keys(), key=lambda k: results[k]['confidence'])
            
            return {
                'success': True,
                'rgb_image': self.image_to_base64(rgb),
                'nir_image': self.image_to_base64(nir),
                'results': results,
                'best_method': best_method,
                'best_shift': results[best_method]['shift'],
                'best_confidence': results[best_method]['confidence']
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

# Global tester instance
tester = AlignmentTester()

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Simple Alignment Comparison</title>
    <style>
        body { font-family: Arial; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
        .input-images { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }
        .results-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin: 20px 0; }
        .image-box { text-align: center; background: #fafafa; padding: 15px; border-radius: 8px; }
        .method-box { border: 2px solid #ddd; }
        .method-box.best { border-color: #4CAF50; background: #e8f5e8; }
        img { max-width: 100%; border-radius: 5px; }
        button { background: #4CAF50; color: white; padding: 15px 30px; 
                border: none; border-radius: 5px; font-size: 16px; cursor: pointer; }
        button:hover { background: #45a049; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .summary { background: #e3f2fd; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .error { background: #ffebee; color: #c62828; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .loading { color: #666; font-style: italic; text-align: center; margin: 20px 0; }
        .stats { font-size: 14px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌿 NDVI Plant Alignment Tester</h1>
        <p style="color: #666; margin-bottom: 20px;">📋 <strong>Best targets:</strong> Leaves with visible veins, potted plants, textured foliage</p>
        
        <button id="testBtn" onclick="testAlignment()">📸 Capture & Test All Methods</button>
        
        <div id="loading" class="loading" style="display:none;">
            ⏳ Capturing images and testing alignment methods...
        </div>
        
        <div id="error" class="error" style="display:none;"></div>
        
        <div id="summary" class="summary" style="display:none;">
            <h3>📊 Best Result:</h3>
            <p><strong>Winner:</strong> <span id="bestMethod"></span></p>
            <p><strong>Shift:</strong> <span id="bestShift"></span> pixels (<span id="shiftMagnitude"></span> total)</p>
            <p><strong>Confidence:</strong> <span id="bestConfidence"></span></p>
            <div id="shiftWarning" style="color: #d32f2f; margin-top: 10px; display: none;">
                ⚠️ <strong>Large shift detected!</strong> Try a textured object instead of smooth surfaces.
            </div>
        </div>
        
        <div id="inputImages" class="input-images" style="display:none;">
            <div class="image-box">
                <h3>📸 RGB Camera</h3>
                <img id="rgbImg" alt="RGB">
            </div>
            <div class="image-box">
                <h3>🔴 NIR Camera</h3>
                <img id="nirImg" alt="NIR">
            </div>
        </div>
        
        <div id="resultsGrid" class="results-grid" style="display:none;">
            <div id="pcBox" class="image-box method-box">
                <h3>Phase Correlation</h3>
                <img id="pcOverlay" alt="PC Overlay">
                <div class="stats">
                    <div>Shift: <span id="pcShift"></span></div>
                    <div>Confidence: <span id="pcConf"></span></div>
                </div>
            </div>
            
            <div id="orbBox" class="image-box method-box">
                <h3>ORB Features (Plant-Optimized)</h3>
                <img id="orbOverlay" alt="ORB Overlay">
                <div class="stats">
                    <div>Shift: <span id="orbShift"></span></div>
                    <div>Confidence: <span id="orbConf"></span></div>
                </div>
            </div>
            
            <div id="eccBox" class="image-box method-box">
                <h3>ECC</h3>
                <img id="eccOverlay" alt="ECC Overlay">
                <div class="stats">
                    <div>Shift: <span id="eccShift"></span></div>
                    <div>Confidence: <span id="eccConf"></span></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function testAlignment() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('summary').style.display = 'none';
            document.getElementById('inputImages').style.display = 'none';
            document.getElementById('resultsGrid').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            document.getElementById('testBtn').disabled = true;
            
            fetch('/test')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('testBtn').disabled = false;
                    
                    if (data.success) {
                        // Show input images
                        document.getElementById('rgbImg').src = data.rgb_image;
                        document.getElementById('nirImg').src = data.nir_image;
                        document.getElementById('inputImages').style.display = 'grid';
                        
                        // Show summary
                        const shiftMagnitude = Math.sqrt(data.best_shift[0]*data.best_shift[0] + data.best_shift[1]*data.best_shift[1]);
                        document.getElementById('bestMethod').textContent = data.results[data.best_method].name;
                        document.getElementById('bestShift').textContent = 
                            `X: ${data.best_shift[0].toFixed(2)}, Y: ${data.best_shift[1].toFixed(2)}`;
                        document.getElementById('shiftMagnitude').textContent = `${shiftMagnitude.toFixed(1)}px`;
                        document.getElementById('bestConfidence').textContent = data.best_confidence.toFixed(3);
                        
                        // Show warning for large shifts
                        if (shiftMagnitude > 20) {
                            document.getElementById('shiftWarning').style.display = 'block';
                        } else {
                            document.getElementById('shiftWarning').style.display = 'none';
                        }
                        
                        document.getElementById('summary').style.display = 'block';
                        
                        // Show method results
                        const methods = ['phase_correlation', 'orb', 'ecc'];
                        const prefixes = ['pc', 'orb', 'ecc'];
                        
                        methods.forEach((method, i) => {
                            const prefix = prefixes[i];
                            const result = data.results[method];
                            
                            document.getElementById(prefix + 'Overlay').src = result.overlay;
                            document.getElementById(prefix + 'Shift').textContent = 
                                `X: ${result.shift[0].toFixed(2)}, Y: ${result.shift[1].toFixed(2)}`;
                            document.getElementById(prefix + 'Conf').textContent = result.confidence.toFixed(3);
                            
                            // Highlight best method
                            const box = document.getElementById(prefix + 'Box');
                            if (method === data.best_method) {
                                box.classList.add('best');
                                box.querySelector('h3').textContent += ' 🏆';
                            } else {
                                box.classList.remove('best');
                                box.querySelector('h3').textContent = result.name;
                            }
                        });
                        
                        document.getElementById('resultsGrid').style.display = 'grid';
                        
                    } else {
                        document.getElementById('error').textContent = 'Error: ' + data.error;
                        document.getElementById('error').style.display = 'block';
                    }
                })
                .catch(error => {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('testBtn').disabled = false;
                    document.getElementById('error').textContent = 'Network error: ' + error.message;
                    document.getElementById('error').style.display = 'block';
                });
        }
    </script>
</body>
</html>
    ''')

@app.route('/test')
def test():
    return jsonify(tester.capture_and_test())

if __name__ == '__main__':
    print("🌐 Simple Alignment Tester")
    print("📱 Open: http://192.168.1.3:5000")
    print("🔍 Compare overlays visually to see which method works best")
    app.run(host='0.0.0.0', port=5000, debug=False)
