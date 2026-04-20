from flask import Flask, render_template_string, jsonify
import cv2
import numpy as np
from picamera2 import Picamera2
import base64
import threading
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)

class SimpleAlignmentTester:
    def __init__(self):
        self.rgb_cam = Picamera2(0)
        self.nir_cam = Picamera2(1)
        self.rgb_cam.configure(self.rgb_cam.create_preview_configuration(main={"size": (640, 480)}))
        self.nir_cam.configure(self.nir_cam.create_preview_configuration(main={"size": (640, 480)}))
        self.rgb_cam.start()
        self.nir_cam.start()
        self.results = {}
    
    def capture_and_align(self):
        try:
            # Capture frames
            rgb = self.rgb_cam.capture_array()
            nir = self.nir_cam.capture_array()
            
            # Simple phase correlation alignment
            rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            nir_gray = cv2.cvtColor(nir, cv2.COLOR_RGB2GRAY)
            
            shift, response = cv2.phaseCorrelate(np.float32(nir_gray), np.float32(rgb_gray))
            
            # Convert shift tuple to list - FIX for the error
            shift_list = [float(shift[0]), float(shift[1])]
            
            # Apply shift
            M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
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
            
            self.results = {
                'rgb_image': f"data:image/jpeg;base64,{rgb_b64}",
                'nir_image': f"data:image/jpeg;base64,{nir_b64}",
                'overlay_image': f"data:image/jpeg;base64,{overlay_b64}",
                'shift': shift_list,  # Fixed: now it's a list
                'confidence': float(response),
                'similarity': float(similarity),
                'success': True
            }
            
        except Exception as e:
            print(f"Error in capture_and_align: {e}")
            self.results = {
                'success': False,
                'error': str(e)
            }

tester = SimpleAlignmentTester()

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Simple Alignment Tester</title>
    <style>
        body { font-family: Arial; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
        .images { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-top: 20px; }
        .image-box { text-align: center; background: #fafafa; padding: 15px; border-radius: 8px; }
        img { max-width: 100%; border: 2px solid #ddd; border-radius: 5px; }
        button { background: #4CAF50; color: white; padding: 15px 30px; 
                border: none; border-radius: 5px; font-size: 16px; cursor: pointer; margin: 10px; }
        button:hover { background: #45a049; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .results { background: #e8f5e8; padding: 15px; margin: 20px 0; border-radius: 5px; border-left: 4px solid #4CAF50; }
        .error { background: #f8d7da; color: #721c24; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .loading { color: #666; font-style: italic; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📷 Simple Camera Alignment Tester</h1>
        
        <button id="testBtn" onclick="testAlignment()">🎯 Test Alignment</button>
        
        <div id="loading" class="loading" style="display:none;">
            ⏳ Testing alignment...
        </div>
        
        <div id="error" class="error" style="display:none;"></div>
        
        <div id="results" class="results" style="display:none;">
            <h3>📊 Results:</h3>
            <p><strong>Shift:</strong> <span id="shift"></span> pixels</p>
            <p><strong>Confidence:</strong> <span id="confidence"></span></p>
            <p><strong>Similarity:</strong> <span id="similarity"></span></p>
            <div id="quality"></div>
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
            <div class="image-box">
                <h3>🎯 Overlay (50% blend)</h3>
                <img id="overlay" src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjQwIiBoZWlnaHQ9IjQ4MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxOCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPk5vIGltYWdlPC90ZXh0Pjwvc3ZnPg==" alt="Overlay">
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
                        
                        // Update results - Fixed the undefined error
                        if (data.shift && data.shift.length >= 2) {
                            document.getElementById('shift').textContent = 
                                `X: ${data.shift[0].toFixed(2)}, Y: ${data.shift[1].toFixed(2)}`;
                        }
                        
                        document.getElementById('confidence').textContent = data.confidence.toFixed(3);
                        document.getElementById('similarity').textContent = data.similarity.toFixed(3);
                        
                        // Quality assessment
                        let quality = '';
                        if (data.similarity > 0.7 && Math.abs(data.shift[0]) < 5 && Math.abs(data.shift[1]) < 5) {
                            quality = '✅ <strong>Good alignment!</strong>';
                        } else if (data.similarity > 0.4) {
                            quality = '⚠️ <strong>Fair alignment</strong>';
                        } else {
                            quality = '❌ <strong>Poor alignment</strong>';
                        }
                        document.getElementById('quality').innerHTML = quality;
                        
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
    print("🌐 Simple Alignment Tester")
    print("📱 Open: http://192.168.1.3:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
