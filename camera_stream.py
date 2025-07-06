from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import atexit

app = Flask(__name__)

# Initialize two cameras (camera 0 and camera 1)
picam2_1 = Picamera2(0)
picam2_1.start()
picam2_2 = Picamera2(1)
picam2_2.start()

def cleanup():
    print("Releasing cameras...")
    picam2_1.stop()
    picam2_2.stop()

atexit.register(cleanup)

def gen_frames(picam):
    while True:
        frame = picam.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', frame_bgr)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def align_image(base_img, img_to_align, dx=0, dy=0):
    # Shift img_to_align by (dx, dy) pixels
    rows, cols = base_img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned = cv2.warpAffine(img_to_align, M, (cols, rows))
    return aligned

def align_image_ecc(base_img, img_to_align):
    # Convert to grayscale for ECC
    base_gray = cv2.cvtColor(base_img, cv2.COLOR_RGB2GRAY)
    align_gray = cv2.cvtColor(img_to_align, cv2.COLOR_RGB2GRAY)
    sz = base_gray.shape
    # Initialize warp matrix to identity
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    # Define the motion model (translation)
    warp_mode = cv2.MOTION_TRANSLATION
    # ECC criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
    try:
        cc, warp_matrix = cv2.findTransformECC(base_gray, align_gray, warp_matrix, warp_mode, criteria)
        aligned = cv2.warpAffine(img_to_align, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned
    except cv2.error:
        # If ECC fails, return the original image
        return img_to_align

def align_image_orb(base_img, img_to_align):
    # Convert to grayscale
    base_gray = cv2.cvtColor(base_img, cv2.COLOR_RGB2GRAY)
    align_gray = cv2.cvtColor(img_to_align, cv2.COLOR_RGB2GRAY)
    # ORB detector
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(base_gray, None)
    kp2, des2 = orb.detectAndCompute(align_gray, None)
    # Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return img_to_align
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) < 4:
        return img_to_align
    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:20]]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:20]]).reshape(-1, 1, 2)
    # Find homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is not None:
        h, w = base_gray.shape
        aligned = cv2.warpPerspective(img_to_align, M, (w, h))
        return aligned
    else:
        return img_to_align

def align_image_phasecorr(base_img, img_to_align):
    # Convert to grayscale
    base_gray = cv2.cvtColor(base_img, cv2.COLOR_RGB2GRAY)
    align_gray = cv2.cvtColor(img_to_align, cv2.COLOR_RGB2GRAY)
    # Ensure float32 for phaseCorrelate
    base_gray = np.float32(base_gray)
    align_gray = np.float32(align_gray)
    # Compute shift
    shift, response = cv2.phaseCorrelate(align_gray, base_gray)
    dx, dy = shift
    # Apply translation
    rows, cols = base_img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned = cv2.warpAffine(img_to_align, M, (cols, rows))
    return aligned

def gen_ndvi_frames():
    while True:
        rgb = picam2_1.capture_array()
        nir = picam2_2.capture_array()
        # Align NIR to RGB using phase correlation
        nir_aligned = align_image_phasecorr(rgb, nir)
        nir_channel = nir_aligned[:, :, 0].astype(float)
        red_channel = rgb[:, :, 0].astype(float)
        # NDVI calculation
        bottom = (nir_channel + red_channel)
        bottom[bottom == 0] = 0.01  # avoid division by zero
        ndvi = (nir_channel - red_channel) / bottom
        # Scale NDVI to 0-255 for display
        ndvi_normalized = ((ndvi + 1) / 2 * 255).astype(np.uint8)
        ndvi_colored = cv2.applyColorMap(ndvi_normalized, cv2.COLORMAP_JET)
        ret, buffer = cv2.imencode('.jpg', ndvi_colored)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def gen_evi_frames():
    while True:
        rgb = picam2_1.capture_array()
        nir = picam2_2.capture_array()
        # Align NIR to RGB using phase correlation
        nir_aligned = align_image_phasecorr(rgb, nir)
        nir_channel = nir_aligned[:, :, 0].astype(float)
        red_channel = rgb[:, :, 0].astype(float)
        blue_channel = rgb[:, :, 2].astype(float)
        # Apply median filter to reduce noise
        nir_channel = cv2.GaussianBlur(nir_channel.astype(np.uint8), (5,5), 0).astype(float)
        red_channel = cv2.GaussianBlur(red_channel.astype(np.uint8), (5,5), 0).astype(float)
        blue_channel = cv2.GaussianBlur(blue_channel.astype(np.uint8), (5,5), 0).astype(float)
        denominator = (nir_channel + 6 * red_channel - 7.5 * blue_channel + 1)
        denominator[denominator == 0] = 0.01  # avoid division by zero
        evi = 2.5 * (nir_channel - red_channel) / denominator
        # Bound EVI to [-1, 1] and mask extreme values
        evi = np.clip(evi, -1, 1)
        evi_masked = np.where((evi > 1) | (evi < -1), 0, evi)
        # Scale EVI to 0-255 for display
        evi_normalized = ((evi_masked + 1) / 2 * 255).astype(np.uint8)
        evi_colored = cv2.applyColorMap(evi_normalized, cv2.COLORMAP_JET)
        ret, buffer = cv2.imencode('.jpg', evi_colored)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed1')
def video_feed1():
    return Response(gen_frames(picam2_1),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(gen_frames(picam2_2),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ndvi_feed')
def ndvi_feed():
    return Response(gen_ndvi_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/evi_feed')
def evi_feed():
    return Response(gen_evi_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
    <head>
        <style>
            .grid-container {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                justify-items: center;
                align-items: center;
            }
            .grid-item {
                text-align: center;
            }
            img {
                max-width: 100%;
                height: auto;
                border: 2px solid #333;
                border-radius: 8px;
            }
        </style>
    </head>
    <body>
        <div class="grid-container">
            <div class="grid-item">
                <h2>Camera 1 (RGB)</h2>
                <img src="/video_feed1">
            </div>
            <div class="grid-item">
                <h2>Camera 2 (NIR)</h2>
                <img src="/video_feed2">
            </div>
            <div class="grid-item">
                <h2>NDVI</h2>
                <img src="/ndvi_feed">
            </div>
            <div class="grid-item">
                <h2>EVI</h2>
                <img src="/evi_feed">
            </div>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
