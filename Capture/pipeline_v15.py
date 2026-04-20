import cv2
import numpy as np
import time
import datetime
import threading
import os
import io
import base64
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from torchvision import transforms
from model import CustomModel # Assuming your model is defined in a file named model.py

# --- Configuration ---
UPLOAD_FOLDER = 'captured_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Global Variables ---
camera_rgb = None
camera_nir = None
# Global variables to store the latest captured images
latest_rgb_image = None
latest_nir_image = None
aligned_rgb_image = None
aligned_nir_image = None
ndvi_heatmap = None
rdvi_heatmap = None

# --- Camera Initialization (Placeholder - Replace with actual camera setup) ---
def init_cameras():
    global camera_rgb, camera_nir
    # Replace these with your actual camera initialization code for Raspberry Pi 5
    # For testing, we'll use dummy cameras or pre-recorded videos
    print("Initializing RGB Camera...")
    camera_rgb = cv2.VideoCapture(0)  # Adjust camera index as needed
    if not camera_rgb.isOpened():
        print("Error: Could not open RGB camera.")
        camera_rgb = None
    else:
        camera_rgb.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera_rgb.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Initializing NIR Camera...")
    camera_nir = cv2.VideoCapture(1)  # Adjust camera index as needed
    if not camera_nir.isOpened():
        print("Error: Could not open NIR camera.")
        camera_nir = None
    else:
        camera_nir.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera_nir.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- V2 Preprocessing (Alignment) ---
def v2_preprocessing(rgb_img, nir_img):
    """
    Performs alignment of NIR image to RGB image.
    This is a placeholder, you'll need to implement your actual alignment logic here.
    For demonstration, we'll just resize to match if dimensions differ.
    """
    if rgb_img is None or nir_img is None:
        return None, None

    h_rgb, w_rgb, _ = rgb_img.shape
    h_nir, w_nir = nir_img.shape if len(nir_img.shape) == 2 else nir_img.shape[:2]

    # Simple resize for demonstration. Implement actual alignment (e.g., using SIFT/ORB + homography)
    if h_rgb != h_nir or w_rgb != w_nir:
        nir_img_resized = cv2.resize(nir_img, (w_rgb, h_rgb), interpolation=cv2.INTER_AREA)
    else:
        nir_img_resized = nir_img

    # If NIR is grayscale, convert to 3 channels to match RGB for display
    if len(nir_img_resized.shape) == 2:
        nir_img_resized = cv2.cvtColor(nir_img_resized, cv2.COLOR_GRAY2BGR)

    # For a more robust solution, you would use feature matching and homography:
    # Example (conceptual, requires libraries like OpenCV's SIFT/ORB):
    # orb = cv2.ORB_create()
    # kp1, des1 = orb.detectAndCompute(rgb_img, None)
    # kp2, des2 = orb.detectAndCompute(nir_img, None)
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = bf.match(des1, des2)
    # matches = sorted(matches, key=lambda x: x.distance)
    # # Use RANSAC to find homography
    # src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    # dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # nir_img_aligned = cv2.warpPerspective(nir_img, M, (w_rgb, h_rgb))

    # For now, just return the resized versions as a placeholder for "aligned"
    return rgb_img, nir_img_resized

# --- NDVI & RDVI Calculation ---
def calculate_indices(rgb_img, nir_img):
    """
    Calculates NDVI and RDVI from RGB and NIR images.
    Assumes NIR image is single channel or has the relevant NIR band.
    Assumes RGB image is BGR.
    """
    if rgb_img is None or nir_img is None:
        return None, None

    # Ensure NIR is single channel for calculation
    if len(nir_img.shape) == 3:
        nir_single_channel = nir_img[:, :, 0] # Assuming NIR is in the first channel
    else:
        nir_single_channel = nir_img

    # Extract Red channel from BGR RGB image
    red_channel = rgb_img[:, :, 2]

    # Convert to float for calculations
    red_channel = red_channel.astype(np.float32)
    nir_single_channel = nir_single_channel.astype(np.float32)

    # Avoid division by zero
    denominator_ndvi = (nir_single_channel + red_channel)
    denominator_ndvi[denominator_ndvi == 0] = 1e-6 # Small epsilon to prevent zero division

    denominator_rdvi = (np.sqrt(nir_single_channel + red_channel))
    denominator_rdvi[denominator_rdvi == 0] = 1e-6 # Small epsilon

    # Calculate NDVI
    ndvi = (nir_single_channel - red_channel) / denominator_ndvi

    # Calculate RDVI
    rdvi = (nir_single_channel - red_channel) / denominator_rdvi

    return ndvi, rdvi

# --- Visualization for NDVI/RDVI ---
def create_heatmap_display(index_map, title):
    """
    Generates a clearer heatmap visualization for NDVI/RDVI.
    Maps values from -1 to 1 (or 0 to 1 for some indices) to a colormap.
    """
    if index_map is None:
        return np.zeros((480, 640, 3), dtype=np.uint8) # Return black image if no data

    # Normalize to 0-255 for colormap application
    normalized_map = cv2.normalize(index_map, None, 0, 255, cv2.NORM_MINMAX)
    normalized_map = normalized_map.astype(np.uint8)

    # Apply a colormap (e.g., COLORMAP_JET, COLORMAP_VIRIDIS, COLORMAP_TURBO)
    heatmap = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)

    # Add title
    cv2.putText(heatmap, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return heatmap

# --- AI Inference ---
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Adjust size to your model's input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_ai_model():
    global model
    try:
        # Instantiate your CustomModel
        model = CustomModel() # Make sure CustomModel expects the correct input channels (e.g., 3 for RGB)
        model.load_state_dict(torch.load("your_model_weights.pth", map_location=device))
        model.to(device)
        model.eval()
        print("AI Model loaded successfully.")
    except Exception as e:
        print(f"Error loading AI model: {e}")
        model = None

def run_inference(image_path):
    if model is None:
        load_ai_model() # Try loading if not loaded
        if model is None:
            return "AI Model not loaded."

    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            # Assuming your model outputs probabilities for classes
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_class_idx = torch.argmax(probabilities).item()
            # Replace with your actual class names
            class_names = ["Class A", "Class B", "Class C"] # Example class names
            prediction = class_names[predicted_class_idx]
            confidence = probabilities[predicted_class_idx].item()

        return f"Prediction: {prediction}, Confidence: {confidence:.2f}"
    except Exception as e:
        return f"Error during inference: {e}"

# --- Camera Streaming Functions ---
def generate_frames(camera_obj):
    while True:
        if camera_obj is None or not camera_obj.isOpened():
            print("Camera not available for streaming.")
            time.sleep(1)
            continue
        success, frame = camera_obj.read()
        if not success:
            print("Failed to read frame from camera.")
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

def is_local_request():
    # Check if the request originated from the same machine (localhost or 127.0.0.1)
    # This is a simple check; for production, you might want more robust authentication.
    client_ip = request.remote_addr
    return client_ip == '127.0.0.1' or client_ip == 'localhost'

@app.route('/video_feed_rgb')
def video_feed_rgb():
    return Response(generate_frames(camera_rgb),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_nir')
def video_feed_nir():
    return Response(generate_frames(camera_nir),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_and_process', methods=['POST'])
def capture_and_process():
    global latest_rgb_image, latest_nir_image, aligned_rgb_image, aligned_nir_image, ndvi_heatmap, rdvi_heatmap

    if not is_local_request():
        return jsonify({"status": "error", "message": "Access denied. Only local SSH connection can trigger capture."}), 403

    if camera_rgb is None or camera_nir is None:
        return jsonify({"status": "error", "message": "Cameras not initialized."}), 500

    print("Capturing images...")
    ret_rgb, frame_rgb = camera_rgb.read()
    ret_nir, frame_nir = camera_nir.read()

    if not ret_rgb or not ret_nir:
        return jsonify({"status": "error", "message": "Failed to capture one or both images."}), 500

    latest_rgb_image = frame_rgb
    latest_nir_image = frame_nir

    # Align images
    aligned_rgb_image, aligned_nir_image = v2_preprocessing(latest_rgb_image, latest_nir_image)
    if aligned_rgb_image is None or aligned_nir_image is None:
        return jsonify({"status": "error", "message": "Image alignment failed."}), 500

    # Calculate NDVI and RDVI
    ndvi, rdvi = calculate_indices(aligned_rgb_image, aligned_nir_image)

    # Create heatmap visualizations
    ndvi_heatmap = create_heatmap_display(ndvi, "NDVI Heatmap")
    rdvi_heatmap = create_heatmap_display(rdvi, "RDVI Heatmap")

    # Save images to datahub (simulate saving to a directory)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rgb_filename = f"rgb_{timestamp}.jpg"
    nir_filename = f"nir_{timestamp}.jpg"
    ndvi_filename = f"ndvi_heatmap_{timestamp}.jpg"
    rdvi_filename = f"rdvi_heatmap_{timestamp}.jpg"

    rgb_path = os.path.join(app.config['UPLOAD_FOLDER'], rgb_filename)
    nir_path = os.path.join(app.config['UPLOAD_FOLDER'], nir_filename)
    ndvi_path = os.path.join(app.config['UPLOAD_FOLDER'], ndvi_filename)
    rdvi_path = os.path.join(app.config['UPLOAD_FOLDER'], rdvi_filename)

    cv2.imwrite(rgb_path, latest_rgb_image)
    cv2.imwrite(nir_path, latest_nir_image)
    cv2.imwrite(ndvi_path, ndvi_heatmap)
    cv2.imwrite(rdvi_path, rdvi_heatmap)

    print(f"Captured images saved to {app.config['UPLOAD_FOLDER']}")

    # Return success status
    return jsonify({"status": "success", "message": "Images captured, processed, and saved."})

@app.route('/display_processed_images')
def display_processed_images():
    # This route will serve the processed images for display on the webpage
    # We will encode them as base64 strings to embed directly in the HTML
    global latest_rgb_image, latest_nir_image, aligned_rgb_image, aligned_nir_image, ndvi_heatmap, rdvi_heatmap

    images_data = {}

    def img_to_base64(img_array):
        if img_array is None:
            return ""
        _, buffer = cv2.imencode('.jpg', img_array)
        return base64.b64encode(buffer).decode('utf-8')

    images_data['latest_rgb'] = img_to_base64(latest_rgb_image)
    images_data['latest_nir'] = img_to_base64(latest_nir_image)
    images_data['aligned_rgb'] = img_to_base64(aligned_rgb_image)
    images_data['aligned_nir'] = img_to_base64(aligned_nir_image)
    images_data['ndvi_heatmap'] = img_to_base64(ndvi_heatmap)
    images_data['rdvi_heatmap'] = img_to_base64(rdvi_heatmap)

    return jsonify(images_data)

@app.route('/run_ai_inference', methods=['POST'])
def run_ai_inference():
    if not is_local_request():
        return jsonify({"status": "error", "message": "Access denied. Only local SSH connection can trigger AI inference."}), 403

    # Get the latest RGB image path that was saved
    # This assumes that the capture_and_process route saves images,
    # and we want to run inference on the most recently saved RGB image.
    # You might want to make this more robust by passing an image ID or path.
    all_files = sorted(os.listdir(app.config['UPLOAD_FOLDER']), reverse=True)
    latest_rgb_file = None
    for f in all_files:
        if f.startswith('rgb_') and f.endswith('.jpg'):
            latest_rgb_file = f
            break

    if latest_rgb_file is None:
        return jsonify({"status": "error", "message": "No RGB image found for inference. Please capture and process first."}), 400

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], latest_rgb_file)
    print(f"Running inference on: {image_path}")
    inference_result = run_inference(image_path)

    return jsonify({"status": "success", "result": inference_result})

@app.route('/shutdown', methods=['POST'])
def shutdown():
    if not is_local_request():
        return jsonify({"status": "error", "message": "Access denied. Only local SSH connection can shut down."}), 403

    print("Shutting down the server...")
    func = request.environ.get('werkzeug.server.shutdown')
    if func is not None:
        func()
    return "Server shutting down..."

# --- Main execution ---
if __name__ == '__main__':
    # Initialize cameras in a separate thread if they block, or here directly
    init_cameras()
    # Load AI model at startup
    load_ai_model()

    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True) # debug=True is for development, set to False for production
