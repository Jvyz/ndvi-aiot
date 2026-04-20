# -*- coding: utf-8 -*-
"""
Simplified Dataset Capture: Healthy vs Infected Tomato Leaves
NO REAL-TIME STREAMING - Alignment only on capture for better performance

Categories:
  - Healthy: Leaves from disease-free plants
  - Infected: Leaves showing bacterial spot symptoms

Usage:
  python dataset_capture.py --rgb 0 --ir 1 --lan
"""

import os, time, threading, argparse
import numpy as np
import socket
import csv
from datetime import datetime
import cv2 as cv
from flask import Flask, jsonify, render_template_string
from picamera2 import Picamera2

from opencv_registration import OpenCVAligner, alignment_quality

# ---------------- Camera Setup ----------------

def setup_camera(cam_index, size, fps, exposure_us, analogue_gain, awb=True, ae=False, warmup_s=0.4):
    """Initialize and configure PiCamera2"""
    cam = Picamera2(camera_num=cam_index)
    cfg = cam.create_video_configuration(main={"size": size, "format": "RGB888"}, buffer_count=4)
    cam.configure(cfg)
    cam.start()
    time.sleep(warmup_s)

    frame_period_us = int(1_000_000 / fps)
    exp = min(exposure_us, frame_period_us - 1000)
    
    try:
        cam.set_controls({"AeEnable": ae, "AwbEnable": awb})
    except Exception:
        pass
    
    cam.set_controls({
        "ExposureTime": exp,
        "AnalogueGain": analogue_gain,
        "ColourGains": (1.0, 1.0),
        "FrameDurationLimits": (frame_period_us, frame_period_us),
    })
    return cam

def get_frame_bgr(cam):
    """Capture frame from PiCamera2 and convert RGB -> BGR for OpenCV"""
    arr = cam.capture_array("main")
    return cv.cvtColor(arr, cv.COLOR_RGB2BGR)

# ---------------- Global State ----------------

class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.healthy_count = 0
        self.infected_count = 0
        self.last_message = "Ready to capture"
        self.last_capture_info = {}
        self.metadata_file = None
        self.rgb_cam = None
        self.ir_cam = None
        self.aligner = None

S = State()

# ---------------- Web Interface (NO STREAMING) ----------------

HTML_PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Dataset Capture - X. perforans Detection</title>
<style>
 *{box-sizing:border-box}
 body{margin:0;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Arial;background:#0a0a0a;color:#e8e8e8}
 header{padding:24px;border-bottom:2px solid #2a2a2a;background:linear-gradient(135deg,#1a1a1a,#0f0f0f);text-align:center}
 h1{margin:0;font-size:32px;font-weight:700;background:linear-gradient(135deg,#10b981,#059669);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
 .subtitle{margin:8px 0 0 0;font-size:15px;color:#888}
 .container{max-width:900px;margin:40px auto;padding:0 24px}
 .capture-section{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-bottom:32px}
 .capture-card{background:#1a1a1a;border-radius:12px;padding:32px;border:2px solid #2a2a2a;transition:all 0.3s}
 .capture-card.healthy{border-color:#10b981}
 .capture-card.healthy:hover{border-color:#059669;box-shadow:0 8px 24px rgba(16,185,129,0.2)}
 .capture-card.infected{border-color:#f59e0b}
 .capture-card.infected:hover{border-color:#d97706;box-shadow:0 8px 24px rgba(245,158,11,0.2)}
 .card-icon{font-size:48px;margin-bottom:16px}
 .card-header{font-size:24px;font-weight:700;margin-bottom:12px}
 .card-header.healthy{color:#10b981}
 .card-header.infected{color:#f59e0b}
 .card-desc{font-size:14px;color:#888;margin-bottom:24px;line-height:1.6}
 .btn-capture{width:100%;border:none;padding:16px 24px;border-radius:8px;cursor:pointer;font-size:18px;font-weight:600;transition:all 0.2s}
 .btn-healthy{background:linear-gradient(135deg,#10b981,#059669);color:#fff}
 .btn-healthy:hover{transform:translateY(-2px);box-shadow:0 8px 20px rgba(16,185,129,0.4)}
 .btn-infected{background:linear-gradient(135deg,#f59e0b,#d97706);color:#fff}
 .btn-infected:hover{transform:translateY(-2px);box-shadow:0 8px 20px rgba(245,158,11,0.4)}
 .btn-capture:active{transform:translateY(0)}
 .btn-capture:disabled{opacity:0.5;cursor:not-allowed;transform:none}
 .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:32px}
 .stat-card{background:#1a1a1a;padding:24px;border-radius:10px;border:1px solid #2a2a2a;text-align:center}
 .stat-label{font-size:12px;color:#888;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px}
 .stat-value{font-size:36px;font-weight:700}
 .count-healthy{color:#10b981}
 .count-infected{color:#f59e0b}
 .count-total{color:#0ea5e9}
 .status-box{background:#1e293b;border-left:4px solid #0ea5e9;padding:20px 24px;border-radius:8px;margin-bottom:32px}
 .status-message{font-size:15px;line-height:1.6;color:#e8e8e8}
 .capture-info{margin-top:12px;font-size:13px;color:#94a3b8}
 .cam-settings{background:#1a1a1a;padding:24px;border-radius:10px;border:1px solid #2a2a2a;margin-bottom:32px}
 .cam-settings h3{margin:0 0 16px 0;font-size:18px;color:#e8e8e8}
 .settings-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
 .setting-group{display:flex;flex-direction:column;gap:8px}
 .setting-group label{font-size:13px;color:#888}
 .setting-group input{padding:10px;background:#0a0a0a;border:1px solid #444;color:#e8e8e8;border-radius:6px;font-size:14px}
 .btn-small{background:#374151;color:#fff;border:none;padding:10px 16px;border-radius:6px;cursor:pointer;font-size:14px;font-weight:500;margin-top:8px}
 .btn-small:hover{background:#4b5563}
 .instructions{background:#1e293b;border-left:4px solid #0ea5e9;padding:20px 24px;border-radius:8px;margin-bottom:32px}
 .instructions h3{margin:0 0 12px 0;font-size:16px;color:#38bdf8}
 .instructions ul{margin:8px 0;padding-left:20px;color:#94a3b8;line-height:1.8}
</style>
</head>
<body>
<header>
  <h1>🍃 Tomato Leaf Dataset Collection</h1>
  <p class="subtitle">X. perforans Detection Training Data | RGB + NIR Imaging</p>
  <p class="subtitle">⚡ Optimized Mode - No Streaming for Better Performance</p>
</header>

<div class="container">
  <div class="instructions">
    <h3>📋 Collection Protocol</h3>
    <ul>
      <li><strong>Sample:</strong> Detach individual leaves, place on black fabric background</li>
      <li><strong>Lighting:</strong> Use consistent lighting (sunlight OK with consistent time/weather)</li>
      <li><strong>Healthy:</strong> Disease-free plants, no visible symptoms</li>
      <li><strong>Infected:</strong> Leaves with bacterial spot (dark spots, yellowing)</li>
      <li><strong>Click button:</strong> Cameras capture, align, and save automatically</li>
    </ul>
  </div>

  <div class="stats">
    <div class="stat-card">
      <div class="stat-label">Healthy Captured</div>
      <div class="stat-value count-healthy" id="healthy-count">0</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Infected Captured</div>
      <div class="stat-value count-infected" id="infected-count">0</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Total Dataset</div>
      <div class="stat-value count-total" id="total-count">0</div>
    </div>
  </div>

  <div class="status-box">
    <div class="status-message" id="message">Ready to capture dataset images</div>
    <div class="capture-info" id="capture-info"></div>
  </div>

  <div class="capture-section">
    <div class="capture-card healthy">
      <div class="card-icon">✅</div>
      <div class="card-header healthy">Healthy Leaf</div>
      <div class="card-desc">
        Capture leaves from disease-free plants with no visible symptoms.
      </div>
      <button class="btn-capture btn-healthy" onclick="capture('healthy')">
        Capture Healthy Leaf
      </button>
    </div>
    
    <div class="capture-card infected">
      <div class="card-icon">⚠️</div>
      <div class="card-header infected">Infected Leaf</div>
      <div class="card-desc">
        Capture leaves showing bacterial spot symptoms (spots, lesions, yellowing).
      </div>
      <button class="btn-capture btn-infected" onclick="capture('infected')">
        Capture Infected Leaf
      </button>
    </div>
  </div>

  <div class="cam-settings">
    <h3>⚙️ Camera Settings</h3>
    <div class="settings-grid">
      <div>
        <div class="setting-group">
          <label>RGB Exposure (µs):</label>
          <input id="rgbexp" type="number" value="6000" step="500">
        </div>
        <div class="setting-group">
          <label>RGB Gain:</label>
          <input id="rgbgain" type="number" value="2.0" step="0.1">
        </div>
        <button class="btn-small" onclick="applySettings('rgb')">Apply RGB Settings</button>
      </div>
      <div>
        <div class="setting-group">
          <label>NIR Exposure (µs):</label>
          <input id="irexp" type="number" value="6000" step="500">
        </div>
        <div class="setting-group">
          <label>NIR Gain:</label>
          <input id="irgain" type="number" value="4.0" step="0.1">
        </div>
        <button class="btn-small" onclick="applySettings('ir')">Apply NIR Settings</button>
      </div>
    </div>
  </div>
</div>

<script>
 async function capture(category){
   const btn = event.target;
   btn.disabled = true;
   const originalText = btn.textContent;
   btn.textContent = '⏳ Capturing & Aligning...';
   document.getElementById('message').textContent = 'Capturing images and computing alignment...';
   
   try{
     const r = await (await fetch(`/capture?category=${category}`)).json();
     if(r.ok){
       document.getElementById('message').textContent = r.msg || 'Captured successfully';
       const info = `NCC Quality: ${r.quality.toFixed(4)} | Align Time: ${r.align_time_ms.toFixed(1)}ms`;
       document.getElementById('capture-info').textContent = info;
       if(category === 'healthy'){
         document.getElementById('healthy-count').textContent = r.healthy_count || 0;
       } else {
         document.getElementById('infected-count').textContent = r.infected_count || 0;
       }
       const total = parseInt(document.getElementById('healthy-count').textContent) + 
                     parseInt(document.getElementById('infected-count').textContent);
       document.getElementById('total-count').textContent = total;
     } else {
       document.getElementById('message').textContent = '❌ ' + (r.msg || 'Capture failed');
       document.getElementById('capture-info').textContent = '';
     }
   }catch(e){
     document.getElementById('message').textContent = '❌ Capture failed: ' + e;
     document.getElementById('capture-info').textContent = '';
   }finally{
     btn.disabled = false;
     btn.textContent = originalText;
   }
 }

 async function applySettings(cam){
   const exp = document.getElementById(cam + 'exp').value;
   const gain = document.getElementById(cam + 'gain').value;
   try{
     const r = await (await fetch(`/settings?camera=${cam}&exp_us=${exp}&gain=${gain}`)).json();
     document.getElementById('message').textContent = r.msg || 'Settings applied';
     document.getElementById('capture-info').textContent = '';
   }catch(e){
     document.getElementById('message').textContent = 'Failed to apply settings';
   }
 }

 // Update counts on load
 async function updateCounts(){
   try{
     const s = await (await fetch('/status')).json();
     document.getElementById('healthy-count').textContent = s.healthy_count || 0;
     document.getElementById('infected-count').textContent = s.infected_count || 0;
     document.getElementById('total-count').textContent = s.total_count || 0;
     if(s.last_message) document.getElementById('message').textContent = s.last_message;
   }catch(e){}
 }
 updateCounts();
</script>
</body>
</html>
"""

def create_app(args):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(HTML_PAGE)

    @app.route("/status")
    def status():
        with S.lock:
            return jsonify({
                "healthy_count": S.healthy_count,
                "infected_count": S.infected_count,
                "total_count": S.healthy_count + S.infected_count,
                "last_message": S.last_message
            })

    @app.route("/capture")
    def capture():
        """Capture, align, and save RGB + NIR pair"""
        from flask import request
        category = request.args.get("category", "healthy")
        
        if category not in ["healthy", "infected"]:
            return jsonify({"ok": False, "msg": "Invalid category"}), 400
        
        try:
            # Capture frames
            rgb_bgr = get_frame_bgr(S.rgb_cam)
            ir_rgb = get_frame_bgr(S.ir_cam)
            ir_gray = cv.cvtColor(ir_rgb, cv.COLOR_BGR2GRAY)

            # Align NIR to RGB
            t0 = time.time()
            ir_aligned, align_info = S.aligner.align_ir_to_rgb(rgb_bgr, ir_gray)
            t1 = time.time()
            align_time_ms = (t1 - t0) * 1000.0

            if ir_aligned is None:
                return jsonify({
                    "ok": False,
                    "msg": "❌ Alignment failed - check camera positioning"
                }), 400

            # Compute alignment quality
            ncc = alignment_quality(rgb_bgr, ir_aligned)

            # Save images
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            category_dir = os.path.join(args.dataset_dir, category)
            os.makedirs(category_dir, exist_ok=True)

            rgb_path = os.path.join(category_dir, f"rgb_{category}_{timestamp}.png")
            nir_path = os.path.join(category_dir, f"nir_{category}_{timestamp}.png")

            cv.imwrite(rgb_path, rgb_bgr)
            cv.imwrite(nir_path, cv.cvtColor(ir_aligned, cv.COLOR_GRAY2BGR))

            # Update counts
            with S.lock:
                if category == "healthy":
                    S.healthy_count += 1
                    count = S.healthy_count
                else:
                    S.infected_count += 1
                    count = S.infected_count

                total = S.healthy_count + S.infected_count

                # Log metadata
                log_metadata(
                    timestamp=timestamp,
                    category=category,
                    rgb_path=rgb_path,
                    nir_path=nir_path,
                    ncc_score=ncc,
                    align_time_ms=align_time_ms,
                    sample_number=count
                )

                S.last_message = f"✅ Saved {category} leaf #{count} | NCC: {ncc:.4f} | Total: {total}"

            return jsonify({
                "ok": True,
                "msg": S.last_message,
                "healthy_count": S.healthy_count if category == "healthy" else None,
                "infected_count": S.infected_count if category == "infected" else None,
                "quality": float(ncc),
                "align_time_ms": align_time_ms,
                "files": [rgb_path, nir_path]
            })

        except Exception as e:
            return jsonify({"ok": False, "msg": f"Error: {str(e)}"}), 500

    @app.route("/settings")
    def settings():
        """Update camera settings"""
        from flask import request
        cam = request.args.get("camera", "rgb")
        exp = request.args.get("exp_us")
        gain = request.args.get("gain")

        target = S.rgb_cam if cam == "rgb" else S.ir_cam
        
        to_set = {}
        if exp: to_set["ExposureTime"] = max(100, int(exp))
        if gain: to_set["AnalogueGain"] = max(1.0, float(gain))
        
        try:
            target.set_controls(to_set)
            msg = f"✓ Updated {cam.upper()}: {to_set}"
            with S.lock:
                S.last_message = msg
            return jsonify({"ok": True, "msg": msg})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    return app

# ---------------- Metadata Logging ----------------

def init_metadata_file(dataset_dir):
    """Initialize CSV metadata file"""
    metadata_path = os.path.join(dataset_dir, "dataset_metadata.csv")
    
    if not os.path.exists(metadata_path):
        with open(metadata_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'capture_id', 'timestamp', 'datetime', 'category', 'sample_number',
                'rgb_filename', 'nir_filename', 'ncc_score', 'alignment_time_ms',
                'image_width', 'image_height', 'notes'
            ])
    
    return metadata_path

def log_metadata(timestamp, category, rgb_path, nir_path, ncc_score, align_time_ms, sample_number):
    """Log capture metadata to CSV"""
    if S.metadata_file is None:
        return
    
    capture_id = f"{category}_{sample_number:03d}"
    dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        img = cv.imread(rgb_path)
        height, width = img.shape[:2]
    except:
        width, height = 0, 0
    
    try:
        with open(S.metadata_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                capture_id, timestamp, dt_str, category, sample_number,
                os.path.basename(rgb_path), os.path.basename(nir_path),
                f"{ncc_score:.4f}", f"{align_time_ms:.2f}",
                width, height, ''
            ])
    except Exception as e:
        print(f"Warning: Failed to log metadata: {e}")

# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser(description="Dataset capture (no streaming)")
    
    parser.add_argument("--rgb", type=int, default=0)
    parser.add_argument("--ir", type=int, default=1)
    parser.add_argument("--size", default="1920x1080")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--exp_us_rgb", type=int, default=6000)
    parser.add_argument("--exp_us_ir", type=int, default=6000)
    parser.add_argument("--gain_rgb", type=float, default=2.0)
    parser.add_argument("--gain_ir", type=float, default=4.0)
    parser.add_argument("--rgb_awb", action="store_true", default=True)
    parser.add_argument("--grid_step", type=int, default=24)
    parser.add_argument("--down_for_H", type=float, default=0.5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--lan", action="store_true")
    parser.add_argument("--dataset_dir", default="datasets")

    args = parser.parse_args()
    args.W, args.H = map(int, args.size.split("x"))

    if args.lan:
        args.host = "0.0.0.0"

    # Load calibration
    rgb_mapx = np.load("rgb_mapx.npy")
    rgb_mapy = np.load("rgb_mapy.npy")
    ir_mapx = np.load("ir_mapx.npy")
    ir_mapy = np.load("ir_mapy.npy")

    # Initialize aligner
    S.aligner = OpenCVAligner(
        rectify_maps=(rgb_mapx, rgb_mapy, ir_mapx, ir_mapy),
        grid_step=args.grid_step,
        down_for_H=args.down_for_H
    )

    # Setup cameras
    S.rgb_cam = setup_camera(args.rgb, (args.W, args.H), args.fps,
                             args.exp_us_rgb, args.gain_rgb, awb=args.rgb_awb)
    S.ir_cam = setup_camera(args.ir, (args.W, args.H), args.fps,
                            args.exp_us_ir, args.gain_ir, awb=False)

    # Initialize metadata
    os.makedirs(args.dataset_dir, exist_ok=True)
    S.metadata_file = init_metadata_file(args.dataset_dir)

    # Start web server
    app = create_app(args)
    
    print(f"\n{'='*70}")
    print(f"🍃 Tomato Leaf Dataset Collection (Optimized - No Streaming)")
    print(f"{'='*70}")
    print(f"📁 Dataset: {args.dataset_dir}/")
    print(f"🌐 Web UI: http://{args.host}:{args.port}/")
    
    if args.host == "0.0.0.0":
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            print(f"🔗 LAN: http://{local_ip}:{args.port}/")
        except:
            pass
    
    print(f"⚡ Performance: Cameras idle until capture button clicked")
    print(f"{'='*70}\n")
    
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)

if __name__ == "__main__":
    main()
