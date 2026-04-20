# -*- coding: utf-8 -*-
"""
Simplified Dataset Capture: Healthy vs Infected Tomato Leaves
Purpose: Collect aligned RGB + NIR image pairs for X. perforans detection

Categories:
  - Healthy: Leaves from disease-free plants
  - Infected: Leaves showing bacterial spot symptoms

Usage:
  python dataset_capture.py --rgb 0 --ir 1 --lan
  
Add --ae_rgb or --ae_ir for auto exposure on specific cameras
"""

import os, time, threading, argparse
import numpy as np
import socket
import csv
from datetime import datetime
import cv2 as cv
from flask import Flask, Response, jsonify, render_template_string
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
    
    # Try to set AE and AWB separately with error handling
    try:
        cam.set_controls({"AeEnable": ae})
    except Exception as e:
        print(f"Warning: AeEnable not supported on camera {cam_index}: {e}")
    
    try:
        cam.set_controls({"AwbEnable": awb})
    except Exception as e:
        print(f"Warning: AwbEnable not supported on camera {cam_index}: {e}")
    
    # Only set manual exposure if AE is disabled
    if not ae:
        cam.set_controls({
            "ExposureTime": exp,
            "AnalogueGain": analogue_gain,
            "ColourGains": (1.0, 1.0),
            "FrameDurationLimits": (frame_period_us, frame_period_us),
        })
    else:
        # For auto exposure, just set frame duration limits
        cam.set_controls({
            "FrameDurationLimits": (frame_period_us, frame_period_us),
        })
    
    return cam

def get_frame_bgr(cam):
    """Capture frame from PiCamera2 and convert RGB -> BGR for OpenCV"""
    arr = cam.capture_array("main")
    return cv.cvtColor(arr, cv.COLOR_RGB2BGR)

# ---------------- Shared State ----------------

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.rgb_jpg = None
        self.ir_jpg = None
        self.overlay_jpg = None
        self.info = {"status": "initializing"}
        self.last_message = ""
        self.healthy_count = 0
        self.infected_count = 0
        self.metadata_file = None

S = SharedState()

# ---------------- Web Interface ----------------

HTML_PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Leaf Dataset Collection - X. perforans Detection</title>
<style>
 *{box-sizing:border-box}
 body{margin:0;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Arial;background:#0a0a0a;color:#e8e8e8}
 header{padding:20px 24px;border-bottom:2px solid #2a2a2a;background:linear-gradient(135deg,#1a1a1a,#0f0f0f)}
 h1{margin:0;font-size:26px;font-weight:700;background:linear-gradient(135deg,#10b981,#059669);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
 .subtitle{margin:6px 0 0 0;font-size:14px;color:#888}
 .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;padding:20px;max-width:1900px;margin:0 auto}
 figure{margin:0;background:#1a1a1a;border-radius:10px;overflow:hidden;border:1px solid #2a2a2a;box-shadow:0 4px 12px rgba(0,0,0,0.4)}
 figcaption{padding:14px 16px;font-weight:600;font-size:15px;border-bottom:1px solid #2a2a2a;background:#222;display:flex;align-items:center;gap:8px}
 img{width:100%;display:block;min-height:280px;background:#0a0a0a}
 .controls{padding:24px;border-top:2px solid #2a2a2a;background:#1a1a1a;max-width:1900px;margin:0 auto}
 .capture-section{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:24px}
 .capture-card{background:#222;border-radius:10px;padding:20px;border:2px solid #2a2a2a}
 .capture-card.healthy{border-color:#10b981}
 .capture-card.infected{border-color:#f59e0b}
 .card-header{font-size:18px;font-weight:700;margin-bottom:12px;display:flex;align-items:center;gap:10px}
 .card-header.healthy{color:#10b981}
 .card-header.infected{color:#f59e0b}
 .card-desc{font-size:13px;color:#888;margin-bottom:16px;line-height:1.5}
 .btn-capture{width:100%;border:none;padding:14px 24px;border-radius:8px;cursor:pointer;font-size:16px;font-weight:600;transition:all 0.2s;box-shadow:0 4px 12px rgba(0,0,0,0.3)}
 .btn-healthy{background:linear-gradient(135deg,#10b981,#059669);color:#fff}
 .btn-healthy:hover{transform:translateY(-2px);box-shadow:0 6px 16px rgba(16,185,129,0.4)}
 .btn-infected{background:linear-gradient(135deg,#f59e0b,#d97706);color:#fff}
 .btn-infected:hover{transform:translateY(-2px);box-shadow:0 6px 16px rgba(245,158,11,0.4)}
 .btn-capture:active{transform:translateY(0)}
 .btn-capture:disabled{opacity:0.5;cursor:not-allowed;transform:none}
 .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:24px}
 .stat-card{background:#222;padding:16px 20px;border-radius:8px;border:1px solid #2a2a2a}
 .stat-label{font-size:12px;color:#888;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px}
 .stat-value{font-size:24px;font-weight:700}
 .stat-value.ok{color:#10b981}
 .stat-value.warn{color:#f59e0b}
 .count-healthy{color:#10b981}
 .count-infected{color:#f59e0b}
 .message{padding:14px 18px;background:#2a2a2a;border-radius:8px;font-size:14px;border-left:4px solid #0ea5e9}
 .cam-settings{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-bottom:24px;padding:16px;background:#222;border-radius:8px}
 .cam-group{display:flex;gap:8px;align-items:center;padding:8px 12px;background:#1a1a1a;border-radius:6px}
 .cam-group label{font-size:13px;color:#888}
 .cam-group input{width:80px;padding:6px 10px;background:#0a0a0a;border:1px solid #444;color:#e8e8e8;border-radius:4px;font-size:13px}
 .btn-small{background:#374151;color:#fff;border:none;padding:7px 14px;border-radius:6px;cursor:pointer;font-size:13px;font-weight:500}
 .btn-small:hover{background:#4b5563}
 .icon{font-size:20px}
 .instructions{background:#1e293b;border-left:4px solid #0ea5e9;padding:16px 20px;border-radius:8px;margin-bottom:24px}
 .instructions h3{margin:0 0 10px 0;font-size:16px;color:#38bdf8}
 .instructions ul{margin:8px 0;padding-left:20px;color:#94a3b8}
 .instructions li{margin:6px 0;line-height:1.5}
</style>
</head>
<body>
<header>
  <h1>🍃 Tomato Leaf Dataset Collection</h1>
  <p class="subtitle">X. perforans Detection Training Data | RGB + NIR Imaging</p>
</header>

<div class="grid">
  <figure>
    <figcaption><span class="icon">📸</span> RGB Camera</figcaption>
    <img src="/stream/rgb" alt="RGB Camera Feed">
  </figure>
  <figure>
    <figcaption><span class="icon">🔴</span> NIR Camera (720nm+)</figcaption>
    <img src="/stream/ir" alt="NIR Camera Feed">
  </figure>
  <figure>
    <figcaption><span class="icon">🔗</span> Aligned Overlay</figcaption>
    <img src="/stream/overlay" alt="Alignment Preview">
  </figure>
</div>

<div class="controls">
  <div class="instructions">
    <h3>📋 Collection Protocol</h3>
    <ul>
      <li><strong>Sample:</strong> Detach individual leaves, place on black fabric background</li>
      <li><strong>Lighting:</strong> Consistent lighting (sunlight OK if time/weather consistent)</li>
      <li><strong>Healthy:</strong> Disease-free plants, no visible symptoms</li>
      <li><strong>Infected:</strong> Bacterial spot symptoms (dark spots, yellowing)</li>
      <li><strong>Quality:</strong> Check NCC score before capture (should be >0.7)</li>
    </ul>
  </div>

  <div class="stats">
    <div class="stat-card">
      <div class="stat-label">Alignment Status</div>
      <div class="stat-value" id="status">—</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Quality (NCC)</div>
      <div class="stat-value" id="ncc">—</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Processing Time</div>
      <div class="stat-value" id="ms">—</div>
    </div>
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
      <div class="stat-value" id="total-count" style="color:#0ea5e9">0</div>
    </div>
  </div>

  <div class="capture-section">
    <div class="capture-card healthy">
      <div class="card-header healthy">✅ Healthy Leaf</div>
      <div class="card-desc">
        Capture leaves from disease-free plants with no visible symptoms. 
        Ensure leaf is flat on black background.
      </div>
      <button class="btn-capture btn-healthy" onclick="capture('healthy')">
        Capture Healthy Leaf
      </button>
    </div>
    
    <div class="capture-card infected">
      <div class="card-header infected">⚠️ Infected Leaf</div>
      <div class="card-desc">
        Capture leaves showing bacterial spot symptoms (dark spots, lesions, yellowing). 
        Include both symptomatic and surrounding areas.
      </div>
      <button class="btn-capture btn-infected" onclick="capture('infected')">
        Capture Infected Leaf
      </button>
    </div>
  </div>

  <div class="cam-settings">
    <div class="cam-group">
      <label>RGB Exp (µs):</label>
      <input id="rgbexp" type="number" value="6000" step="500">
      <label>Gain:</label>
      <input id="rgbgain" type="number" value="2.0" step="0.1">
      <button class="btn-small" onclick="applySettings('rgb')">Set RGB</button>
    </div>
    <div class="cam-group">
      <label>NIR Exp (µs):</label>
      <input id="irexp" type="number" value="6000" step="500">
      <label>Gain:</label>
      <input id="irgain" type="number" value="4.0" step="0.1">
      <button class="btn-small" onclick="applySettings('ir')">Set NIR</button>
    </div>
  </div>

  <div class="message" id="message">Ready to capture dataset images. Place leaf on black background and ensure good alignment quality.</div>
</div>

<script>
 async function capture(category){
   const btn = event.target;
   btn.disabled = true;
   const originalText = btn.textContent;
   btn.textContent = '⏳ Capturing...';
   
   try{
     const r = await (await fetch(`/capture?category=${category}`)).json();
     document.getElementById('message').textContent = r.msg || 'Captured successfully';
     if(category === 'healthy'){
       document.getElementById('healthy-count').textContent = r.healthy_count || 0;
     } else {
       document.getElementById('infected-count').textContent = r.infected_count || 0;
     }
     const total = (r.healthy_count || 0) + (r.infected_count || 0);
     document.getElementById('total-count').textContent = total;
   }catch(e){
     document.getElementById('message').textContent = 'Capture failed: ' + e;
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
   }catch(e){
     document.getElementById('message').textContent = 'Failed to apply settings';
   }
 }

 async function pollStatus(){
   try{
     const s = await (await fetch('/status')).json();
     const statusEl = document.getElementById('status');
     statusEl.textContent = s.status === 'ok' ? 'Good' : 'Check alignment';
     statusEl.className = s.status === 'ok' ? 'stat-value ok' : 'stat-value warn';
     
     const ncc = s.ncc;
     const nccEl = document.getElementById('ncc');
     if(ncc !== undefined && ncc !== null){
       nccEl.textContent = ncc.toFixed(4);
       nccEl.className = ncc > 0.7 ? 'stat-value ok' : 'stat-value warn';
     } else {
       nccEl.textContent = '—';
       nccEl.className = 'stat-value';
     }
     
     const msEl = document.getElementById('ms');
     if(s.ms_align){
       msEl.textContent = s.ms_align.toFixed(1) + ' ms';
     }
     
     if(s.last_message) document.getElementById('message').textContent = s.last_message;
     if(s.healthy_count !== undefined) document.getElementById('healthy-count').textContent = s.healthy_count;
     if(s.infected_count !== undefined) document.getElementById('infected-count').textContent = s.infected_count;
     if(s.total_count !== undefined) document.getElementById('total-count').textContent = s.total_count;
   }catch(e){}
   setTimeout(pollStatus, 500);
 }
 pollStatus();
</script>
</body>
</html>
"""

def create_app(args):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(HTML_PAGE)

    def mjpeg_stream(stream_attr):
        """Generate MJPEG stream from shared state attribute"""
        def gen():
            while True:
                with S.lock:
                    buf = getattr(S, stream_attr)
                if buf is None:
                    time.sleep(0.05)
                    continue
                yield b"--frame\r\n"
                yield b"Content-Type: image/jpeg\r\n"
                yield b"Content-Length: " + str(len(buf)).encode() + b"\r\n\r\n"
                yield buf + b"\r\n"
                time.sleep(0.04)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/stream/rgb")
    def stream_rgb():
        return mjpeg_stream("rgb_jpg")

    @app.route("/stream/ir")
    def stream_ir():
        return mjpeg_stream("ir_jpg")

    @app.route("/stream/overlay")
    def stream_overlay():
        return mjpeg_stream("overlay_jpg")

    @app.route("/status")
    def status():
        with S.lock:
            total = S.healthy_count + S.infected_count
            return jsonify({
                **S.info,
                "last_message": S.last_message,
                "healthy_count": S.healthy_count,
                "infected_count": S.infected_count,
                "total_count": total
            })

    @app.route("/capture")
    def capture():
        """Save RGB + NIR pair as either healthy or infected"""
        from flask import request
        category = request.args.get("category", "healthy")
        
        if category not in ["healthy", "infected"]:
            return jsonify({"ok": False, "msg": "Invalid category"}), 400
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        category_dir = os.path.join(args.dataset_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        with S.lock:
            rgb_img = cv.imdecode(np.frombuffer(S.rgb_jpg, np.uint8), cv.IMREAD_COLOR) if S.rgb_jpg else None
            ir_img = cv.imdecode(np.frombuffer(S.ir_jpg, np.uint8), cv.IMREAD_COLOR) if S.ir_jpg else None
            quality = S.info.get("ncc", 0)
            align_time = S.info.get("ms_align", 0)
        
        if rgb_img is None or ir_img is None:
            return jsonify({"ok": False, "msg": "No frames available"}), 400
        
        rgb_path = os.path.join(category_dir, f"rgb_{category}_{timestamp}.png")
        nir_path = os.path.join(category_dir, f"nir_{category}_{timestamp}.png")
        
        cv.imwrite(rgb_path, rgb_img)
        cv.imwrite(nir_path, ir_img)
        
        with S.lock:
            if category == "healthy":
                S.healthy_count += 1
                count = S.healthy_count
            else:
                S.infected_count += 1
                count = S.infected_count
            
            total_count = S.healthy_count + S.infected_count
            
            log_metadata(
                timestamp=timestamp,
                category=category,
                rgb_path=rgb_path,
                nir_path=nir_path,
                ncc_score=quality,
                align_time_ms=align_time,
                sample_number=count
            )
            
            S.last_message = f"✅ Saved {category} leaf #{count} | NCC: {quality:.4f} | Total: {total_count}"
        
        return jsonify({
            "ok": True,
            "msg": S.last_message,
            "healthy_count": S.healthy_count,
            "infected_count": S.infected_count,
            "quality": float(quality),
            "files": [rgb_path, nir_path]
        })

    @app.route("/settings")
    def settings():
        """Update camera settings on-the-fly"""
        from flask import request
        cam = request.args.get("camera", "rgb")
        exp = request.args.get("exp_us")
        gain = request.args.get("gain")

        target = capture_worker.rgb_cam if cam == "rgb" else capture_worker.ir_cam
        
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
    """Initialize CSV metadata file with headers"""
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
    """Append capture metadata to CSV file"""
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

# ---------------- Capture Worker Thread ----------------

def capture_worker(args):
    """Continuously capture and align frames for preview"""
    
    rgb_mapx = np.load("rgb_mapx.npy")
    rgb_mapy = np.load("rgb_mapy.npy")
    ir_mapx = np.load("ir_mapx.npy")
    ir_mapy = np.load("ir_mapy.npy")

    aligner = OpenCVAligner(
        rectify_maps=(rgb_mapx, rgb_mapy, ir_mapx, ir_mapy),
        grid_step=args.grid_step,
        down_for_H=args.down_for_H
    )

    rgb_cam = setup_camera(
        args.rgb, (args.W, args.H), args.fps,
        args.exp_us_rgb, args.gain_rgb, awb=args.rgb_awb, ae=args.ae_rgb
    )
    ir_cam = setup_camera(
        args.ir, (args.W, args.H), args.fps,
        args.exp_us_ir, args.gain_ir, awb=False, ae=args.ae_ir
    )

    capture_worker.rgb_cam = rgb_cam
    capture_worker.ir_cam = ir_cam

    try:
        while True:
            rgb_bgr = get_frame_bgr(rgb_cam)
            ir_rgb = get_frame_bgr(ir_cam)
            ir_gray = cv.cvtColor(ir_rgb, cv.COLOR_BGR2GRAY)

            t0 = time.time()
            ir_aligned, align_info = aligner.align_ir_to_rgb(rgb_bgr, ir_gray)
            t1 = time.time()

            if ir_aligned is None:
                info = {"status": "alignment_failed", "ms_align": (t1-t0)*1000.0}
                overlay = np.zeros_like(rgb_bgr)
                cv.putText(overlay, "ALIGNMENT FAILED", (30, 50),
                          cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                ncc = alignment_quality(rgb_bgr, ir_aligned)
                info = {
                    "status": "ok",
                    "ncc": float(ncc),
                    "ms_align": (t1-t0)*1000.0,
                    **align_info
                }
                overlay = cv.addWeighted(
                    rgb_bgr, 0.7,
                    cv.cvtColor(ir_aligned, cv.COLOR_GRAY2BGR), 0.3, 0
                )

            rgb_vis = rgb_bgr.copy()
            ir_vis = cv.cvtColor(ir_gray, cv.COLOR_GRAY2BGR)
            
            if args.ir_colormap:
                ir_vis = cv.applyColorMap(cv.cvtColor(ir_vis, cv.COLOR_BGR2GRAY), cv.COLORMAP_TURBO)

            if args.vis_scale != 1.0:
                rgb_vis = cv.resize(rgb_vis, None, fx=args.vis_scale, fy=args.vis_scale)
                ir_vis = cv.resize(ir_vis, None, fx=args.vis_scale, fy=args.vis_scale)
                overlay = cv.resize(overlay, None, fx=args.vis_scale, fy=args.vis_scale)

            cv.putText(rgb_vis, "RGB", (12, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv.putText(ir_vis, "NIR", (12, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if info["status"] == "ok":
                cv.putText(overlay, f"NCC: {info['ncc']:.4f}", (12, 30),
                          cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Convert BGR to RGB for web display
            rgb_vis_web = cv.cvtColor(rgb_vis, cv.COLOR_BGR2RGB)
            overlay_web = cv.cvtColor(overlay, cv.COLOR_BGR2RGB)

            _, j_rgb = cv.imencode(".jpg", rgb_vis_web, [cv.IMWRITE_JPEG_QUALITY, 85])
            _, j_ir = cv.imencode(".jpg", ir_vis, [cv.IMWRITE_JPEG_QUALITY, 85])
            _, j_overlay = cv.imencode(".jpg", overlay_web, [cv.IMWRITE_JPEG_QUALITY, 85])

            with S.lock:
                S.rgb_jpg = j_rgb.tobytes()
                S.ir_jpg = j_ir.tobytes()
                S.overlay_jpg = j_overlay.tobytes()
                S.info = info

            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        rgb_cam.stop()
        ir_cam.stop()

# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser(
        description="Dataset collection system for healthy vs infected tomato leaves"
    )
    
    parser.add_argument("--rgb", type=int, default=0, help="RGB camera index")
    parser.add_argument("--ir", type=int, default=1, help="NIR camera index")
    parser.add_argument("--size", default="1920x1080", help="Image resolution")
    parser.add_argument("--fps", type=int, default=5, help="Frame rate")
    parser.add_argument("--exp_us_rgb", type=int, default=6000, help="RGB exposure (µs)")
    parser.add_argument("--exp_us_ir", type=int, default=6000, help="NIR exposure (µs)")
    parser.add_argument("--gain_rgb", type=float, default=2.0, help="RGB gain")
    parser.add_argument("--gain_ir", type=float, default=4.0, help="NIR gain")
    parser.add_argument("--rgb_awb", action="store_true", default=True, help="Enable AWB on RGB")
    parser.add_argument("--ae_rgb", action="store_true", help="Enable auto exposure on RGB camera")
    parser.add_argument("--ae_ir", action="store_true", help="Enable auto exposure on IR camera")
    parser.add_argument("--grid_step", type=int, default=24)
    parser.add_argument("--down_for_H", type=float, default=0.5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--lan", action="store_true", help="Expose on LAN")
    parser.add_argument("--vis_scale", type=float, default=0.75)
    parser.add_argument("--ir_colormap", action="store_true")
    parser.add_argument("--dataset_dir", default="datasets", 
                       help="Root directory for dataset (creates healthy/ and infected/ subdirs)")

    args = parser.parse_args()
    args.W, args.H = map(int, args.size.split("x"))

    if args.lan:
        args.host = "0.0.0.0"

    # Initialize metadata file
    os.makedirs(args.dataset_dir, exist_ok=True)
    S.metadata_file = init_metadata_file(args.dataset_dir)

    # Start worker
    worker = threading.Thread(target=capture_worker, args=(args,), daemon=True)
    worker.start()

    # Start web server
    app = create_app(args)
    
    print(f"\n{'='*70}")
    print(f"🍃 Tomato Leaf Dataset Collection System")
    print(f"{'='*70}")
    print(f"📁 Dataset structure:")
    print(f"   {args.dataset_dir}/")
    print(f"   ├── healthy/      (disease-free leaves)")
    print(f"   ├── infected/     (leaves with bacterial spot symptoms)")
    print(f"   └── dataset_metadata.csv  (quality tracking & statistics)")
    print(f"\n⚙️  Camera Settings:")
    print(f"   RGB: {'Auto Exposure' if args.ae_rgb else f'{args.exp_us_rgb}µs exposure'}, Gain {args.gain_rgb}")
    print(f"   NIR: {'Auto Exposure' if args.ae_ir else f'{args.exp_us_ir}µs exposure'}, Gain {args.gain_ir}")
    print(f"\n🌐 Web UI: http://{args.host}:{args.port}/")
    
    if args.host == "0.0.0.0":
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            print(f"🔗 LAN: http://{local_ip}:{args.port}/")
        except:
            pass
    
    print(f"{'='*70}\n")
    
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)

if __name__ == "__main__":
    main()
