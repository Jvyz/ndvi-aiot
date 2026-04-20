# ndvi_utils.py
import numpy as np
import cv2 as cv

def linearize_u8(u8):
    """Approx inverse gamma for 8-bit images. Prefer RAW if available."""
    f = u8.astype(np.float32) / 255.0
    return np.power(np.clip(f, 1e-6, 1.0), 2.2)

def match_global_brightness(nir_lin, red_lin, method="median", mask=None):
    """
    Returns scale s to multiply NIR by so that NIR and Red have matched global levels.
    Use a mask for a known reference (e.g., calibration panel) if available.
    """
    if mask is None:
        mask = np.ones_like(nir_lin, dtype=bool)
    a = nir_lin[mask]; b = red_lin[mask]
    if method == "median":
        denom = np.median(a) + 1e-6
        s = (np.median(b) + 1e-6) / denom
    else:  # percentile
        qa = np.percentile(a, 90); qb = np.percentile(b, 90)
        s = (qb + 1e-6) / (qa + 1e-6)
    return float(s)

def compute_ndvi_from_rgb_ir(rgb_bgr, ir_gray, normalize=False, mask=None, eps=1e-3):
    """
    rgb_bgr: BGR uint8 (OpenCV)
    ir_gray: GRAY uint8 aligned to RGB geometry
    """
    red_lin = linearize_u8(rgb_bgr[:,:,2])
    nir_lin = linearize_u8(ir_gray)
    if normalize:
        s = match_global_brightness(nir_lin, red_lin, method="median", mask=mask)
        nir_lin = nir_lin * s
    ndvi = (nir_lin - red_lin) / (nir_lin + red_lin + eps)
    return np.clip(ndvi, -1.0, 1.0)

def colorize_ndvi(ndvi):
    ndvi_u8 = ((ndvi + 1.0) * 127.5).astype(np.uint8)
    return cv.applyColorMap(ndvi_u8, cv.COLORMAP_TURBO)
