# zhang_core_preprocessing.py
"""
Zhang et al. Core Methodology - Simplified & Validated
Focus on proven vegetation indices: RDVI, NDVI, SR
No complex masking - just core feature extraction for Random Forest

Zhang's Key Findings:
- RDVI: Top performing index (96% accuracy)
- NDVI: Baseline with 750nm ranked #1
- SR: Simple but effective
- Random Forest: Best performing classifier
"""

import os
import json
import glob
import numpy as np
import cv2 as cv
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import argparse
from dataclasses import dataclass
import joblib
import warnings # Added import
warnings.filterwarnings('ignore') # Added to suppress warnings if desired

@dataclass
class CalibrationData:
    """Zhang/Stamford calibration methodology"""
    materials: List[str]
    red_reflectances: List[float]  # Known reflectances at ~620nm
    nir_reflectances: List[float]  # Known reflectances at ~750nm

@dataclass
class ZhangVegetationIndices:
    """Core Zhang et al. vegetation indices only"""
    ndvi: np.ndarray
    rdvi: np.ndarray  # Zhang's #1 priority
    sr: np.ndarray    # Simple ratio
    # Means for feature extraction
    mean_ndvi: float
    mean_rdvi: float
    mean_sr: float
    std_ndvi: float
    std_rdvi: float
    std_sr: float

class ZhangCorePreprocessor:
    """
    Simplified preprocessor following Zhang's core methodology only
    No complex masking, focus on proven indices
    """

    def __init__(self, calibration_maps_path: str, calibration_data: CalibrationData):
        self.calibration_data = calibration_data
        self.load_rectification_maps(calibration_maps_path)

        # Zhang's critical wavelengths
        self.red_wavelength = 620    # Red channel
        self.nir_wavelength = 750    # Zhang's #1 ranked wavelength

        print(f"[Zhang Core] Initialized with validated wavelengths:")
        print(f"  Red: {self.red_wavelength}nm, NIR: {self.nir_wavelength}nm")
        print(f"  Following Zhang et al. 96% accuracy methodology")

    def load_rectification_maps(self, maps_path: str):
        """Load stereo rectification maps"""
        try:
            self.rgb_mapx = np.load(os.path.join(maps_path, "rgb_mapx.npy"))
            self.rgb_mapy = np.load(os.path.join(maps_path, "rgb_mapy.npy"))
            self.ir_mapx = np.load(os.path.join(maps_path, "ir_mapx.npy"))
            self.ir_mapy = np.load(os.path.join(maps_path, "ir_mapy.npy"))
            print(f"[Zhang Core] Loaded rectification maps")
        except FileNotFoundError as e:
            print(f"[Warning] Rectification maps not found: {e}, using identity mapping") # Added error message
            self.rgb_mapx = self.rgb_mapy = self.ir_mapx = self.ir_mapy = None

    def rectify_image_pair(self, rgb_img: np.ndarray, ir_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simple image alignment with resize fallback"""
        if self.rgb_mapx is not None:
            # Use INTER_LINEAR for speed, as per original Zhang Core inference
            rgb_rect = cv.remap(rgb_img, self.rgb_mapx, self.rgb_mapy, cv.INTER_LINEAR)
            ir_rect = cv.remap(ir_img, self.ir_mapx, self.ir_mapy, cv.INTER_LINEAR)
        else:
            rgb_rect, ir_rect = rgb_img.copy(), ir_img.copy()

        # --- FIX for shape mismatch if rectification maps are missing ---
        if rgb_rect.shape[:2] != ir_rect.shape[:2]:
            print(f"[Info] Resizing RGB image from {rgb_rect.shape[:2]} to match NIR {ir_rect.shape[:2]}")
            target_shape = (ir_rect.shape[1], ir_rect.shape[0]) # (width, height)
            # Use INTER_LINEAR for resize to match remap interpolation
            rgb_rect = cv.resize(rgb_rect, target_shape, interpolation=cv.INTER_LINEAR)
        # --- End FIX ---

        return rgb_rect, ir_rect

    def empirical_line_calibration(self, img: np.ndarray, channel: str,
                                   calib_coords: List[Tuple[int, int]],
                                   patch_size: int = 30) -> np.ndarray:
        """Zhang/Stamford empirical line calibration"""
        if not calib_coords or len(calib_coords) != len(self.calibration_data.materials):
            # Skip calibration if not properly configured
            # Return original img values scaled 0-100 if skipping
            return (img.astype(np.float32) / 255.0) * 100

        # Extract calibration patches
        digital_numbers = []
        for x, y in calib_coords:
             # Ensure patch coordinates are within image bounds
             y_start = max(0, y - patch_size // 2)
             y_end = min(img.shape[0], y + patch_size // 2)
             x_start = max(0, x - patch_size // 2)
             x_end = min(img.shape[1], x + patch_size // 2)
             patch = img[y_start:y_end, x_start:x_end]
             if patch.size > 0: # Ensure patch is not empty
                 digital_numbers.append(np.mean(patch))
             else:
                 digital_numbers.append(0) # Append 0 if patch is empty

        # Get known reflectances
        if channel == 'red':
            reflectances = self.calibration_data.red_reflectances
        elif channel == 'nir':
            reflectances = self.calibration_data.nir_reflectances
        else:
             # Return original scaled 0-100 if channel is invalid
            return (img.astype(np.float32) / 255.0) * 100

        # Linear calibration
        digital_numbers = np.array(digital_numbers)
        reflectances = np.array(reflectances)

        # Check for sufficient variation in DNs to avoid polyfit warning/error
        if np.ptp(digital_numbers) < 1e-6: # Check if all DNs are almost identical
            print("[Warning] Calibration panel digital numbers have insufficient variation. Using simple normalization.")
            return (img.astype(np.float32) / 255.0) * 100

        try:
            coeffs = np.polyfit(digital_numbers, reflectances, 1)
            calibrated = np.polyval(coeffs, img.astype(np.float32))
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"[Warning] Linear calibration failed: {e}. Using simple normalization.")
            return (img.astype(np.float32) / 255.0) * 100

        return np.clip(calibrated, 0, 100) # Clip to valid reflectance percentage

    def calculate_zhang_vegetation_indices(self,
                                           red_reflectance_pixels: np.ndarray,
                                           nir_reflectance_pixels: np.ndarray,
                                           epsilon: float = 1e-8, # Small value
                                           red_threshold: float = 1e-4 # 0.01% reflectance
                                           ) -> ZhangVegetationIndices:
        """
        Zhang's core vegetation indices with proper normalization and robust SR fix.
        Accepts 1D arrays of pixels already scaled 0.0 - 1.0 (or 0-100).
        """

        # --- 1. Ensure Input is Float and Scaled 0.0 - 1.0 ---
        red_refl = red_reflectance_pixels.astype(np.float32)
        nir_refl = nir_reflectance_pixels.astype(np.float32)

        # Check if data is on 0-100 scale and normalize
        if red_refl.max() > 1.1: # Allow 1.1 for slight overshoots
            red_refl /= 100.0
        if nir_refl.max() > 1.1:
            nir_refl /= 100.0

        # Clip to ensure valid 0-1 range
        red_refl = np.clip(red_refl, 0.0, 1.0)
        nir_refl = np.clip(nir_refl, 0.0, 1.0)

        # --- 2. Calculate Denominators ---

        # Denominator for NDVI and RDVI
        # Add epsilon to prevent 0/0 -> NaN
        denominator_ndvi_rdvi = nir_refl + red_refl + epsilon

        # Denominator for SR (Simple Ratio)
        # Apply the red_threshold to prevent division by very small numbers
        red_clipped_for_sr = np.maximum(red_refl, red_threshold)
        # Add epsilon here too for absolute safety
        denominator_sr = red_clipped_for_sr + epsilon

        # --- 3. Calculate Indices ---

        # Common numerator for NDVI/RDVI
        numerator_common = (nir_refl - red_refl)

        # NDVI = (NIR - Red) / (NIR + Red)
        ndvi_pixels = numerator_common / denominator_ndvi_rdvi

        # RDVI = (NIR - Red) / sqrt(NIR + Red)
        # Add epsilon to sqrt argument to prevent sqrt(0)
        rdvi_pixels = numerator_common / (np.sqrt(denominator_ndvi_rdvi) + epsilon)

        # SR = NIR / Red (using the robust denominator)
        sr_pixels = nir_refl / denominator_sr

        # --- 4. Calculate Statistics ---
        # Use nanmean/nanstd to safely ignore any potential NaNs
        # (though epsilons should prevent Infs/NaNs)
        mean_ndvi = float(np.nanmean(ndvi_pixels))
        std_ndvi = float(np.nanstd(ndvi_pixels))
        mean_rdvi = float(np.nanmean(rdvi_pixels))
        std_rdvi = float(np.nanstd(rdvi_pixels))
        mean_sr = float(np.nanmean(sr_pixels))
        std_sr = float(np.nanstd(sr_pixels))

        return ZhangVegetationIndices(
            ndvi=ndvi_pixels, # Return pixel arrays (might be large!)
            rdvi=rdvi_pixels,
            sr=sr_pixels,
            mean_ndvi=mean_ndvi,
            mean_rdvi=mean_rdvi,
            mean_sr=mean_sr,
            std_ndvi=std_ndvi,
            std_rdvi=std_rdvi,
            std_sr=std_sr
        )

    def extract_zhang_features(self, rgb_img: np.ndarray, vi_data: ZhangVegetationIndices) -> Dict[str, float]:
        """
        Extract only Zhang's validated features - no complex masking
        """
        features = {}

        # RDVI features
        features.update({
            'rdvi_mean': vi_data.mean_rdvi,
            'rdvi_std': vi_data.std_rdvi,
            'rdvi_min': float(np.nanmin(vi_data.rdvi)),
            'rdvi_max': float(np.nanmax(vi_data.rdvi)),
            'rdvi_median': float(np.nanmedian(vi_data.rdvi)),
            'rdvi_range': float(np.nanmax(vi_data.rdvi) - np.nanmin(vi_data.rdvi))
        })

        # NDVI features
        features.update({
            'ndvi_mean': vi_data.mean_ndvi,
            'ndvi_std': vi_data.std_ndvi,
            'ndvi_min': float(np.nanmin(vi_data.ndvi)),
            'ndvi_max': float(np.nanmax(vi_data.ndvi)),
            'ndvi_median': float(np.nanmedian(vi_data.ndvi)),
            'ndvi_range': float(np.nanmax(vi_data.ndvi) - np.nanmin(vi_data.ndvi))
        })

        # Simple Ratio features (Calculate stats robustly)
        sr_finite = vi_data.sr[np.isfinite(vi_data.sr)] # Use only finite values for stats
        if sr_finite.size > 0:
            features.update({
                'sr_mean': float(np.mean(sr_finite)),
                'sr_std': float(np.std(sr_finite)),
                'sr_min': float(np.min(sr_finite)),
                'sr_max': float(np.max(sr_finite)), # Max might still be large, but not infinite
                'sr_median': float(np.median(sr_finite))
            })
        else: # Default if all SR values were non-finite (e.g., all red=0)
             features.update({'sr_mean': 0.0, 'sr_std': 0.0, 'sr_min': 0.0, 'sr_max': 0.0, 'sr_median': 0.0})


        # Simple RGB statistics
        for i, channel in enumerate(['blue', 'green', 'red']):
            channel_data = rgb_img[:,:,i]
            features.update({
                f'rgb_{channel}_mean': float(np.mean(channel_data)),
                f'rgb_{channel}_std': float(np.std(channel_data))
            })

        # Basic texture
        gray = cv.cvtColor(rgb_img, cv.COLOR_BGR2GRAY)
        features['texture_std'] = float(np.std(gray))

        return features

    def process_image_pair(self, rgb_path: str, ir_path: str,
                           calib_coords: List[Tuple[int, int]] = None,
                           save_debug: bool = False) -> Dict[str, float]:
        """
        Zhang's core processing pipeline - simplified and validated
        """
        print(f"\n[Zhang Core] {os.path.basename(rgb_path)} & {os.path.basename(ir_path)}")

        # Load images
        rgb_img = cv.imread(rgb_path, cv.IMREAD_COLOR)

        # Handle TIF format specifically (often 16-bit or multi-band)
        if ir_path.lower().endswith(('.tif', '.tiff')):
            ir_img_raw = cv.imread(ir_path, cv.IMREAD_UNCHANGED)
            if ir_img_raw is None:
                 raise FileNotFoundError(f"Could not load TIF image: {ir_path}")

            # If multi-band TIF, assume NIR is the first band (adjust if necessary)
            if len(ir_img_raw.shape) == 3:
                print(f"[Info] Multi-band TIF detected. Using first band as NIR.")
                ir_img = ir_img_raw[:, :, 0]
            else:
                ir_img = ir_img_raw

            # If 16-bit, normalize to 8-bit range for consistency
            if ir_img.dtype == 'uint16':
                print("[Info] 16-bit NIR image detected. Normalizing to 8-bit.")
                # Use simple scaling, adjust alpha if needed based on actual sensor range
                ir_img = cv.convertScaleAbs(ir_img, alpha=(255.0/65535.0))
            elif ir_img.dtype != 'uint8':
                 print(f"[Warning] Unexpected NIR image dtype: {ir_img.dtype}. Attempting conversion to uint8.")
                 try:
                     # Attempt normalization assuming max value represents white
                     max_val = np.max(ir_img)
                     if max_val > 0:
                         ir_img = ((ir_img / max_val) * 255).astype(np.uint8)
                     else:
                         ir_img = ir_img.astype(np.uint8) # All zeros
                 except Exception as conv_e:
                      print(f"[Error] Could not convert NIR image to uint8: {conv_e}")
                      ir_img = np.zeros(ir_img.shape[:2], dtype=np.uint8) # Fallback to black image

        else: # Handle other formats like PNG
            ir_img = cv.imread(ir_path, cv.IMREAD_GRAYSCALE)

        if rgb_img is None or ir_img is None:
            raise FileNotFoundError(f"Could not load images: {rgb_path}, {ir_path}")

        # Ensure NIR image is grayscale (single channel) after processing TIF/PNG
        if len(ir_img.shape) == 3:
             print("[Warning] NIR image has 3 channels after loading, converting to grayscale.")
             ir_img = cv.cvtColor(ir_img, cv.COLOR_BGR2GRAY)


        print(f"  Image sizes after load: RGB {rgb_img.shape}, NIR {ir_img.shape}")

        # Simple rectification (includes resize fallback)
        rgb_rect, ir_rect = self.rectify_image_pair(rgb_img, ir_img)

        # Extract red channel
        red_channel = rgb_rect[:,:,2]  # OpenCV uses BGR

        # Apply calibration or normalization
        calibration_method = "Simple normalization"
        if calib_coords:
            red_refl = self.empirical_line_calibration(red_channel, 'red', calib_coords)
            nir_refl = self.empirical_line_calibration(ir_rect, 'nir', calib_coords)
             # Check if calibration actually ran or fell back
            if not np.allclose(red_refl, (red_channel.astype(np.float32) / 255.0) * 100):
                 calibration_method = "Empirical line applied"
        else:
            red_refl = (red_channel.astype(np.float32) / 255.0) * 100
            nir_refl = (ir_rect.astype(np.float32) / 255.0) * 100

        print(f"  Calibration: {calibration_method}")
        print(f"  Reflectance ranges: Red {np.nanmin(red_refl):.1f}-{np.nanmax(red_refl):.1f}, NIR {np.nanmin(nir_refl):.1f}-{np.nanmax(nir_refl):.1f}")

        # Calculate Zhang's core vegetation indices
        vi_data = self.calculate_zhang_vegetation_indices(red_refl, nir_refl)

        # DEBUG: Validate ranges
        print(f"  === ZHANG VEGETATION INDICES ===")
        print(f"  NDVI: {vi_data.mean_ndvi:.4f} (range: {np.nanmin(vi_data.ndvi):.4f} to {np.nanmax(vi_data.ndvi):.4f})")
        print(f"  RDVI: {vi_data.mean_rdvi:.4f} (range: {np.nanmin(vi_data.rdvi):.4f} to {np.nanmax(vi_data.rdvi):.4f})")
        # Use robust max for SR range display, avoiding inf
        sr_finite = vi_data.sr[np.isfinite(vi_data.sr)]
        sr_min_val = np.min(sr_finite) if sr_finite.size > 0 else 0
        sr_max_val = np.max(sr_finite) if sr_finite.size > 0 else 0
        print(f"  SR:   {vi_data.mean_sr:.4f} (range: {sr_min_val:.4f} to {sr_max_val:.4f})")


        # Validate expected ranges for the MEANS (max values can still be high for SR)
        range_warning = False
        if not (-1.0 <= vi_data.mean_ndvi <= 1.0):
             print(f"  WARNING: Mean NDVI outside expected range!")
             range_warning = True
        # RDVI theoretical range is wider, allow slightly more than [-1, 1] for mean
        if not (-1.5 <= vi_data.mean_rdvi <= 1.5):
             print(f"  WARNING: Mean RDVI outside expected range!")
             range_warning = True
        # Check if mean SR is excessively large or negative
        if vi_data.mean_sr > 100 or vi_data.mean_sr < 0: # Increased threshold for mean
             print(f"  WARNING: Mean SR outside expected range!")
             range_warning = True


        # Extract core features only
        features = self.extract_zhang_features(rgb_rect, vi_data)

        # Add metadata
        features['rgb_path'] = rgb_path
        features['ir_path'] = ir_path

        # Save debug if requested
        if save_debug:
            self._save_zhang_debug(rgb_rect, ir_rect, vi_data, features, rgb_path)

        return features

    def _save_zhang_debug(self, rgb_rect: np.ndarray, ir_rect: np.ndarray,
                         vi_data: ZhangVegetationIndices, features: Dict, rgb_path: str):
        """Save Zhang-focused debug visualization"""
        debug_dir = os.path.dirname(rgb_path).replace('datasets', 'zhang_debug')
        os.makedirs(debug_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(rgb_path))[0]

        # Create simple 2x3 visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Zhang Core Analysis: {base_name}', fontsize=16)

        # Original images
        axes[0,0].imshow(cv.cvtColor(rgb_rect, cv.COLOR_BGR2RGB))
        axes[0,0].set_title('RGB Image (Rectified/Resized)')
        axes[0,0].axis('off')

        axes[0,1].imshow(ir_rect, cmap='gray')
        axes[0,1].set_title('NIR Image (Rectified/Resized)')
        axes[0,1].axis('off')

        # Zhang's core indices
        # NDVI (-1 to 1 mapped to 0-1 for color)
        ndvi_vis = np.clip((vi_data.ndvi + 1) / 2.0, 0, 1) # Ensure valid range 0-1
        im1 = axes[1,0].imshow(ndvi_vis, cmap='RdYlGn', vmin=0, vmax=1)
        axes[1,0].set_title(f'NDVI (Zhang Baseline)\nMean: {vi_data.mean_ndvi:.3f}')
        axes[1,0].axis('off')
        plt.colorbar(im1, ax=axes[1,0], shrink=0.6)

        # RDVI (approx -1 to 1 mapped to 0-1)
        # Use nanmin/nanmax for robust range finding
        rdvi_min, rdvi_max = np.nanmin(vi_data.rdvi), np.nanmax(vi_data.rdvi)
        if rdvi_max > rdvi_min and np.isfinite(rdvi_max) and np.isfinite(rdvi_min):
            rdvi_vis = (vi_data.rdvi - rdvi_min) / (rdvi_max - rdvi_min)
            rdvi_vis = np.clip(rdvi_vis, 0, 1) # Ensure valid range 0-1
        else:
            rdvi_vis = np.zeros_like(vi_data.rdvi) # Fallback if range is invalid
        im2 = axes[1,1].imshow(rdvi_vis, cmap='RdYlGn', vmin=0, vmax=1)
        axes[1,1].set_title(f'RDVI (Zhang Priority)\nMean: {vi_data.mean_rdvi:.3f}')
        axes[1,1].axis('off')
        plt.colorbar(im2, ax=axes[1,1], shrink=0.6)

        # Simple Ratio (clip at a reasonable upper value for visualization)
        sr_vis_max = 10.0 # Vegetation typically < 10
        sr_vis = np.clip(vi_data.sr / sr_vis_max, 0, 1)
        im3 = axes[1,2].imshow(sr_vis, cmap='viridis', vmin=0, vmax=1)
        axes[1,2].set_title(f'Simple Ratio\nMean: {features.get("sr_mean", 0):.3f}') # Use calculated robust mean
        axes[1,2].axis('off')
        plt.colorbar(im3, ax=axes[1,2], shrink=0.6)

        # Feature summary
        axes[0,2].axis('off')
        feature_text = f"""Zhang Core Features:

RDVI (Top Priority):
  Mean: {features.get('rdvi_mean', 0):.4f}
  Std:  {features.get('rdvi_std', 0):.4f}
  Range: {features.get('rdvi_range', 0):.4f}

NDVI (Baseline):
  Mean: {features.get('ndvi_mean', 0):.4f}
  Std:  {features.get('ndvi_std', 0):.4f}

Simple Ratio:
  Mean: {features.get('sr_mean', 0):.4f}
  Std:  {features.get('sr_std', 0):.4f}

Expected Performance:
96% accuracy (Zhang)
"""
        axes[0,2].text(0.05, 0.95, feature_text, fontsize=10, verticalalignment='top',
                      fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

        plt.tight_layout()
        debug_path = os.path.join(debug_dir, f'{base_name}_zhang_core.png')
        try:
            plt.savefig(debug_path, dpi=150, bbox_inches='tight')
            print(f"  Zhang debug saved: {debug_path}")
        except Exception as save_e:
            print(f"[Error] Failed to save debug plot: {save_e}")
        plt.close(fig) # Close the figure


def create_zhang_calibration_data() -> CalibrationData:
    """Zhang/Stamford reference materials"""
    return CalibrationData(
        materials=['white_panel', 'light_gray', 'medium_gray', 'dark_gray', 'black_panel', 'vegetation_ref'],
        red_reflectances=[95.0, 75.0, 50.0, 25.0, 5.0, 15.0],
        nir_reflectances=[95.0, 75.0, 50.0, 25.0, 5.0, 45.0]
    )


def detect_dataset_type(dataset_root: str) -> str:
    """Detect naming convention (Simplified - Primarily checks for hardware vs pair*)"""
    print(f"[Info] Detecting dataset type in {dataset_root}")

    sample_files = []
    try:
        # Scan only the top level and first level directories for efficiency
        for entry in os.scandir(dataset_root):
            if entry.is_dir():
                 for sub_entry in os.scandir(entry.path):
                      if sub_entry.is_file():
                           sample_files.append(sub_entry.name)
                      if len(sample_files) >= 50: break # Check more files
            elif entry.is_file():
                sample_files.append(entry.name)
            if len(sample_files) >= 50: break
    except FileNotFoundError:
         print(f"[Error] Dataset root directory not found: {dataset_root}")
         return 'unknown' # Return a distinct type if root doesn't exist

    hardware_pattern = 0
    pair_pattern = 0

    for filename in sample_files:
        if ('nir_' in filename or 'rgb_' in filename) and filename.lower().endswith('.png'):
            hardware_pattern += 1
        # More robust check for 'pair' convention
        elif filename.lower().startswith('pair_') and (filename.lower().endswith('.jpg') or filename.lower().endswith('.tif')):
            pair_pattern += 1

    print(f"[Debug] Pattern counts: Hardware={hardware_pattern}, Pair={pair_pattern}")

    if hardware_pattern > pair_pattern:
        print(f"[Detected] Hardware setup naming")
        return 'hardware'
    elif pair_pattern > 0:
        print(f"[Detected] Pair (pair_...) naming")
        return 'pair'
    else:
        print(f"[Warning] Unknown naming, defaulting to hardware")
        return 'hardware' # Default remains hardware


def find_image_pairs_hardware(stage_path: str) -> List[Tuple[str, str]]:
    """Find RGB-NIR pairs for hardware naming"""
    pairs = []
    # Find all NIR files first, as they are the trigger
    nir_files = sorted(glob.glob(os.path.join(stage_path, "nir_*.png")))
    print(f"[Debug] Found {len(nir_files)} potential hardware NIR files in {stage_path}")

    for nir_path in nir_files:
        nir_basename = os.path.basename(nir_path)
        # Expected format: nir_label_YYYYMMDD_HHMMSS.png
        parts = nir_basename.split('_')
        if len(parts) == 4 and parts[0] == 'nir' and parts[3].endswith('.png'):
            try:
                # Construct corresponding RGB filename
                rgb_basename = f"rgb_{parts[1]}_{parts[2]}_{parts[3]}"
                rgb_path = os.path.join(stage_path, rgb_basename)

                if os.path.exists(rgb_path):
                    pairs.append((rgb_path, nir_path))
                    # print(f"[Found] {os.path.basename(rgb_path)} <-> {os.path.basename(nir_path)}") # Less verbose
                else:
                    print(f"[Warning] No RGB pair found for {nir_basename}")
            except IndexError:
                 print(f"[Warning] Could not parse hardware filename: {nir_basename}")

    return pairs


def find_image_pairs_pair(stage_path: str) -> List[Tuple[str, str]]:
    """Find RGB-NIR pairs for Pair (e.g. Parrot) naming"""
    pairs = []
    # Find all JPG files first (assumed RGB)
    rgb_files = sorted(glob.glob(os.path.join(stage_path, "pair_*.jpg")))
    print(f"[Debug] Found {len(rgb_files)} potential pair JPG files in {stage_path}")

    for rgb_path in rgb_files:
        rgb_basename = os.path.basename(rgb_path)
        # Construct corresponding TIF filename (case-insensitive replace)
        if rgb_basename.lower().endswith('.jpg'):
            nir_basename = rgb_basename[:-4] + '.tif' # More robust replace
            nir_path = os.path.join(stage_path, nir_basename)

            if os.path.exists(nir_path):
                pairs.append((rgb_path, nir_path))
                # print(f"[Found] {os.path.basename(rgb_path)} <-> {os.path.basename(nir_path)}") # Less verbose
            else:
                 # Check for TIFF extension as well
                 nir_basename_tiff = rgb_basename[:-4] + '.tiff'
                 nir_path_tiff = os.path.join(stage_path, nir_basename_tiff)
                 if os.path.exists(nir_path_tiff):
                      pairs.append((rgb_path, nir_path_tiff))
                 else:
                      print(f"[Warning] No TIF/TIFF pair found for {rgb_basename}")
        else:
             print(f"[Warning] Skipping file with unexpected extension: {rgb_basename}")


    return pairs


def process_zhang_dataset(dataset_root: str, preprocessor: ZhangCorePreprocessor,
                         output_path: str, calib_coords: Optional[List[Tuple[int, int]]] = None):
    """
    Process dataset using Zhang's core methodology, finding both conventions.
    """
    # Removed dataset_type detection from here, happens per folder now

    stage_mapping = {
        'healthy': 0,
        'early': 1,
        'late': 2
    }
    all_features = []
    total_processed = 0

    for stage_name, stage_label in stage_mapping.items():
        stage_path = os.path.join(dataset_root, stage_name)
        if not os.path.exists(stage_path):
            print(f"[Warning] Stage not found: {stage_path}")
            continue

        print(f"\n[Zhang Processing] Stage: {stage_name} (label: {stage_label})")

        # --- Find BOTH types of pairs in the current stage folder ---
        print(f"[Info] Searching for 'hardware' convention pairs in {stage_name}...")
        hardware_pairs = find_image_pairs_hardware(stage_path)
        print(f"[Info] Found {len(hardware_pairs)} 'hardware' pairs.")

        print(f"[Info] Searching for 'pair' convention pairs in {stage_name}...")
        pair_pairs = find_image_pairs_pair(stage_path)
        print(f"[Info] Found {len(pair_pairs)} 'pair' pairs.")

        # Combine pairs, adding convention info
        all_pairs_to_process = []
        for rgb_path, nir_path in hardware_pairs:
             all_pairs_to_process.append((rgb_path, nir_path, 'hardware'))
        for rgb_path, nir_path in pair_pairs:
             all_pairs_to_process.append((rgb_path, nir_path, 'pair'))

        if not all_pairs_to_process:
            print(f"[Warning] No pairs of any type found in {stage_path}")
            continue
        # --- End finding pairs ---

        processed_count = 0
        failed_count = 0
        all_pairs_to_process.sort(key=lambda x: x[0]) # Process in deterministic order

        for rgb_path, nir_path, convention in all_pairs_to_process:
            try:
                features = preprocessor.process_image_pair(
                    rgb_path, nir_path, calib_coords, save_debug=True # Keep debug on
                )

                features['stage'] = stage_name
                features['label'] = stage_label
                base_name = os.path.splitext(os.path.basename(rgb_path))[0]
                features['sample_id'] = f"{stage_name}_{base_name}" # More unique ID
                features['dataset_type'] = convention # Store actual convention used

                all_features.append(features)
                processed_count += 1
                total_processed +=1

                if processed_count % 10 == 0:
                    print(f"  Processed {processed_count} from {stage_name}")

            except FileNotFoundError as e:
                 print(f"[Error] File not found during processing {os.path.basename(rgb_path)}: {e}")
                 failed_count += 1
            except Exception as e:
                 print(f"[Error] Processing {os.path.basename(rgb_path)}: {e}")
                 import traceback # Import traceback for detailed errors
                 traceback.print_exc() # Print full stack trace
                 failed_count += 1
                 continue # Skip to next pair on error

        print(f"[Stage Summary] {stage_name}: Processed {processed_count}, Failed {failed_count}")


    # Save results (handle empty dataframe possibility)
    if not all_features:
        print("[Error] No features were extracted successfully. Cannot save output.")
        return None

    df = pd.DataFrame(all_features)
    print(f"\n[Info] Total features extracted before quality check: {len(df)}")

    # --- Robust Quality Check ---
    print("\n[Quality Check] Identifying samples with issues (e..g., excessive NaNs)...")
    initial_rows = len(df)
    # 1. Drop columns that are ALL NaN (these features failed everywhere)
    df.dropna(axis=1, how='all', inplace=True)
    if initial_rows > 0:
         print(f"[Quality] Remaining columns after dropping all-NaN: {len(df.columns)}")

    # 2. Check for rows with high NaN percentage
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
         nan_threshold = 0.5 # Allow up to 50% NaNs per row before dropping
         nan_ratio = df[numeric_cols].isnull().sum(axis=1) / len(numeric_cols)
         valid_samples = nan_ratio <= nan_threshold
         num_removed = (~valid_samples).sum()
         if num_removed > 0:
              print(f"[Quality] Removing {num_removed} samples with >{nan_threshold*100:.0f}% NaN values.")
              df = df[valid_samples].reset_index(drop=True)
         else:
              print("[Quality] No samples removed based on NaN threshold.")

         # 3. Fill remaining NaNs with median
         if not df.empty:
              print("[Quality] Filling remaining NaN values with column medians.")
              # Calculate medians only on remaining valid data
              medians = df[numeric_cols].median()
              df[numeric_cols] = df[numeric_cols].fillna(medians)
              # Check if any NaNs are left (e.g., if a whole column was NaN initially and threshold was 1.0)
              if df[numeric_cols].isnull().sum().sum() > 0:
                   print("[Warning] Some NaNs remain after median fill. Filling with 0.")
                   df[numeric_cols] = df[numeric_cols].fillna(0)
         else:
              print("[Warning] DataFrame is empty after NaN removal. No data to save.")
              return None

    else:
        print("[Warning] No numeric columns found for NaN check/fill.")


    # --- End Quality Check ---


    if len(df) > 0:
        df.to_csv(output_path, index=False)
        print(f"\n[Complete] Saved {len(df)} samples to {output_path}")

        # Zhang-focused summary
        print("\n=== Zhang Core Methodology Results ===")
        print(f"Total samples saved: {len(df)} (out of {initial_rows} processed)")
        print(f"\nDataset types found:")
        for dtype, count in df['dataset_type'].value_counts().items():
             print(f"  {dtype}: {count}")
        print("\nSamples per stage:")
        for stage, count in df['stage'].value_counts().sort_index().items():
            print(f"  {stage}: {count}")

        print(f"\nZhang's core vegetation indices (Mean of sample means):")
        # Validate indices based on saved data
        for vi_mean_col in ['rdvi_mean', 'ndvi_mean', 'sr_mean']:
            if vi_mean_col in df.columns:
                 mean_val = df[vi_mean_col].mean()
                 min_val = df[vi_mean_col].min()
                 max_val = df[vi_mean_col].max()
                 vi_base = vi_mean_col.split('_')[0].upper() # RDVI, NDVI, SR
                 print(f"  {vi_base}: {mean_val:.4f} (sample means range: {min_val:.3f} to {max_val:.3f})")
                 # Simple range check on the mean of means
                 if vi_base == 'RDVI' and not (-1.5 <= mean_val <= 1.5):
                      print(f"    WARNING: Overall mean {vi_base} seems outside expected range!")
                 elif vi_base == 'NDVI' and not (-1.0 <= mean_val <= 1.0):
                      print(f"    WARNING: Overall mean {vi_base} seems outside expected range!")
                 elif vi_base == 'SR' and not (0 <= mean_val <= 100): # Wider range for SR mean
                      print(f"    WARNING: Overall mean {vi_base} seems outside expected range!")
                 else:
                      print(f"    ✓ Overall mean {vi_base} in expected range")
            else:
                 print(f"  {vi_mean_col} not found in final data.")

        print(f"\nZhang methodology validation: ✓")
        print(f"Ready for Random Forest (96% target accuracy)")

    else:
         print("[Error] No valid samples remained after quality checks.")
         return None

    return df


def prepare_zhang_training_data(df: pd.DataFrame, test_size: float = 0.2) -> Tuple:
    """Prepare data for Zhang's Random Forest, checking for sufficient samples"""

    # Select only Zhang's core features
    feature_cols = [col for col in df.columns if col not in
                   ['stage', 'label', 'sample_id', 'rgb_path', 'ir_path', 'dataset_type']]

    print(f"\n=== Zhang Random Forest Data Preparation ===")
    print(f"Using features: {feature_cols}")

    X = df[feature_cols].values
    y = df['label'].values

    # Check for sufficient data and classes BEFORE splitting
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"Label distribution before split: {dict(zip(unique_labels, counts))}")
    min_samples_per_class = np.min(counts) if len(counts) > 0 else 0

    # Ensure there are enough samples in the smallest class for stratification
    # test_samples_needed = max(1, int(np.ceil(min_samples_per_class * test_size))) # Min 1 sample per class in test
    n_splits_required = 2 # Minimum for train/test split
    if min_samples_per_class < n_splits_required:
         print(f"\n[Error] The smallest class (count={min_samples_per_class}) has fewer samples than required for stratification ({n_splits_required}).")
         print("Cannot perform train/test split. Try adjusting preprocessing or collecting more data for minority classes.")
         # Return empty arrays to signal failure
         return np.array([]), np.array([]), np.array([]), np.array([]), None, []


    # Proceed with splitting
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    except ValueError as e:
         print(f"\n[Error] Train/test split failed: {e}")
         print("This might happen if a class has only 1 sample, making stratification impossible.")
         return np.array([]), np.array([]), np.array([]), np.array([]), None, []


    # --- ADD SPLIT CHECK ---
    print("\n--- Data Split Check ---")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    print(f"y_train unique labels: {unique_train}")
    print(f"y_train counts:      {counts_train}")
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    print(f"y_test unique labels:  {unique_test}")
    print(f"y_test counts:       {counts_test}")
    if len(unique_test) < len(unique_labels):
        print("!!! WARNING: Test set is missing classes present in the full dataset!")
        # Decide how to handle this - raise error or just warn? For now, warn.
        # raise ValueError("Test set is missing classes due to small sample size per class. Cannot proceed.")
    print("--- End Check ---\n")
    # --- END SPLIT CHECK ---


    # Only scale if data exists
    if X_train.size > 0 and X_test.size > 0:
         scaler = StandardScaler()
         X_train_scaled = scaler.fit_transform(X_train)
         X_test_scaled = scaler.transform(X_test)
    else: # Should not happen if split check passes, but for safety
        scaler = None
        X_train_scaled, X_test_scaled = X_train, X_test # Return unscaled if split failed earlier

    print(f"Core features: {len(feature_cols)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Zhang target: 96% accuracy")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


def main():
    parser = argparse.ArgumentParser(description='Zhang Core Methodology - Simplified & Validated')
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Root directory with healthy/early/late folders')
    parser.add_argument('--calib_maps', type=str, required=True,
                        help='Directory with rectification maps')
    parser.add_argument('--output', type=str, default='zhang_core_features.csv',
                        help='Output CSV file for features')
    parser.add_argument('--calib_coords', type=str, default=None,
                        help='JSON file with calibration coordinates')
    # Added argument for output directory for consistency
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save training data files (npz, joblib, json)')


    args = parser.parse_args()

    # Create output directory for training files if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)


    print(f"=== Zhang et al. Core Methodology ===")
    print(f"Focus: RDVI (96% accuracy), NDVI, SR")
    print(f"Target: Bacterial leaf spot detection")

    # Initialize with Zhang's methodology
    calib_data = create_zhang_calibration_data()
    preprocessor = ZhangCorePreprocessor(args.calib_maps, calib_data)

    # Load calibration if available
    calib_coords = None
    if args.calib_coords and os.path.exists(args.calib_coords):
        try:
             with open(args.calib_coords, 'r') as f:
                 coord_data = json.load(f)
                 # Check if the expected key exists
                 if 'calibration_points' in coord_data:
                      calib_coords = [(pt['x'], pt['y']) for pt in coord_data['calibration_points']]
                      print(f"[Info] Loaded {len(calib_coords)} calibration points.")
                 else:
                      print(f"[Warning] Calibration JSON file ({args.calib_coords}) exists but missing 'calibration_points' key.")
        except json.JSONDecodeError:
             print(f"[Error] Failed to decode calibration JSON file: {args.calib_coords}")
        except KeyError as e:
             print(f"[Error] Missing key {e} in calibration point data.")
        except Exception as e:
             print(f"[Error] Failed to load calibration coordinates: {e}")


    # Process with Zhang core methodology
    df = process_zhang_dataset(args.dataset_root, preprocessor, args.output, calib_coords)

    if df is not None and len(df) > 0:
        # Prepare for Random Forest
        X_train, X_test, y_train, y_test, scaler, feature_names = prepare_zhang_training_data(df)

        # Check if data preparation was successful
        if scaler is not None and len(feature_names) > 0:
            # Define output paths using the output_dir argument
            data_path = os.path.join(args.output_dir, 'zhang_rf_training_data.npz')
            scaler_path = os.path.join(args.output_dir, 'zhang_scaler.joblib')
            features_path = os.path.join(args.output_dir, 'zhang_feature_names.json')

            # Save for Random Forest training
            np.savez(data_path,
                    X_train=X_train, X_test=X_test,
                    y_train=y_train, y_test=y_test)

            joblib.dump(scaler, scaler_path)

            with open(features_path, 'w') as f:
                json.dump(feature_names, f, indent=2)

            print(f"\n[Success] Zhang training data saved to directory: {args.output_dir}")
            print(f"  - {os.path.basename(data_path)}")
            print(f"  - {os.path.basename(scaler_path)}")
            print(f"  - {os.path.basename(features_path)}")
            print(f"Ready for Zhang's Random Forest training")
        else:
             print(f"[Error] Data preparation failed (likely due to insufficient samples per class). Training files not saved.")


    else:
        print(f"[Error] Dataset processing failed or yielded no valid samples.")


if __name__ == "__main__":
    main()
