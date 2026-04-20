# segmentation_pipeline_kmeans.py
"""
Integrates alignment from v2_preprocessing.py with a UNIFIED segmentation pipeline.
- K-Means + Watershed is used for ALL stages (healthy, early, late).
This fixes the data leakage problem where segmentation method was
correlated with the label.
"""

import os
import cv2 as cv
import numpy as np
import torch # Still needed for device check
try:
    from sklearn.cluster import MiniBatchKMeans
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    import pandas as pd
    import joblib
except ImportError as e:
    print(f"[Error] Missing required libraries (scikit-learn, scikit-image, pandas): {e}")
    print("Please install them: pip install scikit-learn scikit-image pandas joblib")
    exit()

from typing import Tuple, List, Dict, Optional, Any
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import time
import json
import glob
import re

# --- Import necessary components from your alignment script ---
try:
    from v2_preprocessing import ZhangCorePreprocessor, ZhangVegetationIndices, CalibrationData, create_zhang_calibration_data
    print("[Info] Successfully imported from v2_preprocessing.py")
except ImportError as e:
    print(f"[Error] Could not import from v2_preprocessing.py: {e}")
    print("Ensure v2_preprocessing.py is in the same directory or Python path.")
    exit()
# --- End Import ---

# --- Determine Device ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Info] Using device check (CPU primarily for this script): {DEVICE}")
# --- End Device ---


# --- *** ROBUST SEGMENTATION FUNCTION (REPLACES K-MEANS) *** ---
def segment_leaves_kmeans_watershed(
    rgb_aligned_bgr: np.ndarray,
    nir_aligned: np.ndarray,
    min_leaf_area: int = 500,
    watershed_compactness: float = 0,
    watershed_min_distance: int = 20,
    **kwargs # Accepts extra args like n_clusters but ignores them
) -> Tuple[List[Tuple[str, np.ndarray]], np.ndarray, np.ndarray]:
    """
    Segments leaves using HSV + NIR thresholding, then separates with Watershed.
    This is much more robust to background changes than K-Means.
    Returns: Tuple ([(leaf_id, mask)], hsv_nir_debug_img, watershed_debug_img)
    """
    start_time = time.time()
    
    # 1. --- Green (HSV) Mask ---
    # Convert the RGB image to HSV (Hue, Saturation, Value)
    hsv_image = cv.cvtColor(rgb_aligned_bgr, cv.COLOR_BGR2HSV)
    
    # Define the range for "green" in HSV
    # These values are common, but you may need to tune them
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    
    # Create a mask for green pixels
    hsv_mask = cv.inRange(hsv_image, lower_green, upper_green)

    # 2. --- Bright NIR Mask ---
    # Find a good threshold for the NIR image.
    # Otsu's method is good at finding a split between "bright" (leaf) and "dark" (background)
    nir_threshold, nir_mask = cv.threshold(nir_aligned, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    # We want pixels *above* the threshold (the bright leaves)
    # If Otsu picks a very low threshold (e.g., all bright), we might need to adjust
    # For now, this is a good starting point.
    print(f"  [Segment Robust] NIR Threshold: {nir_threshold}")

    # 3. --- Combine Masks ---
    # The final mask includes pixels that are BOTH green AND bright in NIR
    binary_leaf_mask = cv.bitwise_and(hsv_mask, nir_mask)
    
    # --- The rest of the function is the SAME as before (cleaning + watershed) ---

    # 4. --- Clean the Mask ---
    kernel_open = np.ones((3,3), np.uint8)
    kernel_close = np.ones((7,7), np.uint8)
    cleaned_mask = cv.morphologyEx(binary_leaf_mask, cv.MORPH_OPEN, kernel_open, iterations=2)
    cleaned_mask = cv.morphologyEx(cleaned_mask, cv.MORPH_CLOSE, kernel_close, iterations=3)

    # Create a debug image for this step
    hsv_nir_debug_img = rgb_aligned_bgr.copy()
    hsv_nir_debug_img[cleaned_mask == 0] = (0, 0, 0) # Black out background
    
    print(f"  [Segment Robust] HSV+NIR mask generated in {time.time() - start_time:.3f}s")
    
    # 5. --- Watershed (to separate touching leaves) ---
    print(f"  [Segment Watershed] Applying Watershed (min_distance={watershed_min_distance})...")
    watershed_start = time.time()
    
    # Calculate the distance transform
    distance = ndi.distance_transform_edt(cleaned_mask > 0)
    
    # Find local peaks (markers) for watershed
    coords = peak_local_max(distance, min_distance=watershed_min_distance, labels=cleaned_mask)
    
    if coords.size == 0:
        print("  [Segment Watershed] No local maxima found. Treating as single object.")
        if np.sum(cleaned_mask) < min_leaf_area: 
            print("  [Segment Watershed] Cleaned mask area too small.")
            return [], hsv_nir_debug_img, np.zeros_like(rgb_aligned_bgr)
        markers, _ = ndi.label(cleaned_mask)
        watershed_labels = markers
    else:
        mask_markers = np.zeros(distance.shape, dtype=bool)
        mask_markers[tuple(coords.T)] = True
        markers, _ = ndi.label(mask_markers)
        watershed_labels = watershed(-distance, markers, mask=cleaned_mask, compactness=watershed_compactness)
        
    print(f"  [Segment Watershed] Watershed done in {time.time() - watershed_start:.3f}s")
    
    # 6. --- Extract Individual Leaf Masks ---
    unique_labels = np.unique(watershed_labels)
    all_leaf_masks_watershed = []
    valid_leaf_count = 0
    
    watershed_debug_img = np.zeros_like(rgb_aligned_bgr)
    np.random.seed(42)
    
    # Generate random colors for each label
    num_labels = len(unique_labels[unique_labels > 0])
    # Use a visually distinct colormap like 'tab20'
    cmap = plt.get_cmap('tab20')
    label_colors = {label: tuple((np.array(cmap(i / max(1,num_labels)))[:3] * 255).astype(int)) for i, label in enumerate(unique_labels) if label != 0}
    label_colors[0] = (0, 0, 0) # Background is black
    
    for label in unique_labels:
         if label == 0: continue # Skip background
         
         mask = np.where(watershed_labels == label, 1, 0).astype(np.uint8)
         area = np.sum(mask)
         
         if area > min_leaf_area:
             leaf_id = f"leaf_{valid_leaf_count:03d}"
             all_leaf_masks_watershed.append((leaf_id, mask))
             color_bgr = tuple(reversed(label_colors.get(label, (128, 128, 128)))) # Convert RGB from cmap to BGR
             watershed_debug_img[mask == 1] = color_bgr
             valid_leaf_count += 1
         else:
             # Mark small, discarded segments as gray
             watershed_debug_img[mask == 1] = (100, 100, 100)
             
    total_segment_time = time.time() - start_time
    print(f"  [Segment] Robust HSV+NIR + Watershed finished in {total_segment_time:.3f}s. Found {len(all_leaf_masks_watershed)} valid leaves.")
    
    # Return: 
    # 1. List of (id, mask) tuples
    # 2. The combined (HSV+NIR) mask for debugging (replaces kmeans_debug)
    # 3. The final color-coded watershed image
    return all_leaf_masks_watershed, hsv_nir_debug_img, watershed_debug_img
# --- *** END OF ROBUST FUNCTION *** ---


# --- Simple Thresholding Segmentation (REMOVED FROM MAIN PIPELINE TO PREVENT DATA LEAK) ---
# --- This function is no longer called by main() but is kept here for reference ---
def segment_single_leaf_threshold(
    image_aligned: np.ndarray,
    min_leaf_area: int = 500,
    use_nir: bool = True,
    blur_ksize: int = 5,
    debug_save_path: Optional[str] = None
) -> Tuple[List[Tuple[str, np.ndarray]], Optional[np.ndarray]]:
    """
    Segments a single leaf from a dark background using blurring, thresholding and largest contour.
    *** WARNING: DO NOT USE IN TRAINING PIPELINE IF MIXED WITH K-MEANS, as it causes data leakage. ***
    """
    # (This function is unchanged as it's not used by the main pipeline)
    start_time = time.time()
    h, w = image_aligned.shape[:2]
    all_leaf_masks = []
    debug_img_threshold = None

    if use_nir and len(image_aligned.shape) == 3 and image_aligned.shape[2] >= 3:
        gray_img = image_aligned[:,:,-1] if image_aligned.shape[2] > 2 else image_aligned
        print("  [Segment Threshold] Using NIR/Last channel or single channel input.")
    else:
        gray_img = cv.cvtColor(image_aligned, cv.COLOR_BGR2GRAY) if len(image_aligned.shape) == 3 else image_aligned
        print("  [Segment Threshold] Using Grayscale.")

    if gray_img.dtype != np.uint8:
         gray_img = cv.normalize(gray_img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    if blur_ksize > 0:
        ksize = blur_ksize if blur_ksize % 2 != 0 else blur_ksize + 1
        blurred_img = cv.GaussianBlur(gray_img, (ksize, ksize), 0)
    else: blurred_img = gray_img

    threshold_value, binary_mask = cv.threshold(blurred_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    kernel_open = np.ones((3,3), np.uint8); kernel_close = np.ones((5,5), np.uint8)
    cleaned_mask = cv.morphologyEx(binary_mask, cv.MORPH_OPEN, kernel_open, iterations=1)
    cleaned_mask = cv.morphologyEx(cleaned_mask, cv.MORPH_CLOSE, kernel_close, iterations=1)

    contours, _ = cv.findContours(cleaned_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    largest_contour = None; max_area = 0; valid_contours_found = []
    if contours:
        sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
        for cnt in sorted_contours:
            area = cv.contourArea(cnt)
            if area < min_leaf_area: break
            valid_contours_found.append(cnt)
            if largest_contour is None: largest_contour = cnt; max_area = area

    if largest_contour is not None:
        leaf_id = "leaf_000"
        final_mask = np.zeros((h, w), dtype=np.uint8)
        cv.drawContours(final_mask, [largest_contour], -1, 1, thickness=cv.FILLED)
        all_leaf_masks.append((leaf_id, final_mask))

    if debug_save_path:
        debug_img_threshold = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)
        debug_img_threshold[binary_mask == 255] = (255, 0, 0)
        debug_img_threshold[cleaned_mask == 255] = (0, 255, 255)
        if largest_contour is not None: cv.drawContours(debug_img_threshold, [largest_contour], -1, (0, 255, 0), 2)
    else:
         debug_img_threshold = None

    total_segment_time = time.time() - start_time
    print(f"  [Segment Threshold] Finished in {total_segment_time:.3f}s. Found {len(all_leaf_masks)} leaves.")
    return all_leaf_masks, debug_img_threshold
# --- End Simple Thresholding ---


# --- Structural Analysis (Function remains the same) ---
def segment_leaf_structure(
    leaf_mask: np.ndarray,
    edge_thickness: int = 5,
    erode_kernel_size: int = 5
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """
    Separates a leaf mask into edge and interveinal regions.
    Returns Tuple (edge_mask, interveinal_mask, debug_vis). Masks are uint8 or None.
    """
    if leaf_mask is None or np.sum(leaf_mask) == 0: return None, None, np.zeros( (*leaf_mask.shape, 3) if leaf_mask is not None else (100,100,3) , dtype=np.uint8)
    h, w = leaf_mask.shape
    debug_vis = np.zeros((h, w, 3), dtype=np.uint8)
    contours, _ = cv.findContours(leaf_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours: return None, None, debug_vis
    contour_img = np.zeros_like(leaf_mask); cv.drawContours(contour_img, contours, -1, 1, thickness=-1)
    ext_contours, _ = cv.findContours(contour_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not ext_contours: return None, None, debug_vis
    edge_mask = np.zeros_like(leaf_mask); cv.drawContours(edge_mask, ext_contours, -1, 1, thickness=edge_thickness)
    edge_mask = cv.bitwise_and(edge_mask, edge_mask, mask=leaf_mask)
    kernel_erode = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
    interveinal_mask = cv.erode(contour_img, kernel_erode, iterations=1)
    interveinal_mask[edge_mask == 1] = 0
    interveinal_mask = cv.bitwise_and(interveinal_mask, interveinal_mask, mask=leaf_mask)
    debug_vis[leaf_mask == 1] = (100, 100, 100)
    if interveinal_mask is not None and np.sum(interveinal_mask) > 10: debug_vis[interveinal_mask == 1] = (0, 255, 0)
    if edge_mask is not None and np.sum(edge_mask) > 10: debug_vis[edge_mask == 1] = (0, 0, 255)
    edge_mask = edge_mask if np.sum(edge_mask) > 10 else None
    interveinal_mask = interveinal_mask if np.sum(interveinal_mask) > 10 else None
    return edge_mask, interveinal_mask, debug_vis


# --- Main Processing Function (This is for the batch-processing script, not used by pipeline_v3.py) ---
def process_and_extract_features(
    rgb_path: str,
    nir_path: str,
    preprocessor: ZhangCorePreprocessor,
    segmentation_results: Tuple[List[Tuple[str, np.ndarray]], List[Optional[np.ndarray]]],
    rgb_aligned_bgr: np.ndarray,
    nir_aligned: np.ndarray,
    output_dir: str,
    base_name: str,
    calib_coords: Optional[List[Tuple[int, int]]] = None,
    save_debug_images: bool = True
) -> Optional[Dict[str, Dict]]:
    """Analyzes structure and extracts VIs from pre-segmented leaves."""
    # (This function is unchanged)
    all_leaf_masks, debug_images_list = segmentation_results
    pair_output_dir = os.path.join(output_dir, "debug_images", base_name + "_" + Path(rgb_path).stem.split('_')[-1])
    if not all_leaf_masks:
        print("  [VI Calc] No valid leaves from segmentation, skipping VI calculation.")
        return None
    image_leaf_features = {}
    try:
        start_time = time.time()
        red_channel_aligned = rgb_aligned_bgr[:,:,2]
        if calib_coords:
            red_refl_full = preprocessor.empirical_line_calibration(red_channel_aligned, 'red', calib_coords)
            nir_refl_full = preprocessor.empirical_line_calibration(nir_aligned, 'nir', calib_coords)
            print(f"  [Calibrate] Empirical line applied (0-100 scale).")
            red_refl_full_01 = np.clip(red_refl_full / 100.0, 0.0, 1.0)
            nir_refl_full_01 = np.clip(nir_refl_full / 100.0, 0.0, 1.0)
        else:
            red_refl_full_01 = (red_channel_aligned.astype(np.float32) / 255.0)
            nir_refl_full_01 = (nir_aligned.astype(np.float32) / 255.0)
            print(f"  [Calibrate] Simple normalization (0-1) applied.")
        print(f"  [Debug] Reflectance ranges (0-1): Red {red_refl_full_01.min():.3f}-{red_refl_full_01.max():.3f}, NIR {nir_refl_full_01.min():.3f}-{nir_refl_full_01.max():.3f}")
        for leaf_id, leaf_mask in all_leaf_masks:
             print(f"   --- Analyzing {leaf_id} ---")
             leaf_feature_vector = {
                 'pixel_count_total': int(np.sum(leaf_mask)),
                 'edge_ndvi_mean': np.nan, 'edge_ndvi_std': np.nan, 'edge_rdvi_mean': np.nan, 'edge_rdvi_std': np.nan,
                 'edge_sr_mean': np.nan, 'edge_sr_std': np.nan, 'edge_pixel_count': 0,
                 'inter_ndvi_mean': np.nan, 'inter_ndvi_std': np.nan, 'inter_rdvi_mean': np.nan, 'inter_rdvi_std': np.nan,
                 'inter_sr_mean': np.nan, 'inter_sr_std': np.nan, 'inter_pixel_count': 0
             }
             edge_mask, interveinal_mask, structure_debug_img = segment_leaf_structure(leaf_mask)
             leaf_output_dir = None
             if save_debug_images:
                  os.makedirs(pair_output_dir, exist_ok=True)
                  leaf_output_dir = os.path.join(pair_output_dir, leaf_id)
                  os.makedirs(leaf_output_dir, exist_ok=True)
                  cv.imwrite(os.path.join(leaf_output_dir, f"{base_name}_{leaf_id}_structure_debug.png"), structure_debug_img)
             def calculate_region_vis(region_mask, region_name):
                 stats = {}
                 if region_mask is not None:
                      mask_bool = region_mask.astype(bool)
                      if np.any(mask_bool):
                         red_refl = red_refl_full_01[mask_bool]; nir_refl = nir_refl_full_01[mask_bool]
                         pixel_count = int(red_refl.size)
                         if pixel_count > 10:
                              vi_data = preprocessor.calculate_zhang_vegetation_indices(red_refl, nir_refl)
                              stats = { f'{region_name}_ndvi_mean': vi_data.mean_ndvi, f'{region_name}_ndvi_std': vi_data.std_ndvi,
                                        f'{region_name}_rdvi_mean': vi_data.mean_rdvi, f'{region_name}_rdvi_std': vi_data.std_rdvi,
                                        f'{region_name}_sr_mean': vi_data.mean_sr, f'{region_name}_sr_std': vi_data.std_sr,
                                        f'{region_name}_pixel_count': pixel_count }
                              stats = {k: round(v, 6) if isinstance(v, float) else v for k,v in stats.items()}
                              print(f"     {region_name.capitalize()} VIs: NDVI={stats.get(f'{region_name}_ndvi_mean', np.nan):.3f} (Pixels={pixel_count})")
                         else: print(f"     {region_name.capitalize()} region too small ({pixel_count} pixels)")
                      else: print(f"     {region_name.capitalize()} mask was empty.")
                 else: print(f"     No valid {region_name} region found.")
                 return stats
             edge_stats = calculate_region_vis(edge_mask, 'edge')
             inter_stats = calculate_region_vis(interveinal_mask, 'inter')
             leaf_feature_vector.update(edge_stats)
             leaf_feature_vector.update(inter_stats)
             image_leaf_features[leaf_id] = leaf_feature_vector
             if save_debug_images and leaf_output_dir:
                  rgb_masked = cv.bitwise_and(rgb_aligned_bgr, rgb_aligned_bgr, mask=leaf_mask)
                  nir_aligned_mono = nir_aligned if len(nir_aligned.shape) == 2 else nir_aligned[:,:,0]
                  nir_masked_mono = cv.bitwise_and(nir_aligned_mono, nir_aligned_mono, mask=leaf_mask)
                  y_coords, x_coords = np.where(leaf_mask == 1)
                  if y_coords.size > 0:
                    ymin, ymax = y_coords.min(), y_coords.max()
                    xmin, xmax = x_coords.min(), x_coords.max()
                    padding = 5
                    ymin = max(0, ymin - padding); ymax = min(rgb_aligned_bgr.shape[0], ymax + padding)
                    xmin = max(0, xmin - padding); xmax = min(rgb_aligned_bgr.shape[1], xmax + padding)
                    rgb_crop = rgb_masked[ymin:ymax, xmin:xmax]
                    nir_crop = nir_masked_mono[ymin:ymax, xmin:xmax]
                    mask_crop = leaf_mask[ymin:ymax, xmin:xmax] * 255
                    cv.imwrite(os.path.join(leaf_output_dir, f"{base_name}_{leaf_id}_rgb_masked.png"), rgb_crop)
                    cv.imwrite(os.path.join(leaf_output_dir, f"{base_name}_{leaf_id}_nir_masked.png"), nir_crop)
                    cv.imwrite(os.path.join(leaf_output_dir, f"{base_name}_{leaf_id}_mask.png"), mask_crop)
        vi_calc_time = time.time() - start_time
        print(f"  [VI Calc & Structure] Processed {len(image_leaf_features)} leaves in {vi_calc_time:.3f}s")
    except Exception as e:
        print(f"[Error] Failed VI calculation for {base_name}: {e}")
        import traceback; traceback.print_exc()
        return None
    return image_leaf_features
# --- End Main Processing Function ---


# --- Main Execution (This is for the batch-processing script, not used by pipeline_v3.py) ---
def main():
    parser = argparse.ArgumentParser(description='Align, UNIFIED Segment, Extract Structural VIs, Output CSV')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory containing stage folders (healthy, early, late)')
    parser.add_argument('--calib_maps', type=str, required=True, help='Directory with rectification maps')
    parser.add_argument('--output_dir', type=str, default='feature_extraction_output', help='Base directory for outputs (CSV and debug images)')
    parser.add_argument('--output_csv', type=str, default='full_dataset_structural_features.csv', help='Filename for the output CSV')
    parser.add_argument('--calib_coords', type=str, default=None, help='JSON file with calibration target coordinates (optional)')
    parser.add_argument('--no_debug_images', action='store_true', help='Disable saving intermediate debug images')
    args = parser.parse_args()

    print("--- Starting UNIFIED Feature Extraction Pipeline ---")
    # (This function is unchanged)
    os.makedirs(args.output_dir, exist_ok=True)
    debug_img_base_path = os.path.join(args.output_dir, "debug_images")
    if not args.no_debug_images:
        os.makedirs(debug_img_base_path, exist_ok=True)
    calib_data = create_zhang_calibration_data()
    preprocessor = ZhangCorePreprocessor(args.calib_maps, calib_data)
    calib_coords_list = None
    if args.calib_coords and os.path.exists(args.calib_coords):
        try:
            with open(args.calib_coords, 'r') as f:
                coord_data = json.load(f); calib_coords_list = [(pt['x'], pt['y']) for pt in coord_data.get('calibration_points', [])]
                if not calib_coords_list: print("[Warning] 'calibration_points' key not found or empty in JSON.")
                else: print(f"[Info] Loaded {len(calib_coords_list)} calibration coordinates.")
        except Exception as e: print(f"[Warning] Failed to load calib coords: {e}."); calib_coords_list = None
    all_image_pairs_with_labels = []
    expected_stages = {'healthy': 0, 'early': 1, 'late': 2}
    print("\n--- Searching for Image Pairs (Simplified Logic) ---")
    total_pairs_found = 0
    for stage_name, label in expected_stages.items():
        stage_path = os.path.join(args.dataset_root, stage_name); stage_pairs_count = 0; processed_rgb_files = set()
        if os.path.isdir(stage_path):
            print(f"Searching in stage: {stage_name} (Label: {label}) at path: {stage_path}")
            parrot_rgb_files = glob.glob(os.path.join(stage_path, "pair_*_0000.jpg"))
            for rgb_path in parrot_rgb_files:
                nir_path = rgb_path.replace(".jpg", ".tif")
                if os.path.exists(nir_path): all_image_pairs_with_labels.append({'rgb': rgb_path, 'nir': nir_path, 'label': label, 'stage': stage_name}); processed_rgb_files.add(rgb_path); stage_pairs_count += 1
            hardware_rgb_files = glob.glob(os.path.join(stage_path, "rgb_*.png"))
            for rgb_path in hardware_rgb_files:
                if rgb_path in processed_rgb_files: continue
                rgb_filename = Path(rgb_path).name; nir_filename = rgb_filename.replace("rgb_", "nir_", 1); nir_path = os.path.join(stage_path, nir_filename)
                if os.path.exists(nir_path): all_image_pairs_with_labels.append({'rgb': rgb_path, 'nir': nir_path, 'label': label, 'stage': stage_name}); processed_rgb_files.add(rgb_path); stage_pairs_count += 1
            print(f"  Total pairs found in '{stage_name}': {stage_pairs_count}."); total_pairs_found += stage_pairs_count
        else: print(f"[Warning] Stage directory not found: {stage_path}")
    if not all_image_pairs_with_labels: print("\n[Error] No image pairs found."); return
    print(f"\n[Info] Found a total of {total_pairs_found} image pairs.")
    ordered_cols = [
        'label', 'image', 'leaf_id', 'pixel_count_total',
        'edge_ndvi_mean', 'edge_ndvi_std', 'edge_rdvi_mean', 'edge_rdvi_std',
        'edge_sr_mean', 'edge_sr_std', 'edge_pixel_count',
        'inter_ndvi_mean', 'inter_ndvi_std', 'inter_rdvi_mean', 'inter_rdvi_std',
        'inter_sr_mean', 'inter_sr_std', 'inter_pixel_count'
    ]
    all_features_for_csv = []
    print("\n--- Processing Image Pairs ---")
    processed_count = 0
    for pair_info in all_image_pairs_with_labels:
        rgb_file_name = Path(pair_info['rgb']).name
        if rgb_file_name.startswith('pair_'): base_name = rgb_file_name.split('_0000')[0]
        elif rgb_file_name.startswith('rgb_'): parts = rgb_file_name.split('_'); base_name = "_".join(parts[0:3]) if len(parts) >=3 else Path(pair_info['rgb']).stem
        else: base_name = Path(pair_info['rgb']).stem
        print(f"\nProcessing pair: {base_name} (Stage: {pair_info['stage']})")
        leaf_features_dict = None
        try:
            start_align = time.time()
            rgb_img_bgr = cv.imread(pair_info['rgb'], cv.IMREAD_COLOR)
            nir_path = pair_info['nir']
            nir_img_8bit = None
            if nir_path.lower().endswith(('.tif', '.tiff')):
                nir_img_raw = cv.imread(nir_path, cv.IMREAD_UNCHANGED)
                if nir_img_raw is None: raise FileNotFoundError(f"NIR not found: {nir_path}")
                nir_img = nir_img_raw[:, :, 0] if len(nir_img_raw.shape) == 3 else nir_img_raw
                if nir_img.dtype == 'uint16':
                    nir_img_8bit = cv.convertScaleAbs(nir_img, alpha=(255.0/65535.0))
                elif nir_img.dtype != np.uint8:
                     max_val = np.max(nir_img)
                     if max_val > 0: nir_img_8bit = ((nir_img.astype(np.float32) / max_val) * 255).astype(np.uint8)
                     else: nir_img_8bit = nir_img.astype(np.uint8)
                else:
                    nir_img_8bit = nir_img
            else:
                nir_img_raw = cv.imread(nir_path, cv.IMREAD_UNCHANGED)
                if nir_img_raw is None: raise FileNotFoundError(f"NIR not found: {nir_path}")
                if nir_img_raw.dtype != np.uint8 or len(nir_img_raw.shape) > 2 :
                    if nir_img_raw.max() > 255: nir_img_8bit = cv.normalize(nir_img_raw, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
                    else:
                        nir_img_8bit = nir_img_raw[:,:,0] if len(nir_img_raw.shape) == 3 else nir_img_raw
                        if nir_img_8bit.dtype != np.uint8: nir_img_8bit = (nir_img_8bit * 255).astype(np.uint8)
                else: nir_img_8bit = nir_img_raw
            if len(nir_img_8bit.shape) == 3:
                 nir_img_8bit = cv.cvtColor(nir_img_8bit, cv.COLOR_BGR2GRAY)
            if rgb_img_bgr is None: raise FileNotFoundError(f"RGB not found: {pair_info['rgb']}")
            if rgb_img_bgr.size == 0: raise ValueError("RGB empty")
            if nir_img_8bit.size == 0: raise ValueError("NIR empty")
            rgb_aligned_bgr, nir_aligned = preprocessor.rectify_image_pair(rgb_img_bgr, nir_img_8bit)
            print(f"  [Align] Alignment done in {time.time() - start_align:.3f}s")
            segmentation_results = None; debug_images_list = []
            pair_debug_dir = os.path.join(debug_img_base_path, base_name + "_" + Path(pair_info['rgb']).stem.split('_')[-1])
            
            # --- *** USING ROBUST SEGMENTATION *** ---
            print(f"  [Segment] Using UNIFIED Robust HSV+NIR + Watershed logic...")
            robust_masks, hsv_nir_debug, watershed_debug = segment_leaves_kmeans_watershed(
                rgb_aligned_bgr, nir_aligned
            )
            segmentation_results = (robust_masks, [hsv_nir_debug, watershed_debug])
            debug_images_list.append(("hsv_nir_mask", hsv_nir_debug))
            debug_images_list.append(("watershed_leaves", watershed_debug))
            # --- *** END FIX *** ---

            if not args.no_debug_images:
                 os.makedirs(pair_debug_dir, exist_ok=True)
                 cv.imwrite(os.path.join(pair_debug_dir, f"{base_name}_rgb_aligned.png"), rgb_aligned_bgr)
                 cv.imwrite(os.path.join(pair_debug_dir, f"{base_name}_nir_aligned.png"), nir_aligned)
                 for img_key, img_data in debug_images_list:
                      if img_data is not None:
                           cv.imwrite(os.path.join(pair_debug_dir, f"{base_name}_{img_key}.png"), img_data)
            leaf_features_dict = process_and_extract_features(
                pair_info['rgb'], pair_info['nir'], preprocessor,
                segmentation_results, rgb_aligned_bgr, nir_aligned,
                args.output_dir, base_name,
                calib_coords=calib_coords_list,
                save_debug_images = not args.no_debug_images
            )
            if leaf_features_dict:
                processed_count += 1
                for leaf_id, features in leaf_features_dict.items():
                    row_data = {'label': pair_info['label'], 'image': base_name, 'leaf_id': leaf_id}
                    for col in ordered_cols[3:]:
                        row_data[col] = features.get(col, np.nan)
                    all_features_for_csv.append(row_data)
        except Exception as e:
            print(f"[Error] Failed processing pair {base_name}: {e}")
            import traceback; traceback.print_exc()
    print(f"\n--- Feature Extraction Complete ---")
    print(f"Successfully processed and extracted features from {processed_count} / {total_pairs_found} image pairs.")
    if all_features_for_csv:
        feature_df = pd.DataFrame(all_features_for_csv)
        csv_output_path = os.path.join(args.output_dir, args.output_csv)
        feature_df = feature_df.reindex(columns=ordered_cols)
        feature_df.to_csv(csv_output_path, index=False, na_rep='NaN')
        print(f"\n[Info] Aggregated features saved to: {csv_output_path}")
        print(f"  Total leaves in CSV: {len(feature_df)}")
    else:
        print("\n[Warning] No features were extracted to save to CSV.")
    print("\n--- Feature Extraction Pipeline Finished ---")


if __name__ == "__main__":
    # This check is important!
    # This 'main' function is for batch processing.
    # The 'pipeline_v3.py' script (the web app) does NOT call this.
    # It only *imports* functions from this file.
    main()
