import cv2 as cv
import numpy as np

# ================== Utility ==================
def to_gray_bgr(img_bgr):
    if img_bgr.ndim == 2:
        return img_bgr
    return cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

def gradient_mag(gray):
    # Gradient magnitude is more stable across spectra than raw intensity
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    mag = cv.magnitude(gx, gy)
    return cv.convertScaleAbs(mag)

def normalize_std(x):
    xf = x.astype(np.float32)
    return (xf - xf.mean()) / (xf.std() + 1e-6)

def alignment_quality(bgr_ref, ir_aligned_gray):
    # NCC of gradient magnitudes (range ~ -1..1). Good pairs often 0.35–0.7
    g1 = gradient_mag(to_gray_bgr(bgr_ref))
    g2 = gradient_mag(ir_aligned_gray)
    return float((normalize_std(g1) * normalize_std(g2)).mean())

# ================== Global model: Homography on gradient-ORB ==================
def estimate_homography_orb_grad(rgb_rect_gray, ir_rect_gray, down=0.5,
                                 nfeatures=5000, ratio=0.85, ransac_thr=4.0):
    # Build gradient-domain images
    g1 = gradient_mag(rgb_rect_gray)
    g2 = gradient_mag(ir_rect_gray)

    if down != 1.0:
        g1s = cv.resize(g1, None, fx=down, fy=down, interpolation=cv.INTER_AREA)
        g2s = cv.resize(g2, None, fx=down, fy=down, interpolation=cv.INTER_AREA)
        scale = 1.0 / down
    else:
        g1s, g2s, scale = g1, g2, 1.0

    orb = cv.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8,
                        edgeThreshold=15, WTA_K=2, patchSize=31, fastThreshold=7)
    k1, d1 = orb.detectAndCompute(g1s, None)
    k2, d2 = orb.detectAndCompute(g2s, None)

    if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
        return None, 0

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    if len(good) < 10:
        return None, 0

    pts1 = np.float32([k1[m.queryIdx].pt for m in good]) * scale
    pts2 = np.float32([k2[m.trainIdx].pt for m in good]) * scale

    # If your pair was stereo-rectified first, vertical disparity should be small; keep optional filter:
    # mask_y = np.abs(pts1[:,1] - pts2[:,1]) < 4.0; pts1, pts2 = pts1[mask_y], pts2[mask_y]

    if len(pts1) < 10:
        return None, 0

    H, inliers = cv.findHomography(pts2, pts1, cv.RANSAC, ransacReprojThreshold=ransac_thr,
                                   maxIters=4000, confidence=0.995)
    inl = int(inliers.sum()) if inliers is not None else 0
    return H, inl

# ================== Local model: Grid LK mesh refinement ==================
def mesh_refine_grad(rgb_rect_gray, ir_warp_gray, grid_step=48, win=21, pyr=3):
    h, w = rgb_rect_gray.shape
    xs = np.arange(grid_step//2, w, grid_step)
    ys = np.arange(grid_step//2, h, grid_step)
    if len(xs) < 2 or len(ys) < 2:
        return None, None, 0, 0.0

    G1 = gradient_mag(rgb_rect_gray)
    G2 = gradient_mag(ir_warp_gray)

    gx, gy = np.meshgrid(xs, ys)
    pts0 = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

    pts1, st, err = cv.calcOpticalFlowPyrLK(
        G1, G2, pts0, None,
        winSize=(win, win), maxLevel=pyr,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 40, 0.01)
    )
    ok = (st.flatten() == 1)
    if ok.sum() < 8:
        return None, None, 0, 0.0

    p0 = pts0[ok]; p1 = pts1[ok]
    flow = p1 - p0

    # Robust outlier removal via median/MAD
    med = np.median(flow, axis=0)
    mad = np.median(np.abs(flow - med), axis=0) + 1e-6
    keep = (np.abs(flow - med) < 3.5 * mad).all(axis=1)
    p0, flow = p0[keep], flow[keep]

    if len(p0) < 8:
        med_err = float(np.median(err[ok])) if err is not None else 0.0
        return None, None, ok.sum(), med_err

    # Bin sparse flow to coarse grid and upsample to dense
    gh, gw = len(ys), len(xs)
    flow_x = np.zeros((gh, gw), np.float32)
    flow_y = np.zeros((gh, gw), np.float32)
    counts  = np.zeros((gh, gw), np.float32)

    # nearest grid bin indices for each point
    ix = np.clip(((p0[:,0] - xs[0]) / grid_step).round().astype(int), 0, gw-1)
    iy = np.clip(((p0[:,1] - ys[0]) / grid_step).round().astype(int), 0, gh-1)
    for (i, j), (dx, dy) in zip(zip(iy, ix), flow):
        flow_x[i, j] += dx; flow_y[i, j] += dy; counts[i, j] += 1.0
    counts[counts == 0] = 1.0
    flow_x /= counts; flow_y /= counts

    dense_dx = cv.resize(flow_x, (w, h), interpolation=cv.INTER_CUBIC)
    dense_dy = cv.resize(flow_y, (w, h), interpolation=cv.INTER_CUBIC)
    XX, YY = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mapx = XX + dense_dx
    mapy = YY + dense_dy

    med_err = float(np.median(err[ok])) if err is not None else 0.0
    return mapx, mapy, len(p0), med_err

# ================== Main Aligner ==================
class OpenCVAligner:
    """
    Pure-OpenCV aligner for RGB(BGR) vs IR(gray) frames.
    Optional stereo rectification maps can be provided.
    """
    def __init__(self, rectify_maps=None, grid_step=48, down_for_H=0.5):
        """
        rectify_maps: tuple of (rgb_mapx, rgb_mapy, ir_mapx, ir_mapy) or None
        grid_step: control point spacing for mesh refinement (32 = tighter, 64 = faster)
        down_for_H: downscale factor for homography feature stage (0.5 ~ 0.75)
        """
        self.maps = rectify_maps
        self.grid_step = grid_step
        self.down_for_H = down_for_H

    def rectify(self, img_bgr, ir_gray):
        if self.maps is None:
            return img_bgr, ir_gray
        rgb_mx, rgb_my, ir_mx, ir_my = self.maps
        rgb_rect = cv.remap(img_bgr, rgb_mx, rgb_my, cv.INTER_LINEAR)
        ir_rect  = cv.remap(ir_gray,  ir_mx,  ir_my,  cv.INTER_LINEAR)
        return rgb_rect, ir_rect

    def align_ir_to_rgb(self, img_bgr, ir_gray):
        """
        Returns: (ir_aligned_gray, info_dict) or (None, {"status": "..."} on failure)
        """
        # 1) (optional) Rectify
        rgb_rect_bgr, ir_rect = self.rectify(img_bgr, ir_gray)
        rgb_rect_gray = to_gray_bgr(rgb_rect_bgr)

        # 2) Global homography (IR -> RGB) on gradient ORB
        H, inliers = estimate_homography_orb_grad(
            rgb_rect_gray, ir_rect, down=self.down_for_H
        )
        if H is None:
            return None, {"status":"no_homography"}

        h, w = rgb_rect_gray.shape
        ir_warp = cv.warpPerspective(
            ir_rect, H, (w, h),
            flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0
        )

        # 3) Local mesh refinement via grid LK on gradients
        mapx, mapy, tracks, med_err = mesh_refine_grad(
            rgb_rect_gray, ir_warp, grid_step=self.grid_step
        )
        if mapx is not None:
            ir_fine = cv.remap(ir_warp, mapx, mapy, cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT)
            info = {"status":"ok", "H_inliers":inliers, "tracks":tracks, "lk_med_err":med_err}
            return ir_fine, info
        else:
            info = {"status":"ok_no_mesh", "H_inliers":inliers, "tracks":0, "lk_med_err":None}
            return ir_warp, info
