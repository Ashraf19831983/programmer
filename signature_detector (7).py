# -*- coding: utf-8 -*-
"""
Signature Forgery Detection System — Enhanced Edition
=============================================
Implemented Improvements:
  1.  Relative path instead of hard-coded path
  2.  Window size based on actual screen dimensions
  3.  Safe class_names[pred] access with IndexError guard
  4.  Organised data/ folder next to the script file
  5.  Mathematically correct Eccentricity (uses mu11)
  6.  Feature extraction from ALL contours, not just the largest
  7.  Optimal k selection via cross-validation instead of fixed k=20
  8.  Confidence threshold with "Unrecognised" fallback message
  9.  Replaced all print() calls with proper logging
  10. SignatureCanvasWidget — shared widget to eliminate drawing code duplication
  11. Safe algorithm selection dialog (handles None / cancel)
  12. Training runs in a background thread to prevent UI freeze
  13. Default sample count raised to 15 with progress bar
  14. One-Class SVM model added for genuine forgery detection
  15. Automatic retrain warning when a new person is added
  16. True accuracy via cross-validation when data is scarce
"""

import os
import sys
import cv2
import logging
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from PIL import Image, ImageDraw, ImageTk
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════
# General Configuration
# ═══════════════════════════════════════════════════════════════

# ── Data folder next to script file (relative, not absolute) ────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "signatures_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILE      = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_FILE     = os.path.join(MODELS_DIR, "scaler.pkl")
SELECTOR_FILE   = os.path.join(MODELS_DIR, "selector.pkl")
OC_MODELS_DIR   = os.path.join(MODELS_DIR, "one_class")
CLASS_NAMES_FILE= os.path.join(MODELS_DIR, "class_names.txt")

CANVAS_SIZE    = 400
SIGNATURE_SIZE = 200
THUMB_SIZE     = 180   # Small image size (features / result panels)
THUMB_BIG      = 340   # Enlarged grayscale / binary image size
NUM_SAMPLES  = 15          # Raised from 5 to 15
CONFIDENCE_THRESHOLD = 0.55  # Below this → "Unrecognised"
K_SELECT             = 30    # Features selected from 152 total

# ── Logging setup ───────────────────────────────────────────────
LOG_FILE = os.path.join(BASE_DIR, "app.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════

def ensure_dir(path: str):
    """Creates the directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def list_people() -> list:
    """Returns a sorted list of registered person names."""
    ensure_dir(DATA_DIR)
    return sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    )


def screen_fraction(root: tk.Tk, w_frac: float, h_frac: float) -> tuple:
    """Computes window dimensions as a fraction of screen size."""
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    return int(sw * w_frac), int(sh * h_frac)


def center_window(win: tk.Toplevel, width: int, height: int):
    """Centers the window on the screen."""
    win.update_idletasks()
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    x  = (sw - width)  // 2
    y  = (sh - height) // 2
    win.geometry(f"{width}x{height}+{x}+{y}")


# ═══════════════════════════════════════════════════════════════
# Image Processing and Feature Extraction
# ═══════════════════════════════════════════════════════════════

def preprocess_signature(img_pil: Image.Image) -> np.ndarray:
    """
    Converts the image to a fixed-size binary image.
    Uses Otsu thresholding while preserving the aspect ratio.
    """
    img = np.array(img_pil.convert('L'))
    _, binary = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    h, w = binary.shape
    if h == 0 or w == 0:
        return np.zeros((SIGNATURE_SIZE, SIGNATURE_SIZE), dtype=np.uint8)

    if h > w:
        new_h = SIGNATURE_SIZE
        new_w = max(1, int(w * SIGNATURE_SIZE / h))
    else:
        new_w = SIGNATURE_SIZE
        new_h = max(1, int(h * SIGNATURE_SIZE / w))

    resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)
    square  = np.zeros((SIGNATURE_SIZE, SIGNATURE_SIZE), dtype=np.uint8)
    y_off   = (SIGNATURE_SIZE - new_h) // 2
    x_off   = (SIGNATURE_SIZE - new_w) // 2
    square[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    square  = (square > 128).astype(np.uint8) * 255
    return square


# ══════════════════════════════════════════════════════════════
# Feature Names — 152 features across 7 groups
# ══════════════════════════════════════════════════════════════
# Group 1 : 7  log-scale Hu moments
# Group 2 : 10 geometric shape features
# Group 3 : 16 4x4 grid features (ink density per cell)
# Group 4 : 64 8x8 grid features (higher resolution)
# Group 5 : 20 horizontal projection (row distribution)
# Group 6 : 20 vertical projection (column distribution)
# Group 7 : 15 statistical and additional features
# ══════════════════════════════════════════════════════════════
FEATURE_NAMES = (
    # 7 Hu moments
    [f"Hu{i}"       for i in range(1, 8)]   +
    # 10 shape features
    ["Area", "Perimeter", "Compactness", "Eccentricity",
     "Solidity", "Extent", "AspectRatio", "NumContours",
     "MeanCntArea", "StdCntArea"]            +
    # 16 4x4 grid
    [f"G4x4_{i}"    for i in range(1, 17)]  +
    # 64 8x8 grid
    [f"G8x8_{i}"    for i in range(1, 65)]  +
    # 20 horizontal projection
    [f"ProjH_{i}"   for i in range(1, 21)]  +
    # 20 vertical projection
    [f"ProjV_{i}"   for i in range(1, 21)]  +
    # 15 statistical/additional
    ["Px_Mean", "Px_Std", "Px_Skew", "Px_Kurt", "Px_Entropy",
     "Trans_H_Mean", "Trans_H_Std", "Trans_V_Mean", "Trans_V_Std",
     "BBW", "BBH", "BBCx", "BBCy",
     "HullPerim", "ConvexityDefect"]
)
# Total: 7 + 10 + 16 + 64 + 20 + 20 + 15 = 152 features
N_FEATURES = len(FEATURE_NAMES)   # 152 total features


def _safe_stats(arr: np.ndarray) -> tuple:
    """Safely computes basic statistics for a 1-D array."""
    arr = arr.astype(float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean  = float(np.mean(arr))
    std   = float(np.std(arr))
    # skewness
    if std > 1e-12:
        skew = float(np.mean(((arr - mean) / std) ** 3))
        kurt = float(np.mean(((arr - mean) / std) ** 4) - 3)
    else:
        skew, kurt = 0.0, 0.0
    return mean, std, skew, kurt


def _entropy(arr: np.ndarray) -> float:
    """Shannon entropy of the density array."""
    arr   = arr.flatten().astype(float)
    total = arr.sum()
    if total <= 0:
        return 0.0
    p    = arr / total
    p    = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _projection_profile(img: np.ndarray, axis: int, n: int = 20) -> np.ndarray:
    """
    Computes the projection along the given axis and returns n sampled points.
    axis=0 → horizontal projection (row sums)
    axis=1 → vertical projection (column sums)
    """
    proj  = img.sum(axis=axis).astype(float)
    total = proj.sum()
    if total > 0:
        proj /= total
    # Resample to n points
    indices = np.linspace(0, len(proj) - 1, n).astype(int)
    return proj[indices]


def _transition_stats(img: np.ndarray) -> tuple:
    """
    Computes mean and std of 0-to-1 transitions along rows and columns.
    Transitions reflect signature complexity.
    """
    # Horizontal transitions (along rows)
    h_trans = np.diff(img, axis=1)
    h_per_row = (h_trans != 0).sum(axis=1).astype(float)
    h_mean = float(h_per_row.mean()) if h_per_row.size > 0 else 0.0
    h_std  = float(h_per_row.std())  if h_per_row.size > 0 else 0.0

    # Vertical transitions (along columns)
    v_trans = np.diff(img, axis=0)
    v_per_col = (v_trans != 0).sum(axis=0).astype(float)
    v_mean = float(v_per_col.mean()) if v_per_col.size > 0 else 0.0
    v_std  = float(v_per_col.std())  if v_per_col.size > 0 else 0.0

    return h_mean, h_std, v_mean, v_std


def extract_features(binary_img: np.ndarray) -> np.ndarray:
    """
    Extracts 152 features from the binary image across 7 groups:

    Group 1 — Hu Moments (7):
        Log-scale Hu moments invariant to rotation and scale.

    Group 2 — Geometric Shape Features (10):
        Area, perimeter, compactness, eccentricity, solidity, extent,
        aspect ratio, contour count, mean and std of contour areas.

    Group 3 — 4x4 Grid (16):
        Ink density in 16 cells.

    Group 4 — 8x8 Grid (64):
        Ink density in 64 cells — higher resolution than the 4x4 grid.

    Group 5 — Horizontal Projection (20):
        Ink distribution along the row axis (20 sampled points).

    Group 6 — Vertical Projection (20):
        Ink distribution along the column axis (20 sampled points).

    Group 7 — Statistical and Additional Features (15):
        Pixel statistics, transitions, bounding box, convex hull.
    """
    img = (binary_img > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        (img * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.zeros(N_FEATURES)

    # ── Group 1: Hu Moments ─────────────────────────────────────
    all_points = np.vstack(contours)
    moments    = cv2.moments(all_points)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    # ── Group 2: Shape Features ────────────────────────────────
    # Mathematically correct Eccentricity
    mu20 = moments.get("mu20", 0)
    mu02 = moments.get("mu02", 0)
    mu11 = moments.get("mu11", 0)
    disc = 4 * mu11**2 + (mu20 - mu02)**2
    if (mu20 + mu02) > 0 and disc >= 0:
        lam1 = (mu20 + mu02 + np.sqrt(disc)) / 2
        lam2 = max((mu20 + mu02 - np.sqrt(disc)) / 2, 0.0)
        eccentricity = float(np.sqrt(1 - lam2 / lam1)) if lam1 > 0 else 0.0
    else:
        eccentricity = 0.0

    area      = float(np.sum(img))
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0.0

    cnt_largest = max(contours, key=cv2.contourArea)
    hull        = cv2.convexHull(cnt_largest)
    hull_area   = float(cv2.contourArea(hull))
    hull_perim  = float(cv2.arcLength(hull, True))
    solidity    = area / hull_area if hull_area > 0 else 0.0

    x, y, w, h  = cv2.boundingRect(cnt_largest)
    bbox_area   = w * h
    extent      = area / bbox_area if bbox_area > 0 else 0.0
    aspect_ratio = float(w) / h if h > 0 else 0.0

    # Bounding-box centre normalised to image dimensions
    H_img, W_img = img.shape
    bb_cx = (x + w / 2) / W_img if W_img > 0 else 0.0
    bb_cy = (y + h / 2) / H_img if H_img > 0 else 0.0
    bb_w_norm = float(w) / W_img if W_img > 0 else 0.0
    bb_h_norm = float(h) / H_img if H_img > 0 else 0.0

    num_contours = float(len(contours))
    cnt_areas    = np.array([cv2.contourArea(c) for c in contours], dtype=float)
    mean_cnt_area= float(cnt_areas.mean())
    std_cnt_area = float(cnt_areas.std())

    # Convexity defect — complexity measure of the largest contour
    try:
        hull_idx = cv2.convexHull(cnt_largest, returnPoints=False)
        defects  = cv2.convexityDefects(cnt_largest, hull_idx)
        if defects is not None and len(defects) > 0:
            depths = defects[:, 0, 3].astype(float) / 256.0
            convexity_defect = float(depths.mean())
        else:
            convexity_defect = 0.0
    except Exception:
        convexity_defect = 0.0

    shape_feats = np.array([
        area, perimeter, compactness, eccentricity,
        solidity, extent, aspect_ratio, num_contours,
        mean_cnt_area, std_cnt_area
    ])

    # ── Group 3: 4x4 Grid ──────────────────────────────────────
    grid4_features = []
    for rb in np.array_split(img, 4, axis=0):
        for cb in np.array_split(rb, 4, axis=1):
            grid4_features.append(float(np.sum(cb)))

    # ── Group 4: 8x8 Grid ──────────────────────────────────────
    grid8_features = []
    for rb in np.array_split(img, 8, axis=0):
        for cb in np.array_split(rb, 8, axis=1):
            grid8_features.append(float(np.sum(cb)))

    # ── Groups 5+6: Projection Profiles ───────────────────────
    proj_h = _projection_profile(img, axis=1, n=20)  # horizontal
    proj_v = _projection_profile(img, axis=0, n=20)  # vertical

    # ── Group 7: Additional Statistics ────────────────────────
    px_vals = img.flatten().astype(float)
    px_mean, px_std, px_skew, px_kurt = _safe_stats(px_vals)
    px_entropy = _entropy(img)

    h_mean_tr, h_std_tr, v_mean_tr, v_std_tr = _transition_stats(img)

    extra_feats = np.array([
        px_mean, px_std, px_skew, px_kurt, px_entropy,
        h_mean_tr, h_std_tr, v_mean_tr, v_std_tr,
        bb_w_norm, bb_h_norm, bb_cx, bb_cy,
        hull_perim, convexity_defect
    ])

    # ── Concatenate all features ───────────────────────────────
    features = np.concatenate([
        hu_moments,          # 7
        shape_feats,         # 10
        grid4_features,      # 16
        grid8_features,      # 64
        proj_h,              # 20
        proj_v,              # 20
        extra_feats          # 15
    ])                       # = 152

    assert len(features) == N_FEATURES,         f"Feature count mismatch: {len(features)} vs expected {N_FEATURES}"

    return features



def _set_text(widget: tk.Text, content: str):
    """Safely updates a Text widget (read-only after update)."""
    widget.config(state=tk.NORMAL)
    widget.delete("1.0", tk.END)
    widget.insert(tk.END, content)
    widget.config(state=tk.DISABLED)


# ═══════════════════════════════════════════════════════════════
# Visual Processing Functions for Diagnostic Panels
# ═══════════════════════════════════════════════════════════════

def make_contour_image(binary_img: np.ndarray, size: int) -> Image.Image:
    """
    Draws colour-coded contours on a white background.
    Each contour gets a distinct colour from a preset palette.
    """
    COLORS = [
        (220,  20,  60), ( 30, 144, 255), ( 50, 205,  50),
        (255, 140,   0), (148,   0, 211), (  0, 206, 209),
        (255,  20, 147), (  0, 128,   0), (255, 215,   0),
        (100, 149, 237),
    ]
    canvas_bgr = np.ones((binary_img.shape[0], binary_img.shape[1], 3),
                          dtype=np.uint8) * 255
    contours, _ = cv2.findContours(
        binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Sort contours largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for i, cnt in enumerate(contours):
        color_rgb = COLORS[i % len(COLORS)]
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        cv2.drawContours(canvas_bgr, [cnt], -1, color_bgr, 2)
        # Label contour if large enough
        if cv2.contourArea(cnt) > 50:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(canvas_bgr, str(i+1), (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            color_bgr, 1, cv2.LINE_AA)
    img_pil = Image.fromarray(canvas_bgr).resize((size, size), Image.LANCZOS)
    return img_pil


def make_skeleton_image(binary_img: np.ndarray, size: int) -> Image.Image:
    """
    Produces the morphological skeleton of the signature.
    Result: single-pixel-wide lines representing the signature structure.
    """
    img = (binary_img > 0).astype(np.uint8)
    skeleton = np.zeros_like(img)
    kernel   = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = img.copy()
    itr  = 0
    while True:
        eroded   = cv2.erode(temp, kernel)
        opened   = cv2.dilate(eroded, kernel)
        diff     = cv2.subtract(temp, opened)
        skeleton = cv2.bitwise_or(skeleton, diff)
        temp     = eroded.copy()
        if cv2.countNonZero(temp) == 0 or itr > 60:
            break
        itr += 1
    # Cyan skeleton on a light-grey background
    skel_colored = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    skel_colored[skeleton > 0] = [0, 200, 255]   # cyan
    # light-grey background
    bg = np.full_like(skel_colored, 240)
    mask = skeleton == 0
    skel_colored[mask] = bg[mask]
    img_pil = Image.fromarray(skel_colored).resize((size, size), Image.LANCZOS)
    return img_pil, itr


def make_heatmap_image(binary_img: np.ndarray, size: int,
                        grid: int = 8) -> Image.Image:
    """
    Creates a colour heatmap from a grid x grid partition.
    High ink-density cells appear red, empty cells blue (TURBO palette).
    """
    img  = (binary_img > 0).astype(np.float32)
    rows = np.array_split(img, grid, axis=0)
    heat = np.zeros((grid, grid), dtype=np.float32)
    for r, row_block in enumerate(rows):
        cols = np.array_split(row_block, grid, axis=1)
        for c, block in enumerate(cols):
            heat[r, c] = float(block.sum())

    # Normalise 0-255
    mx = heat.max()
    if mx > 0:
        heat_norm = (heat / mx * 255).astype(np.uint8)
    else:
        heat_norm = heat.astype(np.uint8)

    # Apply TURBO colour map
    heat_bgr   = cv2.applyColorMap(heat_norm, cv2.COLORMAP_TURBO)
    heat_rgb   = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    # Upscale with nearest-neighbour and draw grid lines
    cell_size  = size // grid
    canvas_big = np.zeros((size, size, 3), dtype=np.uint8)
    for r in range(grid):
        for c in range(grid):
            y0 = r * cell_size
            x0 = c * cell_size
            canvas_big[y0:y0+cell_size, x0:x0+cell_size] = heat_rgb[r, c]

    # Draw grid lines
    for i in range(1, grid):
        cv2.line(canvas_big, (0, i*cell_size), (size, i*cell_size),
                 (50, 50, 50), 1)
        cv2.line(canvas_big, (i*cell_size, 0), (i*cell_size, size),
                 (50, 50, 50), 1)

    img_pil = Image.fromarray(canvas_big)
    return img_pil, heat


def draw_projection_chart(master_frame, proj_h: np.ndarray,
                           proj_v: np.ndarray, size: int):
    """
    Draws horizontal and vertical projection bar charts
    directly onto tkinter Canvas widgets inside the given frame.
    """
    CHART_W = size
    CHART_H = size // 2 - 6
    PAD     = 22

    # ── Horizontal projection ──────────────────────────────────
    cv_h = tk.Canvas(master_frame, width=CHART_W, height=CHART_H,
                     bg="#1A237E", highlightthickness=0)
    cv_h.pack(pady=(4, 2))
    tk.Label(master_frame, text="▶ Horizontal Projection (row distribution)",
             font=("Arial", 8, "bold"), fg="#1A237E",
             bg="white").pack()

    n     = len(proj_h)
    mx_h  = proj_h.max() if proj_h.max() > 0 else 1
    bar_w = max(1, (CHART_W - PAD*2) // n)
    for i, val in enumerate(proj_h):
        bar_h = int((val / mx_h) * (CHART_H - PAD))
        x0 = PAD + i * bar_w
        y0 = CHART_H - PAD - bar_h
        x1 = x0 + bar_w - 1
        y1 = CHART_H - PAD
        # Gradient: blue to yellow by bar height
        intensity = int(val / mx_h * 255)
        color = f"#{intensity:02x}{255-intensity//2:02x}ff"
        cv_h.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
    # Axes
    cv_h.create_line(PAD, CHART_H-PAD, CHART_W-PAD, CHART_H-PAD,
                      fill="white", width=1)
    cv_h.create_text(CHART_W//2, 6, text="Horizontal Distribution",
                      fill="white", font=("Arial", 7, "bold"))

    # ── Vertical projection ────────────────────────────────────
    cv_v = tk.Canvas(master_frame, width=CHART_W, height=CHART_H,
                     bg="#1B5E20", highlightthickness=0)
    cv_v.pack(pady=(2, 4))
    tk.Label(master_frame, text="▶ Vertical Projection (column distribution)",
             font=("Arial", 8, "bold"), fg="#1B5E20",
             bg="white").pack()

    mx_v  = proj_v.max() if proj_v.max() > 0 else 1
    bar_w = max(1, (CHART_W - PAD*2) // len(proj_v))
    for i, val in enumerate(proj_v):
        bar_h = int((val / mx_v) * (CHART_H - PAD))
        x0 = PAD + i * bar_w
        y0 = CHART_H - PAD - bar_h
        x1 = x0 + bar_w - 1
        y1 = CHART_H - PAD
        intensity = int(val / mx_v * 255)
        color = f"#{intensity//3:02x}{intensity:02x}{intensity//3:02x}"
        cv_v.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
    cv_v.create_line(PAD, CHART_H-PAD, CHART_W-PAD, CHART_H-PAD,
                      fill="white", width=1)
    cv_v.create_text(CHART_W//2, 6, text="Vertical Distribution",
                      fill="white", font=("Arial", 7, "bold"))

    return cv_h, cv_v


# ═══════════════════════════════════════════════════════════════
# Shared Drawing Canvas Widget (eliminates code duplication)
# ═══════════════════════════════════════════════════════════════

class SignatureCanvasWidget(tk.Frame):
    """
    Self-contained signature drawing widget.
    Shared by the collection window and recognition window
    to eliminate code duplication.
    """

    def __init__(self, parent, size: int = CANVAS_SIZE, line_width: int = 5, **kwargs):
        super().__init__(parent, **kwargs)
        self.size       = size
        self.line_width = line_width
        self._drawing   = False
        self._last_x    = None
        self._last_y    = None

        self.canvas = tk.Canvas(
            self, width=size, height=size,
            bg='white', cursor='pencil',
            relief=tk.SOLID, bd=2
        )
        self.canvas.pack()

        self._pil_img  = Image.new("RGB", (size, size), "white")
        self._draw_obj = ImageDraw.Draw(self._pil_img)

        self.canvas.bind("<Button-1>",       self._start_draw)
        self.canvas.bind("<B1-Motion>",      self._draw)
        self.canvas.bind("<ButtonRelease-1>",self._stop_draw)

    def _start_draw(self, event):
        self._drawing = True
        self._last_x  = event.x
        self._last_y  = event.y

    def _draw(self, event):
        if not self._drawing:
            return
        x, y = event.x, event.y
        self.canvas.create_line(
            self._last_x, self._last_y, x, y,
            width=self.line_width, fill='black',
            capstyle=tk.ROUND, smooth=True
        )
        self._draw_obj.line(
            [self._last_x, self._last_y, x, y],
            fill='black', width=self.line_width
        )
        self._last_x, self._last_y = x, y

    def _stop_draw(self, event):
        self._drawing = False

    def clear(self):
        """Clears the canvas and resets the PIL image."""
        self.canvas.delete("all")
        self._pil_img  = Image.new("RGB", (self.size, self.size), "white")
        self._draw_obj = ImageDraw.Draw(self._pil_img)

    def get_image(self) -> Image.Image:
        """Returns a copy of the current PIL image."""
        return self._pil_img.copy()

    def is_blank(self) -> bool:
        """Returns True if the canvas is blank."""
        arr = np.array(self._pil_img.convert('L'))
        return arr.min() > 250


# ═══════════════════════════════════════════════════════════════
# Signature Collection Window
# ═══════════════════════════════════════════════════════════════

class SignatureCollector:
    """
    Window for collecting signature samples from one person.
    Uses the shared SignatureCanvasWidget.
    """

    def __init__(self, parent, person_name: str,
                 callback=None, num_samples: int = NUM_SAMPLES):
        self.parent      = parent
        self.person_name = person_name
        self.num_samples = num_samples
        self.collected   = 0
        self.images      = []
        self.callback    = callback

        self.window = tk.Toplevel(parent)
        self.window.title(f"Collect Signatures — {person_name}")
        self.window.resizable(False, False)
        self.window.grab_set()

        # ── Header ──────────────────────────────────────────────────
        header = tk.Frame(self.window, bg="#1565C0", pady=10)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text=f"📝  Collecting Signatures: {person_name}",
            bg="#1565C0", fg="white",
            font=("Arial", 14, "bold")
        ).pack()

        # ── Drawing canvas ─────────────────────────────────────────
        self.canvas_widget = SignatureCanvasWidget(self.window)
        self.canvas_widget.pack(pady=10, padx=20)

        # ── Progress bar ───────────────────────────────────────────
        prog_frame = tk.Frame(self.window)
        prog_frame.pack(fill=tk.X, padx=20, pady=(0, 5))
        self.progress = ttk.Progressbar(
            prog_frame, maximum=num_samples,
            mode='determinate', length=CANVAS_SIZE
        )
        self.progress.pack(fill=tk.X)

        # ── Instructions ───────────────────────────────────────────
        self.instr = tk.Label(
            self.window,
            text=self._instr_text(),
            font=("Arial", 11),
            fg="#1565C0"
        )
        self.instr.pack(pady=5)

        # ── Buttons ─────────────────────────────────────────────────
        btn_frame = tk.Frame(self.window, pady=8)
        btn_frame.pack()

        tk.Button(
            btn_frame, text="🧹 Clear",
            command=self._clear,
            width=12, bg="#FF9800", fg="white",
            font=("Arial", 11), relief=tk.FLAT
        ).pack(side=tk.LEFT, padx=6)

        self.save_btn = tk.Button(
            btn_frame, text="💾 Save Signature",
            command=self._save_one,
            width=14, bg="#4CAF50", fg="white",
            font=("Arial", 11, "bold"), relief=tk.FLAT
        )
        self.save_btn.pack(side=tk.LEFT, padx=6)

        self.done_btn = tk.Button(
            btn_frame, text="✅ Done",
            command=self._done,
            width=12, bg="#1565C0", fg="white",
            font=("Arial", 11), relief=tk.FLAT,
            state=tk.DISABLED
        )
        self.done_btn.pack(side=tk.LEFT, padx=6)

        center_window(self.window, CANVAS_SIZE + 60, CANVAS_SIZE + 220)

    def _instr_text(self) -> str:
        remaining = self.num_samples - self.collected
        return (
            f"Signature {self.collected + 1} of {self.num_samples}  "
            f"— Remaining: {remaining}"
        )

    def _clear(self):
        self.canvas_widget.clear()

    def _save_one(self):
        if self.canvas_widget.is_blank():
            messagebox.showwarning(
                "Warning", "Canvas is empty! Please draw a signature first.",
                parent=self.window
            )
            return

        if self.collected >= self.num_samples:
            messagebox.showwarning(
                "Warning", "You have already collected all required signatures.",
                parent=self.window
            )
            return

        self.images.append(self.canvas_widget.get_image())
        self.collected += 1
        self.progress['value'] = self.collected
        self.instr.config(text=self._instr_text())
        self._clear()

        if self.collected == self.num_samples:
            self.save_btn.config(state=tk.DISABLED)
            self.done_btn.config(state=tk.NORMAL)
            messagebox.showinfo(
                "Collection Complete",
                f"✅ {self.num_samples} signatures collected.\nPress Done to save.",
                parent=self.window
            )

    def _done(self):
        if self.collected < self.num_samples:
            messagebox.showwarning(
                "Warning",
                f"Please collect {self.num_samples} signatures first.\n"
                f"You have {self.collected} so far.",
                parent=self.window
            )
            return

        person_dir = os.path.join(DATA_DIR, self.person_name)
        ensure_dir(person_dir)

        for i, img in enumerate(self.images):
            path = os.path.join(person_dir, f"sig_{i + 1:02d}.png")
            img.save(path)
            logger.info("Saved signature: %s", path)

        self.window.destroy()

        messagebox.showinfo(
            "Saved",
            f"\u2705 Saved {self.num_samples} signatures for: {self.person_name}\\n"
            f"Folder: {person_dir}\\n\\n"
            f"\u26a0\ufe0f  Remember to retrain the models to include this new person!",
            parent=self.parent
        )

        if self.callback:
            self.callback()


# ═══════════════════════════════════════════════════════════════
# Main Application Class
# ═══════════════════════════════════════════════════════════════

class ForgeryDetectionApp:
    """
    Main application for signature forgery detection.

    Combines:
    - Multi-person classifier (SVM / KNN / Naive Bayes)
    - One-Class SVM models for genuine forgery detection
    - Minimum confidence threshold before declaring a result
    - Asynchronous training in a background thread
    """

    def __init__(self, root: tk.Tk):
        self.root  = root
        self.model         = None
        self.scaler        = None
        self.selector      = None
        self.class_names   = []
        self.oc_models     = {}   # person -> OneClassSVM
        self.chosen_algorithm = None
        self._model_dirty  = False  # True if new person added since last training

        self._build_ui()
        ensure_dir(DATA_DIR)
        ensure_dir(MODELS_DIR)
        ensure_dir(OC_MODELS_DIR)
        self._try_load_models()
        self._update_status()

    # ── Build UI ─────────────────────────────────────────────────

    def _build_ui(self):
        self.root.title("Signature Forgery Detection System")
        self.root.resizable(False, False)
        w, h = screen_fraction(self.root, 0.35, 0.60)
        w, h = max(w, 500), max(h, 500)
        center_window(self.root, w, h)

        # ── Header ──────────────────────────────────────────────────
        header = tk.Frame(self.root, bg="#0D47A1", pady=18)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text="🔐  Signature Forgery Detection System",
            bg="#0D47A1", fg="white",
            font=("Arial", 18, "bold")
        ).pack()

        # ── Main Buttons ────────────────────────────────────────────
        body = tk.Frame(self.root, bg="#F5F5F5", pady=20)
        body.pack(fill=tk.BOTH, expand=True)

        btn_cfg = [
            ("➕  Add New Person",           "#388E3C", self._add_person),
            ("🧠  Train Models",             "#1565C0", self._train_models),
            ("📂  Train from External Dataset","#00695C", self._train_from_external),
            ("🔍  Recognise / Detect Forgery","#E65100", self._recognize_signature),
            ("👥  Manage People",            "#6A1B9A", self._manage_people),
            ("🚪  Exit",                     "#B71C1C", self.root.quit),
        ]

        for text, color, cmd in btn_cfg:
            tk.Button(
                body, text=text, command=cmd,
                width=30, height=2,
                bg=color, fg="white",
                font=("Arial", 12, "bold"),
                relief=tk.FLAT, cursor="hand2",
                activebackground=color
            ).pack(pady=6)

        # ── Status bar ──────────────────────────────────────────────
        status_frame = tk.Frame(self.root, bg="#E3F2FD", pady=8, padx=12)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_lbl = tk.Label(
            status_frame, text="",
            bg="#E3F2FD", fg="#0D47A1",
            font=("Arial", 10), wraplength=460, justify=tk.RIGHT
        )
        self.status_lbl.pack()

    # ── Update status bar ───────────────────────────────────────

    def _update_status(self):
        people = list_people()
        n      = len(people)
        model_ok = self.model is not None

        if n == 0:
            msg = "No people registered yet. Start by adding a new person."
            color = "#B71C1C"
        elif n == 1:
            msg = f"Only one person ({people[0]}). Add another person to enable training."
            color = "#E65100"
        else:
            model_status = "✅ Model loaded" if model_ok else "⚠️ Not trained yet"
            dirty_warn   = " | ⚠️ New people added — please retrain" if self._model_dirty else ""
            msg = f"{n} people: {', '.join(people)}\n{model_status}{dirty_warn}"
            color = "#1B5E20" if (model_ok and not self._model_dirty) else "#E65100"

        self.status_lbl.config(text=msg, fg=color)

    # ── Load models on startup ──────────────────────────────────

    def _try_load_models(self):
        """Attempts to load saved models automatically at startup."""
        files = [MODEL_FILE, SCALER_FILE, SELECTOR_FILE, CLASS_NAMES_FILE]
        if not all(os.path.exists(f) for f in files):
            logger.info("No saved models found.")
            return
        try:
            # ── Detect feature-count mismatch ─────────────────────
            scaler_tmp = joblib.load(SCALER_FILE)
            expected_n = getattr(scaler_tmp, "n_features_in_", None)
            if expected_n is not None and expected_n != N_FEATURES:
                logger.warning(
                    "Mismatch: saved model expects %d features, current code produces %d."
                    " Old model files will be deleted.", expected_n, N_FEATURES
                )
                for f_ in [MODEL_FILE, SCALER_FILE, SELECTOR_FILE, CLASS_NAMES_FILE]:
                    try:
                        if os.path.exists(f_): os.remove(f_)
                    except Exception: pass
                return  # Force user to retrain

            self.model    = joblib.load(MODEL_FILE)
            self.scaler   = scaler_tmp
            self.selector = joblib.load(SELECTOR_FILE)
            with open(CLASS_NAMES_FILE, "r", encoding="utf-8") as f:
                self.class_names = [ln.strip() for ln in f if ln.strip()]
            self.chosen_algorithm = "Loaded from file"
            # Load One-Class models
            for person in self.class_names:
                oc_path = os.path.join(OC_MODELS_DIR, f"{person}.pkl")
                if os.path.exists(oc_path):
                    self.oc_models[person] = joblib.load(oc_path)
            logger.info(
                "Models loaded. Algorithm: %s | People: %s",
                self.chosen_algorithm,
                self.class_names
            )
        except Exception as exc:
            logger.error("Failed to load models: %s", exc)
            self.model = None

    # ── Add New Person ───────────────────────────────────────────

    def _add_person(self):
        name = simpledialog.askstring(
            "Add Person", "Enter the person's name:",
            parent=self.root
        )
        if not name:
            return
        name = name.strip()
        if not name:
            messagebox.showwarning("Warning", "Name cannot be empty.", parent=self.root)
            return
        if os.path.exists(os.path.join(DATA_DIR, name)):
            messagebox.showwarning(
                "Warning",
                f"Person '{name}' already exists.",
                parent=self.root
            )
            return

        def _after_collect():
            self._model_dirty = True
            self._update_status()

        SignatureCollector(
            self.root, name,
            callback=_after_collect,
            num_samples=NUM_SAMPLES
        )

    # ── Manage People ────────────────────────────────────────────

    def _manage_people(self):
        """Window for viewing and deleting registered people."""
        people = list_people()
        if not people:
            messagebox.showinfo("Information", "No people registered yet.", parent=self.root)
            return

        win = tk.Toplevel(self.root)
        win.title("Manage People")
        win.grab_set()
        center_window(win, 380, 400)

        tk.Label(win, text="Registered People",
                 font=("Arial", 13, "bold")).pack(pady=12)

        listbox = tk.Listbox(win, font=("Arial", 11), width=35, height=14,
                             selectmode=tk.SINGLE)
        listbox.pack(padx=20)
        for p in people:
            sigs = len([
                f for f in os.listdir(os.path.join(DATA_DIR, p))
                if f.lower().endswith('.png')
            ])
            listbox.insert(tk.END, f"{p}  ({sigs} signatures)")

        def _delete():
            sel = listbox.curselection()
            if not sel:
                messagebox.showwarning("Warning", "Please select a person first.", parent=win)
                return
            person = people[sel[0]]
            if not messagebox.askyesno(
                "Confirm",
                f"Delete '{person}' and all their signatures?\\n"
                "This action cannot be undone.",
                parent=win
            ):
                return
            import shutil
            shutil.rmtree(os.path.join(DATA_DIR, person), ignore_errors=True)
            oc_path = os.path.join(OC_MODELS_DIR, f"{person}.pkl")
            if os.path.exists(oc_path):
                os.remove(oc_path)
            self._model_dirty = True
            messagebox.showinfo(
                "Done",
                f"✅ Deleted '{person}'.\n⚠️ Please retrain the models.",
                parent=win
            )
            win.destroy()
            self._update_status()

        tk.Button(
            win, text="🗑️  Delete Selected Person",
            command=_delete,
            bg="#B71C1C", fg="white",
            font=("Arial", 11), relief=tk.FLAT,
            width=22, pady=6
        ).pack(pady=10)

    # ── Train Models ─────────────────────────────────────────────

    def _train_models(self):
        """Starts training in a background thread to prevent UI freeze."""
        people = list_people()
        if len(people) < 2:
            messagebox.showerror(
                "Error",
                "At least two people are required for training.",
                parent=self.root
            )
            return

        # ── Progress window ──────────────────────────────────────────
        prog_win = tk.Toplevel(self.root)
        prog_win.title("Training in progress...")
        prog_win.protocol("WM_DELETE_WINDOW", lambda: None)
        center_window(prog_win, 420, 180)

        tk.Label(prog_win, text="🧠  Training models...",
                 font=("Arial", 13, "bold")).pack(pady=14)
        prog_bar = ttk.Progressbar(prog_win, mode='indeterminate', length=360)
        prog_bar.pack(padx=20)
        prog_bar.start(12)
        self._status_lbl_train = tk.Label(prog_win, text="Extracting features...",
                                          font=("Arial", 10))
        self._status_lbl_train.pack(pady=8)

        def _update_train_status(msg):
            try:
                self._status_lbl_train.config(text=msg)
                prog_win.update_idletasks()
            except Exception:
                pass

        def _run_training():
            try:
                result = self._do_training(_update_train_status)
            except Exception as exc:
                logger.error("Training error: %s", exc, exc_info=True)
                result = {"error": str(exc)}
            self.root.after(0, lambda: self._on_training_done(prog_win, result))

        threading.Thread(target=_run_training, daemon=True).start()

    def _do_training(self, status_cb) -> dict:
        """
        Actual training function — runs in a background thread.
        Returns a dict with training results or an 'error' key.
        """
        status_cb("Extracting features from images...")
        X, y, class_names = self._extract_all_features()
        if X is None:
            return {"error": "No valid images found."}

        unique = np.unique(y)
        if len(unique) < 2:
            return {"error": "At least two different people are required."}

        min_per_class = int(np.bincount(y).min())
        if min_per_class < 2:
            return {"error": f"Each person needs at least 2 samples. Minimum found: {min_per_class}."}

        status_cb("Normalising features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ── Select optimal k via cross-validation ─────────────
        status_cb("Selecting optimal number of features...")
        best_k, best_cv_score = self._select_best_k(X_scaled, y)
        logger.info("Best k=%d with CV accuracy=%.4f", best_k, best_cv_score)

        selector = SelectKBest(f_classif, k=best_k)

        # ── Safe train/test split (small dataset) ─────────────
        n_samples = len(y)
        if n_samples < 10:
            # CV only, no split
            X_sel = selector.fit_transform(X_scaled, y)
            X_train_sel = X_sel
            X_test_sel  = X_sel
            y_train = y_test = y
            use_cv_only = True
        else:
            test_sz = max(0.15, min(0.25, 2 / n_samples))
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y,
                test_size=test_sz,
                random_state=42,
                stratify=y
            )
            X_train_sel = selector.fit_transform(X_train, y_train)
            X_test_sel  = selector.transform(X_test)
            use_cv_only = False

        # ── Train three classifiers ──────────────────────────────
        status_cb("Training classifiers (SVM / KNN / Naive Bayes)...")
        models = {
            'SVM':        SVC(kernel='rbf', gamma='scale', probability=True, C=5),
            'KNN':        KNeighborsClassifier(n_neighbors=max(3, min_per_class)),
            'Naive Bayes': GaussianNB()
        }
        trained  = {}
        results  = {}
        for name, clf in models.items():
            clf.fit(X_train_sel, y_train)
            if use_cv_only:
                # CV for scarce data
                n_splits = min(5, min_per_class)
                cv_acc   = cross_val_score(
                    clf, X_train_sel, y_train,
                    cv=StratifiedKFold(n_splits=n_splits),
                    scoring='accuracy'
                ).mean()
                acc = cv_acc
                trained[name] = (clf, acc, None)
                results[name] = acc
            else:
                y_pred = clf.predict(X_test_sel)
                acc    = accuracy_score(y_test, y_pred)
                trained[name] = (clf, acc, y_pred)
                results[name] = acc
                logger.info(
                    "%s — accuracy: %.4f\n%s",
                    name, acc,
                    classification_report(
                        y_test, y_pred,
                        target_names=class_names,
                        zero_division=0
                    )
                )

        # ── Train One-Class SVM models for forgery detection ──────
        status_cb("Training forgery-detection models (One-Class SVM)...")
        oc_models = {}
        X_sel_full = selector.transform(X_scaled)
        for label, person in enumerate(class_names):
            idx = np.where(y == label)[0]
            if len(idx) >= 3:
                oc = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
                oc.fit(X_sel_full[idx])
                oc_models[person] = oc
                logger.info("One-Class SVM trained for %s (%d samples)", person, len(idx))

        return {
            "trained":     trained,
            "results":     results,
            "scaler":      scaler,
            "selector":    selector,
            "class_names": class_names,
            "oc_models":   oc_models,
            "use_cv":      use_cv_only,
        }

    def _select_best_k(self, X_scaled: np.ndarray, y: np.ndarray) -> tuple:
        """
        Selects the optimal k for SelectKBest.
        Target: K_SELECT=30 features from 152.
        If sufficient data, CV searches a narrow range [25-35]
        for best accuracy; otherwise k=30 is used directly.
        Returns (best_k, best_score).
        """
        n_features = X_scaled.shape[1]
        best_k     = min(K_SELECT, n_features)
        best_score = -1.0

        n_classes  = len(np.unique(y))
        n_splits   = min(5, max(2, len(y) // n_classes))

        if len(y) >= n_classes * n_splits * 2:
            cv        = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            probe_clf = SVC(kernel='rbf', probability=True, gamma='scale')
            k_lo = max(K_SELECT - 5, 5)
            k_hi = min(K_SELECT + 6, n_features + 1)
            for k in range(k_lo, k_hi):
                sel = SelectKBest(f_classif, k=k)
                try:
                    X_sel  = sel.fit_transform(X_scaled, y)
                    scores = cross_val_score(
                        probe_clf, X_sel, y, cv=cv, scoring='accuracy'
                    )
                    mean_s = float(scores.mean())
                    if mean_s > best_score:
                        best_score = mean_s
                        best_k     = k
                except Exception:
                    continue
            logger.info("CV selected k=%d (range %d-%d) accuracy=%.4f",
                        best_k, k_lo, k_hi - 1, best_score)
        else:
            logger.info("Scarce data -> fixed k = %d", best_k)

        return best_k, best_score

    def _extract_all_features(self):
        """Extracts features from all images in the database."""
        people = list_people()
        if not people:
            return None, None, None

        X, y = [], []
        class_names = people

        for label, person in enumerate(people):
            person_dir = os.path.join(DATA_DIR, person)
            images = [
                f for f in os.listdir(person_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            if not images:
                logger.warning("Folder %s contains no images.", person)
                continue

            for fname in images:
                path = os.path.join(person_dir, fname)
                try:
                    img     = Image.open(path).convert('RGB')
                    binary  = preprocess_signature(img)
                    feats   = extract_features(binary)
                    X.append(feats)
                    y.append(label)
                except Exception as exc:
                    logger.error("Error reading %s: %s", path, exc)

        if not X:
            return None, None, None

        return np.array(X), np.array(y), class_names

    # ── Train from External Dataset ──────────────────────────────────

    def _extract_features_from_folder(self, root_dir: str,
                                       status_cb) -> tuple:
        """
        Scans root_dir for sub-folders (one per person) and extracts
        features from every image inside them.
        Also COPIES all images into DATA_DIR so they are merged with
        the local database and become available for future training.

        Expected layout:
            root_dir/
                PersonA/  ← sub-folder name = person name
                    sig_01.png
                    sig_02.png
                    ...
                PersonB/
                    ...

        Returns (X, y, class_names) or (None, None, None) on failure.
        """
        if not os.path.isdir(root_dir):
            return None, None, None

        people = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )
        if not people:
            return None, None, None

        IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        X, y, class_names = [], [], []

        for label, person in enumerate(people):
            src_person_dir = os.path.join(root_dir, person)
            dst_person_dir = os.path.join(DATA_DIR, person)
            ensure_dir(dst_person_dir)

            images = [
                f for f in os.listdir(src_person_dir)
                if f.lower().endswith(IMG_EXTS)
            ]
            if not images:
                logger.warning("External folder %s has no images.", src_person_dir)
                continue

            status_cb(f"Processing person {label+1}/{len(people)}: {person} "
                      f"({len(images)} images)...")

            copied = 0
            for fname in images:
                src_path = os.path.join(src_person_dir, fname)
                dst_path = os.path.join(dst_person_dir, fname)
                try:
                    img    = Image.open(src_path).convert('RGB')
                    binary = preprocess_signature(img)
                    feats  = extract_features(binary)
                    X.append(feats)
                    y.append(label)
                    class_names_tmp = person   # track
                    # Copy image to local DATA_DIR (skip if already there)
                    if not os.path.exists(dst_path):
                        img.save(dst_path)
                        copied += 1
                except Exception as exc:
                    logger.error("Error reading %s: %s", src_path, exc)

            class_names.append(person)
            logger.info("External — %s: %d features extracted, %d images copied",
                        person, sum(1 for lb in y if lb == label),
                        copied)

        if not X:
            return None, None, None
        return np.array(X), np.array(y), class_names

    def _train_from_external(self):
        """
        Opens a folder-browser dialog so the user can select the root
        dataset folder. Defaults to the pre-configured path but can be changed freely.

        After selecting, shows a summary of what was found, then trains
        exactly like the normal training flow.
        exactly like the normal training flow.
        """
        DEFAULT_PATH = os.path.join("C:\\", "Users", "Ashraf", "Desktop", "منتظر", "data")


        # ── Ask user to confirm / change the path ─────────────────
        dlg = tk.Toplevel(self.root)
        dlg.title("Train from External Dataset")
        center_window(dlg, 580, 260)
        dlg.grab_set()

        tk.Label(dlg,
                 text="📂  Train from External Dataset",
                 font=("Arial", 13, "bold"), pady=10).pack()

        tk.Label(dlg,
                 text="Select the root folder that contains one sub-folder per person.\n"
                      "Images will be copied into the local database automatically.",
                 font=("Arial", 10), fg="#555", wraplength=520,
                 justify=tk.CENTER).pack(pady=(0, 8))

        path_var = tk.StringVar(value=DEFAULT_PATH)

        path_frame = tk.Frame(dlg)
        path_frame.pack(fill=tk.X, padx=20, pady=4)

        path_entry = tk.Entry(path_frame, textvariable=path_var,
                              font=("Arial", 10), width=52,
                              relief=tk.SOLID, bd=1)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        def _browse():
            chosen = filedialog.askdirectory(
                title="Select dataset root folder",
                initialdir=DEFAULT_PATH if os.path.isdir(DEFAULT_PATH)
                           else os.path.expanduser("~"),
                parent=dlg
            )
            if chosen:
                path_var.set(chosen)

        tk.Button(path_frame, text="🗁 Browse",
                  font=("Arial", 10), relief=tk.FLAT,
                  bg="#1565C0", fg="white", padx=8, pady=3,
                  cursor="hand2", command=_browse).pack(side=tk.LEFT, padx=(6, 0))

        # ── Preview: scan folder and show what was found ──────────
        preview_lbl = tk.Label(dlg, text="", font=("Arial", 9),
                               fg="#1B5E20", wraplength=520)
        preview_lbl.pack(pady=4)

        def _preview(*_):
            p = path_var.get().strip()
            if not os.path.isdir(p):
                preview_lbl.config(text="⚠️  Folder not found.", fg="#B71C1C")
                return
            people = sorted(d for d in os.listdir(p)
                            if os.path.isdir(os.path.join(p, d)))
            if not people:
                preview_lbl.config(text="⚠️  No sub-folders found.", fg="#B71C1C")
                return
            IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
            total_imgs = sum(
                len([f for f in os.listdir(os.path.join(p, person))
                     if f.lower().endswith(IMG_EXTS)])
                for person in people
            )
            preview_lbl.config(
                text=f"✅  Found {len(people)} people — "
                     f"{total_imgs} total images: {', '.join(people[:8])}"
                     + (" ..." if len(people) > 8 else ""),
                fg="#1B5E20"
            )

        path_var.trace_add("write", _preview)
        _preview()   # initial scan

        result_holder = [None]   # [path] or [None] if cancelled

        btn_row = tk.Frame(dlg)
        btn_row.pack(pady=10)

        def _start():
            p = path_var.get().strip()
            if not os.path.isdir(p):
                messagebox.showerror("Error",
                    f"Folder not found:\n{p}", parent=dlg)
                return
            result_holder[0] = p
            dlg.destroy()

        def _cancel():
            dlg.destroy()

        tk.Button(btn_row, text="🚀  Start Training",
                  font=("Arial", 11, "bold"), relief=tk.FLAT,
                  bg="#00695C", fg="white", padx=16, pady=6,
                  cursor="hand2", command=_start).pack(side=tk.LEFT, padx=8)

        tk.Button(btn_row, text="Cancel",
                  font=("Arial", 11), relief=tk.FLAT,
                  bg="#757575", fg="white", padx=16, pady=6,
                  cursor="hand2", command=_cancel).pack(side=tk.LEFT, padx=8)

        self.root.wait_window(dlg)

        chosen_path = result_holder[0]
        if chosen_path is None:
            return   # user cancelled

        # ── Progress window ───────────────────────────────────────
        prog_win = tk.Toplevel(self.root)
        prog_win.title("Training from External Dataset...")
        prog_win.protocol("WM_DELETE_WINDOW", lambda: None)
        center_window(prog_win, 460, 200)

        tk.Label(prog_win,
                 text="📂  Loading & training from external dataset...",
                 font=("Arial", 12, "bold")).pack(pady=14)
        prog_bar = ttk.Progressbar(prog_win, mode='indeterminate', length=400)
        prog_bar.pack(padx=20)
        prog_bar.start(12)
        self._status_lbl_train = tk.Label(prog_win,
                                          text="Scanning folder...",
                                          font=("Arial", 10))
        self._status_lbl_train.pack(pady=8)

        def _update_status(msg):
            try:
                self._status_lbl_train.config(text=msg)
                prog_win.update_idletasks()
            except Exception:
                pass

        def _run():
            try:
                result = self._do_training_on_folder(chosen_path, _update_status)
            except Exception as exc:
                logger.error("External training error: %s", exc, exc_info=True)
                result = {"error": str(exc)}
            self.root.after(0, lambda: self._on_training_done(prog_win, result))

        threading.Thread(target=_run, daemon=True).start()

    def _do_training_on_folder(self, folder_path: str, status_cb) -> dict:
        """
        Full training pipeline using an external folder as the data source.
        Images are copied to DATA_DIR and then the normal pipeline runs.
        """
        status_cb("Scanning and extracting features from external folder...")
        X, y, class_names = self._extract_features_from_folder(
            folder_path, status_cb
        )
        if X is None:
            return {"error": f"No valid images found in:\n{folder_path}"}

        unique = np.unique(y)
        if len(unique) < 2:
            return {"error": "At least two different people are required."}

        min_per_class = int(np.bincount(y).min())
        if min_per_class < 2:
            return {"error": f"Each person needs at least 2 samples. "
                             f"Minimum found: {min_per_class}."}

        status_cb("Normalising features...")
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        status_cb("Selecting optimal number of features...")
        best_k, best_cv_score = self._select_best_k(X_scaled, y)
        logger.info("External training — best k=%d accuracy=%.4f",
                    best_k, best_cv_score)

        selector = SelectKBest(f_classif, k=best_k)

        n_samples = len(y)
        use_cv_only = n_samples < 10
        if use_cv_only:
            selector.fit(X_scaled, y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            selector.fit(X_train, y_train)
            X_test_sel = selector.transform(X_test)

        X_sel_train = selector.transform(
            X_scaled if use_cv_only else X_train
        )

        # ── Train three classifiers ──────────────────────────────
        status_cb("Training classifiers (SVM / KNN / Naive Bayes)...")
        classifiers = {
            "SVM":         SVC(kernel='rbf', probability=True,
                               class_weight='balanced', gamma='scale'),
            "KNN":         KNeighborsClassifier(n_neighbors=3),
            "Naive Bayes": GaussianNB(),
        }
        trained, results = {}, {}
        cv_folds = StratifiedKFold(
            n_splits=min(5, max(2, n_samples // len(class_names))),
            shuffle=True, random_state=42
        )

        for name, clf in classifiers.items():
            status_cb(f"Training {name}...")
            if use_cv_only:
                scores  = cross_val_score(clf, X_sel_train, y,
                                          cv=cv_folds, scoring='accuracy')
                acc     = float(scores.mean())
                y_pred  = y.copy()
                clf.fit(X_sel_train, y)
            else:
                clf.fit(X_sel_train, y_train)
                y_pred  = clf.predict(X_test_sel)
                acc     = accuracy_score(y_test, y_pred)
            trained[name] = (clf, acc, y_pred)
            results[name] = acc
            logger.info("External %s — accuracy: %.4f", name, acc)

        # ── One-Class SVM ────────────────────────────────────────
        status_cb("Training forgery-detection models (One-Class SVM)...")
        oc_models   = {}
        X_sel_full  = selector.transform(X_scaled)
        for label_idx, person in enumerate(class_names):
            idx = np.where(y == label_idx)[0]
            if len(idx) >= 3:
                oc = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
                oc.fit(X_sel_full[idx])
                oc_models[person] = oc
                logger.info("One-Class SVM trained for %s (%d samples)",
                            person, len(idx))

        status_cb("Done!")
        return {
            "trained":     trained,
            "results":     results,
            "scaler":      scaler,
            "selector":    selector,
            "class_names": class_names,
            "oc_models":   oc_models,
            "use_cv":      use_cv_only,
            "source":      folder_path,
        }

    def _on_training_done(self, prog_win: tk.Toplevel, result: dict):
        """Handles training result in the main thread."""
        try:
            prog_win.destroy()
        except Exception:
            pass

        if "error" in result:
            messagebox.showerror("Training Error", result["error"], parent=self.root)
            return

        trained     = result["trained"]
        results     = result["results"]
        class_names = result["class_names"]
        use_cv      = result.get("use_cv", False)

        # ── Display results and select algorithm ─────────────────
        results_str = "\n".join(
            f"{'[CV] ' if use_cv else ''}{n}: {acc:.2%}"
            for n, acc in results.items()
        )
        note = "\n(* cross-validation accuracy — scarce data)" if use_cv else ""
        msg  = f"Model accuracies:\n\n{results_str}{note}\n\nSelect an algorithm:"

        # Button selection — avoids askstring returning None
        choice = self._ask_algorithm_choice(trained.keys())
        if choice is None:
            # User cancelled -> auto-select best
            choice = max(results, key=results.get)
            messagebox.showinfo(
                "Information",
                f"Best algorithm selected automatically: {choice} ({results[choice]:.2%})",
                parent=self.root
            )

        # ── Save models ────────────────────────────────────────────
        ensure_dir(MODELS_DIR)
        ensure_dir(OC_MODELS_DIR)

        self.model        = trained[choice][0]
        self.scaler       = result["scaler"]
        self.selector     = result["selector"]
        self.class_names  = class_names
        self.oc_models    = result["oc_models"]
        self.chosen_algorithm = choice
        self._model_dirty = False

        joblib.dump(self.model,    MODEL_FILE)
        joblib.dump(self.scaler,   SCALER_FILE)
        joblib.dump(self.selector, SELECTOR_FILE)
        with open(CLASS_NAMES_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(class_names))
        for person, oc in self.oc_models.items():
            joblib.dump(oc, os.path.join(OC_MODELS_DIR, f"{person}.pkl"))

        logger.info("Models saved. Algorithm: %s", choice)

        messagebox.showinfo(
            "Training Complete",
            f"✅ Models trained and saved.\n\n{results_str}\n\nUsing: {choice}",
            parent=self.root
        )
        self._update_status()

    def _ask_algorithm_choice(self, options) -> str | None:
        """Algorithm selection dialog with clear buttons."""
        choice_var = [None]

        dlg = tk.Toplevel(self.root)
        dlg.title("Select Algorithm")
        dlg.grab_set()
        center_window(dlg, 340, 220)

        tk.Label(
            dlg,
            text="Select recognition algorithm:",
            font=("Arial", 12, "bold")
        ).pack(pady=14)

        colors = {"SVM": "#1565C0", "KNN": "#2E7D32", "Naive Bayes": "#6A1B9A"}

        for opt in options:
            def _pick(o=opt):
                choice_var[0] = o
                dlg.destroy()

            tk.Button(
                dlg, text=opt, command=_pick,
                width=20, bg=colors.get(opt, "#455A64"),
                fg="white", font=("Arial", 11),
                relief=tk.FLAT, pady=6
            ).pack(pady=4)

        self.root.wait_window(dlg)
        return choice_var[0]

    # ── Signature Recognition and Forgery Detection ────────────

    def _recognize_signature(self):
        """Opens the drawing and recognition window."""
        if self.model is None:
            messagebox.showerror(
                    "Error",
                    "No trained model found or version mismatch.\\n\\n"
                    "Click  '\U0001f9e0 Train Models'  first\\n"
                    "to create a new model compatible with 152 features.",
                    parent=self.root
                )
            return

        if self._model_dirty:
            if not messagebox.askyesno(
                "Warning",
                "\u26a0\ufe0f New people were added since the last training.\\n"
                "Results may be inaccurate.\\n"
                "Do you want to continue anyway?",
                parent=self.root
            ):
                return

        self._open_recognition_window()

    def _open_recognition_window(self):
        win = tk.Toplevel(self.root)
        win.title("Signature Recognition and Forgery Detection")

        # Proportional to screen size
        sw, sh = screen_fraction(self.root, 0.92, 0.90)
        center_window(win, sw, sh)

        # ── Drawing section ─────────────────────────────────────────
        left = tk.Frame(win, bg="#FAFAFA", bd=2, relief=tk.GROOVE)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Label(
            left, text="✏️  Draw Signature Here",
            font=("Arial", 12, "bold"), bg="#FAFAFA"
        ).pack(pady=(10, 4))

        canvas_widget = SignatureCanvasWidget(left)
        canvas_widget.pack(padx=10, pady=6)

        btn_row = tk.Frame(left, bg="#FAFAFA")
        btn_row.pack(pady=8)

        tk.Button(
            btn_row, text="🧹 Clear",
            command=canvas_widget.clear,
            bg="#FF9800", fg="white",
            font=("Arial", 11), relief=tk.FLAT,
            width=10, pady=5
        ).pack(side=tk.LEFT, padx=5)

        predict_btn = tk.Button(
            btn_row, text="🔍 Analyse",
            bg="#1B5E20", fg="white",
            font=("Arial", 12, "bold"), relief=tk.FLAT,
            width=12, pady=5,
            cursor="hand2"
        )
        predict_btn.pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_row, text="❌ Close",
            command=win.destroy,
            bg="#B71C1C", fg="white",
            font=("Arial", 11), relief=tk.FLAT,
            width=10, pady=5
        ).pack(side=tk.LEFT, padx=5)

        # ── Results (processing steps) ──────────────────────────────
        right = tk.Frame(win)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=10)

        tk.Label(
            right, text="📊  Processing Steps and Result",
            font=("Arial", 12, "bold")
        ).pack(pady=(6, 4))

        steps_frame = tk.Frame(right)
        steps_frame.pack(fill=tk.BOTH, expand=True)

        col_names = [
            "Grayscale", "Binary",
            "Features", "Selected Features", "Result"
        ]
        # First two columns (Grayscale/Binary) use THUMB_BIG
        BIG_COLS  = {"Grayscale", "Binary"}
        step_labels = {}

        for i, cname in enumerate(col_names):
            is_big = cname in BIG_COLS
            tsize  = THUMB_BIG if is_big else THUMB_SIZE

            subf = tk.Frame(steps_frame, relief=tk.RIDGE, bd=2, bg="white")
            subf.grid(row=0, column=i, padx=4, pady=4, sticky="nsew")
            steps_frame.columnconfigure(i, weight=2 if is_big else 1)

            # Column header
            hdr_color = "#1565C0" if is_big else "#E3F2FD"
            hdr_fg    = "white"   if is_big else "#212121"
            tk.Label(
                subf, text=cname,
                font=("Arial", 12 if is_big else 11, "bold"),
                bg=hdr_color, fg=hdr_fg, pady=5
            ).pack(fill=tk.X)

            # Image container — fixed size
            img_frame = tk.Frame(subf, bg="#ECEFF1",
                                 width=tsize, height=tsize)
            img_frame.pack_propagate(False)
            img_frame.pack(pady=4, padx=4)

            img_lbl = tk.Label(img_frame, bg="#ECEFF1")
            img_lbl.place(relx=0.5, rely=0.5, anchor="center")

            # Zoom button for large images
            if is_big:
                def _make_zoom_btn(lbl_ref, title_ref):
                    def _zoom():
                        ph = getattr(lbl_ref, "image", None)
                        if ph is None:
                            return
                        popup = tk.Toplevel(win)
                        popup.title(title_ref)
                        center_window(popup, CANVAS_SIZE + 40, CANVAS_SIZE + 40)
                        full_lbl = tk.Label(popup)
                        full_lbl.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                        # Store full-res image in zoom_image attribute
                        zoom_ph = getattr(lbl_ref, "zoom_image", ph)
                        full_lbl.config(image=zoom_ph)
                        full_lbl.image = zoom_ph
                    return _zoom
                tk.Button(
                    subf, text="🔍 Zoom",
                    font=("Arial", 9), relief=tk.FLAT,
                    bg="#1565C0", fg="white", pady=2, padx=6,
                    cursor="hand2",
                    command=_make_zoom_btn(img_lbl, cname)
                ).pack(pady=(0, 4))

            txt_frame = tk.Frame(subf)
            txt_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

            txt_h = 6 if is_big else 14
            txt_w = 30 if is_big else 22
            txt = tk.Text(
                txt_frame, height=txt_h, width=txt_w,
                font=("Courier", 9 if is_big else 8), wrap=tk.WORD,
                bg="white", fg="#212121",
                relief=tk.SOLID, bd=1
            )
            sb = tk.Scrollbar(txt_frame, command=txt.yview)
            txt.config(yscrollcommand=sb.set)
            sb.pack(side=tk.RIGHT, fill=tk.Y)
            txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            step_labels[cname] = {"img": img_lbl, "text": txt,
                                   "tsize": tsize}

        # ══════════════════════════════════════════════════════
        # Second row — Visual Diagnostic Panels
        # ══════════════════════════════════════════════════════
        row2_frame = tk.Frame(right)
        row2_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        VIS_COLS = [
            ("🔴 Contours",         "#B71C1C", "white", False),
            ("🦴 Skeleton",          "#37474F", "white", False),
            ("🌡️ Heat Map",          "#E65100", "white", False),
            ("📊 Projections",       "#1A237E", "white", True),
        ]
        vis_labels = {}

        for i, (vname, hdr_bg, hdr_fg, is_chart) in enumerate(VIS_COLS):
            subf2 = tk.Frame(row2_frame, relief=tk.RIDGE, bd=2, bg="white")
            subf2.grid(row=0, column=i, padx=4, pady=4, sticky="nsew")
            row2_frame.columnconfigure(i, weight=1)

            tk.Label(
                subf2, text=vname,
                font=("Arial", 10, "bold"),
                bg=hdr_bg, fg=hdr_fg, pady=4
            ).pack(fill=tk.X)

            if not is_chart:
                img_frame2 = tk.Frame(subf2, bg="#ECEFF1",
                                      width=THUMB_SIZE, height=THUMB_SIZE)
                img_frame2.pack_propagate(False)
                img_frame2.pack(pady=4, padx=4)
                img_lbl2 = tk.Label(img_frame2, bg="#ECEFF1")
                img_lbl2.place(relx=0.5, rely=0.5, anchor="center")

                # Zoom button
                def _make_zoom2(lbl_ref2, title2):
                    def _z2():
                        ph2 = getattr(lbl_ref2, "image", None)
                        if ph2 is None: return
                        pop2 = tk.Toplevel(win)
                        pop2.title(title2)
                        center_window(pop2, CANVAS_SIZE+40, CANVAS_SIZE+40)
                        fl2 = tk.Label(pop2)
                        fl2.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                        zp2 = getattr(lbl_ref2, "zoom_image", ph2)
                        fl2.config(image=zp2); fl2.image = zp2
                    return _z2
                tk.Button(
                    subf2, text="🔍 Zoom",
                    font=("Arial", 9), relief=tk.FLAT,
                    bg=hdr_bg, fg="white", pady=2, padx=6,
                    cursor="hand2",
                    command=_make_zoom2(img_lbl2, vname)
                ).pack(pady=(0, 2))

                txt2_frame = tk.Frame(subf2)
                txt2_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
                txt2 = tk.Text(
                    txt2_frame, height=5, width=20,
                    font=("Courier", 8), wrap=tk.WORD,
                    bg="white", fg="#212121",
                    relief=tk.SOLID, bd=1
                )
                sb2 = tk.Scrollbar(txt2_frame, command=txt2.yview)
                txt2.config(yscrollcommand=sb2.set)
                sb2.pack(side=tk.RIGHT, fill=tk.Y)
                txt2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                vis_labels[vname] = {"img": img_lbl2, "text": txt2}
            else:
                # Bar-chart panel — no image Label
                chart_inner = tk.Frame(subf2, bg="white")
                chart_inner.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
                vis_labels[vname] = {"chart_frame": chart_inner}

        # ── Prediction function ──────────────────────────────────────
        def _predict():
            if canvas_widget.is_blank():
                messagebox.showwarning(
                    "Warning", "Canvas is empty! Please draw a signature first.",
                    parent=win
                )
                return

            predict_btn.config(state=tk.DISABLED, text="⏳ Analysing...")
            win.update_idletasks()

            try:
                pil_img = canvas_widget.get_image()

                # ── Grayscale (enlarged) ─────────────────────────────────────
                gray      = pil_img.convert('L')
                g_big     = gray.resize((THUMB_BIG, THUMB_BIG), Image.LANCZOS)
                g_photo   = ImageTk.PhotoImage(g_big)
                step_labels["Grayscale"]["img"].config(image=g_photo)
                step_labels["Grayscale"]["img"].image      = g_photo
                step_labels["Grayscale"]["img"].zoom_image = g_photo
                # Grayscale image statistics
                gray_arr = np.array(gray)
                _set_text(step_labels["Grayscale"]["text"],
                          f"Grayscale\n"
                          f"Dimensions: {gray_arr.shape[1]}x{gray_arr.shape[0]}\n"
                          f"Mean    : {gray_arr.mean():.1f}\n"
                          f"Std Dev : {gray_arr.std():.1f}\n"
                          f"Min/Max : {gray_arr.min()}/{gray_arr.max()}")

                # ── Binary (enlarged) ────────────────────────────────────────
                binary    = preprocess_signature(pil_img)
                b_big     = Image.fromarray(binary).resize(
                    (THUMB_BIG, THUMB_BIG), Image.NEAREST
                )
                b_photo   = ImageTk.PhotoImage(b_big)
                step_labels["Binary"]["img"].config(image=b_photo)
                step_labels["Binary"]["img"].image      = b_photo
                step_labels["Binary"]["img"].zoom_image = b_photo
                px_white = int(np.sum(binary > 0))
                px_total = binary.size
                _set_text(step_labels["Binary"]["text"],
                          f"Binary — Otsu\n"
                          f"Dimensions: {binary.shape[1]}x{binary.shape[0]}\n"
                          f"Ink (white): {px_white}\n"
                          f"Background: {px_total - px_white}\n"
                          f"Density  : {px_white/px_total:.3f}")

                # ══════════════════════════════════════════════
                # ── Visual Diagnostic Panels (Second Row) ──
                # ══════════════════════════════════════════════

                # ── ① Colour-coded Contours ─────────────────────────────────
                try:
                    cnt_pil  = make_contour_image(binary, THUMB_SIZE)
                    cnt_big  = cnt_pil.resize((CANVAS_SIZE, CANVAS_SIZE),
                                               Image.LANCZOS)
                    cnt_ph   = ImageTk.PhotoImage(cnt_pil)
                    cnt_ph_z = ImageTk.PhotoImage(cnt_big)
                    vl_cnt   = vis_labels["🔴 Contours"]
                    vl_cnt["img"].config(image=cnt_ph)
                    vl_cnt["img"].image      = cnt_ph
                    vl_cnt["img"].zoom_image = cnt_ph_z
                    ctrs, _  = cv2.findContours(binary,
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    areas    = sorted(
                        [cv2.contourArea(c) for c in ctrs], reverse=True
                    )
                    cnt_info_lines = [f"Contour count: {len(ctrs)}"]
                    cnt_info_lines += [
                        f"  K{i+1}: {a:.0f}px²"
                        for i, a in enumerate(areas[:8])
                    ]
                    cnt_info = "\n".join(cnt_info_lines)
                    _set_text(vl_cnt["text"], cnt_info)
                except Exception as _e:
                    logger.warning("Contour panel error: %s", _e)

                # ── ② Morphological Skeleton ────────────────────────────────
                try:
                    skel_pil, skel_itr = make_skeleton_image(binary, THUMB_SIZE)
                    skel_big = skel_pil.resize((CANVAS_SIZE, CANVAS_SIZE),
                                               Image.LANCZOS)
                    skel_ph  = ImageTk.PhotoImage(skel_pil)
                    skel_phz = ImageTk.PhotoImage(skel_big)
                    vl_sk    = vis_labels["🦴 Skeleton"]
                    vl_sk["img"].config(image=skel_ph)
                    vl_sk["img"].image      = skel_ph
                    vl_sk["img"].zoom_image = skel_phz
                    skel_arr = np.array(skel_pil.convert("L"))
                    skel_px  = int((skel_arr < 200).sum())
                    _set_text(vl_sk["text"],
                              f"Iterations: {skel_itr}\nSkeleton pixels: {skel_px}\n(fewer = simpler signature)")
                except Exception as _e:
                    logger.warning("Skeleton panel error: %s", _e)

                # ── ③ 8x8 Heat Map ─────────────────────────────────────────
                try:
                    heat_pil, heat_mat = make_heatmap_image(binary, THUMB_SIZE)
                    heat_big = heat_pil.resize((CANVAS_SIZE, CANVAS_SIZE),
                                               Image.NEAREST)
                    heat_ph  = ImageTk.PhotoImage(heat_pil)
                    heat_phz = ImageTk.PhotoImage(heat_big)
                    vl_ht    = vis_labels["🌡️ Heat Map"]
                    vl_ht["img"].config(image=heat_ph)
                    vl_ht["img"].image      = heat_ph
                    vl_ht["img"].zoom_image = heat_phz
                    r_max, c_max = np.unravel_index(heat_mat.argmax(),
                                                     heat_mat.shape)
                    r_min, c_min = np.unravel_index(heat_mat.argmin(),
                                                     heat_mat.shape)
                    _set_text(vl_ht["text"],
                              f"Densest cell: ({r_max+1},{c_max+1})\\nEmptiest cell: ({r_min+1},{c_min+1})\\nMean density: {heat_mat.mean():.1f}\\nRed=dense  Blue=empty")
                except Exception as _e:
                    logger.warning("Heat map panel error: %s", _e)

                # ── ④ Projection Profiles ─────────────────────────────────
                try:
                    vl_proj = vis_labels["📊 Projections"]
                    cf      = vl_proj["chart_frame"]
                    # Clear previous charts
                    for w in cf.winfo_children():
                        w.destroy()
                    img_norm = (binary > 0).astype(np.uint8)
                    ph  = _projection_profile(img_norm, axis=1, n=20)
                    pv  = _projection_profile(img_norm, axis=0, n=20)
                    draw_projection_chart(cf, ph, pv, THUMB_SIZE)
                except Exception as _e:
                    logger.warning("Projection panel error: %s", _e)

                # ── Feature Extraction (152 features) ────────────────────────
                features   = extract_features(binary)
                grp_labels = [
                    ("Hu Moments",     7),
                    ("Shape",          10),
                    ("Grid 4x4",       16),
                    ("Grid 8x8",       64),
                    ("Horiz. Proj.",   20),
                    ("Vert. Proj.",    20),
                    ("Statistical",    15),
                ]
                feat_lines = [f"● Total features: {len(features)}\n"]
                idx = 0
                for grp_name, grp_n in grp_labels:
                    grp_feats = features[idx:idx+grp_n]
                    feat_lines.append(
                        f"── {grp_name} ({grp_n}) ──"
                    )
                    for fi in range(grp_n):
                        feat_lines.append(
                            f"  {FEATURE_NAMES[idx+fi]}: "
                            f"{grp_feats[fi]:+.3f}"
                        )
                    idx += grp_n
                _set_text(step_labels["Features"]["text"],
                          "\n".join(feat_lines))

                # ── Feature Selection ────────────────────────────────────────
                feats_scaled = self.scaler.transform([features])
                feats_sel    = self.selector.transform(feats_scaled)
                support      = self.selector.get_support()
                sel_names    = [FEATURE_NAMES[i] for i, s in enumerate(support) if s]
                sel_lines    = "\n".join(
                    f"{n}: {v:+.3f}"
                    for n, v in zip(sel_names, feats_sel[0])
                )
                _set_text(
                    step_labels["Selected Features"]["text"],
                    f"Selected {len(sel_names)} from {len(features)}:\n\n{sel_lines}"
                )

                # ── Prediction ────────────────────────────────────────────
                pred_label = self.model.predict(feats_sel)[0]
                proba      = self.model.predict_proba(feats_sel)[0]
                max_prob   = float(proba.max())

                # Safe class_names index check
                if 0 <= pred_label < len(self.class_names):
                    person_name = self.class_names[pred_label]
                else:
                    person_name = f"[Unknown #{pred_label}]"
                    logger.warning(
                        "pred_label=%d out of range (len=%d)",
                        pred_label, len(self.class_names)
                    )

                # ── Confidence threshold ──────────────────────────────────
                if max_prob < CONFIDENCE_THRESHOLD:
                    verdict      = "❓ Unrecognised"
                    verdict_color= "#E65100"
                    forgery_note = "Low confidence — cannot identify person"
                else:
                    # ── One-Class SVM forgery detection ────────────────────────
                    is_genuine = True
                    if person_name in self.oc_models:
                        oc_pred = self.oc_models[person_name].predict(feats_sel)[0]
                        is_genuine = (oc_pred == 1)

                    if is_genuine:
                        verdict       = f"✅ {person_name}"
                        verdict_color = "#1B5E20"
                        forgery_note  = "Signature appears genuine"
                    else:
                        verdict       = f"⚠️ Forged / Suspicious ({person_name})"
                        verdict_color = "#B71C1C"
                        forgery_note  = "One-Class SVM: deviates from the genuine pattern"

                # ── Display all probabilities ────────────────────────────
                proba_lines = "\n".join(
                    f"{'← ' if i == pred_label else '   '}"
                    f"{(self.class_names[i] if i < len(self.class_names) else '?')}: "
                    f"{p:.2%}"
                    for i, p in enumerate(proba)
                )
                result_text = (
                    f"{verdict}\n\n"
                    f"Confidence: {max_prob:.2%}\n\n"
                    f"{forgery_note}\n\n"
                    f"─── All Probabilities ───\n{proba_lines}"
                )
                _set_text(step_labels["Result"]["text"], result_text)
                step_labels["Result"]["text"].tag_configure(
                    "verdict", foreground=verdict_color,
                    font=("Arial", 10, "bold")
                )

                # Thumbnail for result panel
                orig_thumb  = pil_img.resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
                orig_photo  = ImageTk.PhotoImage(orig_thumb)
                step_labels["Result"]["img"].config(image=orig_photo)
                step_labels["Result"]["img"].image = orig_photo

                logger.info(
                    "Recognition: %s | confidence: %.3f | verdict: %s",
                    person_name, max_prob, verdict
                )

            except Exception as exc:
                logger.error("Predict error: %s", exc, exc_info=True)
                messagebox.showerror(
                    "Error", f"An error occurred during analysis:\n{exc}",
                    parent=win
                )
            finally:
                predict_btn.config(state=tk.NORMAL, text="🔍 Analyse")

        predict_btn.config(command=_predict)



# ═══════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ensure_dir(DATA_DIR)
    ensure_dir(MODELS_DIR)
    ensure_dir(OC_MODELS_DIR)

    root = tk.Tk()
    app  = ForgeryDetectionApp(root)
    logger.info("Application started")
    root.mainloop()
    logger.info("Application closed")
