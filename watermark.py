"""
Watermark embedding and extraction module.
Supports:
  - LSB (Least Significant Bit) in spatial domain
    - Visible text watermark overlay
  - DWT + SVD in transform domain
"""

import numpy as np
import cv2
import pywt
import joblib
from pathlib import Path

try:
    from deep_watermark import deep_watermark_probability, deep_watermark_threshold
except Exception:
    deep_watermark_probability = None
    deep_watermark_threshold = None

try:
    import torch
    import torchvision.transforms as T
except Exception:
    torch = None
    T = None

from features import extract_features


# ──────────────────────────────────────────────
# LSB Watermarking
# ──────────────────────────────────────────────

# Watermark seed — shared secret between embed and extract
_WM_SEED = 20260309


def embed_lsb(image: np.ndarray, message: str = "COPYRIGHT") -> np.ndarray:
    """
    Embed a full-image LSB watermark using a seeded PRNG pattern.
    Replaces ALL blue-channel LSBs with a reproducible pseudo-random
    bit stream, creating a strong, statistically detectable signature.
    A short ASCII header encodes the message in the first pixels.
    """
    img = image.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    n_pixels = h * w

    # Generate full-image PRNG bit pattern (seeded → reproducible)
    rng = np.random.default_rng(_WM_SEED)
    pattern = rng.integers(0, 2, size=n_pixels, dtype=np.uint8)

    # Embed message header in first pixels (for extraction display)
    header_bits = []
    for ch in (message + '\x00'):
        header_bits.extend([(ord(ch) >> (7 - i)) & 1 for i in range(8)])
    for i, bit in enumerate(header_bits[:n_pixels]):
        pattern[i] = bit

    flat_b = img[:, :, 0].flatten().astype(np.uint8)
    flat_b = (flat_b & 0xFE) | pattern   # replace LSB with pattern
    img[:, :, 0] = flat_b.reshape(h, w)
    return img


def extract_lsb(image: np.ndarray, msg_length_hint: int = 32) -> str:
    """Extract the ASCII header message from LSBs of the blue channel."""
    flat_b = image[:, :, 0].flatten().astype(np.uint8)
    bits = flat_b & 1
    chars = []
    for i in range(0, min(msg_length_hint * 8, len(bits)), 8):
        byte_bits = bits[i:i+8]
        if len(byte_bits) < 8:
            break
        val = int(''.join(str(b) for b in byte_bits), 2)
        if val == 0:
            break
        if 32 <= val <= 126:
            chars.append(chr(val))
    return ''.join(chars)


# ──────────────────────────────────────────────
# Visible Watermark Embedding
# ──────────────────────────────────────────────

VISIBLE_TEXTS = [
    "© Copyright", "All Rights Reserved", "Watermarked",
    "Protected", "© 2026", "CopyRight Protected", "DO NOT USE"
]


def _draw_text_overlay(base: np.ndarray, text: str, x: int, y: int, alpha: float = 0.45) -> np.ndarray:
    overlay = base.copy()
    h, w = base.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = max(0.7, w / 220)
    thickness = max(2, int(font_scale * 1.8))
    cv2.putText(overlay, text, (x + 2, y + 2), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(overlay, text, (x, y), font, font_scale, (220, 220, 220), thickness, cv2.LINE_AA)
    alpha = float(np.clip(alpha, 0.2, 0.8))
    return cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)


def embed_visible_watermark(image: np.ndarray, text: str = None, alpha: float = 0.45) -> np.ndarray:
    import random
    img = image.copy()
    h, w = img.shape[:2]
    if text is None:
        text = random.choice(VISIBLE_TEXTS)

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = max(0.7, w / 220)
    thickness = max(2, int(font_scale * 1.8))
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    positions = [
        (max(4, w // 2 - tw // 2), max(th + 4, h // 2 + th // 2)),
        (max(4, int(w * 0.05)), max(th + 4, int(h * 0.2))),
        (max(4, w - tw - 6), max(th + 4, int(h * 0.2))),
        (max(4, int(w * 0.05)), max(th + 4, int(h * 0.8))),
        (max(4, w - tw - 6), max(th + 4, int(h * 0.8))),
    ]
    x, y = random.choice(positions)
    return _draw_text_overlay(img, text, x, y, alpha=alpha)


def embed_tiled_visible_watermark(image: np.ndarray, text: str = "©", alpha: float = 0.28) -> np.ndarray:
    img = image.copy()
    h, w = img.shape[:2]
    step_x = max(60, w // 4)
    step_y = max(50, h // 4)
    for yy in range(step_y // 2, h, step_y):
        for xx in range(step_x // 3, w, step_x):
            img = _draw_text_overlay(img, text, xx, yy, alpha=alpha)
    return img


def embed_logo_stamp(image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    img = image.copy()
    h, w = img.shape[:2]
    overlay = img.copy()
    cx, cy = int(w * 0.82), int(h * 0.82)
    radius = max(18, min(h, w) // 9)
    cv2.circle(overlay, (cx, cy), radius, (230, 230, 230), -1, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), radius, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(overlay, "C", (cx - radius // 3, cy + radius // 3), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.8, radius / 24), (40, 40, 40), 2, cv2.LINE_AA)
    return cv2.addWeighted(overlay, float(np.clip(alpha, 0.2, 0.8)), img, 1 - float(np.clip(alpha, 0.2, 0.8)), 0)


def embed_lsb_variant(image: np.ndarray, bits: int = 1, channel_index: int = 0) -> np.ndarray:
    img = image.copy().astype(np.uint8)
    h, w = img.shape[:2]
    total = h * w
    bits = int(np.clip(bits, 1, 2))
    channel_index = int(np.clip(channel_index, 0, 2))
    rng = np.random.default_rng(_WM_SEED + bits + channel_index * 7)
    payload = rng.integers(0, 1 << bits, size=total, dtype=np.uint8)
    mask = np.uint8((1 << bits) - 1)
    flat = img[:, :, channel_index].flatten()
    flat = (flat & np.uint8(0xFF ^ mask)) | payload
    img[:, :, channel_index] = flat.reshape(h, w)
    return img


def embed_pixel_modulation(image: np.ndarray, delta: int = 2) -> np.ndarray:
    img = image.copy().astype(np.int16)
    h, w = img.shape[:2]
    rng = np.random.default_rng(_WM_SEED + 99)
    pattern = rng.integers(0, 2, size=(h, w), dtype=np.int16)
    sign = np.where(pattern == 0, -1, 1)
    for ch in range(3):
        img[:, :, ch] = img[:, :, ch] + sign * delta
    return np.clip(img, 0, 255).astype(np.uint8)


def embed_ycbcr_chroma(image: np.ndarray) -> np.ndarray:
    bgr = image.copy().astype(np.uint8)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    h, w = ycrcb.shape[:2]
    rng = np.random.default_rng(_WM_SEED + 17)
    pattern = rng.integers(0, 2, size=(h, w), dtype=np.uint8)
    cb = ycrcb[:, :, 2]
    cb = (cb & 0xFE) | pattern
    ycrcb[:, :, 2] = cb
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def embed_hsv_value(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image.copy().astype(np.uint8), cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    rng = np.random.default_rng(_WM_SEED + 23)
    pattern = rng.integers(0, 2, size=(h, w), dtype=np.int16)
    delta = np.where(pattern == 0, -2, 2)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.int16) + delta, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def embed_dct_watermark(image: np.ndarray, alpha: float = 0.04) -> np.ndarray:
    img = image.copy().astype(np.float32)
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    d = cv2.dct(gray)
    h, w = d.shape
    rng = np.random.default_rng(_WM_SEED + 31)
    patch_h, patch_w = h // 4, w // 4
    y0, y1 = h // 3, h // 3 + patch_h
    x0, x1 = w // 3, w // 3 + patch_w
    noise = rng.standard_normal((patch_h, patch_w)).astype(np.float32)
    d[y0:y1, x0:x1] += alpha * np.std(d[y0:y1, x0:x1]) * noise
    rec = cv2.idct(d)
    rec = np.clip(rec, 0, 255).astype(np.uint8)
    out = image.copy()
    out[:, :, 1] = rec
    return out


_MODEL_CACHE = None
_DEEP_CACHE = None
_DEEP_MLP_CACHE = None
_PROBA_POLARITY_CACHE = {}


def _get_watermark_class_index(model) -> int:
    """
    Resolve which predict_proba column corresponds to the watermark class.
    Defaults to index 1 for binary models when metadata is unavailable.
    """
    classes = None
    if hasattr(model, "classes_"):
        classes = getattr(model, "classes_", None)
    elif hasattr(model, "named_steps") and "clf" in getattr(model, "named_steps", {}):
        clf = model.named_steps["clf"]
        classes = getattr(clf, "classes_", None)

    if classes is None or len(classes) == 0:
        return 1

    labels = [str(c).strip().lower() for c in classes]
    for i, lbl in enumerate(labels):
        if lbl in {"1", "watermarked", "wm", "positive", "pos", "true"}:
            return i

    # For binary numeric labels this usually maps to class "1" at index 1.
    if len(classes) == 2:
        return 1
    return max(0, len(classes) - 1)


def _should_invert_model_probability(model) -> bool:
    """
    Calibrate polarity once using known clean/watermarked folders.
    If clean scores are higher than watermarked scores, the model polarity is inverted.
    """
    key = id(model)
    if key in _PROBA_POLARITY_CACHE:
        return _PROBA_POLARITY_CACHE[key]

    clean_dir = Path("dataset/train/clean")
    wm_dir = Path("dataset/train/watermarked")
    if not clean_dir.exists() or not wm_dir.exists():
        _PROBA_POLARITY_CACHE[key] = False
        return False

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    clean_files = [p for p in clean_dir.iterdir() if p.suffix.lower() in exts][:6]
    wm_files = [p for p in wm_dir.iterdir() if p.suffix.lower() in exts][:6]
    if len(clean_files) < 2 or len(wm_files) < 2:
        _PROBA_POLARITY_CACHE[key] = False
        return False

    wm_idx = _get_watermark_class_index(model)
    clean_scores, wm_scores = [], []

    try:
        for p in clean_files:
            img = cv2.imread(str(p))
            if img is None:
                continue
            feat = extract_features(img)
            proba = model.predict_proba(feat.reshape(1, -1))[0]
            if len(proba) > wm_idx:
                clean_scores.append(float(proba[wm_idx]))

        for p in wm_files:
            img = cv2.imread(str(p))
            if img is None:
                continue
            feat = extract_features(img)
            proba = model.predict_proba(feat.reshape(1, -1))[0]
            if len(proba) > wm_idx:
                wm_scores.append(float(proba[wm_idx]))
    except Exception:
        _PROBA_POLARITY_CACHE[key] = False
        return False

    if not clean_scores or not wm_scores:
        _PROBA_POLARITY_CACHE[key] = False
        return False

    invert = float(np.mean(clean_scores)) > float(np.mean(wm_scores))
    _PROBA_POLARITY_CACHE[key] = bool(invert)
    return bool(invert)


def _predict_watermark_probability(model, feat_row: np.ndarray) -> float:
    proba = model.predict_proba(feat_row.reshape(1, -1))[0]
    wm_idx = _get_watermark_class_index(model)
    if len(proba) <= wm_idx:
        return 0.0
    p = float(proba[wm_idx])
    if _should_invert_model_probability(model):
        p = 1.0 - p
    return float(np.clip(p, 0.0, 1.0))


def _load_detector_model():
    # Legacy classical model path is intentionally disabled.
    # The app now runs with deep checkpoint + DIP pipeline only.
    return None, 0.50


def _load_deep_model():
    global _DEEP_CACHE
    if _DEEP_CACHE is not None:
        return _DEEP_CACHE

    if torch is None:
        _DEEP_CACHE = (None, None, 0.50)
        return _DEEP_CACHE

    ckpt_path = Path("models/deep_detector.pt")
    if not ckpt_path.exists():
        _DEEP_CACHE = (None, None, 0.50)
        return _DEEP_CACHE

    try:
        from torchvision.models import resnet18

        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        n_classes = int(ckpt.get("n_classes", 2))
        model = resnet18(weights=None)
        in_f = model.fc.in_features
        model.fc = torch.nn.Linear(in_f, n_classes)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        img_size = int(ckpt.get("img_size", 160))
        tfm = T.Compose([
            T.ToTensor(),
            T.Resize((img_size, img_size), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        thr = float(ckpt.get("threshold", 0.50))
        _DEEP_CACHE = (model, tfm, thr)
    except Exception:
        _DEEP_CACHE = (None, None, 0.50)
    return _DEEP_CACHE


def _load_deep_mlp_model():
    global _DEEP_MLP_CACHE
    if _DEEP_MLP_CACHE is not None:
        return _DEEP_MLP_CACHE
    p = Path("models/deep_detector.pkl")
    if not p.exists():
        _DEEP_MLP_CACHE = (None, 0.50)
        return _DEEP_MLP_CACHE
    try:
        obj = joblib.load(p)
        if isinstance(obj, dict) and "pipeline" in obj:
            _DEEP_MLP_CACHE = (obj["pipeline"], float(obj.get("threshold", 0.50)))
        else:
            _DEEP_MLP_CACHE = (obj, 0.50)
    except Exception:
        _DEEP_MLP_CACHE = (None, 0.50)
    return _DEEP_MLP_CACHE


def _predict_deep_score(bgr_img: np.ndarray) -> tuple[float, float]:
    if deep_watermark_probability is not None and deep_watermark_threshold is not None:
        try:
            p = deep_watermark_probability(bgr_img)
            if p is not None:
                thr = float(deep_watermark_threshold(default=0.60))
                return float(np.clip(p, 0.0, 1.0)), float(np.clip(thr, 0.35, 0.90))
        except Exception:
            pass

    return 0.0, 0.50


def detect_visible_watermark(image: np.ndarray) -> dict:
    """
        Hybrid watermark detector:
            PRIMARY: dedicated local watermark CNN (models/wm_deep.pt)
            SUPPORT: DIP T1-T5 structural signals for UI only
    """
    img = cv2.resize(image.copy(), (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- ML PRIMARY PATH ---------------------------------------------------
    clf, model_thr = _load_detector_model()
    ml_score = 0.0
    if clf is not None:
        try:
            feat = extract_features(img)
            ml_score = _predict_watermark_probability(clf, feat)
        except Exception:
            ml_score = 0.0

    deep_score, deep_thr = _predict_deep_score(img)

    if clf is not None:
        final_score = 0.82 * ml_score + 0.18 * deep_score
        fused_thr   = 0.45
        is_wm       = final_score >= fused_thr

        # Lightweight DIP signals for the UI visualization panels only
        clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced  = clahe.apply(gray)
        gx        = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy        = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_p95  = float(np.percentile(np.sqrt(gx * gx + gy * gy), 95))
        edge_den  = float(np.mean(cv2.Canny(gray, 80, 160) > 0))
        blurred   = cv2.GaussianBlur(gray.astype(np.float32), (31, 31), 10)
        residual  = np.abs(gray.astype(np.float32) - blurred)
        res_norm  = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, res_bin = cv2.threshold(res_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        res_ratio  = float(np.mean(res_bin > 0))
        ad = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, -2)
        cnts, _ = cv2.findContours(ad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mser_n = sum(1 for c in cnts if 8 < cv2.contourArea(c) < 2400)

        wm_kind = "none"
        if is_wm:
            wm_kind = "text watermark" if ml_score >= 0.65 else "image watermark"

        return {
            "score":           round(float(final_score), 3),
            "is_visible_wm":   bool(is_wm),
            "watermark_kind":  wm_kind,
            "text_score":      round(float(ml_score), 3),
            "image_score":     round(float(deep_score), 3),
            "ml_score":        round(float(ml_score), 3),
            "deep_score":      round(float(deep_score), 3),
            "threshold":       round(float(fused_thr), 3),
            "edge_density":    round(float(edge_den), 3),
            "anomaly_ratio":   round(float(res_ratio), 3),
            "grad_p95":        round(float(grad_p95), 2),
            "mser_regions":    int(mser_n),
            "bright_ratio":    round(float(res_ratio), 4),
            "hough_ratio":     0.0,
            "res_ratio":       round(float(res_ratio), 3),
            "gray_ratio":      round(float(ml_score), 3),
            "sat_ratio":       0.0,
            "sat_zscore":      0.0,
            "bright_zscore":   0.0,
            "gray_excess":     round(float(ml_score), 4),
            "n_horizontal_lines": 0,
            "residual_excess": round(float(deep_score), 3),
        }

    # --- DIP-ONLY FALLBACK (no model file) ------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # T1: dual adaptive maps for low/high-opacity watermark traces.
    ad_light = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 15, -2)
    ad_dark = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 2)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    all_contours = []
    for bm in (ad_light, ad_dark):
        bm = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(contours)

    text_regions = 0
    centers = []
    quads = set()
    high_contrast_regions = 0
    box_keys = set()
    comp_heights = []
    y_centers = []

    for cnt in all_contours:
        area = cv2.contourArea(cnt)
        if area < 8 or area > 2400:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 3 or h < 3:
            continue
        ar = w / (h + 1e-6)
        if ar < 0.10 or ar > 14.0:
            continue

        fill = area / (w * h + 1e-6)
        if fill < 0.05 or fill > 0.92:
            continue

        # Deduplicate near-identical boxes from light/dark threshold branches.
        key = (x // 3, y // 3, w // 3, h // 3)
        if key in box_keys:
            continue
        box_keys.add(key)

        pad = max(int(max(w, h) * 0.8), 8)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(255, x + w + pad), min(255, y + h + pad)
        nbr = gray[y1:y2, x1:x2]

        cm = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(cm, [cnt], -1, 255, -1)
        cm_crop = cm[y1:y2, x1:x2]
        bg_pix = nbr[cm_crop == 0]
        fg_pix = nbr[cm_crop == 255]

        if len(bg_pix) < 20 or len(fg_pix) < 5:
            continue

        bg_std = float(np.std(bg_pix))
        fg_std = float(np.std(fg_pix))
        delta_mu = float(abs(float(np.mean(fg_pix)) - float(np.mean(bg_pix))))

        # Watermarks are semi-transparent: skip high-contrast natural features
        if delta_mu > 40:
            continue
        # Watermark characters are small relative to image; skip large natural blobs
        if h > 38:
            continue
        if bg_std < 18:
            continue
        # Background must be notably more textured than the uniform watermark fill
        if bg_std < fg_std * 2.0:
            continue
        if delta_mu > 70:
            high_contrast_regions += 1

        text_regions += 1
        cx, cy = x + w // 2, y + h // 2
        centers.append((cx, cy))
        comp_heights.append(h)
        y_centers.append(cy)
        quads.add((0 if cy < 128 else 1, 0 if cx < 128 else 1))

    # T2/T3: repetition and spread analysis.
    density = np.zeros((4, 4), dtype=np.float32)
    for cx, cy in centers:
        density[min(3, cy // 64), min(3, cx // 64)] += 1
    active_cells = int(np.sum(density > 0))

    localized_cluster = False
    spread_area = 0.0
    if len(centers) >= 3:
        xs = np.array([p[0] for p in centers], dtype=np.float32)
        ys = np.array([p[1] for p in centers], dtype=np.float32)
        bw = float(xs.max() - xs.min() + 1.0)
        bh = float(ys.max() - ys.min() + 1.0)
        spread_area = (bw * bh) / (256.0 * 256.0)
        if spread_area < 0.09 and active_cells <= 3:
            localized_cluster = True

    # Text watermarks usually form repeated line structures with somewhat consistent
    # character heights; random textures fail this regularity check.
    align_score = 0.0
    if len(y_centers) >= 4:
        y_hist = np.bincount(np.clip((np.array(y_centers) // 12).astype(np.int32), 0, 21), minlength=22)
        strong_rows = int(np.sum(y_hist >= 2))
        if strong_rows >= 2:
            align_score += 0.45
        if strong_rows >= 3:
            align_score += 0.20
    if len(comp_heights) >= 4:
        h_arr = np.array(comp_heights, dtype=np.float32)
        h_cv = float(np.std(h_arr) / (np.mean(h_arr) + 1e-6))
        if h_cv < 0.65:
            align_score += 0.35

    text_score = 0.0
    # Require strong alignment evidence AND enough characters; this rejects
    # natural images that incidentally produce a few scattered blobs.
    if text_regions >= 6 and len(quads) >= 2 and align_score >= 0.45:
        text_score = min(0.90, 0.12 + 0.018 * text_regions + 0.020 * max(0, active_cells - 1) + 0.60 * align_score)

    # T4: image/logo watermark score.
    # Two sub-approaches: (a) corner-region overlay analysis, (b) residual-map candidates.

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    grad_p95 = float(np.percentile(mag, 95))
    edge_density = float(np.mean(cv2.Canny(gray, 80, 160) > 0))

    # (a) Corner-region analysis — logo watermarks live in corners / near edges.
    # A logo stamp is a semi-transparent overlay region whose mean brightness
    # differs from the same area's high-frequency local surroundings.
    corner_score = 0.0
    corner_regions = [
        (0,  0,  72, 72),   # top-left
        (184, 0,  255, 72),  # top-right
        (0,  184, 72, 255),  # bottom-left
        (184, 184, 255, 255),# bottom-right
        (92, 92, 164, 164),  # centre (full-image centre watermarks)
    ]
    corner_hits = 0
    for (x1, y1, x2, y2) in corner_regions:
        patch = gray[y1:y2, x1:x2].astype(np.float32)
        if patch.size < 100:
            continue
        p_mean = float(np.mean(patch))
        p_std  = float(np.std(patch))
        # Use a wider ring around the corner as reference
        pad = 30
        rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
        rx2, ry2 = min(255, x2 + pad), min(255, y2 + pad)
        ring = gray[ry1:ry2, rx1:rx2].astype(np.float32)
        r_mean = float(np.mean(ring))
        r_std  = float(np.std(ring))

        if r_std < 5:  # flat uniform region — no background to compare against
            continue

        brightness_delta = abs(p_mean - r_mean)
        # Logo watermark creates a noticeable but not huge brightness shift (3-40 grey levels)
        # and has lower internal std than the surrounding complex background.
        # Require non-trivial internal structure (not just blank sky/wall)
        # and stronger smoothness contrast vs surroundings.
        if 5.0 <= brightness_delta <= 28.0 and p_std < r_std * 0.68 and r_std > 22 and p_std > 6:
            corner_hits += 1

    if corner_hits >= 2:
        corner_score = min(0.82, 0.38 + 0.22 * corner_hits)
    elif corner_hits == 1:
        corner_score = 0.32

    # (b) Residual-map candidates (Gaussian subtraction).
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (31, 31), 10)
    residual = np.abs(gray.astype(np.float32) - blurred)
    res_norm = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, res_bin = cv2.threshold(res_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    res_ratio = float(np.mean(res_bin > 0))

    logo_candidates = 0
    res_corner_candidates = 0
    cc, _ = cv2.findContours(res_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cc:
        a = cv2.contourArea(cnt)
        if a < 80 or a > 4500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 8 or h < 8:
            continue
        ar = w / (h + 1e-6)
        if ar < 0.25 or ar > 4.0:
            continue
        pad = max(int(max(w, h) * 0.6), 8)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(255, x + w + pad), min(255, y + h + pad)
        nbr = gray[y1:y2, x1:x2]
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        m = mask[y1:y2, x1:x2]
        fg = nbr[m == 255]
        bg = nbr[m == 0]
        if len(fg) < 20 or len(bg) < 40:
            continue
        delta = abs(float(np.mean(fg)) - float(np.mean(bg)))
        bg_std = float(np.std(bg))
        if delta < 3.0 or delta > 50.0:
            continue
        if bg_std < 10:
            continue
        logo_candidates += 1
        cx, cy = x + w // 2, y + h // 2
        if cx < 64 or cx > 192 or cy < 64 or cy > 192:
            res_corner_candidates += 1

    res_logo_score = 0.0
    # Require at least 2 logo-like blobs to avoid single-object false positives.
    if logo_candidates >= 2 and 0.02 < res_ratio < 0.35 and grad_p95 > 20:
        res_logo_score = min(0.78, 0.18 + 0.09 * logo_candidates + 0.14 * res_corner_candidates + 0.60 * res_ratio)

    logo_score = max(corner_score, res_logo_score)

    # Suppress common false positives: plates/shop boards/symbol blocks.
    if localized_cluster and high_contrast_regions >= 2:
        text_score *= 0.15
        logo_score *= 0.20

    # Additional line-dominance suppression (grilles, signboards).
    lines = cv2.HoughLinesP(cv2.Canny(gray, 100, 180), 1, np.pi / 180, threshold=70,
                            minLineLength=40, maxLineGap=6)
    n_lines = int(0 if lines is None else len(lines))
    if n_lines > 45 and spread_area < 0.20:
        text_score *= 0.35
        logo_score *= 0.35

    # Dedicated model decision path.
    dip_score   = max(text_score, logo_score)
    final_score = float(np.clip(deep_score, 0.0, 1.0))
    ml_score = deep_score
    fused_thr = float(np.clip(deep_thr, 0.55, 0.75))
    is_wm = bool(final_score >= fused_thr)
    wm_kind = "none"
    if is_wm:
        wm_kind = "text watermark" if text_score >= logo_score else "image watermark"

    return {
        "score": round(float(final_score), 3),
        "is_visible_wm": bool(is_wm),
        "watermark_kind": wm_kind,
        "text_score": round(float(text_score), 3),
        "image_score": round(float(logo_score), 3),
        "ml_score": round(float(ml_score), 3),
        "deep_score": round(float(deep_score), 3),
        "threshold": round(float(fused_thr), 3),
        "edge_density": round(float(edge_density), 3),
        "anomaly_ratio": round(float(res_ratio), 3),
        "grad_p95": round(float(grad_p95), 2),
        "mser_regions": int(text_regions),
        "bright_ratio": round(float(res_ratio), 4),
        "hough_ratio": round(float(min(1.0, n_lines / 100.0)), 3),
        "res_ratio": round(float(res_ratio), 3),
        "gray_ratio": round(float(text_score), 3),
        "sat_ratio": 0.0,
        "sat_zscore": round(float(active_cells / 16.0), 3),
        "bright_zscore": round(float(spread_area), 3),
        "gray_excess": round(float(text_score), 4),
        "n_horizontal_lines": int(n_lines),
        "residual_excess": round(float(logo_score), 3),
    }


# ──────────────────────────────────────────────
# DWT + SVD Watermarking
# ──────────────────────────────────────────────

def embed_dwt_svd(image: np.ndarray, watermark: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    Embed a grayscale watermark into a color image using DWT+SVD on the green channel.
    alpha: embedding strength (0.05–0.15 invisible range)
    """
    img = image.copy().astype(np.float64)
    channel = img[:, :, 1]  # green channel — most perceptually important

    # DWT decompose
    coeffs = pywt.dwt2(channel, 'haar')
    cA, (cH, cV, cD) = coeffs

    # SVD on approximation sub-band
    U, S, Vt = np.linalg.svd(cA, full_matrices=False)

    # Resize watermark to match S shape
    wm_gray = cv2.resize(watermark.astype(np.float64), (len(S), 1)).flatten()
    wm_norm = wm_gray / (wm_gray.max() + 1e-8)

    # Embed
    S_wm = S + alpha * wm_norm[:len(S)]

    # Reconstruct
    cA_wm = U @ np.diag(S_wm) @ Vt
    coeffs_wm = cA_wm, (cH, cV, cD)
    channel_wm = pywt.idwt2(coeffs_wm, 'haar')

    # Clip to valid range
    channel_wm = np.clip(channel_wm, 0, 255)
    img[:, :, 1] = channel_wm[:channel.shape[0], :channel.shape[1]]
    return img.astype(np.uint8)


def extract_dwt_svd(watermarked: np.ndarray, original: np.ndarray, wm_size: tuple, alpha: float = 0.1) -> np.ndarray:
    """Extract the watermark from a watermarked image given the original."""
    def get_S(img):
        channel = img[:, :, 1].astype(np.float64)
        coeffs = pywt.dwt2(channel, 'haar')
        cA, _ = coeffs
        _, S, _ = np.linalg.svd(cA, full_matrices=False)
        return S

    S_wm = get_S(watermarked)
    S_orig = get_S(original)

    wm_flat = (S_wm - S_orig) / (alpha + 1e-8)
    wm_flat = np.clip(wm_flat, 0, 1)

    wm_len = wm_size[0] * wm_size[1]
    wm_flat = wm_flat[:wm_len]
    wm = (wm_flat.reshape(wm_size) * 255).astype(np.uint8)
    return cv2.resize(wm, wm_size)


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def compute_psnr(original: np.ndarray, watermarked: np.ndarray) -> float:
    mse = np.mean((original.astype(np.float64) - watermarked.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10((255.0 ** 2) / mse)


def compute_ssim(original: np.ndarray, watermarked: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as ssim
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    wm_gray = cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(orig_gray, wm_gray, full=True)
    return float(score)


def compute_nc(original_wm: np.ndarray, extracted_wm: np.ndarray) -> float:
    """Normalized Correlation between original and extracted watermark."""
    o = original_wm.flatten().astype(np.float64)
    e = cv2.resize(extracted_wm, (original_wm.shape[1], original_wm.shape[0])).flatten().astype(np.float64)
    num = np.sum(o * e)
    den = np.sqrt(np.sum(o ** 2) * np.sum(e ** 2)) + 1e-8
    return float(num / den)
