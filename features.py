"""
Multi-domain feature extraction for watermark detection.
Extracts statistical, frequency, and bit-plane features from an image.
"""

import numpy as np
import cv2
import pywt
from scipy import stats


def chi_square_lsb(channel: np.ndarray) -> float:
    """
    Chi-square test on LSB pairs.
    Natural images fail chi-square when LSB is tampered (watermarked).
    Returns p-value: low p → likely watermarked.
    """
    pixels = channel.flatten().astype(np.int32)
    even = pixels[pixels % 2 == 0]
    odd = pixels[pixels % 2 == 1]

    hist_even, _ = np.histogram(even, bins=128, range=(0, 256))
    hist_odd, _ = np.histogram(odd, bins=128, range=(0, 256))

    expected = (hist_even + hist_odd) / 2.0 + 1e-8
    chi2 = np.sum((hist_odd - expected) ** 2 / expected)
    return float(chi2)


def bit_plane_entropy(channel: np.ndarray, plane: int = 0) -> float:
    """Entropy of the specified bit-plane (0=LSB). High entropy in LSB → possible watermark."""
    bit = (channel >> plane) & 1
    p1 = np.mean(bit)
    p0 = 1 - p1
    if p0 == 0 or p1 == 0:
        return 0.0
    return float(-p0 * np.log2(p0) - p1 * np.log2(p1))


def channel_histogram_entropy(channel: np.ndarray) -> float:
    """Shannon entropy of the full pixel histogram."""
    hist, _ = np.histogram(channel, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def dwt_energy_ratio(channel: np.ndarray) -> float:
    """
    Ratio of high-frequency subband energy to low-frequency.
    Watermarks increase high-freq energy.
    """
    channel_f = channel.astype(np.float64)
    cA, (cH, cV, cD) = pywt.dwt2(channel_f, 'haar')
    high_energy = np.sum(cH ** 2) + np.sum(cV ** 2) + np.sum(cD ** 2)
    low_energy = np.sum(cA ** 2) + 1e-8
    return float(high_energy / low_energy)


def dct_stats(channel: np.ndarray) -> list[float]:
    """Mean and std of DCT coefficients (AC components) — watermarks shift these."""
    channel_f = channel.astype(np.float32)
    dct = cv2.dct(cv2.resize(channel_f, (128, 128)))
    ac = dct.flatten()[1:]  # skip DC
    return [float(np.mean(np.abs(ac))), float(np.std(ac))]


def pixel_pair_correlation(channel: np.ndarray) -> float:
    """
    RS steganalysis-inspired: correlation between adjacent pixel pairs.
    LSB watermarking breaks natural correlations.
    """
    flat = channel.flatten().astype(np.float64)
    if len(flat) < 2:
        return 0.0
    corr = np.corrcoef(flat[:-1], flat[1:])[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def color_channel_correlation(image: np.ndarray) -> list[float]:
    """
    Cross-channel (R-G, G-B, R-B) Pearson correlations.
    Watermarking one channel breaks natural inter-channel correlation.
    """
    r = image[:, :, 2].flatten().astype(np.float64)
    g = image[:, :, 1].flatten().astype(np.float64)
    b = image[:, :, 0].flatten().astype(np.float64)
    rg = float(np.corrcoef(r, g)[0, 1])
    gb = float(np.corrcoef(g, b)[0, 1])
    rb = float(np.corrcoef(r, b)[0, 1])
    return [rg if not np.isnan(rg) else 0.0,
            gb if not np.isnan(gb) else 0.0,
            rb if not np.isnan(rb) else 0.0]


def svd_singular_values(channel: np.ndarray, n: int = 10) -> list[float]:
    """Top-n singular values normalized — DWT+SVD watermarks shift singular values."""
    small = cv2.resize(channel.astype(np.float64), (64, 64))
    _, S, _ = np.linalg.svd(small, full_matrices=False)
    S_norm = S[:n] / (S[0] + 1e-8)
    return S_norm.tolist()


def visible_watermark_features(image: np.ndarray) -> list:
    """
    Features specifically designed to detect visible text/logo watermarks.
    These capture edge density, local contrast anomalies, and gradient patterns
    that semi-transparent text overlays create.
    """
    img = cv2.resize(image, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    # 1. Canny edge density (watermark text = dense edges)
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    edge_density = float(edges.mean())

    # 2. Local variance grid — find anomalous high-contrast patches
    grid = 4
    gh, gw = h // grid, w // grid
    variances = []
    for i in range(grid):
        for j in range(grid):
            patch = gray[i*gh:(i+1)*gh, j*gw:(j+1)*gw]
            variances.append(float(np.var(patch)))
    var_arr = np.array(variances)
    var_mean = float(var_arr.mean())
    var_std  = float(var_arr.std())
    var_max  = float(var_arr.max())
    anomaly_ratio = float(np.sum(var_arr > var_mean + 1.5 * var_std) / len(variances))

    # 3. Gradient magnitude stats
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    grad_mean = float(grad_mag.mean())
    grad_std  = float(grad_mag.std())
    grad_p90  = float(np.percentile(grad_mag, 90))
    grad_p99  = float(np.percentile(grad_mag, 99))

    # 4. Bright pixel ratio (semi-transparent white text bleaches pixels)
    _, bright = cv2.threshold(gray.astype(np.uint8), 200, 255, cv2.THRESH_BINARY)
    bright_ratio = float(bright.mean() / 255)

    # 5. Local texture uniformity: visible WM disrupts smooth regions
    laplacian = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
    lap_var = float(np.var(laplacian))
    lap_mean = float(np.abs(laplacian).mean())

    # 6. Top-half vs bottom-half variance difference
    # (many watermarks placed in center/bottom)
    top_var = float(np.var(gray[:h//2, :]))
    bot_var = float(np.var(gray[h//2:, :]))
    half_var_diff = abs(top_var - bot_var)

    # 7. MSER region count (proxy for text blobs)
    try:
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray.astype(np.uint8))
        mser_count = float(len(regions))
    except Exception:
        mser_count = 0.0

    return [
        edge_density, var_mean, var_std, var_max, anomaly_ratio,
        grad_mean, grad_std, grad_p90, grad_p99, bright_ratio,
        lap_var, lap_mean, half_var_diff, mser_count
    ]


def extract_features(image: np.ndarray) -> np.ndarray:
    """
    Extract a comprehensive 64-dimensional feature vector from an image.
    Covers: LSB/chi-sq, DWT energy, DCT stats, SVD, color correlation,
    YCbCr entropy, global stats, AND visible watermark detection features.
    """
    if image is None or image.size == 0:
        return np.zeros(64)

    # Ensure 3-channel BGR
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, (256, 256))

    features = []

    channels = {
        'B': image[:, :, 0],
        'G': image[:, :, 1],
        'R': image[:, :, 2]
    }

    for ch_name, ch in channels.items():
        # 1. Chi-square LSB test (1 per channel = 3)
        features.append(chi_square_lsb(ch))

        # 2. LSB plane entropy (1 per channel = 3)
        features.append(bit_plane_entropy(ch, plane=0))

        # 3. Histogram entropy (1 per channel = 3)
        features.append(channel_histogram_entropy(ch))

        # 4. DWT energy ratio (1 per channel = 3)
        features.append(dwt_energy_ratio(ch))

        # 5. DCT mean + std (2 per channel = 6)
        features.extend(dct_stats(ch))

        # 6. Pixel pair correlation (1 per channel = 3)
        features.append(pixel_pair_correlation(ch))

        # 7. SVD top-5 singular values (5 per channel = 15)
        features.extend(svd_singular_values(ch, n=5))

    # 8. Cross-channel correlations (3)
    features.extend(color_channel_correlation(image))

    # 9. YCbCr chroma entropy (2)
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    features.append(channel_histogram_entropy(ycbcr[:, :, 1]))  # Cb
    features.append(channel_histogram_entropy(ycbcr[:, :, 2]))  # Cr

    # 10. Global image statistics (4)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features.append(float(np.mean(gray)))
    features.append(float(np.std(gray)))
    features.append(float(stats.skew(gray.flatten())))
    features.append(float(stats.kurtosis(gray.flatten())))

    # 11. Visible watermark features (14) — detects text/logo overlays
    features.extend(visible_watermark_features(image))

    feat_array = np.array(features, dtype=np.float64)
    feat_array = np.nan_to_num(feat_array, nan=0.0, posinf=1e6, neginf=-1e6)

    # Pad or trim to exactly 64
    if len(feat_array) < 64:
        feat_array = np.pad(feat_array, (0, 64 - len(feat_array)))
    else:
        feat_array = feat_array[:64]

    return feat_array
