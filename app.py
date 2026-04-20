"""
WatermarkShield Professional - Visible Watermark Detection System
Detects: Text watermarks, Image/Logo watermarks, Tiled watermarks, Color watermarks
"""

import streamlit as st
import base64
import numpy as np
import cv2
import os
import time
from pathlib import Path
from PIL import Image
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from watermark import detect_visible_watermark


BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "watermarkshield_logo.svg"
if LOGO_PATH.exists():
    LOGO_SRC = "data:image/svg+xml;base64," + base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
else:
    LOGO_SRC = ""


# 
# Page Config
# 
st.set_page_config(
    page_title="WatermarkShield | Visible Detection",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 
# CSS
# 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);color:#e8e8f0;}
#MainMenu,footer{visibility:hidden;}

.hero-banner{background:linear-gradient(90deg,#6a11cb 0%,#2575fc 100%);
 border-radius:18px;padding:2.2rem 3rem;margin-bottom:1.5rem;text-align:center;
 box-shadow:0 8px 32px rgba(100,17,203,.35);}
.hero-title{font-size:2.5rem;font-weight:800;color:white;margin:0;}
.hero-subtitle{font-size:1rem;color:rgba(255,255,255,.85);margin-top:.5rem;}
.hero-badge{display:inline-block;background:rgba(255,255,255,.18);border:1px solid rgba(255,255,255,.3);
 border-radius:20px;padding:.2rem 1rem;font-size:.78rem;color:white;margin-top:.8rem;
 letter-spacing:1px;text-transform:uppercase;}

.glass-card{background:rgba(255,255,255,.07);backdrop-filter:blur(12px);
 border:1px solid rgba(255,255,255,.15);border-radius:16px;padding:1.5rem;margin-bottom:1rem;}

.verdict-yes{background:linear-gradient(135deg,rgba(34,197,94,.2),rgba(16,185,129,.15));
 border:2px solid rgba(34,197,94,.5);border-radius:16px;padding:1.5rem;
 text-align:center;font-size:1.3rem;font-weight:700;color:#4ade80;margin:1rem 0;}
.verdict-no{background:linear-gradient(135deg,rgba(239,68,68,.2),rgba(220,38,38,.15));
 border:2px solid rgba(239,68,68,.5);border-radius:16px;padding:1.5rem;
 text-align:center;font-size:1.3rem;font-weight:700;color:#f87171;margin:1rem 0;}

.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:1rem;margin:1.2rem 0;}
.metric-card{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);
 border-radius:12px;padding:1rem;text-align:center;}
.metric-value{font-size:1.7rem;font-weight:700;color:#a78bfa;}
.metric-label{font-size:.72rem;color:rgba(255,255,255,.5);margin-top:.3rem;
 text-transform:uppercase;letter-spacing:.5px;}

.step-box{background:rgba(255,255,255,.04);border-left:4px solid #6a11cb;
 border-radius:0 12px 12px 0;padding:1rem 1.5rem;margin:.7rem 0;}
.step-title{font-size:1rem;font-weight:700;color:#a78bfa;margin-bottom:.4rem;}
.step-desc{font-size:.87rem;color:rgba(255,255,255,.7);line-height:1.6;}

.dataset-box{background:linear-gradient(135deg,rgba(106,17,203,.15),rgba(37,117,252,.15));
 border:1px solid rgba(167,139,250,.3);border-radius:12px;padding:1.2rem;}

.stat-row{display:flex;justify-content:space-between;padding:.35rem 0;
 border-bottom:1px solid rgba(255,255,255,.08);}
.stat-label{color:rgba(255,255,255,.7);font-size:.85rem;}
.stat-value{color:#a78bfa;font-weight:700;font-size:.85rem;}
</style>
""", unsafe_allow_html=True)


# 
# Utilities
# 

@st.cache_data(ttl=60, show_spinner=False)
def get_dataset_stats():
    stats = {"train_clean": 0, "train_watermarked": 0, "test_clean": 0, "test_watermarked": 0}
    root = Path("dataset")
    for subset in ["train", "test"]:
        for cls in ["clean", "watermarked"]:
            folder = root / subset / cls
            if folder.exists():
                stats[f"{subset}_{cls}"] = len(list(folder.glob("*.*")))
    return stats


def pil_to_bgr(pil_img):
    img = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def detect_color_watermark(bgr_img: np.ndarray) -> dict:
    """
    Detect color-based visible watermarks (semi-transparent tinted overlays, color washes).

    CORE PRINCIPLE: A color watermark tints NEUTRAL/GRAY areas too—it is an additive
    overlay that shifts even near-white and near-gray pixels.  Naturally colorful
    objects (orange cars, red signs, fruit) do NOT tint the image's neutral regions.
    We therefore test neutral-tone pixels separately as the primary discriminator.
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    s_ch = hsv[:, :, 1]

    b_f = bgr_img[:, :, 0].astype(np.float32)
    g_f = bgr_img[:, :, 1].astype(np.float32)
    r_f = bgr_img[:, :, 2].astype(np.float32)

    # -- SIGNAL 1 (PRIMARY): Neutral-tone tinting -----------------------------
    # Near-gray pixels: per-pixel std across BGR channels is small.
    per_px_std = np.std(bgr_img.astype(np.float32), axis=2)
    neutral_mask = per_px_std < 22
    neutral_ratio = float(np.mean(neutral_mask))

    gray_tint_score = 0.0
    neutral_chroma_mean = 0.0
    if neutral_ratio > 0.04:
        nb = b_f[neutral_mask];  ng = g_f[neutral_mask];  nr = r_f[neutral_mask]
        means = np.array([nb.mean(), ng.mean(), nr.mean()])
        gray_imbalance = float(np.std(means) / (np.mean(means) + 1e-6))
        # Any channel shift above ~6 % in the gray pixels signals a color tint.
        gray_tint_score = min(1.0, gray_imbalance * 10.0)

        a_lab = lab[:, :, 1].astype(np.float32) - 128
        b_lab = lab[:, :, 2].astype(np.float32) - 128
        chroma = np.sqrt(a_lab ** 2 + b_lab ** 2)
        neutral_chroma_mean = float(np.mean(chroma[neutral_mask]))

    # -- SIGNAL 2: Global saturation spread -----------------------------------
    # Watermark overlays elevate saturation uniformly across ALL quadrants.
    # A single colored object elevates only its own quadrant.
    H, W = s_ch.shape
    hm, wm = H // 2, W // 2
    q_sats = [
        float(np.mean(s_ch[:hm, :wm])),
        float(np.mean(s_ch[:hm, wm:])),
        float(np.mean(s_ch[hm:, :wm])),
        float(np.mean(s_ch[hm:, wm:])),
    ]
    quads_elevated = sum(1 for q in q_sats if q > 65)
    global_sat_mean = float(np.mean(s_ch))

    # -- SIGNAL 3: Tight global hue peak (color wash) -------------------------
    hist_h = cv2.calcHist([hsv], [0], None, [36], [0, 180]).flatten()
    hist_norm = hist_h / (hist_h.sum() + 1e-6)
    hue_peak = float(hist_norm.max())
    dominant_bin = int(np.argmax(hist_norm))

    # -- SIGNAL 4: Full-image channel imbalance --------------------------------
    full_means = np.array([b_f.mean(), g_f.mean(), r_f.mean()])
    channel_imbalance = float(np.std(full_means) / (np.mean(full_means) + 1e-6))

    # -- SIGNAL 5: chroma P95 across whole image -------------------------------
    a_all = lab[:, :, 1].astype(np.float32) - 128
    b_all = lab[:, :, 2].astype(np.float32) - 128
    chroma_p95 = float(np.percentile(np.sqrt(a_all ** 2 + b_all ** 2), 95))

    # -- Edge density (natural scene veto) ------------------------------------
    gray_u8 = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    edge_density = float(np.mean(cv2.Canny(gray_u8, 80, 160) > 0))

    # -- Scoring --------------------------------------------------------------
    score = 0.0
    fired = 0

    # Primary: neutral tinting is the strongest discriminator.
    if gray_tint_score > 0.45:
        score += 0.50; fired += 1
    elif gray_tint_score > 0.25:
        score += 0.30; fired += 1

    # Neutral-area chroma: tinted grays are chromatic.
    if neutral_chroma_mean > 28:
        score += 0.25; fired += 1
    elif neutral_chroma_mean > 16:
        score += 0.12; fired += 1

    # Global saturation in ALL quadrants (not just one object's region).
    if quads_elevated >= 4 and global_sat_mean > 75:
        score += 0.25; fired += 1
    elif quads_elevated >= 3 and global_sat_mean > 80:
        score += 0.15; fired += 1

    # Very tight global hue concentration (monochrome wash).
    if hue_peak > 0.45 and global_sat_mean > 70:
        score += 0.20; fired += 1

    # Channel imbalance (must be strong to count alone).
    if channel_imbalance > 0.22:
        score += 0.15; fired += 1

    # -- Hard gates -----------------------------------------------------------
    # Require neutral-tinting evidence OR 4+ other signals.
    if gray_tint_score < 0.20 and fired < 4:
        score = 0.0

    # High-edge natural scene with no neutral tinting ? suppress.
    if edge_density > 0.13 and gray_tint_score < 0.20:
        score *= 0.25

    final = min(score, 1.0)
    high_sat_ratio = float(np.mean(s_ch > 150))
    return {
        "score": round(final, 3),
        "is_color_wm": final >= 0.55,
        "high_sat_ratio": round(high_sat_ratio, 3),
        "hue_peak": round(hue_peak, 3),
        "channel_imbalance": round(channel_imbalance, 3),
        "uniform_ratio": round(gray_tint_score, 3),
        "chroma_p95": round(chroma_p95, 2),
        "sat_mean": round(global_sat_mean, 1),
        "hue_concentration": int(dominant_bin),
    }


# 
# SIDEBAR
# 

with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:1rem 0 0.5rem 0;'>
        <div style='display:flex;justify-content:center;margin-bottom:0.75rem;'>
            <img src='{LOGO_SRC}' alt='WatermarkShield logo' style='width:84px;height:84px;border-radius:22px;box-shadow:0 10px 24px rgba(0,0,0,.28);' />
        </div>
        <div style='font-weight:800;font-size:1.1rem;color:white;letter-spacing:0.4px;'>WatermarkShield</div>
        <div style='font-size:.73rem;color:rgba(255,255,255,.5);margin-top:4px;'>Visible Detection System</div>
    </div>
    <hr style='border-color:rgba(255,255,255,.1);'>
    """, unsafe_allow_html=True)

    st.markdown("** Upload Image**")
    uploaded = st.file_uploader(
        "Supported: JPG, PNG, BMP, TIFF, WebP",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("** Dataset Overview**")
    ds = get_dataset_stats()
    train_total = ds['train_clean'] + ds['train_watermarked']
    test_total  = ds['test_clean']  + ds['test_watermarked']
    grand_total = train_total + test_total

    st.markdown(f"""
    <div class="dataset-box">
        <div style='font-weight:700;color:#a78bfa;font-size:.95rem;margin-bottom:.5rem;'> Training Set</div>
        <div class="stat-row"><span class="stat-label">Watermarked</span><span class="stat-value">{ds['train_watermarked']:,}</span></div>
        <div class="stat-row"><span class="stat-label">Clean</span><span class="stat-value">{ds['train_clean']:,}</span></div>
        <div class="stat-row" style='border:none;'><span class="stat-label"><strong>Total</strong></span><span class="stat-value"><strong>{train_total:,}</strong></span></div>
        <br>
        <div style='font-weight:700;color:#a78bfa;font-size:.95rem;margin-bottom:.5rem;'> Test Set</div>
        <div class="stat-row"><span class="stat-label">Watermarked</span><span class="stat-value">{ds['test_watermarked']:,}</span></div>
        <div class="stat-row"><span class="stat-label">Clean</span><span class="stat-value">{ds['test_clean']:,}</span></div>
        <div class="stat-row" style='border:none;'><span class="stat-label"><strong>Total</strong></span><span class="stat-value"><strong>{test_total:,}</strong></span></div>
        <br>
        <div style='text-align:center;background:rgba(167,139,250,.15);border-radius:8px;padding:.5rem;'>
            <span style='color:white;font-size:.9rem;'>Grand Total: <strong style="color:#a78bfa;">{grand_total:,}</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <hr style='border-color:rgba(255,255,255,.1);'>
    <div style='font-size:.7rem;color:rgba(255,255,255,.3);text-align:center;'>
        &copy; Kalana Kuruppu and Haritha Naveen
    </div>
    """, unsafe_allow_html=True)


# 
# HERO BANNER
# 

st.markdown(f"""
<div class="hero-banner">
    <div style='display:flex;justify-content:center;margin-bottom:0.8rem;'>
        <img src='{LOGO_SRC}' alt='WatermarkShield logo' style='width:110px;height:110px;border-radius:28px;box-shadow:0 14px 32px rgba(0,0,0,.32);' />
    </div>
    <p class="hero-title">WatermarkShield Professional</p>
    <p class="hero-subtitle">
        Visible Watermark Detection  Text  Logo/Image  Tiled  Color Overlays<br>
        Rule-based multi-signal sliding window analysis
    </p>
    <span class="hero-badge">3 Detection Techniques  Anywhere in Image  Step-by-Step Pipeline</span>
</div>
""", unsafe_allow_html=True)


# 
# NO IMAGE  show dataset overview
# 

if uploaded is None:
    st.markdown("###  Dataset Overview")
    ds = get_dataset_stats()
    train_total = ds['train_clean'] + ds['train_watermarked']
    test_total  = ds['test_clean']  + ds['test_watermarked']

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("<div class='glass-card'><h4 style='color:#a78bfa;margin-top:0;'> Training Dataset</h4></div>", unsafe_allow_html=True)
        fig = go.Figure(data=[
            go.Bar(name='Watermarked', x=['Training'], y=[ds['train_watermarked']], marker_color='#a78bfa'),
            go.Bar(name='Clean',       x=['Training'], y=[ds['train_clean']],       marker_color='#60a5fa'),
        ])
        fig.update_layout(barmode='group', template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                          height=220, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, width='stretch')
        st.markdown(f"""
        | Category | Count | Source |
        |----------|-------|--------|
        | Watermarked | **{ds['train_watermarked']:,}** | Kaggle + Synthetic |
        | Clean | **{ds['train_clean']:,}** | Kaggle + Synthetic |
        | **Total** | **{train_total:,}** | Combined |
        """)

    with c2:
        st.markdown("<div class='glass-card'><h4 style='color:#34d399;margin-top:0;'> Test Dataset</h4></div>", unsafe_allow_html=True)
        fig2 = go.Figure(data=[
            go.Bar(name='Watermarked', x=['Test'], y=[ds['test_watermarked']], marker_color='#a78bfa'),
            go.Bar(name='Clean',       x=['Test'], y=[ds['test_clean']],       marker_color='#60a5fa'),
        ])
        fig2.update_layout(barmode='group', template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                           height=220, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig2, width='stretch')
        st.markdown(f"""
        | Category | Count | Source |
        |----------|-------|--------|
        | Watermarked | **{ds['test_watermarked']:,}** | Kaggle valid + Synthetic |
        | Clean | **{ds['test_clean']:,}** | Kaggle valid + Synthetic |
        | **Total** | **{test_total:,}** | Combined |
        """)

    c3, c4 = st.columns(2)
    with c3:
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Train WM', 'Train Clean', 'Test WM', 'Test Clean'],
            values=[ds['train_watermarked'], ds['train_clean'], ds['test_watermarked'], ds['test_clean']],
            marker=dict(colors=['#a78bfa','#60a5fa','#f472b6','#34d399']), hole=0.4
        )])
        fig_pie.update_layout(title='Dataset Distribution', template='plotly_dark',
                              paper_bgcolor='rgba(0,0,0,0)', height=300,
                              margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_pie, width='stretch')

    with c4:
        st.markdown("""
        <div class="glass-card">
            <h4 style='color:#a78bfa;'> Sources</h4>
            <ul style='color:rgba(255,255,255,.8);line-height:2;'>
                <li><strong>Kaggle</strong>: 24,987 real-world internet images with stock watermarks</li>
                <li><strong>Synthetic</strong>: 2,570 controlled watermark images</li>
            </ul>
            <h5 style='color:#a78bfa;'> Watermark Types Detected</h5>
            <ul style='color:rgba(255,255,255,.8);line-height:1.8;'>
                <li> <strong>Text Watermarks</strong>: copyright text, URLs, names</li>
                <li> <strong>Logo / Image Watermarks</strong>: brand stamps, seals</li>
                <li> <strong>Tiled Watermarks</strong>: repeating grid patterns</li>
                <li> <strong>Color Watermarks</strong>: tinted overlays, color washes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style='text-align:center;padding:3rem;'>
        <div style='font-size:4rem;margin-bottom:1rem;'></div>
        <h3 style='color:white;'>Upload an Image to Begin Analysis</h3>
        <p style='color:rgba(255,255,255,.6);margin-top:1rem;'>
            Use the sidebar to upload  System runs 3 processing stages and shows each step
        </p>
    </div>
    """, unsafe_allow_html=True)


# 
# IMAGE UPLOADED  full analysis
# 

if uploaded is not None:
    pil_img  = Image.open(uploaded)
    bgr_img  = pil_to_bgr(pil_img)
    bgr_img  = cv2.resize(bgr_img, (256, 256))
    gray     = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    def run_all(img_bytes_data):
        import io
        pil = Image.open(io.BytesIO(img_bytes_data)).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (256, 256))
        t0 = time.time(); vis = detect_visible_watermark(bgr); vt = time.time()-t0
        t1 = time.time(); col = detect_color_watermark(bgr); ct = time.time()-t1
        return {"vis": vis, "col": col, "vt": vt, "ct": ct}

    uploaded.seek(0)
    img_bytes = uploaded.read()
    R = run_all(img_bytes)

    vis_result   = R["vis"]
    col_result   = R["col"]
    vis_detected = vis_result["is_visible_wm"]

    # Quick demo mode (default ON): ignore color branch to avoid false positives
    # on naturally colorful objects (cars, boards, logos) when time is limited.
    quick_demo_mode = os.getenv("WM_DEMO_QUICK", "1") == "1"
    col_detected = (col_result["is_color_wm"] if not quick_demo_mode else False)
    is_watermarked = vis_detected or col_detected

    vis_kind = vis_result.get("watermark_kind", "none")
    if is_watermarked:
        if vis_detected and vis_kind == "text watermark":
            wm_type = "Text Watermark"
        elif vis_detected and vis_kind == "image watermark":
            wm_type = "Image Watermark"
        elif col_detected:
            wm_type = "Image Watermark"
        else:
            wm_type = "Watermarked"
    else:
        wm_type = "No watermark"

    tab1, tab2, tab3 = st.tabs([" Detection Result", " Processing Pipeline", " System Info"])


    # 
    # TAB 1: Result
    # 
    with tab1:
        col_img, col_res = st.columns([1, 1.5], gap="large")

        with col_img:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("** Uploaded Image**")
            st.image(pil_img, width='stretch')
            st.markdown(f"<div style='font-size:.8rem;color:rgba(255,255,255,.5);'>{uploaded.name}  256256 px</div>",
                        unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_res:
            if is_watermarked:
                st.markdown(f"""<div class="verdict-yes"> WATERMARKED<br>
                    <span style='font-size:.95rem;font-weight:400;'>{wm_type}</span></div>""",
                    unsafe_allow_html=True)
            else:
                st.markdown("""<div class="verdict-no"> NO WATERMARK<br>
                    <span style='font-size:1rem;font-weight:400;'>Clean Image</span></div>""",
                    unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-card"><div class="metric-value">{vis_result['score']*100:.1f}%</div><div class="metric-label">Visible Score</div></div>
                <div class="metric-card"><div class="metric-value">{col_result['score']*100:.1f}%</div><div class="metric-label">Color Score</div></div>
                <div class="metric-card"><div class="metric-value">{" YES" if is_watermarked else " NO"}</div><div class="metric-label">Watermark</div></div>
                <div class="metric-card"><div class="metric-value">DIP+DL</div><div class="metric-label">Method</div></div>
            </div>
            """, unsafe_allow_html=True)

            # Dual gauge bar chart
            fig_b = go.Figure()
            fig_b.add_trace(go.Bar(
                x=['Visible WM Score', 'Color WM Score'],
                y=[vis_result['score'], col_result['score']],
                marker_color=['#a78bfa', '#f472b6'],
                text=[f"{vis_result['score']*100:.1f}%", f"{col_result['score']*100:.1f}%"],
                textposition='auto'
            ))
            fig_b.add_hline(y=0.38, line_dash="dash", line_color="#fbbf24",
                            annotation_text="Visible threshold (0.38)", annotation_font_color="#fbbf24")
            fig_b.add_hline(y=0.45, line_dash="dot", line_color="#f472b6",
                            annotation_text="Color threshold (0.45)", annotation_font_color="#f472b6",
                            annotation_position="bottom right")
            fig_b.update_layout(title='Detection Scores', template='plotly_dark',
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                yaxis_range=[0, 1], height=260,
                                margin=dict(l=15,r=15,t=45,b=15), font=dict(family="Inter",color="white"))
            st.plotly_chart(fig_b, width='stretch')

            if vis_detected:
                with st.expander(" Visible Watermark Signals"):
                    st.markdown(f"""
                    | Signal | Value | Status |
                    |--------|-------|--------|
                    | Edge Density | `{vis_result['edge_density']:.3f}` | {' High' if vis_result['edge_density'] > 0.15 else ' Normal'} |
                    | Anomaly Ratio | `{vis_result['anomaly_ratio']:.3f}` | {' High' if vis_result['anomaly_ratio'] > 0.1 else ' Normal'} |
                    | Gradient P95 | `{vis_result['grad_p95']:.2f}` | {' Sharp boundaries' if vis_result['grad_p95'] > 50 else ' Smooth'} |
                    | MSER Regions | `{vis_result['mser_regions']}` | {' Text blobs' if vis_result['mser_regions'] > 100 else ' Normal'} |
                    | Text Score | `{vis_result.get('text_score', 0.0):.3f}` | {' Dominant' if vis_result.get('text_score', 0.0) >= vis_result.get('image_score', 0.0) else ' Secondary'} |
                    | Image Score | `{vis_result.get('image_score', 0.0):.3f}` | {' Dominant' if vis_result.get('image_score', 0.0) > vis_result.get('text_score', 0.0) else ' Secondary'} |
                    | Model Score | `{vis_result.get('ml_score', 0.0):.3f}` | {' Supports watermark' if vis_result.get('ml_score', 0.0) > 0.5 else ' Supports clean'} |
                    | Deep Score | `{vis_result.get('deep_score', 0.0):.3f}` | {' Supports watermark' if vis_result.get('deep_score', 0.0) > 0.5 else ' Supports clean'} |
                    | Fused Threshold | `{vis_result.get('threshold', 0.38):.3f}` | {' Dynamic' if vis_result.get('threshold', 0.38) != 0.38 else ' Default'} |
                    | **Score** | **`{vis_result['score']:.3f}`** | **{' POSITIVE' if vis_detected else ' NEGATIVE'}** |
                    """)

            if col_detected:
                with st.expander(" Color Watermark Signals"):
                    st.markdown(f"""
                    | Signal | Value | Status |
                    |--------|-------|--------|
                    | High-Sat Ratio | `{col_result['high_sat_ratio']:.3f}` | {' Color overlay' if col_result['high_sat_ratio'] > 0.10 else ' Normal'} |
                    | Hue Peak | `{col_result['hue_peak']:.3f}` | {' Color cluster' if col_result['hue_peak'] > 0.20 else ' Normal'} |
                    | Channel Imbalance | `{col_result['channel_imbalance']:.3f}` | {' Tint present' if col_result['channel_imbalance'] > 0.15 else ' Balanced'} |
                    | Uniform Patches | `{col_result['uniform_ratio']:.3f}` | {' Overlay regions' if col_result['uniform_ratio'] > 0.15 else ' Varied'} |
                    | Chroma P95 | `{col_result['chroma_p95']:.1f}` | {' High chroma' if col_result['chroma_p95'] > 70 else ' Normal'} |
                    | **Score** | **`{col_result['score']:.3f}`** | **{' POSITIVE (>0.45)' if col_detected else ' NEGATIVE'}** |
                    """)


    # 
    # TAB 2: Processing Pipeline
    # 
    with tab2:
        st.markdown("##  Step-by-Step Processing Pipeline")
        st.caption("Each step shows: technique used, why we use it, and the output image from that step")

        # STEP 1
        with st.expander("**STEP 1  Image Acquisition & Preprocessing**", expanded=True):
            st.markdown("""
            <div class="step-box">
                <div class="step-title"> DIP Technique: Normalization + Color Space Conversion</div>
                <div class="step-desc">
                     Why: all images must be in the same format and size so later analysis is fair and stable.<br>
                     Result: image is decoded, resized to <strong>256  256</strong>, and converted to grayscale/HSV/L*a*b*.
                </div>
            </div>
            """, unsafe_allow_html=True)

            hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
            lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.image(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB),
                         caption="Original (256256)", width='stretch')
            with c2:
                st.image(gray, caption="Grayscale", width='stretch', clamp=True)
            with c3:
                st.image(hsv_img[:,:,0], caption="Hue Channel (H)", width='stretch', clamp=True)
            with c4:
                st.image(hsv_img[:,:,1], caption="Saturation Channel (S)", width='stretch', clamp=True)

            h_img, w_img = bgr_img.shape[:2]
            b_mean = float(bgr_img[:,:,0].mean())
            g_mean = float(bgr_img[:,:,1].mean())
            r_mean = float(bgr_img[:,:,2].mean())
            st.markdown(f"""
            | Property | Value |
            |----------|-------|
            | Dimensions | `{w_img}  {h_img} px` |
            | Blue mean | `{b_mean:.1f}` |
            | Green mean | `{g_mean:.1f}` |
            | Red mean | `{r_mean:.1f}` |
            | Brightness | `{float(gray.mean()):.1f}` |
            """)

        # STEP 2
        with st.expander("**STEP 2  Visible Watermark Detection (Text / Logo / Tiled)**", expanded=True):
            st.markdown("""
            <div class="step-box">
                <div class="step-title"> DIP Techniques Used</div>
                <div class="step-desc">
                    <strong>1) CLAHE (contrast enhancement)</strong>: boosts faint low-opacity watermark traces.<br>
                    <strong>2) Adaptive Threshold</strong>: separates probable watermark pixels from background.<br>
                    <strong>3) Residual/High-pass map</strong>: highlights overlay patterns not natural to scene texture.<br>
                    <strong>4) Edge & gradient analysis</strong>: captures repeated text/logo stroke structure.<br>
                    Why: visible watermarks are usually semi-transparent overlays with repetitive boundaries.
                </div>
            </div>
            """, unsafe_allow_html=True)

            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            ad = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, -2)
            blurred = cv2.GaussianBlur(gray.astype(np.float32), (31, 31), 10)
            residual = np.abs(gray.astype(np.float32) - blurred)
            res_norm = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            edges = cv2.Canny(gray, 100, 200)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.image(enhanced, caption="CLAHE Result", width='stretch', clamp=True)
            with c2:
                st.image(ad, caption="Adaptive Threshold Result", width='stretch', clamp=True)
            with c3:
                st.image(res_norm, caption="Residual (High-pass) Result", width='stretch', clamp=True)
            with c4:
                st.image(edges, caption="Edge Map (Canny)", width='stretch', clamp=True)

            st.markdown(f"""
            **Sliding Window Results:**
            | Signal | Value | Status |
            |--------|-------|--------|
            | Edge Density | `{vis_result['edge_density']:.3f}` | {' Text/logo edges' if vis_result['edge_density'] > 0.15 else ' Normal'} |
            | Gradient P95 | `{vis_result['grad_p95']:.2f}` | {' Sharp boundaries' if vis_result['grad_p95'] > 50 else ' Smooth'} |
            | MSER Regions | `{vis_result['mser_regions']}` | {' Text blobs detected' if vis_result['mser_regions'] > 100 else ' Normal'} |
            | **Visible Score** | **`{vis_result['score']:.3f}`** | **{' POSITIVE (>0.65)' if vis_detected else ' NEGATIVE'}** |
            | Processing Time | `{R['vt']*1000:.1f} ms` | |
            """)

            if vis_detected:
                st.success(" **Visible watermark pattern detected** from DIP features")
            else:
                st.info(" **No visible watermark pattern detected** in DIP stage")

        # STEP 3
        with st.expander("**STEP 3  Color Watermark Detection**", expanded=True):
            st.markdown("""
            <div class="step-box">
                <div class="step-title"> DIP Technique: Color-Space Signal Analysis (HSV + L*a*b*)</div>
                <div class="step-desc">
                    Why: some watermarks are color tints instead of clear text strokes.<br>
                    We measure saturation, hue concentration, channel imbalance, and chroma strength.<br>
                    Result: color-score says whether a tint/overlay watermark is present.
                </div>
            </div>
            """, unsafe_allow_html=True)

            hsv_disp = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
            lab_disp = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)

            c1, c2, c3 = st.columns(3)
            with c1:
                sat_colored = cv2.applyColorMap(hsv_disp[:,:,1], cv2.COLORMAP_JET)
                st.image(cv2.cvtColor(sat_colored, cv2.COLOR_BGR2RGB),
                         caption="Saturation Map (jet colormap)", width='stretch')
            with c2:
                # Hue histogram
                fig, ax = plt.subplots(figsize=(4, 3))
                fig.patch.set_facecolor('#1a1a2e')
                ax.set_facecolor('#1a1a2e')
                hue_vals = hsv_disp[:,:,0].flatten()
                ax.hist(hue_vals, bins=36, range=(0,180), color='#a78bfa', alpha=0.8)
                ax.set_title('Hue Distribution', color='white', fontsize=9)
                ax.tick_params(colors='white', labelsize=7)
                for sp in ax.spines.values(): sp.set_color((0.4,0.4,0.4,0.5))
                st.pyplot(fig, width='stretch')
                plt.close()
            with c3:
                a_ch = lab_disp[:,:,1].astype(np.float32) - 128
                b_ch = lab_disp[:,:,2].astype(np.float32) - 128
                chroma_map = np.sqrt(a_ch**2 + b_ch**2)
                chroma_norm = (chroma_map / (chroma_map.max()+1e-6) * 255).astype(np.uint8)
                chroma_colored = cv2.applyColorMap(chroma_norm, cv2.COLORMAP_HOT)
                st.image(cv2.cvtColor(chroma_colored, cv2.COLOR_BGR2RGB),
                         caption="L*a*b* Chroma Map (bright = saturated color)", width='stretch')

            st.markdown(f"""
            **Color Analysis Results:**
            | Signal | Value | Status |
            |--------|-------|--------|
            | High-Sat Pixel Ratio | `{col_result['high_sat_ratio']:.3f}` | {' Color overlay present' if col_result['high_sat_ratio'] > 0.10 else ' Normal'} |
            | Hue Peak Ratio | `{col_result['hue_peak']:.3f}` | {' Dominant hue cluster' if col_result['hue_peak'] > 0.20 else ' Normal'} |
            | Channel Imbalance | `{col_result['channel_imbalance']:.3f}` | {' Color tint present' if col_result['channel_imbalance'] > 0.15 else ' Balanced'} |
            | Uniform Patch Ratio | `{col_result['uniform_ratio']:.3f}` | {' Overlay regions' if col_result['uniform_ratio'] > 0.15 else ' Varied'} |
            | Chroma P95 | `{col_result['chroma_p95']:.1f}` | {' High chroma' if col_result['chroma_p95'] > 70 else ' Normal'} |
            | **Color Score** | **`{col_result['score']:.3f}`** | **{' POSITIVE (>0.45)' if col_detected else ' NEGATIVE'}** |
            | Processing Time | `{R['ct']*1000:.1f} ms` | |
            """)

            if col_detected:
                st.success(" **Color watermark detected**  color overlay, tint, or logo present")
            else:
                st.info(" **No color watermark detected**")

        # STEP 4
        with st.expander("**STEP 4  ML Model Decision (Final after DIP)**", expanded=True):
            st.markdown("""
            <div class="step-box">
                <div class="step-title"> ML Technique: Random Forest Classifier on DIP Features</div>
                <div class="step-desc">
                    After DIP features are extracted, the ML model predicts probability of watermark.<br>
                    Why: rule-only thresholds can fail on complex natural photos; ML learns better boundaries.<br>
                    Result: ML probability is fused into the final decision score.
                </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("ML Probability", f"{vis_result.get('ml_score', 0.0):.3f}")
            with c2:
                st.metric("Decision Threshold", f"{vis_result.get('threshold', 0.45):.3f}")
            with c3:
                st.metric("Deep Model Support", f"{vis_result.get('deep_score', 0.0):.3f}")

            st.markdown(f"""
            **Model Output:**
            - Final Visible Score: **{vis_result['score']:.3f}**
            - Visible Decision: **{'WATERMARKED' if vis_detected else 'CLEAN'}**
            - Watermark Type: **{vis_result.get('watermark_kind', 'none').title()}**
            """)

        # Final verdicts summary
        st.markdown("---")
        st.markdown("###  Final Verdict")
        c1, c2, c3 = st.columns(3)
        with c1:
            if vis_detected:
                st.success(f"**STEP 2:**  Visible WM  score {vis_result['score']:.3f}")
            else:
                st.info(f"**STEP 2:**  Clean  score {vis_result['score']:.3f}")
        with c2:
            if col_detected:
                st.success(f"**STEP 3:**  Color WM  score {col_result['score']:.3f}")
            else:
                st.info(f"**STEP 3:**  Clean  score {col_result['score']:.3f}")
        with c3:
            if is_watermarked:
                st.success(f"**FINAL:  WATERMARK DETECTED**\n\n{wm_type}")
            else:
                st.info("**FINAL:  CLEAN IMAGE**")


    # 
    # TAB 3: System Info
    # 
    with tab3:
        st.markdown("## About WatermarkShield Professional")
        ds = get_dataset_stats()
        grand = sum(ds.values())

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            ### Detection Capabilities

            **Text Watermarks**
            - Copyright text overlays
            - URL or website watermark text
            - Photographer or brand-name overlays

            **Image or Logo Watermarks**
            - Transparent logo stamps
            - Brand or copyright seals
            - Semi-transparent image overlays

            **Tiled Pattern Watermarks**
            - Repeated text grid across image
            - Diagonal tile patterns
            - Background watermark patterns

            **Color Watermarks**
            - Color-tinted overlays
            - Saturated color logo stamps
            - Channel-shifted color marks
            """)

        with c2:
            st.markdown("### Dataset Statistics")
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric("Train WM", f"{ds['train_watermarked']:,}")
                st.metric("Test WM", f"{ds['test_watermarked']:,}")
            with mc2:
                st.metric("Train Clean", f"{ds['train_clean']:,}")
                st.metric("Test Clean", f"{ds['test_clean']:,}")
            with mc3:
                st.metric("Grand Total", f"{grand:,}")
                st.metric("Techniques", "4")

            st.markdown("""
            ### Technical Stack
            - **Detection:** ML-first (Random Forest) with DIP fallback
            - **ML Model:** scikit-learn Pipeline from `models/detector.pkl`
            - **Vision / DIP:** OpenCV + NumPy + PyWavelets
            - **UI:** Streamlit + Plotly + Matplotlib
            - **Visible DIP:** CLAHE, adaptive threshold, residual map, Canny, Sobel, Hough
            - **Color DIP:** HSV saturation, hue histogram, L*a*b* chroma
            - **Coverage:** full image (global + local signals)
            - **False-positive guard:** thresholded ML decision + multi-signal checks
            """)

        report_path = "results/training_report.png"
        if os.path.exists(report_path):
            st.markdown("###  Dataset Report")
            st.image(report_path, caption="Training Dataset: Distribution Report")


