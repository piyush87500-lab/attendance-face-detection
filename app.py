import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

from utils.face_utils import (
    load_known_faces, save_known_face, mark_attendance,
    load_attendance, recognize_faces
)

st.set_page_config(
    page_title="FaceAttend",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; background-color: #0a0a0f; color: #e8e8f0; }
.stApp { background-color: #0a0a0f; }
h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }
.main-title { font-size: 2.8rem; font-weight: 800; letter-spacing: -1px;
    background: linear-gradient(135deg, #7fffd4, #00bfff, #9370db);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.1; }
.metric-card { background: #13131f; border: 1px solid #2a2a3f; border-radius: 12px;
    padding: 1.2rem 1.5rem; text-align: center; }
.metric-num { font-family: 'Space Mono', monospace; font-size: 2.4rem; font-weight: 700; color: #7fffd4; }
.metric-label { font-size: 0.78rem; color: #888; text-transform: uppercase; letter-spacing: 2px; }
.status-badge { display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; margin: 4px; }
.badge-present { background: #1a3a2a; color: #7fffd4; border: 1px solid #7fffd4; }
.badge-unknown { background: #3a1a1a; color: #ff6b6b; border: 1px solid #ff6b6b; }
div[data-testid="stSidebar"] { background-color: #0d0d1a; border-right: 1px solid #1e1e2e; }
.stButton > button { background: linear-gradient(135deg, #7fffd4 0%, #00bfff 100%);
    color: #0a0a0f; font-family: 'Space Mono', monospace; font-weight: 700;
    font-size: 0.85rem; letter-spacing: 1px; border: none; border-radius: 8px;
    padding: 0.6rem 1.4rem; transition: opacity 0.2s; }
.stButton > button:hover { opacity: 0.85; }
.stTextInput > div > div > input { background: #13131f; border: 1px solid #2a2a3f;
    border-radius: 8px; color: #e8e8f0; font-family: 'Space Mono', monospace; }
.section-header { font-family: 'Space Mono', monospace; font-size: 0.7rem;
    letter-spacing: 3px; text-transform: uppercase; color: #555; margin-bottom: 1rem;
    border-bottom: 1px solid #1e1e2e; padding-bottom: 0.5rem; }
.face-box { border: 2px solid #2a2a3f; border-radius: 8px; padding: 0.8rem 1rem;
    background: #0d1220; font-family: 'Space Mono', monospace; font-size: 0.85rem; }
.upload-hint { font-family: 'Space Mono', monospace; font-size: 0.75rem; color: #555; margin-top: 0.5rem; }
.info-box { background: #0d1a2a; border: 1px solid #1a3a5a; border-radius: 8px;
    padding: 1rem 1.2rem; font-family: 'Space Mono', monospace; font-size: 0.78rem;
    color: #7fb3d3; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)


def run_detection(frame_rgb: np.ndarray):
    with st.spinner("🔍 Analyzing faces... (first run may take ~30s to load model)"):
        annotated, detected = recognize_faces(frame_rgb)
    st.image(annotated, channels="RGB", use_container_width=True)
    if detected:
        st.markdown("**Detected faces:**")
        for n in detected:
            badge = "badge-present" if n != "Unknown" else "badge-unknown"
            st.markdown(f'<span class="status-badge {badge}">{n}</span>', unsafe_allow_html=True)
    else:
        st.warning("No faces detected in the image.")


if "known_encodings" not in st.session_state:
    st.session_state.known_encodings, st.session_state.known_names = load_known_faces()

with st.sidebar:
    st.markdown('<p class="main-title">Face\nAttend</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#555;font-family:Space Mono,monospace;font-size:0.7rem;letter-spacing:2px;">ATTENDANCE SYSTEM v2.0 • DeepFace</p>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", ["📸 Live Detection", "➕ Register Face", "📋 Attendance Log", "👥 Manage Faces"], label_visibility="collapsed")
    st.markdown("---")
    known_count = len(set(st.session_state.known_names))
    today_df = load_attendance(today_only=True)
    st.markdown(f'<div class="metric-card"><div class="metric-num">{known_count}</div><div class="metric-label">Registered</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="metric-card"><div class="metric-num">{len(today_df)}</div><div class="metric-label">Present Today</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Refresh Database"):
        st.session_state.known_encodings, st.session_state.known_names = load_known_faces()
        st.success("Reloaded!")

# ─── Live Detection ────────────────────────────────────────────────────────────
if page == "📸 Live Detection":
    st.markdown('<h1 style="font-size:2rem;font-weight:800;">Live Face Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">Upload an image or use webcam to mark attendance</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">ℹ️ First scan may take 20–40 seconds while DeepFace downloads its model (~500MB). Subsequent scans are fast.</div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Webcam Snapshot"])
    with tab1:
        uploaded = st.file_uploader("Drop an image to scan", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            run_detection(np.array(img))
    with tab2:
        st.info("📷 Allow camera access, then click **Take Photo**.")
        cam_img = st.camera_input("Snapshot", label_visibility="collapsed")
        if cam_img:
            img = Image.open(cam_img).convert("RGB")
            run_detection(np.array(img))

# ─── Register Face ─────────────────────────────────────────────────────────────
elif page == "➕ Register Face":
    st.markdown('<h1 style="font-size:2rem;font-weight:800;">Register a Face</h1>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">Add a new person to the recognition database</p>', unsafe_allow_html=True)
    name_input = st.text_input("Full Name", placeholder="e.g. Priya Sharma")
    tab1, tab2 = st.tabs(["📁 Upload Photo", "📷 Capture from Webcam"])
    with tab1:
        uploaded = st.file_uploader("Clear front-facing photo (solo, good lighting)", type=["jpg", "jpeg", "png"])
        if uploaded and name_input:
            preview = Image.open(uploaded).convert("RGB")
            st.image(preview, width=220, caption="Preview")
            if st.button("✅ Register from Upload"):
                result = save_known_face(np.array(preview), name_input.strip())
                if result == "success":
                    st.success(f"✅ {name_input} registered successfully!")
                    st.session_state.known_encodings, st.session_state.known_names = load_known_faces()
                elif result == "no_face":
                    st.error("No face detected. Try a clearer, well-lit photo facing the camera.")
                elif result == "multiple_faces":
                    st.error("Multiple faces detected. Upload a solo photo.")
                else:
                    st.error(f"Error: {result}")
    with tab2:
        cam_img = st.camera_input("Look straight into the camera", label_visibility="collapsed")
        if cam_img and name_input:
            if st.button("✅ Register from Webcam"):
                img = Image.open(cam_img).convert("RGB")
                result = save_known_face(np.array(img), name_input.strip())
                if result == "success":
                    st.success(f"✅ {name_input} registered successfully!")
                    st.session_state.known_encodings, st.session_state.known_names = load_known_faces()
                elif result == "no_face":
                    st.error("No face detected. Make sure you're well-lit and facing the camera.")
                elif result == "multiple_faces":
                    st.error("Multiple faces detected. Only one person at a time.")
                else:
                    st.error(f"Error: {result}")
    if not name_input:
        st.markdown('<p class="upload-hint">⚠ Enter a name above before registering.</p>', unsafe_allow_html=True)

# ─── Attendance Log ────────────────────────────────────────────────────────────
elif page == "📋 Attendance Log":
    st.markdown('<h1 style="font-size:2rem;font-weight:800;">Attendance Log</h1>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">View and export attendance records</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        filter_opt = st.selectbox("Filter", ["Today", "Last 7 Days", "All Records"])
    with col2:
        df_all = load_attendance()
        st.download_button("⬇ Export CSV", df_all.to_csv(index=False).encode(), "attendance.csv", "text/csv")
    if filter_opt == "Today":
        df = load_attendance(today_only=True)
    elif filter_opt == "Last 7 Days":
        df = load_attendance(days=7)
    else:
        df = load_attendance()
    if df.empty:
        st.info("No attendance records found for this period.")
    else:
        st.dataframe(df.sort_values("Timestamp", ascending=False), use_container_width=True, hide_index=True)
        st.markdown("---")
        st.markdown("**Attendance frequency by person**")
        counts = df["Name"].value_counts().reset_index()
        counts.columns = ["Name", "Days Present"]
        st.bar_chart(counts.set_index("Name"))

# ─── Manage Faces ──────────────────────────────────────────────────────────────
elif page == "👥 Manage Faces":
    st.markdown('<h1 style="font-size:2rem;font-weight:800;">Registered Faces</h1>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">View and remove individuals from the database</p>', unsafe_allow_html=True)
    faces_dir = "known_faces"
    all_files = [f for f in os.listdir(faces_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
    names = sorted(set(f.rsplit("_", 1)[0].replace("_", " ") for f in all_files))
    if not names:
        st.info("No faces registered yet. Go to **Register Face** to add someone.")
    else:
        for name in names:
            safe = name.replace(" ", "_")
            imgs = [f for f in all_files if f.startswith(safe + "_") or f.startswith(safe)]
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                st.markdown(f'<div class="face-box">👤 {name} <span style="color:#555;font-size:0.72rem;">({len(imgs)} photo{"s" if len(imgs)>1 else ""})</span></div>', unsafe_allow_html=True)
            with col2:
                try:
                    st.image(Image.open(os.path.join(faces_dir, imgs[0])), width=60)
                except Exception:
                    pass
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑 Remove", key=f"del_{safe}"):
                    for f in imgs:
                        try:
                            os.remove(os.path.join(faces_dir, f))
                        except Exception:
                            pass
                    # Clear deepface cache
                    cache = os.path.join(faces_dir, "representations_vgg_face.pkl")
                    if os.path.exists(cache):
                        os.remove(cache)
                    st.success(f"Removed {name}.")
                    st.session_state.known_encodings, st.session_state.known_names = load_known_faces()
                    st.rerun()