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
    load_attendance, recognize_faces, remove_person, clean_orphan_attendance
)

st.set_page_config(
    page_title="FaceAttend — by Piyush Choudhary",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

FOOTER = '<p style="text-align:center;color:#5c3d1e;font-size:0.82rem;font-family:Georgia,serif;margin-top:3rem;padding-top:1rem;border-top:1px solid #d4a96a;">Created by:- <strong>Piyush Choudhary</strong></p>'

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;900&family=Lato:wght@300;400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
    background-color: #f5e6d0;
    color: #2c1a0e;
}
.stApp { background-color: #f5e6d0; }

/* Top nav bar */
.topnav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(90deg, #6b3a1f, #a0522d, #6b3a1f);
    padding: 0.8rem 2rem;
    border-radius: 0 0 16px 16px;
    margin-bottom: 1.8rem;
    box-shadow: 0 4px 18px rgba(107,58,31,0.18);
}
.topnav-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 900;
    color: #fdf0e0;
    letter-spacing: 1px;
}
.topnav-subtitle {
    font-size: 0.7rem;
    color: #d4a96a;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: -2px;
}
.nav-links {
    display: flex;
    gap: 0.4rem;
}
.nav-btn {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(212,169,106,0.4);
    color: #fdf0e0 !important;
    font-family: 'Lato', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    padding: 0.45rem 1rem;
    border-radius: 8px;
    cursor: pointer;
    text-decoration: none;
    transition: background 0.2s;
    white-space: nowrap;
}
.nav-btn:hover { background: rgba(255,255,255,0.22); }
.nav-btn.active {
    background: #d4a96a;
    color: #2c1a0e !important;
    border-color: #d4a96a;
}

/* Page heading */
.page-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 900;
    color: #3d1f0a;
    margin-bottom: 0.2rem;
}
.section-header {
    font-size: 0.72rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #9a6b3a;
    margin-bottom: 1.2rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #d4a96a;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    flex: 1;
    background: #fff8f0;
    border: 1px solid #d4a96a;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(107,58,31,0.07);
}
.metric-num {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 900;
    color: #6b3a1f;
}
.metric-label {
    font-size: 0.72rem;
    color: #9a6b3a;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* Badges */
.status-badge {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin: 4px;
}
.badge-present { background: #e8f5e9; color: #2e7d32; border: 1px solid #66bb6a; }
.badge-unknown { background: #fdecea; color: #c62828; border: 1px solid #ef5350; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6b3a1f, #a0522d);
    color: #fdf0e0 !important;
    font-family: 'Lato', sans-serif;
    font-weight: 700;
    font-size: 0.88rem;
    letter-spacing: 0.5px;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.4rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

/* Inputs */
.stTextInput > div > div > input {
    background: #fff8f0;
    border: 1.5px solid #d4a96a;
    border-radius: 8px;
    color: #2c1a0e;
    font-family: 'Lato', sans-serif;
}
.stSelectbox > div > div {
    background: #fff8f0;
    border: 1.5px solid #d4a96a;
    color: #2c1a0e;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #fff8f0;
    border: 2px dashed #d4a96a;
    border-radius: 10px;
    padding: 1rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #eddec8;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #5c3d1e;
    font-weight: 700;
    border-radius: 8px;
}
.stTabs [aria-selected="true"] {
    background: #6b3a1f !important;
    color: #fdf0e0 !important;
}

/* Face box */
.face-box {
    border: 1.5px solid #d4a96a;
    border-radius: 10px;
    padding: 0.8rem 1.1rem;
    background: #fff8f0;
    font-size: 0.88rem;
    color: #2c1a0e;
    margin-bottom: 0.5rem;
}

/* Info box */
.info-box {
    background: #fff3e0;
    border: 1px solid #ffb74d;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.82rem;
    color: #5c3d1e;
    margin-bottom: 1rem;
}

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #d4a96a; border-radius: 10px; }

/* Hide default sidebar toggle */
div[data-testid="stSidebar"] { display: none; }
[data-testid="collapsedControl"] { display: none; }
.topnav { display: none; }
.topnav { display: none; }
div[data-testid="stSidebar"] { display: none; }
[data-testid="collapsedControl"] { display: none; }

/* Upload hint */
.upload-hint { font-size: 0.78rem; color: #9a6b3a; margin-top: 0.4rem; }

/* Selectbox label */
label { color: #5c3d1e !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
if "known_encodings" not in st.session_state:
    clean_orphan_attendance()  # Remove attendance for unregistered people
    st.session_state.known_encodings, st.session_state.known_names = load_known_faces()
if "page" not in st.session_state:
    st.session_state.page = "📸 Live Detection"


# ── Top Navigation Bar ─────────────────────────────────────────────────────────
known_count = len(set(st.session_state.known_names))
today_df = load_attendance(today_only=True)

pages = ["📸 Live Detection", "➕ Register Face", "📋 Attendance Log", "👥 Manage Faces"]

# ── App Heading ───────────────────────────────────────────────────────────────
st.markdown('''
<div style="text-align:center; padding: 1.5rem 0 0.5rem 0;">
    <span style="
        font-family: Playfair Display, Georgia, serif;
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: 3px;
        background: linear-gradient(135deg, #6b3a1f, #c0622a, #e8a045, #c0622a, #6b3a1f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: none;
        display: inline-block;
    ">🎯 FaceAttend</span>
    <div style="
        font-family: Lato, sans-serif;
        font-size: 0.8rem;
        letter-spacing: 6px;
        text-transform: uppercase;
        color: #9a6b3a;
        margin-top: -6px;
    ">Smart Attendance System</div>
    <div style="width:80px;height:3px;background:linear-gradient(90deg,#6b3a1f,#e8a045,#6b3a1f);margin:0.6rem auto 0 auto;border-radius:2px;"></div>
</div>
''', unsafe_allow_html=True)

# ── Top nav bar (HTML visual only — decorative) ────────────────────────────────
nav_html = '<div class="topnav"><div><div class="topnav-title">🎯 FaceAttend</div><div class="topnav-subtitle">Smart Attendance System</div></div><div class="nav-links">'
for p in pages:
    active = "active" if st.session_state.page == p else ""
    nav_html += f'<span class="nav-btn {active}">{p}</span>'
nav_html += '</div></div>'
st.markdown(nav_html, unsafe_allow_html=True)

# ── Stats + clickable nav buttons (single row, no duplication) ─────────────────
c1, c2, c3, c4, c5 = st.columns([2.2, 1.4, 1.4, 1.4, 1.4])
with c1:
    st.markdown(f'<p style="margin:0.5rem 0;font-weight:700;color:#3d1f0a;">👥 {known_count} Registered &nbsp;|&nbsp; ✅ {len(today_df)} Present Today</p>', unsafe_allow_html=True)
with c2:
    if st.button("📸 Live Detection", use_container_width=True, key="nav1"):
        st.session_state.page = "📸 Live Detection"
        st.rerun()
with c3:
    if st.button("➕ Register Face", use_container_width=True, key="nav2"):
        st.session_state.page = "➕ Register Face"
        st.rerun()
with c4:
    if st.button("📋 Attendance Log", use_container_width=True, key="nav3"):
        st.session_state.page = "📋 Attendance Log"
        st.rerun()
with c5:
    if st.button("👥 Manage Faces", use_container_width=True, key="nav4"):
        st.session_state.page = "👥 Manage Faces"
        st.rerun()

st.markdown("---")
page = st.session_state.page


# ── Helper ─────────────────────────────────────────────────────────────────────
def run_detection(frame_rgb: np.ndarray):
    with st.spinner("🔍 Scanning faces, please wait..."):
        annotated, detected = recognize_faces(frame_rgb)
    st.image(annotated, channels="RGB", use_container_width=True)
    if detected:
        st.markdown("**Detected:**")
        for n in detected:
            badge = "badge-present" if n != "Unknown" else "badge-unknown"
            st.markdown(f'<span class="status-badge {badge}">{n}</span>', unsafe_allow_html=True)
    else:
        st.warning("No faces detected in the image.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Live Detection
# ══════════════════════════════════════════════════════════════════════════════
if page == "📸 Live Detection":
    st.markdown('<p class="page-title">📸 Live Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">Upload an image or use webcam to mark attendance</p>', unsafe_allow_html=True)

    # Metric cards
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-num">{known_count}</div>
            <div class="metric-label">Registered</div>
        </div>
        <div class="metric-card">
            <div class="metric-num">{len(today_df)}</div>
            <div class="metric-label">Present Today</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="info-box">ℹ️ First scan may take 20–40 seconds while the recognition model loads. Subsequent scans are fast.</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Webcam"])
    with tab1:
        st.markdown('''<div class="info-box">📌 Take a photo using your phone or camera app, then upload it here to mark attendance.</div>''', unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop an image to scan", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            run_detection(np.array(img))
    with tab2:
        st.markdown('''<div class="info-box">
        📷 <strong>Steps to use webcam:</strong><br><br>
        1️⃣ Click the <strong>camera icon</strong> button below<br>
        2️⃣ Your browser will ask for camera permission — click <strong>Allow</strong><br>
        3️⃣ Once the camera opens, click <strong>Take Photo</strong><br>
        4️⃣ The image will be scanned automatically<br><br>
        ⚠️ If camera does not open: try on <strong>Chrome browser</strong> and make sure you are on <strong>HTTPS</strong>
        </div>''', unsafe_allow_html=True)
        cam_img = st.camera_input(" ", label_visibility="collapsed")
        if cam_img:
            img = Image.open(cam_img).convert("RGB")
            run_detection(np.array(img))

    st.markdown(FOOTER, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Register Face
# ══════════════════════════════════════════════════════════════════════════════
elif page == "➕ Register Face":
    st.markdown('<p class="page-title">➕ Register a Face</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">Add a new person to the recognition database</p>', unsafe_allow_html=True)

    name_input = st.text_input("Full Name", placeholder="e.g. Priya Sharma")
    tab1, tab2 = st.tabs(["📁 Upload Photo", "📷 Capture from Webcam"])

    with tab1:
        uploaded = st.file_uploader("Clear front-facing solo photo", type=["jpg", "jpeg", "png"])
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
                    st.error("No face detected. Improve lighting and face the camera directly.")
                elif result == "multiple_faces":
                    st.error("Multiple faces detected. Only one person at a time.")
                else:
                    st.error(f"Error: {result}")

    if not name_input:
        st.markdown('<p class="upload-hint">⚠ Enter a name above before registering.</p>', unsafe_allow_html=True)

    st.markdown(FOOTER, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Attendance Log
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Attendance Log":
    st.markdown('<p class="page-title">📋 Attendance Log</p>', unsafe_allow_html=True)
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

    st.markdown(FOOTER, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Manage Faces
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Manage Faces":
    st.markdown('<p class="page-title">👥 Manage Faces</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">View and remove individuals from the database</p>', unsafe_allow_html=True)

    # Read names ONLY from the pickle database — not from image files
    _, db_names = load_known_faces()
    names = sorted(set(db_names))

    if not names:
        st.info("No faces registered yet. Go to **Register Face** to add someone.")
    else:
        for name in names:
            safe = name.replace(" ", "_")
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                st.markdown(f'<div class="face-box">👤 <strong>{name}</strong></div>', unsafe_allow_html=True)
            with col2:
                # Try to show photo if it exists
                faces_dir = "known_faces"
                try:
                    all_files = [f for f in os.listdir(faces_dir) if f.endswith((".jpg",".jpeg",".png"))]
                    imgs = [f for f in all_files if f.startswith(safe + "_") or f.startswith(safe)]
                    if imgs:
                        st.image(Image.open(os.path.join(faces_dir, imgs[0])), width=60)
                except Exception:
                    pass
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑 Remove", key=f"del_{safe}"):
                    remove_person(name)
                    st.success(f"✅ Removed {name} from database.")
                    st.session_state.known_encodings, st.session_state.known_names = load_known_faces()
                    st.rerun()

    st.markdown(FOOTER, unsafe_allow_html=True)
