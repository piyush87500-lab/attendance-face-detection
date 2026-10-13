import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
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

# ── Global CSS (same as before) ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;900&family=Lato:wght@300;400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
    background-color: #f5e6d0;
    color: #2c1a0e;
}
.stApp { background-color: #f5e6d0; }

.topnav {
    display: flex; align-items: center; justify-content: space-between;
    background: linear-gradient(90deg, #6b3a1f, #a0522d, #6b3a1f);
    padding: 0.8rem 2rem; border-radius: 0 0 16px 16px;
    margin-bottom: 1.8rem; box-shadow: 0 4px 18px rgba(107,58,31,0.18);
}
.topnav-title {
    font-family: 'Playfair Display', serif; font-size: 1.6rem;
    font-weight: 900; color: #fdf0e0; letter-spacing: 1px;
}
.topnav-subtitle {
    font-size: 0.7rem; color: #d4a96a; letter-spacing: 2px;
    text-transform: uppercase; margin-top: -2px;
}
.nav-btn {
    background: rgba(255,255,255,0.12); border: 1px solid rgba(212,169,106,0.4);
    color: #fdf0e0 !important; font-family: 'Lato', sans-serif;
    font-size: 0.82rem; font-weight: 700; padding: 0.45rem 1rem;
    border-radius: 8px; cursor: pointer; text-decoration: none;
    transition: background 0.2s; white-space: nowrap;
}
.nav-btn:hover { background: rgba(255,255,255,0.22); }
.nav-btn.active { background: #d4a96a; color: #2c1a0e !important; border-color: #d4a96a; }

.page-title {
    font-family: 'Playfair Display', serif; font-size: 2rem;
    font-weight: 900; color: #3d1f0a; margin-bottom: 0.2rem;
}
.section-header {
    font-size: 0.72rem; letter-spacing: 3px; text-transform: uppercase;
    color: #9a6b3a; margin-bottom: 1.2rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid #d4a96a;
}
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    flex: 1; background: #fff8f0; border: 1px solid #d4a96a;
    border-radius: 12px; padding: 1rem 1.2rem; text-align: center;
    box-shadow: 0 2px 8px rgba(107,58,31,0.07);
}
.metric-num {
    font-family: 'Playfair Display', serif; font-size: 2.2rem;
    font-weight: 900; color: #6b3a1f;
}
.metric-label { font-size: 0.72rem; color: #9a6b3a; text-transform: uppercase; letter-spacing: 2px; }
.status-badge {
    display: inline-block; padding: 5px 16px; border-radius: 20px;
    font-size: 0.8rem; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; margin: 4px;
}
.badge-present { background: #e8f5e9; color: #2e7d32; border: 1px solid #66bb6a; }
.badge-unknown { background: #fdecea; color: #c62828; border: 1px solid #ef5350; }

.stButton > button {
    background: linear-gradient(135deg, #6b3a1f, #a0522d);
    color: #fdf0e0 !important; font-family: 'Lato', sans-serif;
    font-weight: 700; font-size: 0.88rem; letter-spacing: 0.5px;
    border: none; border-radius: 8px; padding: 0.55rem 1.4rem; transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

.stTextInput > div > div > input {
    background: #fff8f0; border: 1.5px solid #d4a96a;
    border-radius: 8px; color: #2c1a0e; font-family: 'Lato', sans-serif;
}
.stSelectbox > div > div { background: #fff8f0; border: 1.5px solid #d4a96a; color: #2c1a0e; }
[data-testid="stFileUploader"] {
    background: #fff8f0; border: 2px dashed #d4a96a; border-radius: 10px; padding: 1rem;
}
.stTabs [data-baseweb="tab-list"] { background: #eddec8; border-radius: 10px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #5c3d1e; font-weight: 700; border-radius: 8px; }
.stTabs [aria-selected="true"] { background: #6b3a1f !important; color: #fdf0e0 !important; }
.face-box {
    border: 1.5px solid #d4a96a; border-radius: 10px; padding: 0.8rem 1.1rem;
    background: #fff8f0; font-size: 0.88rem; color: #2c1a0e; margin-bottom: 0.5rem;
}
.info-box {
    background: #fff3e0; border: 1px solid #ffb74d; border-radius: 8px;
    padding: 0.9rem 1.1rem; font-size: 0.82rem; color: #5c3d1e; margin-bottom: 1rem;
}
[data-testid="stDataFrame"] { border: 1px solid #d4a96a; border-radius: 10px; }
div[data-testid="stSidebar"] { display: none; }
[data-testid="collapsedControl"] { display: none; }
.topnav { display: none; }
.upload-hint { font-family: 'Lato', sans-serif; font-size: 0.75rem; color: #9a6b3a; margin-top: 0.4rem; }
label { color: #5c3d1e !important; font-weight: 600; }

/* ── Login page specific ── */
.login-card {
    background: #fff8f0; border: 1.5px solid #d4a96a;
    border-radius: 16px; padding: 2.5rem 2rem;
    max-width: 420px; margin: 0 auto;
    box-shadow: 0 8px 32px rgba(107,58,31,0.12);
}
.login-title {
    font-family: 'Playfair Display', serif; font-size: 2rem;
    font-weight: 900; color: #3d1f0a; text-align: center; margin-bottom: 0.3rem;
}
.login-subtitle { color: #9a6b3a; font-size: 0.82rem; text-align: center; margin-bottom: 2rem; }
.role-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;
}
.role-admin { background: #fff3e0; color: #6b3a1f; border: 1px solid #d4a96a; }
.role-student { background: #e8f5e9; color: #2e7d32; border: 1px solid #66bb6a; }
.user-info-bar {
    background: #fff8f0; border: 1px solid #d4a96a; border-radius: 10px;
    padding: 0.6rem 1rem; margin-bottom: 1rem;
    display: flex; align-items: center; justify-content: space-between;
    font-size: 0.82rem; color: #5c3d1e;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# AUTH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")


def load_users():
    """Load users from users.json."""
    if not os.path.exists(USERS_FILE):
        default = {
            "admin": {"password": "admin123", "role": "admin", "display_name": "Administrator"}
        }
        with open(USERS_FILE, "w") as f:
            json.dump(default, f, indent=2)
        return default
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def save_users(users):
    """Save users dict to users.json."""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def check_login(username, password):
    """Validate credentials. Returns (True, user_data) or (False, None)."""
    users = load_users()
    username = username.strip().lower()
    if username in users and users[username]["password"] == password:
        return True, users[username]
    return False, None


def signup_student(username, password, confirm_password, display_name):
    """
    Register a new student account.
    Returns: 'success' | 'username_taken' | 'password_mismatch' |
             'weak_password' | 'empty_fields' | 'admin_reserved'
    """
    username = username.strip().lower()
    display_name = display_name.strip()

    if not username or not password or not display_name:
        return "empty_fields"
    if username == "admin":
        return "admin_reserved"
    if password != confirm_password:
        return "password_mismatch"
    if len(password) < 6:
        return "weak_password"

    users = load_users()
    if username in users:
        return "username_taken"

    users[username] = {
        "password": password,
        "role": "student",
        "display_name": display_name
    }
    save_users(users)
    return "success"


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.username = None
    st.session_state.display_name = None

if "page" not in st.session_state:
    st.session_state.page = "📸 Live Detection"


# ══════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

def show_login():
    # App heading
    st.markdown('''
    <div style="text-align:center; padding: 2.5rem 0 1.5rem 0;">
        <span style="
            font-family: Playfair Display, Georgia, serif;
            font-size: 3.2rem; font-weight: 900; letter-spacing: 3px;
            background: linear-gradient(135deg, #6b3a1f, #c0622a, #e8a045, #c0622a, #6b3a1f);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; display: inline-block;
        ">🎯 FaceAttend</span>
        <div style="font-family:Lato,sans-serif;font-size:0.8rem;letter-spacing:6px;
            text-transform:uppercase;color:#9a6b3a;margin-top:-4px;">
            Smart Attendance System
        </div>
        <div style="width:80px;height:3px;background:linear-gradient(90deg,#6b3a1f,#e8a045,#6b3a1f);
            margin:0.6rem auto 0 auto;border-radius:2px;"></div>
    </div>
    ''', unsafe_allow_html=True)

    # Tab styling
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        background: #eddec8; border-radius: 14px;
        padding: 6px; gap: 8px;
        box-shadow: 0 2px 8px rgba(107,58,31,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; color: #5c3d1e;
        font-weight: 700; font-size: 1rem;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        letter-spacing: 0.5px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6b3a1f, #a0522d) !important;
        color: #fdf0e0 !important;
        box-shadow: 0 4px 12px rgba(107,58,31,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([0.5, 3, 0.5])
    with col_c:
        # ── Login / Signup Tabs ────────────────────────────────────────────────
        tab_login, tab_signup = st.tabs(["🔑  Login", "📝  Student Sign Up"])

        # ── LOGIN TAB ──────────────────────────────────────────────────────────
        with tab_login:
            st.markdown('<p class="login-title" style="margin-top:1rem;">Welcome Back</p>', unsafe_allow_html=True)
            st.markdown('<p class="login-subtitle">Sign in to access your dashboard</p>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            username = st.text_input("👤  Username", placeholder="Enter your username", key="login_user")
            st.markdown("<br>", unsafe_allow_html=True)
            password = st.text_input("🔒  Password", type="password", placeholder="Enter your password", key="login_pass")
            st.markdown("<br>", unsafe_allow_html=True)
            role_sel = st.selectbox("🎭  Login as", ["Student", "Admin"], key="login_role")
            st.markdown("<br><br>", unsafe_allow_html=True)

            if st.button("🚀  Login to Dashboard", use_container_width=True, key="btn_login"):
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    ok, user_data = check_login(username, password)
                    if ok:
                        actual_role = user_data["role"]
                        selected_role = role_sel.lower()
                        if actual_role != selected_role:
                            st.error(f"❌ This account is not registered as a {role_sel}.")
                        else:
                            st.session_state.logged_in = True
                            st.session_state.role = actual_role
                            st.session_state.username = username.strip().lower()
                            st.session_state.display_name = user_data["display_name"]
                            if actual_role == "admin":
                                st.session_state.page = "📸 Live Detection"
                            else:
                                st.session_state.page = "➕ Register Face"
                            st.rerun()
                    else:
                        st.error("❌ Invalid username or password. Please try again.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style="padding:0.9rem;background:#fff3e0;border-radius:10px;
                border:1px solid #d4a96a;font-size:0.78rem;color:#6b3a1f;text-align:center;">
                🎓 New student? Click the <strong>Student Sign Up</strong> tab above to create your account.
            </div>
            """, unsafe_allow_html=True)

        # ── SIGNUP TAB ─────────────────────────────────────────────────────────
        with tab_signup:
            st.markdown('<p class="login-title" style="margin-top:1rem;">Create Account</p>', unsafe_allow_html=True)
            st.markdown('<p class="login-subtitle">Register as a new student — free & instant</p>', unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box">
            📌 After signing up, you can <strong>login</strong> and then
            <strong>register your face</strong> so the admin can mark your attendance automatically.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            su_display = st.text_input("👤  Full Name", placeholder="e.g. Rahul Sharma", key="su_display")
            st.markdown("<br>", unsafe_allow_html=True)
            su_user    = st.text_input("🆔  Username", placeholder="Choose a username (no spaces)", key="su_user")
            st.markdown("<br>", unsafe_allow_html=True)
            su_pass    = st.text_input("🔒  Password", type="password", placeholder="Min. 6 characters", key="su_pass")
            st.markdown("<br>", unsafe_allow_html=True)
            su_confirm = st.text_input("🔒  Confirm Password", type="password", placeholder="Re-enter password", key="su_confirm")
            st.markdown("<br><br>", unsafe_allow_html=True)

            if st.button("✅  Create Student Account", use_container_width=True, key="btn_signup"):
                result = signup_student(su_user, su_pass, su_confirm, su_display)
                if result == "success":
                    st.success(f"🎉 Account created for **{su_display}**! You can now login.")
                    st.balloons()
                elif result == "empty_fields":
                    st.error("❌ Please fill in all fields.")
                elif result == "admin_reserved":
                    st.error("❌ Username 'admin' is reserved. Please choose another.")
                elif result == "username_taken":
                    st.error("❌ This username is already taken. Please choose another.")
                elif result == "password_mismatch":
                    st.error("❌ Passwords do not match. Please try again.")
                elif result == "weak_password":
                    st.error("❌ Password must be at least 6 characters long.")

    st.markdown(FOOTER, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SHOW LOGIN IF NOT LOGGED IN
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.logged_in:
    show_login()
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# LOGGED IN — Load data
# ══════════════════════════════════════════════════════════════════════════════

if "known_encodings" not in st.session_state:
    clean_orphan_attendance()
    st.session_state.known_encodings, st.session_state.known_names = load_known_faces()

known_count = len(set(st.session_state.known_names))
today_df = load_attendance(today_only=True)
role = st.session_state.role
display_name = st.session_state.display_name


# ══════════════════════════════════════════════════════════════════════════════
# APP HEADING
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('''
<div style="text-align:center; padding: 1.5rem 0 0.5rem 0;">
    <span style="
        font-family: Playfair Display, Georgia, serif;
        font-size: 3.2rem; font-weight: 900; letter-spacing: 3px;
        background: linear-gradient(135deg, #6b3a1f, #c0622a, #e8a045, #c0622a, #6b3a1f);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; display: inline-block;
    ">🎯 FaceAttend</span>
    <div style="font-family:Lato,sans-serif;font-size:0.8rem;letter-spacing:6px;
        text-transform:uppercase;color:#9a6b3a;margin-top:-6px;">
        Smart Attendance System
    </div>
    <div style="width:80px;height:3px;background:linear-gradient(90deg,#6b3a1f,#e8a045,#6b3a1f);
        margin:0.6rem auto 0 auto;border-radius:2px;"></div>
</div>
''', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# NAV BAR — different for admin vs student
# ══════════════════════════════════════════════════════════════════════════════

if role == "admin":
    pages = ["📸 Live Detection", "➕ Register Face", "📋 Attendance Log", "👥 Manage Faces"]
else:
    pages = ["➕ Register Face", "📋 My Attendance"]

# Ensure page is valid for current role
if st.session_state.page not in pages:
    st.session_state.page = pages[0]

# User info bar + logout
info_col, *nav_cols, logout_col = st.columns([2.5] + [1.3] * len(pages) + [1.2])

with info_col:
    role_badge = "role-admin" if role == "admin" else "role-student"
    role_label = "Admin" if role == "admin" else "Student"
    st.markdown(
        f'<div style="padding:0.4rem 0;font-size:0.82rem;color:#3d1f0a;font-weight:700;">'
        f'👋 {display_name} &nbsp;<span class="role-badge {role_badge}">{role_label}</span></div>',
        unsafe_allow_html=True
    )

for i, (col, pg) in enumerate(zip(nav_cols, pages)):
    with col:
        if st.button(pg, use_container_width=True, key=f"nav{i}"):
            st.session_state.page = pg
            st.rerun()

with logout_col:
    if st.button("🚪 Logout", use_container_width=True, key="logout"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.username = None
        st.session_state.display_name = None
        st.session_state.page = "📸 Live Detection"
        st.rerun()

st.markdown("---")
page = st.session_state.page


# ══════════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════════

def run_detection(frame_rgb):
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
# ADMIN PAGES
# ══════════════════════════════════════════════════════════════════════════════

if role == "admin":

    # ── Live Detection ─────────────────────────────────────────────────────────
    if page == "📸 Live Detection":
        st.markdown('<p class="page-title">📸 Live Detection</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-header">Upload an image or use webcam to mark attendance</p>', unsafe_allow_html=True)

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

        tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Webcam"])
        with tab1:
            st.markdown('<div class="info-box">📌 Take a photo using your phone or camera app, then upload it here to mark attendance.</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader("Drop an image to scan", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            if uploaded:
                img = Image.open(uploaded).convert("RGB")
                run_detection(np.array(img))
        with tab2:
            st.markdown('''<div class="info-box">
            📷 <strong>Steps to use webcam:</strong><br><br>
            1️⃣ Click the <strong>camera icon</strong> button below<br>
            2️⃣ Allow camera permission in your browser<br>
            3️⃣ Click <strong>Take Photo</strong> — it scans automatically<br><br>
            ⚠️ Use <strong>Chrome browser</strong> on <strong>HTTPS</strong> for best results.
            </div>''', unsafe_allow_html=True)
            cam_img = st.camera_input(" ", label_visibility="collapsed")
            if cam_img:
                img = Image.open(cam_img).convert("RGB")
                run_detection(np.array(img))

        st.markdown(FOOTER, unsafe_allow_html=True)

    # ── Register Face ──────────────────────────────────────────────────────────
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
                        st.error("No face detected. Try a clearer, well-lit photo.")
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
                        st.error("No face detected. Improve lighting.")
                    elif result == "multiple_faces":
                        st.error("Multiple faces detected. One person at a time.")
                    else:
                        st.error(f"Error: {result}")

        if not name_input:
            st.markdown('<p class="upload-hint">⚠ Enter a name above before registering.</p>', unsafe_allow_html=True)

        st.markdown(FOOTER, unsafe_allow_html=True)

    # ── Attendance Log ─────────────────────────────────────────────────────────
    elif page == "📋 Attendance Log":
        st.markdown('<p class="page-title">📋 Attendance Log</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-header">View and export all attendance records</p>', unsafe_allow_html=True)

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

    # ── Manage Faces ───────────────────────────────────────────────────────────
    elif page == "👥 Manage Faces":
        st.markdown('<p class="page-title">👥 Manage Faces</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-header">View and remove individuals from the database</p>', unsafe_allow_html=True)

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
                    faces_dir = "known_faces"
                    try:
                        all_files = [f for f in os.listdir(faces_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
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


# ══════════════════════════════════════════════════════════════════════════════
# STUDENT DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

elif role == "student":

    # ── Register Face (Student) ────────────────────────────────────────────────
    if page == "➕ Register Face":
        st.markdown('<p class="page-title">➕ Register Your Face</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-header">Register your face to be recognised during attendance scans</p>', unsafe_allow_html=True)

        st.markdown('''<div class="info-box">
        📌 <strong>How it works:</strong> Register your face here. When the admin runs a scan, your face will be automatically recognised and your attendance will be marked.<br><br>
        ⚠️ Please register with your <strong>exact display name</strong> so attendance links correctly to your account.
        </div>''', unsafe_allow_html=True)

        # Pre-fill name with their display name
        name_input = st.text_input("Full Name", value=display_name, placeholder="e.g. Piyush Choudhary")

        tab1, tab2 = st.tabs(["📁 Upload Photo", "📷 Capture from Webcam"])

        with tab1:
            uploaded = st.file_uploader("Clear front-facing solo photo", type=["jpg", "jpeg", "png"])
            if uploaded and name_input:
                preview = Image.open(uploaded).convert("RGB")
                st.image(preview, width=220, caption="Preview")
                if st.button("✅ Register from Upload"):
                    result = save_known_face(np.array(preview), name_input.strip())
                    if result == "success":
                        st.success(f"✅ {name_input} registered successfully! Admin can now mark your attendance.")
                        st.session_state.known_encodings, st.session_state.known_names = load_known_faces()
                    elif result == "no_face":
                        st.error("No face detected. Try a clearer, well-lit photo.")
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
                        st.success(f"✅ {name_input} registered successfully! Admin can now mark your attendance.")
                        st.session_state.known_encodings, st.session_state.known_names = load_known_faces()
                    elif result == "no_face":
                        st.error("No face detected. Improve lighting.")
                    elif result == "multiple_faces":
                        st.error("Multiple faces detected. One person at a time.")
                    else:
                        st.error(f"Error: {result}")

        if not name_input:
            st.markdown('<p class="upload-hint">⚠ Enter your name above before registering.</p>', unsafe_allow_html=True)

        st.markdown(FOOTER, unsafe_allow_html=True)

    # ── My Attendance ──────────────────────────────────────────────────────────
    elif page == "📋 My Attendance":
        st.markdown('<p class="page-title">📋 My Attendance</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="section-header">Attendance record for {display_name}</p>', unsafe_allow_html=True)

        # Load only this student's attendance
        df_all = load_attendance()

        # Match by display_name or username
        my_df = pd.DataFrame(columns=["Name", "Date", "Timestamp"])
        if not df_all.empty:
            my_df = df_all[
                df_all["Name"].str.lower().str.strip() == display_name.lower().strip()
            ]

        # Metrics
        total_days = len(my_df)
        today_present = len(my_df[my_df["Date"] == pd.Timestamp("today").strftime("%Y-%m-%d")]) if not my_df.empty else 0

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-num">{total_days}</div>
                <div class="metric-label">Total Days Present</div>
            </div>
            <div class="metric-card">
                <div class="metric-num">{"✅" if today_present else "❌"}</div>
                <div class="metric-label">Present Today</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Filter
        filter_opt = st.selectbox("Filter by Period", [
            "Today",
            "Yesterday",
            "Last 2 Days",
            "Last Week (7 Days)",
            "Last Month (30 Days)",
            "All Records"
        ])

        if filter_opt == "Today":
            df_show = load_attendance(today_only=True)
        elif filter_opt == "Yesterday":
            df_show = load_attendance(days=2)
            if not df_show.empty:
                yesterday = (pd.Timestamp("today") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                df_show = df_show[df_show["Date"] == yesterday]
        elif filter_opt == "Last 2 Days":
            df_show = load_attendance(days=2)
        elif filter_opt == "Last Week (7 Days)":
            df_show = load_attendance(days=7)
        elif filter_opt == "Last Month (30 Days)":
            df_show = load_attendance(days=30)
        else:
            df_show = load_attendance()

        # Filter to this student only
        if not df_show.empty:
            df_show = df_show[
                df_show["Name"].str.lower().str.strip() == display_name.lower().strip()
            ]

        if df_show.empty:
            st.info(f"No attendance records found for **{display_name}** in this period.")
            st.markdown('<div class="info-box">📌 Your attendance is marked automatically when your face is detected in the Live Detection scan.</div>', unsafe_allow_html=True)
        else:
            st.dataframe(
                df_show.sort_values("Timestamp", ascending=False),
                use_container_width=True, hide_index=True
            )

        st.markdown(FOOTER, unsafe_allow_html=True)
