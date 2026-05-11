# 👁️ FaceAttend — AI-Powered Attendance System

A real-time face recognition attendance system built with **Python**, **DeepFace**, **OpenCV**, and **Streamlit**. Works on **Python 3.13** — no dlib or cmake required.

---

## ✨ Features

| Feature | Details |
|---|---|
| 📸 Face Registration | Upload photos **or** capture live from webcam |
| 👁️ Live Detection | Scan images/webcam snapshots for known faces |
| 📋 Attendance Log | CSV-based log with Name, Date, Timestamp |
| 📊 Analytics | Attendance count chart per person |
| ⬇️ Export | Download attendance as CSV anytime |
| 🗑️ Manage | View and delete registered faces |

---

## 🗂️ Project Structure

```
attendance-face-detection/
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── packages.txt             # System packages (Streamlit Cloud only)
├── .streamlit/
│   └── config.toml          # Theme & server config
├── utils/
│   ├── __init__.py
│   └── face_utils.py        # Face detection & attendance logic (DeepFace)
├── known_faces/             # Registered face images (git-ignored)
└── attendance/
    └── attendance.csv       # Auto-generated attendance log (git-ignored)
```

---

## 🚀 Local Setup (Windows)

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/attendance-face-detection.git
cd attendance-face-detection
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> ✅ No cmake, no Visual Studio Build Tools, no dlib needed — DeepFace handles everything.

### 4. Run the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

> ⚠️ **First run note:** DeepFace will download the VGG-Face model (~500MB) the first time you scan a face. This is automatic and only happens once.

---

## 🐧 Local Setup (Linux / macOS)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 📤 Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: FaceAttend attendance system"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/attendance-face-detection.git
git push -u origin main
```

---

## ☁️ Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub
2. Click **"New app"**
3. Select your repository: `attendance-face-detection`
4. Set **Main file path**: `app.py`
5. Click **Deploy** 🎉

> Streamlit Cloud reads `packages.txt` automatically to install OpenCV system dependencies.

---

## ⚙️ How It Works

```
Register Phase:
  Photo/Webcam → OpenCV Haar Cascade (face detection)
              → validated (1 face only) → saved as .jpg

Detection Phase:
  Input Image → OpenCV face_locations()
              → DeepFace.find() against known_faces/
              → mark_attendance() → append to CSV (once per day)
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web UI |
| [DeepFace](https://github.com/serengil/deepface) | Face recognition (VGG-Face model) |
| [OpenCV](https://opencv.org) | Face detection & image processing |
| [Pandas](https://pandas.pydata.org) | CSV attendance management |
| [Pillow](https://pillow.readthedocs.io) | Image I/O |

---

## 📝 Notes

- Attendance is marked **once per person per day** to avoid duplicates.
- For best accuracy, register **2–3 photos** per person in different lighting conditions.
- The DeepFace VGG-Face model (~500MB) is downloaded automatically on first use and cached locally.
- `known_faces/` and `attendance/attendance.csv` are git-ignored to protect privacy.
- Recognition tolerance can be tuned in `utils/face_utils.py` via the `DeepFace.find()` call.