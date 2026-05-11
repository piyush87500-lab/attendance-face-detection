import numpy as np
import pandas as pd
import os
import cv2
import pickle
import shutil
from datetime import datetime, timedelta
from PIL import Image

KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_DIR = "attendance"
ATTENDANCE_FILE = os.path.join(ATTENDANCE_DIR, "attendance.csv")
DB_PATH = "known_faces"  # deepface uses folder as DB

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)


def load_known_faces():
    """Return list of registered names (one per image file)."""
    names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            name = filename.rsplit("_", 1)[0].replace("_", " ")
            names.append(name)
    unique = sorted(set(names))
    # Return dummy encodings list (same length) — kept for sidebar count compat
    return list(range(len(unique))), unique


def save_known_face(frame_rgb: np.ndarray, name: str) -> str:
    """
    Detect exactly one face in frame and save it.
    Returns: 'success' | 'no_face' | 'multiple_faces'
    """
    try:
        from deepface import DeepFace
        # Use OpenCV face detector for quick validation
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return "no_face"
        if len(faces) > 1:
            return "multiple_faces"

        safe_name = name.strip().replace(" ", "_")
        existing = [f for f in os.listdir(KNOWN_FACES_DIR) if f.startswith(safe_name + "_")]
        idx = len(existing)
        path = os.path.join(KNOWN_FACES_DIR, f"{safe_name}_{idx}.jpg")
        img = Image.fromarray(frame_rgb)
        img.save(path)

        # Clear deepface's internal DB cache so it re-indexes
        cache_path = os.path.join(KNOWN_FACES_DIR, "representations_vgg_face.pkl")
        if os.path.exists(cache_path):
            os.remove(cache_path)

        return "success"
    except Exception as e:
        return f"error: {e}"


def recognize_faces(frame_rgb: np.ndarray):
    """
    Run DeepFace recognition on an image.
    Returns: (annotated_rgb, list_of_names)
    """
    from deepface import DeepFace

    annotated = frame_rgb.copy()
    detected_names = []

    # First detect faces with OpenCV for bounding boxes
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    registered = [f for f in os.listdir(KNOWN_FACES_DIR) if f.endswith((".jpg", ".jpeg", ".png"))]
    if len(registered) == 0:
        # No registered faces — just draw boxes as Unknown
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 80, 80), 2)
            cv2.rectangle(annotated, (x, y + h - 30), (x + w, y + h), (255, 80, 80), cv2.FILLED)
            cv2.putText(annotated, "Unknown", (x + 6, y + h - 8), cv2.FONT_HERSHEY_DUPLEX, 0.65, (10, 10, 20), 1)
            detected_names.append("Unknown")
        return annotated, detected_names

    for (x, y, w, h) in faces:
        name = "Unknown"
        color = (255, 80, 80)
        try:
            # Crop face region
            face_crop = frame_rgb[y:y+h, x:x+w]
            results = DeepFace.find(
                img_path=face_crop,
                db_path=DB_PATH,
                model_name="VGG-Face",
                enforce_detection=False,
                silent=True,
            )
            if results and len(results[0]) > 0 and not results[0].empty:
                best_match = results[0].iloc[0]["identity"]
                filename = os.path.basename(best_match)
                name = filename.rsplit("_", 1)[0].replace("_", " ")
                color = (127, 255, 212)
                mark_attendance(name)
        except Exception:
            pass

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(annotated, (x, y + h - 30), (x + w, y + h), color, cv2.FILLED)
        cv2.putText(annotated, name, (x + 6, y + h - 8), cv2.FONT_HERSHEY_DUPLEX, 0.65, (10, 10, 20), 1)
        detected_names.append(name)

    if len(faces) == 0:
        return annotated, []

    return annotated, detected_names


def mark_attendance(name: str):
    """Mark attendance once per person per day."""
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Timestamp"])

    already = ((df["Name"] == name) & (df["Date"] == today)).any()
    if not already:
        new_row = pd.DataFrame([{"Name": name, "Date": today, "Timestamp": timestamp}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)


def load_attendance(today_only: bool = False, days: int = None) -> pd.DataFrame:
    """Load attendance CSV with optional filters."""
    if not os.path.exists(ATTENDANCE_FILE):
        return pd.DataFrame(columns=["Name", "Date", "Timestamp"])
    df = pd.read_csv(ATTENDANCE_FILE)
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    if today_only:
        today = pd.Timestamp(datetime.now().date())
        df = df[df["Date"] == today]
    elif days:
        cutoff = pd.Timestamp((datetime.now() - timedelta(days=days)).date())
        df = df[df["Date"] >= cutoff]
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df["Timestamp"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df