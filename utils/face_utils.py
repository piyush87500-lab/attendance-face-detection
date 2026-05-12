import numpy as np
import pandas as pd
import os
import cv2
import pickle
from datetime import datetime, timedelta
from PIL import Image

KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_DIR = "attendance"
ATTENDANCE_FILE = os.path.join(ATTENDANCE_DIR, "attendance.csv")
ENCODINGS_FILE = os.path.join(KNOWN_FACES_DIR, "encodings.pkl")

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# Use OpenCV's LBPH face recognizer — pure OpenCV, no TensorFlow needed
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def _get_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    return recognizer


def _detect_face_gray(frame_rgb):
    """Detect faces and return grayscale crops + locations."""
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return gray, faces


def load_known_faces():
    """Return dummy encodings list and names list for sidebar counts."""
    if not os.path.exists(ENCODINGS_FILE):
        return [], []
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    names = sorted(set(data.get("names", [])))
    return list(range(len(names))), names


def _save_encodings(labels, names, faces_gray):
    """Train and save the LBPH recognizer."""
    if not faces_gray:
        return
    recognizer = _get_recognizer()
    recognizer.train(faces_gray, np.array(labels))
    recognizer.save(os.path.join(KNOWN_FACES_DIR, "lbph_model.yml"))
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"names": names}, f)


def save_known_face(frame_rgb: np.ndarray, name: str) -> str:
    """Register a face. Returns 'success' | 'no_face' | 'multiple_faces'."""
    gray, faces = _detect_face_gray(frame_rgb)
    if len(faces) == 0:
        return "no_face"
    if len(faces) > 1:
        return "multiple_faces"

    # Save the face image
    safe_name = name.strip().replace(" ", "_")
    existing = [f for f in os.listdir(KNOWN_FACES_DIR) if f.startswith(safe_name + "_") and f.endswith(".jpg")]
    idx = len(existing)
    path = os.path.join(KNOWN_FACES_DIR, f"{safe_name}_{idx}.jpg")
    Image.fromarray(frame_rgb).save(path)

    # Retrain recognizer with ALL registered faces
    _retrain_recognizer()
    return "success"


def _retrain_recognizer():
    """Scan known_faces/ and retrain the LBPH model from scratch."""
    all_files = [f for f in os.listdir(KNOWN_FACES_DIR) if f.endswith(".jpg")]
    if not all_files:
        return

    name_to_id = {}
    id_to_name = []
    faces_gray = []
    labels = []

    for filename in all_files:
        name = filename.rsplit("_", 1)[0].replace("_", " ")
        if name not in name_to_id:
            name_to_id[name] = len(id_to_name)
            id_to_name.append(name)
        label = name_to_id[name]

        img_path = os.path.join(KNOWN_FACES_DIR, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
        if len(detected) > 0:
            x, y, w, h = detected[0]
            face_crop = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
            faces_gray.append(face_crop)
            labels.append(label)

    if not faces_gray:
        return

    recognizer = _get_recognizer()
    recognizer.train(faces_gray, np.array(labels))
    recognizer.save(os.path.join(KNOWN_FACES_DIR, "lbph_model.yml"))

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"names": id_to_name}, f)


def recognize_faces(frame_rgb: np.ndarray):
    """Run face recognition and return (annotated_rgb, list_of_names)."""
    annotated = frame_rgb.copy()
    detected_names = []

    gray, faces = _detect_face_gray(frame_rgb)
    model_path = os.path.join(KNOWN_FACES_DIR, "lbph_model.yml")
    has_model = os.path.exists(model_path) and os.path.exists(ENCODINGS_FILE)

    recognizer = None
    id_to_name = []

    if has_model:
        recognizer = _get_recognizer()
        recognizer.read(model_path)
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        id_to_name = data.get("names", [])

    for (x, y, w, h) in faces:
        name = "Unknown"
        color = (255, 80, 80)

        if recognizer and id_to_name:
            face_crop = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
            label, confidence = recognizer.predict(face_crop)
            # LBPH: lower confidence = better match; threshold ~80
            if confidence < 80 and label < len(id_to_name):
                name = id_to_name[label]
                color = (127, 255, 212)
                mark_attendance(name)

        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(annotated, (x, y+h-30), (x+w, y+h), color, cv2.FILLED)
        cv2.putText(annotated, name, (x+6, y+h-8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (10, 10, 20), 1)
        detected_names.append(name)

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