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

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def _extract_histogram(gray_face: np.ndarray) -> np.ndarray:
    """Extract a normalized LBP-style histogram from a grayscale face crop."""
    resized = cv2.resize(gray_face, (100, 100))
    # Compute histogram of pixel intensities as simple face descriptor
    hist = cv2.calcHist([resized], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _detect_faces(frame_rgb: np.ndarray):
    """Return grayscale image and face bounding boxes."""
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    return gray, faces


def load_known_faces():
    """Return (encodings_list, names_list) for sidebar count."""
    if not os.path.exists(ENCODINGS_FILE):
        return [], []
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    names = sorted(set(data.get("names", [])))
    return list(range(len(names))), names


def _load_encodings_db():
    """Load saved face encodings database."""
    if not os.path.exists(ENCODINGS_FILE):
        return [], []
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    return data.get("encodings", []), data.get("names", [])


def _save_encodings_db(encodings, names):
    """Save face encodings database."""
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)


def save_known_face(frame_rgb: np.ndarray, name: str) -> str:
    """Register a face. Returns 'success' | 'no_face' | 'multiple_faces'."""
    gray, faces = _detect_faces(frame_rgb)
    if len(faces) == 0:
        return "no_face"
    if len(faces) > 1:
        return "multiple_faces"

    # Save the image
    safe_name = name.strip().replace(" ", "_")
    existing = [f for f in os.listdir(KNOWN_FACES_DIR)
                if f.startswith(safe_name + "_") and f.endswith(".jpg")]
    idx = len(existing)
    path = os.path.join(KNOWN_FACES_DIR, f"{safe_name}_{idx}.jpg")
    Image.fromarray(frame_rgb).save(path)

    # Extract histogram encoding from the face
    x, y, w, h = faces[0]
    face_crop = gray[y:y+h, x:x+w]
    encoding = _extract_histogram(face_crop)

    # Append to database
    encodings, names = _load_encodings_db()
    encodings.append(encoding)
    names.append(name.strip())
    _save_encodings_db(encodings, names)

    return "success"


def recognize_faces(frame_rgb: np.ndarray):
    """Recognize faces and return (annotated_rgb, list_of_names)."""
    annotated = frame_rgb.copy()
    detected_names = []

    gray, faces = _detect_faces(frame_rgb)
    encodings, names = _load_encodings_db()

    for (x, y, w, h) in faces:
        name = "Unknown"
        color = (255, 80, 80)

        if encodings:
            face_crop = gray[y:y+h, x:x+w]
            query_enc = _extract_histogram(face_crop)

            # Compare with all known encodings using correlation
            best_score = -1
            best_idx = -1
            for i, enc in enumerate(encodings):
                score = cv2.compareHist(
                    query_enc.reshape(-1, 1).astype(np.float32),
                    enc.reshape(-1, 1).astype(np.float32),
                    cv2.HISTCMP_CORREL
                )
                if score > best_score:
                    best_score = score
                    best_idx = i

            # Threshold: correlation > 0.7 = match
            if best_score > 0.7 and best_idx >= 0:
                name = names[best_idx]
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


def remove_person(name: str):
    """Remove all encodings for a person from the pickle database."""
    encodings, names = _load_encodings_db()
    if not names:
        return

    # Filter out all entries matching this name
    filtered = [(e, n) for e, n in zip(encodings, names) if n != name]

    if filtered:
        new_encodings, new_names = zip(*filtered)
        _save_encodings_db(list(new_encodings), list(new_names))
    else:
        # No faces left — save empty database
        _save_encodings_db([], [])

    # Remove attendance records for this person
    try:
        if os.path.exists(ATTENDANCE_FILE):
            df = pd.read_csv(ATTENDANCE_FILE)
            df = df[df["Name"] != name]
            df.to_csv(ATTENDANCE_FILE, index=False)
    except Exception:
        pass

    # Also try to remove image files (works locally, silently fails on cloud)
    faces_dir = KNOWN_FACES_DIR
    safe = name.strip().replace(" ", "_")
    try:
        all_files = [f for f in os.listdir(faces_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
        for f in all_files:
            if f.startswith(safe + "_") or f.startswith(safe):
                try:
                    os.remove(os.path.join(faces_dir, f))
                except Exception:
                    pass
    except Exception:
        pass
