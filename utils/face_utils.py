import io
import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from PIL import Image
from pymongo import MongoClient

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

_client = None

def get_db():
    global _client
    if _client is None:
        _client = MongoClient(os.environ["MONGO_URI"], serverSelectionTimeoutMS=8000)
    return _client["faceattend"]

# ── Internal helpers ──
def _load_encodings_db():
    docs = list(get_db().faces.find({}, {"name": 1, "encoding": 1}))
    encodings = [np.array(d["encoding"], dtype=np.float32) for d in docs]
    names = [d["name"] for d in docs]
    return encodings, names

def _detect_faces(frame_rgb):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    return gray, faces

def _extract_histogram(gray_face):
    resized = cv2.resize(gray_face, (100, 100))
    hist = cv2.calcHist([resized], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# ── Public API ──
def load_known_faces():
    _, names = _load_encodings_db()
    unique = sorted(set(names))
    return list(range(len(unique))), unique

def get_face_image(name):
    doc = get_db().faces.find_one({"name": name}, {"image": 1})
    return bytes(doc["image"]) if doc and doc.get("image") else None

def save_known_face(frame_rgb, name):
    gray, faces = _detect_faces(frame_rgb)
    if len(faces) == 0:
        return "no_face"
    if len(faces) > 1:
        return "multiple_faces"

    x, y, w, h = faces[0]
    encoding = _extract_histogram(gray[y:y+h, x:x+w])

    img = Image.fromarray(frame_rgb)
    img.thumbnail((300, 300))
    buf = io.BytesIO()
    img.save(buf, "JPEG")

    get_db().faces.insert_one({
        "name": name.strip(),
        "encoding": [float(v) for v in encoding],
        "image": buf.getvalue(),
    })
    return "success"

def recognize_faces(frame_rgb):
    annotated = frame_rgb.copy()
    detected_names = []
    gray, faces = _detect_faces(frame_rgb)
    encodings, names = _load_encodings_db()

    for (x, y, w, h) in faces:
        name = "Unknown"
        color = (255, 80, 80)
        if encodings:
            query_enc = _extract_histogram(gray[y:y+h, x:x+w])
            best_score, best_idx = -1, -1
            for i, enc in enumerate(encodings):
                score = cv2.compareHist(
                    query_enc.reshape(-1, 1).astype(np.float32),
                    enc.reshape(-1, 1).astype(np.float32),
                    cv2.HISTCMP_CORREL,
                )
                if score > best_score:
                    best_score, best_idx = score, i
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

def mark_attendance(name):
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    # upsert = one row per person per day, never duplicates
    get_db().attendance.update_one(
        {"Name": name, "Date": today},
        {"$setOnInsert": {"Timestamp": timestamp}},
        upsert=True,
    )

def load_attendance(today_only=False, days=None):
    rows = list(get_db().attendance.find({}, {"_id": 0, "Name": 1, "Date": 1, "Timestamp": 1}))
    df = pd.DataFrame(rows, columns=["Name", "Date", "Timestamp"])
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

def remove_person(name):
    db = get_db()
    db.faces.delete_many({"name": name})
    db.attendance.delete_many({"Name": name})

def clean_orphan_attendance():
    try:
        _, registered = _load_encodings_db()
        get_db().attendance.delete_many({"Name": {"$nin": list(set(registered))}})
    except Exception:
        pass