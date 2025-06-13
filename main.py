from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
import numpy as np
import cv2
import face_recognition
import datetime
from typing import List, Dict, Optional, Any
import asyncio

app = FastAPI()

# In-memory storage for users, attendance logs, and advanced trainer data
users: Dict[str, Dict[str, Any]] = {}
attendance_logs: List[Dict[str, Any]] = []
trainer_shifts: Dict[str, List[Dict[str, str]]] = {}  # Schedule: list of dicts with "start" and "end" ISO datetime strings
trainer_presence: Dict[str, Dict[str, datetime.datetime]] = {}  # Tracks check-in/out times per day

# Extended tracking data structures
trainer_zone_time: Dict[str, Dict[str, float]] = {}  # user_id -> zone_name -> cumulative seconds
trainer_idle_time: Dict[str, Dict[str, float]] = {}  # user_id -> day -> idle seconds
trainer_interaction_time: Dict[str, Dict[str, float]] = {}  # user_id -> day -> seconds with members
trainer_simultaneous_members: Dict[str, Dict[str, int]] = {}  # user_id -> timestamp -> count of members interacting
trainer_greet_counts: Dict[str, int] = {}  # user_id -> greeting counts
trainer_emotion_log: Dict[str, List[Dict[str, Any]]] = {}  # user_id -> list of emotion analysis entries


# User model
class User(BaseModel):
    name: str
    photo: str  # Base64 encoded JPEG/PNG image string
    membership_status: str  # active, expired, blacklisted (for members)
    expiry_date: str  # YYYY-MM-DD format
    role: Optional[str] = None  # e.g. "trainer", None for members
    schedule: Optional[List[Dict[str, str]]] = None  # List of shifts: [{"start": ISO_datetime, "end": ISO_datetime}]


@app.post("/upload_user/")
async def upload_user(user: User):
    # Decode base64 image
    try:
        image_data = base64.b64decode(user.photo)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 photo data.")

    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode the photo.")

    face_encodings = face_recognition.face_encodings(img)
    if not face_encodings:
        raise HTTPException(status_code=400, detail="No face found in the image.")

    embedding = face_encodings[0].tolist()

    user_id = user.name.lower().replace(" ", "_")
    users[user_id] = {
        "name": user.name,
        "embedding": embedding,
        "membership_status": user.membership_status,
        "expiry_date": user.expiry_date,
        "role": user.role,
        "schedule": user.schedule,
    }

    if user.role == "trainer" and user.schedule:
        trainer_shifts[user_id] = user.schedule

    return {"message": f"User '{user.name}' uploaded successfully."}


@app.post("/process_frame/")
async def process_frame(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)

    now = datetime.datetime.utcnow()

    recognized_users = []

    for face_encoding in face_encodings:
        for user_id, user_data in users.items():
            stored_embedding = np.array(user_data["embedding"])
            # Calculate cosine similarity
            similarity = np.dot(stored_embedding, face_encoding) / (
                np.linalg.norm(stored_embedding) * np.linalg.norm(face_encoding) + 1e-10
            )
            # Threshold for cosine similarity (typical threshold ~0.6 to 0.7)
            if similarity > 0.65:
                # Log attendance
                log_entry = {
                    "name": user_data["name"],
                    "timestamp": now.isoformat(),
                    "user_type": "trainer" if user_data["role"] else "member",
                    "membership_status": user_data["membership_status"],
                }

                # Handle check-in/check-out logic for trainers
                if user_data["role"] == "trainer":
                    manage_trainer_presence(user_id, now)

                    # Attendance compliance check
                    compliance_alert = check_attendance_compliance(user_id, now)
                    if compliance_alert:
                        log_entry["compliance_alert"] = compliance_alert

                attendance_logs.append(log_entry)
                recognized_users.append(user_data["name"])

                # Alert for members with expired or blacklisted status
                if user_data["membership_status"] in ["expired", "blacklisted"]:
                    # This is a placeholder for actual alert mechanism like WebSocket broadcast
                    print(f"Alert: Member {user_data['name']} has {user_data['membership_status']} status detected.")

                break

    return {"message": "Frame processed.", "recognized_users": recognized_users}


def manage_trainer_presence(user_id: str, current_time: datetime.datetime):
    """
    Manage check-in and check-out times for trainers.
    Logic: If no check-in exists today, mark check-in.
    If check-in exists but no check-out, update check-out.
    """
    today_str = current_time.date().isoformat()
    if user_id not in trainer_presence:
        trainer_presence[user_id] = {}

    presence = trainer_presence[user_id]

    check_in_key = f"check_in_{today_str}"
    check_out_key = f"check_out_{today_str}"

    # If no check-in today, mark check-in
    if check_in_key not in presence:
        presence[check_in_key] = current_time
    else:
        # Update check-out time if later than check-in
        if check_out_key not in presence or current_time > presence[check_out_key]:
            presence[check_out_key] = current_time


def check_attendance_compliance(user_id: str, current_time: datetime.datetime) -> Optional[str]:
    """
    Check if the trainer's presence aligns with their shift schedule.
    Returns alert message if late or absent.
    """
    shifts = trainer_shifts.get(user_id)
    if not shifts:
        return None

    # Find today's shift(s)
    today_date_str = current_time.date().isoformat()

    for shift in shifts:
        # Parse ISO datetime strings from shift
        try:
            shift_start = datetime.datetime.fromisoformat(shift["start"])
            shift_end = datetime.datetime.fromisoformat(shift["end"])
        except Exception:
            continue

        # Only consider shifts for today
        if shift_start.date().isoformat() != today_date_str:
            continue

        # Get trainer's check-in time today
        presence = trainer_presence.get(user_id, {})
        check_in_time = presence.get(f"check_in_{today_date_str}")
        if not check_in_time:
            return f"Absence Alert: Trainer '{users[user_id]['name']}' has no check-in for scheduled shift today."

        # Determine lateness (more than 10 minutes after shift start)
        late_threshold = shift_start + datetime.timedelta(minutes=10)
        if check_in_time > late_threshold:
            return f"Late Arrival Alert: Trainer '{users[user_id]['name']}' checked in late at {check_in_time.isoformat()}."

    return None


@app.post("/zone_tracking/")
async def zone_tracking(data: Dict):
    """
    Receive JSON data for trainer zone tracking.
    Expected Format:
    {
      "user_id": "trainer_name",
      "zone": "weights_section",
      "timestamp": "ISO_datetime",
      "duration_seconds": float
    }
    """
    user_id = data.get("user_id")
    zone = data.get("zone")
    duration = data.get("duration_seconds")

    if not user_id or not zone or duration is None:
        raise HTTPException(status_code=400, detail="Missing parameters.")

    if user_id not in trainer_zone_time:
        trainer_zone_time[user_id] = {}

    trainer_zone_time[user_id][zone] = trainer_zone_time[user_id].get(zone, 0) + float(duration)
    return {"message": f"Tracked {duration} seconds for user {user_id} in zone {zone}."}


@app.post("/interaction_tracking/")
async def interaction_tracking(data: Dict):
    """
    Receive JSON data for trainer-member interaction time.
    Expected Format:
    {
      "user_id": "trainer_name",
      "timestamp": "ISO_datetime",
      "interaction_duration_seconds": float,
      "member_count": int
    }
    """
    user_id = data.get("user_id")
    timestamp_str = data.get("timestamp")
    interaction_duration = data.get("interaction_duration_seconds")
    member_count = data.get("member_count")

    if not user_id or not timestamp_str or interaction_duration is None or member_count is None:
        raise HTTPException(status_code=400, detail="Missing parameters.")

    try:
        timestamp = datetime.datetime.fromisoformat(timestamp_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid timestamp format.")

    day_str = timestamp.date().isoformat()

    # Track interaction time
    if user_id not in trainer_interaction_time:
        trainer_interaction_time[user_id] = {}
    trainer_interaction_time[user_id][day_str] = trainer_interaction_time[user_id].get(day_str, 0) + float(interaction_duration)

    # Track simultaneous members
    if user_id not in trainer_simultaneous_members:
        trainer_simultaneous_members[user_id] = {}
    trainer_simultaneous_members[user_id][timestamp_str] = int(member_count)

    return {"message": f"Recorded interaction for user {user_id} with {member_count} members over {interaction_duration} seconds."}


@app.post("/idle_time_tracking/")
async def idle_time_tracking(data: Dict):
    """
    Receive JSON data for idle time detection.
    Expected Format:
    {
      "user_id": "trainer_name",
      "timestamp": "ISO_datetime",
      "idle_duration_seconds": float
    }
    """
    user_id = data.get("user_id")
    timestamp_str = data.get("timestamp")
    idle_duration = data.get("idle_duration_seconds")

    if not user_id or not timestamp_str or idle_duration is None:
        raise HTTPException(status_code=400, detail="Missing parameters.")

    try:
        timestamp = datetime.datetime.fromisoformat(timestamp_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid timestamp format.")

    day_str = timestamp.date().isoformat()

    if user_id not in trainer_idle_time:
        trainer_idle_time[user_id] = {}

    trainer_idle_time[user_id][day_str] = trainer_idle_time[user_id].get(day_str, 0) + float(idle_duration)
    return {"message": f"Recorded {idle_duration} seconds idle time for user {user_id}."}


@app.post("/greet_detection/")
async def greet_detection(data: Dict):
    """
    Track how often trainers greet members.
    Expected Format:
    {
      "user_id": "trainer_name",
      "timestamp": "ISO_datetime"
    }
    """
    user_id = data.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing user_id.")

    trainer_greet_counts[user_id] = trainer_greet_counts.get(user_id, 0) + 1
    return {"message": f"Greet count incremented for {user_id}. Total greetings: {trainer_greet_counts[user_id]}"}


@app.post("/emotion_analysis/")
async def emotion_analysis(data: Dict):
    """
    Track facial emotion analysis results for trainers.
    Expected Format:
    {
      "user_id": "trainer_name",
      "timestamp": "ISO_datetime",
      "emotion": "attentive" | "distracted" | "unfriendly",
      "confidence": float 0-1
    }
    """
    user_id = data.get("user_id")
    timestamp = data.get("timestamp")
    emotion = data.get("emotion")
    confidence = data.get("confidence")

    if not all([user_id, timestamp, emotion]) or confidence is None:
        raise HTTPException(status_code=400, detail="Missing parameters.")

    if user_id not in trainer_emotion_log:
        trainer_emotion_log[user_id] = []

    trainer_emotion_log[user_id].append({
        "timestamp": timestamp,
        "emotion": emotion,
        "confidence": confidence
    })

    return {"message": f"Emotion data recorded for {user_id}."}


@app.get("/analytics/trainer_report/{user_id}")
async def trainer_report(user_id: str):
    """
    Generate a report for trainer including:
    - Total check-in hours today
    - Total idle time today
    - Interaction time today
    - Number of greetings
    - Zone engagement breakdown
    - Average emotion confidence scores
    """

    today_str = datetime.datetime.utcnow().date().isoformat()

    presence = trainer_presence.get(user_id, {})
    check_in = presence.get(f"check_in_{today_str}")
    check_out = presence.get(f"check_out_{today_str}")

    total_hours = None
    if check_in and check_out:
        total_hours = (check_out - check_in).total_seconds() / 3600

    idle_time = trainer_idle_time.get(user_id, {}).get(today_str, 0)
    interaction_time = trainer_interaction_time.get(user_id, {}).get(today_str, 0)
    greet_count = trainer_greet_counts.get(user_id, 0)
    zone_times = trainer_zone_time.get(user_id, {})

    emotions = trainer_emotion_log.get(user_id, [])
    emotion_summary = {}
    if emotions:
        counts = {}
        sums = {}
        for entry in emotions:
            emo = entry["emotion"]
            conf = entry["confidence"]
            counts[emo] = counts.get(emo, 0) + 1
            sums[emo] = sums.get(emo, 0) + conf
        emotion_summary = {emo: sums[emo]/counts[emo] for emo in counts}

    report = {
        "total_hours_present_today": total_hours,
        "idle_time_seconds_today": idle_time,
        "interaction_time_seconds_today": interaction_time,
        "greet_count": greet_count,
        "zone_time_seconds": zone_times,
        "average_emotion_confidence": emotion_summary
    }
    return report


@app.get("/attendance_logs/")
async def get_attendance_logs():
    return attendance_logs


# WebSocket for real-time alerts (placeholder implementation)
@app.websocket("/ws/alerts/")
async def websocket_alerts(websocket: WebSocket):
    await websocket.accept()
    # For demo, just send a hello message and keep connection open
    while True:
        await websocket.send_text("Real-time alert channel active.")
        await asyncio.sleep(10)


# Example of generating embeddings from a local image path
def generate_embedding(image_path: str) -> np.ndarray:
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    if face_encodings:
        return face_encodings[0]
    else:
        raise ValueError("No face found in the image.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

